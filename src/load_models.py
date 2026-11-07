"""Load the five models through a shared interface for embeddings and hidden states."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
from PIL import Image

from src import settings


@dataclass
class ImageEncoding:
    pooled: torch.Tensor                # The unnormalized embeddings have shape (B, D_proj).
    patch_states: list[torch.Tensor]    # Each layer supplies (B, n_patches, D_layer) states without the CLS token.


@dataclass
class TextEncoding:
    pooled: torch.Tensor                # The unnormalized embeddings have shape (B, D_proj).
    pooled_states: list[torch.Tensor]   # Each layer supplies (B, D_layer) states under standard pooling.


@contextmanager
def capture_block_outputs(blocks: nn.ModuleList, indices: Iterable[int]):
    """Yield a dictionary that collects the detached outputs as the requested blocks run."""
    captured: dict[int, torch.Tensor] = {}

    def hook_for(index: int):
        def hook(_module, _inputs, out):
            captured[index] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    handles = [blocks[i].register_forward_hook(hook_for(i)) for i in indices]
    try:
        yield captured
    finally:
        for h in handles:
            h.remove()


def pool_text_tokens(h: torch.Tensor, tokens: torch.Tensor, pooling: str, *, end: str = "argmax", pad_id: int = 0) -> torch.Tensor:
    """Pool the token states (B, T, D) into caption vectors (B, D).

    The "argmax" mode selects CLIP's highest-ID end token. The "last" mode selects
    the final position, even when it is padding. The "mean" mode averages from the
    first token through the end token when end="argmax", or through the last
    non-padding token otherwise.
    """
    rows = torch.arange(h.shape[0], device=h.device)
    if pooling == "argmax":
        return h[rows, tokens.argmax(dim=-1)]
    if pooling == "last":
        return h[:, -1, :]
    if pooling != "mean":
        raise ValueError(f"unknown pooling {pooling!r}")
    positions = torch.arange(h.shape[1], device=h.device).unsqueeze(0)  # The positions have shape (1, T).
    if end == "argmax":
        last = tokens.argmax(dim=-1)
    else:
        last = (((tokens != pad_id).long() * (positions + 1)).max(dim=-1).values - 1).clamp(min=0)
    keep = positions <= last.unsqueeze(1)  # The mask has shape (B, T).
    total = (h * keep.unsqueeze(-1).to(h.dtype)).sum(dim=1)
    return total / keep.sum(dim=1, keepdim=True).clamp(min=1).to(h.dtype)


class ModelAdapter:
    """Share the encoding and pooling logic across model-specific adapters.

    Adapter constructors supply the model, preprocessing, transformer blocks,
    pooling rules, and final text normalization and projection.
    """

    @property
    def n_image_layers(self) -> int:
        return len(self.image_blocks)

    @property
    def n_text_layers(self) -> int:
        return len(self.text_blocks)

    def proportional_depths(self, fractions: Sequence[float], side: str) -> list[int]:
        """Map the layer fractions to zero-based block indices, with 1.0 selecting the last block."""
        n = self.n_image_layers if side == "image" else self.n_text_layers
        return [max(0, min(n - 1, round(max(0.0, min(1.0, float(f))) * n) - 1)) for f in fractions]

    @torch.no_grad()
    def encode_image(self, images: Sequence[Image.Image], depths: Sequence[int]) -> ImageEncoding:
        pixels = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        with capture_block_outputs(self.image_blocks, depths) as captured:
            pooled = self.model.encode_image(pixels)
        start = 1 if self.image_has_cls else 0
        return ImageEncoding(pooled=pooled, patch_states=[captured[d][:, start:, :] for d in depths])

    @torch.no_grad()
    def encode_text(self, texts: Sequence[str], depths: Sequence[int]) -> TextEncoding:
        tokens = self.tokenizer(list(texts)).to(self.device)
        with capture_block_outputs(self.text_blocks, depths) as captured:
            pooled = self.model.encode_text(tokens)
        return TextEncoding(pooled=pooled,
                            pooled_states=[pool_text_tokens(captured[d], tokens, self.text_pooling) for d in depths])

    def _project(self, pooled: torch.Tensor) -> torch.Tensor:
        """The text projection is a Linear layer in some models and a plain matrix in others."""
        if isinstance(self.text_projection, nn.Linear):
            return self.text_projection(pooled)
        return pooled @ self.text_projection

    @torch.no_grad()
    def encode_text_both_poolings(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        """Apply standard and mean pooling after the final layer normalization,
        using the same trained projection."""
        tokens = self.tokenizer(list(texts)).to(self.device)
        captured = {}
        handle = self.ln_final.register_forward_hook(lambda _m, _i, out: captured.__setitem__("h", out.detach()))
        try:
            self.model.encode_text(tokens)
        finally:
            handle.remove()
        h = captured["h"]
        return {"standard": self._project(pool_text_tokens(h, tokens, self.text_pooling)),
                "mean": self._project(pool_text_tokens(h, tokens, "mean", end=self.mean_pooling_end, pad_id=self.pad_id))}


@torch.no_grad()
def encode_images(adapter: ModelAdapter, images: Sequence[Image.Image], batch_size: int = settings.IMAGE_BATCH_SIZE) -> torch.Tensor:
    """Encode the images in batches and return an (N, D) embedding tensor."""
    last = [adapter.n_image_layers - 1]
    return torch.cat([adapter.encode_image(list(images[i:i + batch_size]), last).pooled
                      for i in range(0, len(images), batch_size)])


@torch.no_grad()
def encode_captions(adapter: ModelAdapter, captions: Sequence[str], batch_size: int = settings.CAPTION_BATCH_SIZE) -> torch.Tensor:
    """Encode the captions in batches and return an (N, D) embedding tensor."""
    last = [adapter.n_text_layers - 1]
    return torch.cat([adapter.encode_text(list(captions[i:i + batch_size]), last).pooled
                      for i in range(0, len(captions), batch_size)])


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(name: str, device: torch.device | None = None) -> ModelAdapter:
    cfg = settings.MODEL_SETTINGS[name]
    device = device or pick_device()
    # The adapters import this module, so import them inside this function.
    if cfg["backend"] == "open_clip":
        from src.clip_and_siglip_models import OpenCLIPAdapter
        return OpenCLIPAdapter(name, cfg["arch"], cfg["pretrained"], device)
    from src.pecore_model import PECoreAdapter
    return PECoreAdapter(name, cfg["arch"], device)
