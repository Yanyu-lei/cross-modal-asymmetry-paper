"""open_clip-backed adapter, covers four of the five models:

- openai_clip_b32  (ViT-B-32, openai)
- openai_clip_l14  (ViT-L-14, openai)
- openclip_l14_laion2b  (ViT-L-14, laion2b_s32b_b82k)
- siglip2_so400m_384  (ViT-SO400M-16-SigLIP2-384, webli)

Auto-detects whether the visual tower is a CLIP-style VisionTransformer
(`visual.transformer.resblocks`) or a timm-backed TimmModel
(`visual.trunk.blocks`). Auto-detects symmetric vs asymmetric text path
(`transformer.resblocks` vs `text.transformer.resblocks`).

Text pooling follows each model's contrastive convention:
- CLIP/OpenCLIP: argmax-of-token-id (EOT position)
- SigLIP 2:      last position
"""
from __future__ import annotations

from typing import List, Sequence

import open_clip
import torch
import torch.nn as nn
from PIL import Image

from .base import ImageEncoding, ModelAdapter, TextEncoding
from ._hooks import capture_block_outputs
from ._text_pooling import alt_pool


def _has_path(obj, path: str) -> bool:
    cur = obj
    for p in path.split("."):
        if not hasattr(cur, p):
            return False
        cur = getattr(cur, p)
    return True


def _get_path(obj, path: str):
    cur = obj
    for p in path.split("."):
        cur = getattr(cur, p)
    return cur


class OpenCLIPAdapter(ModelAdapter):
    def __init__(
        self,
        name: str,
        arch: str,
        pretrained: str,
        device: torch.device,
    ):
        self.name = name
        self.device = device
        self.arch = arch

        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        model.eval()
        model.to(device)
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(arch)

        # --- locate image blocks ---
        if _has_path(model, "visual.transformer.resblocks"):
            self._image_blocks_path = "visual.transformer.resblocks"
            self._visual_kind = "clip_vit"
        elif _has_path(model, "visual.trunk.blocks"):
            self._image_blocks_path = "visual.trunk.blocks"
            self._visual_kind = "timm"
        else:
            raise RuntimeError(f"{arch}: cannot locate image transformer blocks")
        self._image_blocks: nn.ModuleList = _get_path(model, self._image_blocks_path)
        self.n_image_layers = len(self._image_blocks)

        # --- locate text blocks ---
        if _has_path(model, "text.transformer.resblocks"):
            self._text_blocks_path = "text.transformer.resblocks"
            self._text_kind = "siglip"
        elif _has_path(model, "transformer.resblocks"):
            self._text_blocks_path = "transformer.resblocks"
            self._text_kind = "clip"
        else:
            raise RuntimeError(f"{arch}: cannot locate text transformer blocks")
        self._text_blocks: nn.ModuleList = _get_path(model, self._text_blocks_path)
        self.n_text_layers = len(self._text_blocks)

        # --- text pooling convention ---
        # SigLIP / SigLIP 2 use last-position pooling; CLIP-family uses EOT (argmax) pooling.
        # Read from the model's own pool_type when available, fallback by family.
        text_module = model.text if self._text_kind == "siglip" else model
        self._text_pool_type = getattr(text_module, "pool_type", None) or (
            "last" if self._text_kind == "siglip" else "argmax"
        )

        # context length for tokenizer (SigLIP 2 = 64, CLIP = 77; tokenizer enforces it)
        self.context_length = getattr(text_module, "context_length", None)

        # vocab size for the "replace" corruption
        self.vocab_size = self._infer_vocab_size()

    def _infer_vocab_size(self) -> int:
        # SigLIP 2 tokenizer wraps a HF tokenizer; CLIP tokenizers have .vocab_size or len()
        tok = self.tokenizer
        for attr in ("vocab_size", "n_vocab"):
            if hasattr(tok, attr):
                v = getattr(tok, attr)
                if isinstance(v, int) and v > 0:
                    return v
        # Try on the inner tokenizer (open_clip wraps HF for SigLIP 2)
        for inner in ("tokenizer", "_tokenizer"):
            if hasattr(tok, inner):
                t2 = getattr(tok, inner)
                for attr in ("vocab_size", "n_vocab"):
                    if hasattr(t2, attr):
                        v = getattr(t2, attr)
                        if isinstance(v, int) and v > 0:
                            return v
        # Last resort: the embedding table
        if hasattr(self.model, "token_embedding"):
            return self.model.token_embedding.num_embeddings
        if hasattr(self.model, "text") and hasattr(self.model.text, "token_embedding"):
            return self.model.text.token_embedding.num_embeddings
        raise RuntimeError(f"{self.arch}: cannot determine vocab size")

    # =========================================================================
    # Image
    # =========================================================================
    @torch.no_grad()
    def encode_image(
        self,
        images: Sequence[Image.Image],
        depths: Sequence[int],
    ) -> ImageEncoding:
        pixel_values = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        with capture_block_outputs(self._image_blocks, depths) as cap:
            pooled = self.model.encode_image(pixel_values)

        # open_clip 3.x is batch-first throughout for both CLIP-style ViT and timm trunks:
        # block outputs are (B, T, D). CLIP-style ViT prepends a CLS at position 0;
        # SigLIP 2 SO400M uses an attention pooler with no CLS, so patches start at 0.
        patch_states: List[torch.Tensor] = []
        for d in depths:
            h = cap[d]
            if h.ndim != 3 or h.shape[0] != pixel_values.shape[0]:
                raise RuntimeError(
                    f"{self.arch}: unexpected image hidden shape {tuple(h.shape)} "
                    f"(expected (B={pixel_values.shape[0]}, T, D))"
                )
            if self._visual_kind == "clip_vit":
                h = h[:, 1:, :]  # drop CLS
            patch_states.append(h)

        return ImageEncoding(pooled=pooled, patch_states=patch_states)

    # =========================================================================
    # Text
    # =========================================================================
    @torch.no_grad()
    def encode_text(
        self,
        texts: Sequence[str],
        depths: Sequence[int],
    ) -> TextEncoding:
        tokens = self.tokenizer(list(texts)).to(self.device)
        with capture_block_outputs(self._text_blocks, depths) as cap:
            pooled = self.model.encode_text(tokens)

        pooled_states: List[torch.Tensor] = []
        for d in depths:
            h = cap[d]
            if h.ndim != 3 or h.shape[0] != tokens.shape[0]:
                raise RuntimeError(
                    f"{self.arch}: unexpected text hidden shape {tuple(h.shape)} "
                    f"(expected (B={tokens.shape[0]}, T, D))"
                )
            pooled_states.append(self._pool_text(h, tokens))

        return TextEncoding(pooled=pooled, pooled_states=pooled_states)

    def _get_text_module_and_projection(self):
        """Resolve (ln_final_module, text_projection) for argmax / mean / etc.

        For CLIP/OpenCLIP, ln_final and text_projection live on the model itself.
        For SigLIP-family, they live on model.text.
        """
        if self._text_kind == "siglip":
            text_module = self.model.text
        else:
            text_module = self.model
        ln_final = text_module.ln_final
        text_proj = text_module.text_projection
        return ln_final, text_proj

    @staticmethod
    def _apply_projection(pooled: torch.Tensor, text_proj) -> torch.Tensor:
        """text_projection can be nn.Linear or nn.Parameter (matrix). Handle both."""
        if isinstance(text_proj, nn.Linear):
            return text_proj(pooled)
        # nn.Parameter / Tensor: matrix multiply
        return pooled @ text_proj

    @torch.no_grad()
    def encode_text_alt_pools(
        self,
        texts: Sequence[str],
        pool_types: Sequence[str] = ("standard", "mean"),
    ):
        """For the EOT-bottleneck mechanistic probe.

        Captures ln_final's output (the input to the standard pool +
        projection), pools it with each requested pool_type, and applies the
        model's text_projection so the result is comparable to the model's
        standard image embeddings.

        Pool type semantics:
          "standard" — the model's NATIVE pool (argmax for CLIP-family,
                       last for SigLIP). This is the baseline.
          "mean"     — masked mean over content tokens. Mechanistic alternative.
          "argmax", "last", "first" — explicit literal pool ops, regardless of
                       model convention.

        Returns: dict {pool_type: (B, D_proj)} with the requested labels as keys.
        """
        tokens = self.tokenizer(list(texts)).to(self.device)

        ln_final, text_proj = self._get_text_module_and_projection()
        captured: dict[str, torch.Tensor] = {}

        def _hook(_module, _inp, out):
            captured["ln_final"] = out.detach()

        handle = ln_final.register_forward_hook(_hook)
        try:
            _ = self.model.encode_text(tokens)
        finally:
            handle.remove()

        h = captured["ln_final"]  # (B, T, D)

        # Tokenizer pad/eot conventions
        if self._text_kind == "siglip":
            pad_id = 1
            mean_strategy = "last_non_pad"
        else:
            pad_id = 0
            mean_strategy = "argmax"

        out: dict[str, torch.Tensor] = {}
        for pt in pool_types:
            if pt == "standard":
                actual_pt = self._text_pool_type   # "argmax" for CLIP-family, "last" for SigLIP
                strat = "argmax"   # ignored when actual_pt != "mean"
            elif pt == "mean":
                actual_pt = "mean"
                strat = mean_strategy
            else:
                actual_pt = pt
                strat = "argmax"
            pooled = alt_pool(h, tokens, actual_pt, eot_strategy=strat, pad_id=pad_id)
            out[pt] = self._apply_projection(pooled, text_proj)
        return out

    def _pool_text(self, h: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Apply the model's contrastive-pooling convention to one layer's hidden state."""
        if self._text_pool_type == "argmax":
            # CLIP convention: pooled = h[batch, argmax_token_id_position]
            idx = tokens.argmax(dim=-1)  # (B,)
            return h[torch.arange(h.shape[0], device=h.device), idx]
        if self._text_pool_type == "last":
            return h[:, -1, :]
        if self._text_pool_type == "first":
            return h[:, 0, :]
        # Fallback: mean over sequence
        return h.mean(dim=1)
