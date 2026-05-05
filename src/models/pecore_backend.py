"""PE-Core adapter (Meta perception_models).

Loads via `pe.CLIP.from_config(name, pretrained=True)`. PE-Core inherits from
TextTransformer, so the model object IS the text encoder; image lives at
`model.visual`. Both vision and text use ResidualAttentionBlock lists at
`*.transformer.resblocks`, same shape as open_clip CLIP.

Caveats specific to PE-Core L14-336:
- Vision uses attention pooling (no CLS *output*) but DOES prepend a CLS token
  internally (`use_cls_token=True`); we strip it for patch_states.
- Text context_length is 32 for L14-336 (much shorter than CLIP's 77).
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
from PIL import Image

from core.vision_encoder import pe as _pe
from core.vision_encoder.transforms import get_image_transform, get_text_tokenizer

from .base import ImageEncoding, ModelAdapter, TextEncoding
from ._hooks import capture_block_outputs
from ._text_pooling import alt_pool


class PECoreAdapter(ModelAdapter):
    def __init__(
        self,
        name: str,
        config_name: str,
        pretrained: bool,
        device: torch.device,
    ):
        self.name = name
        self.device = device
        self.config_name = config_name

        model = _pe.CLIP.from_config(config_name, pretrained=pretrained)
        model.eval()
        model.to(device)
        self.model = model

        vcfg = _pe.PE_VISION_CONFIG[config_name]
        tcfg = _pe.PE_TEXT_CONFIG[config_name]
        self.image_size = vcfg.image_size
        self.context_length = tcfg.context_length
        self.vocab_size = tcfg.vocab_size
        self._use_cls_token = bool(getattr(vcfg, "use_cls_token", False))

        self.preprocess = get_image_transform(image_size=vcfg.image_size, center_crop=True)
        self.tokenizer = get_text_tokenizer(context_length=tcfg.context_length)

        # Block lists
        self._image_blocks: nn.ModuleList = model.visual.transformer.resblocks
        self._text_blocks: nn.ModuleList = model.transformer.resblocks
        self.n_image_layers = len(self._image_blocks)
        self.n_text_layers = len(self._text_blocks)

        # PE-Core text uses argmax-of-EOT pooling (CLIP convention)
        self._text_pool_type = "argmax"

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

        patch_states: List[torch.Tensor] = []
        for d in depths:
            h = cap[d]  # (B, T, D) for PE-Core (it preserves batch-first)
            if h.ndim != 3:
                raise RuntimeError(f"PE-Core image hidden state has unexpected shape: {tuple(h.shape)}")
            if self._use_cls_token:
                h = h[:, 1:, :]
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
                    f"PE-Core text hidden state shape {tuple(h.shape)} "
                    f"does not match expected (B={tokens.shape[0]}, T, D)"
                )
            idx = tokens.argmax(dim=-1)
            pooled_states.append(h[torch.arange(h.shape[0], device=h.device), idx])

        return TextEncoding(pooled=pooled, pooled_states=pooled_states)

    @torch.no_grad()
    def encode_text_alt_pools(
        self,
        texts: Sequence[str],
        pool_types: Sequence[str] = ("standard", "mean"),
    ):
        """Mechanistic probe: pool from ln_final's output, then apply text_projection.
        See openclip_backend.encode_text_alt_pools for the same contract.
        PE-Core's text path: ln_final → pool → text_projection. Standard pool = argmax (EOT)."""
        tokens = self.tokenizer(list(texts)).to(self.device)

        ln_final = self.model.ln_final
        text_proj = self.model.text_projection
        captured: dict[str, torch.Tensor] = {}

        def _hook(_m, _i, out):
            captured["ln_final"] = out.detach()

        handle = ln_final.register_forward_hook(_hook)
        try:
            _ = self.model.encode_text(tokens)
        finally:
            handle.remove()

        h = captured["ln_final"]
        out: dict[str, torch.Tensor] = {}
        for pt in pool_types:
            if pt == "standard":
                actual_pt = self._text_pool_type      # "argmax" for PE-Core
            elif pt == "mean":
                actual_pt = "mean"
            else:
                actual_pt = pt
            pooled = alt_pool(h, tokens, actual_pt, eot_strategy="argmax", pad_id=0)
            if isinstance(text_proj, nn.Linear):
                out[pt] = text_proj(pooled)
            else:
                out[pt] = pooled @ text_proj
        return out
