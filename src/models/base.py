"""Uniform model-adapter interface.

Each adapter loads one of the five models and exposes a single contract:
    encode_image(image, image_depths) -> ImageEncoding
    encode_text(text, text_depths)    -> TextEncoding

Spoke code (image fidelity, text fidelity, match retention, retrieval) only
sees this interface; per-model quirks (pooling convention, block path,
tokenizer context length) live inside the adapter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import torch
from PIL import Image


@dataclass
class ImageEncoding:
    pooled: torch.Tensor              # (B, D_proj) final image embedding (un-normalized)
    patch_states: List[torch.Tensor]  # one tensor per requested depth, shape (B, n_patches, D_layer); CLS already removed


@dataclass
class TextEncoding:
    pooled: torch.Tensor              # (B, D_proj) final text embedding (un-normalized)
    pooled_states: List[torch.Tensor] # one tensor per requested depth, shape (B, D_layer); model-correct pooling already applied


class ModelAdapter:
    """Subclass contract. Concrete adapters live in openclip_backend.py and pecore_backend.py."""

    name: str
    device: torch.device
    n_image_layers: int
    n_text_layers: int

    def encode_image(
        self,
        images: Sequence[Image.Image],
        depths: Sequence[int],
    ) -> ImageEncoding:
        raise NotImplementedError

    def encode_text(
        self,
        texts: Sequence[str],
        depths: Sequence[int],
    ) -> TextEncoding:
        raise NotImplementedError

    def proportional_depths(
        self,
        fractions: Sequence[float],
        modality: str,  # "image" | "text"
    ) -> List[int]:
        """Map [0.25, 0.60, 1.0] → concrete block indices for this model."""
        n = self.n_image_layers if modality == "image" else self.n_text_layers
        # 1-indexed semantics: fraction 1.0 → last block (index n-1, after the last block runs)
        out = []
        for f in fractions:
            f = max(0.0, min(1.0, float(f)))
            idx = max(0, min(n - 1, round(f * n) - 1))
            out.append(idx)
        return out
