"""Adapt the CLIP and SigLIP 2 models while preserving their CLS-token and text-pooling conventions."""
from __future__ import annotations

from operator import attrgetter

import open_clip
import torch

from src.load_models import ModelAdapter


class OpenCLIPAdapter(ModelAdapter):
    def __init__(self, name: str, arch: str, pretrained: str, device: torch.device):
        self.name, self.device = name, device
        model, _, self.preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.model = model.eval().to(device)
        self.tokenizer = open_clip.get_tokenizer(arch)

        is_siglip = hasattr(model, "text")
        text_tower = model.text if is_siglip else model
        self.image_has_cls = not is_siglip
        self.image_blocks = attrgetter("visual.trunk.blocks" if is_siglip else "visual.transformer.resblocks")(model)
        self.text_blocks = text_tower.transformer.resblocks
        self.text_pooling = getattr(text_tower, "pool_type", None) or ("last" if is_siglip else "argmax")
        self.context_length = text_tower.context_length
        self.ln_final, self.text_projection = text_tower.ln_final, text_tower.text_projection
        # SigLIP 2 uses 1 for EOS and 0 for padding, so mean pooling stops at the last non-padding token.
        if is_siglip:
            self.mean_pooling_end, self.pad_id = "last_non_pad", self.tokenizer.tokenizer.pad_token_id
        else:
            self.mean_pooling_end, self.pad_id = "argmax", 0
