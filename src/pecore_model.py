"""Adapt the PE-Core L/14-336 model with its 32-token context and end-of-text pooling."""
from __future__ import annotations

import torch
from core.vision_encoder import pe
from core.vision_encoder.transforms import get_image_transform, get_text_tokenizer

from src.load_models import ModelAdapter


class PECoreAdapter(ModelAdapter):
    def __init__(self, name: str, config_name: str, device: torch.device):
        self.name, self.device = name, device
        self.model = pe.CLIP.from_config(config_name, pretrained=True).eval().to(device)
        image_cfg, text_cfg = pe.PE_VISION_CONFIG[config_name], pe.PE_TEXT_CONFIG[config_name]
        self.preprocess = get_image_transform(image_size=image_cfg.image_size, center_crop=True)
        self.context_length = text_cfg.context_length
        self.tokenizer = get_text_tokenizer(context_length=text_cfg.context_length)

        self.image_has_cls = bool(getattr(image_cfg, "use_cls_token", False))
        self.image_blocks = self.model.visual.transformer.resblocks
        self.text_blocks = self.model.transformer.resblocks
        self.text_pooling = "argmax"
        self.mean_pooling_end, self.pad_id = "argmax", 0
        self.ln_final, self.text_projection = self.model.ln_final, self.model.text_projection
