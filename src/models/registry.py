"""Model registry: load adapters by name, reading configs/models.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import yaml

from .base import ModelAdapter
from .openclip_backend import OpenCLIPAdapter
from .pecore_backend import PECoreAdapter


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODELS_YAML = _PROJECT_ROOT / "configs" / "models.yaml"


def _load_models_config() -> Dict[str, dict]:
    with open(_MODELS_YAML) as f:
        cfg = yaml.safe_load(f)
    return cfg["models"]


def list_models() -> list[str]:
    return list(_load_models_config().keys())


def get_model_config(name: str) -> dict:
    cfg = _load_models_config()
    if name not in cfg:
        raise KeyError(f"unknown model {name!r}; known: {list(cfg.keys())}")
    return cfg[name]


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(name: str, device: torch.device | None = None) -> ModelAdapter:
    cfg = get_model_config(name)
    device = device or pick_device()
    backend = cfg["backend"]
    if backend == "open_clip":
        return OpenCLIPAdapter(
            name=name,
            arch=cfg["arch"],
            pretrained=cfg["pretrained"],
            device=device,
        )
    if backend == "pe_core":
        return PECoreAdapter(
            name=name,
            config_name=cfg["arch"],
            pretrained=bool(cfg["pretrained"]),
            device=device,
        )
    raise ValueError(f"unknown backend {backend!r} for model {name!r}")
