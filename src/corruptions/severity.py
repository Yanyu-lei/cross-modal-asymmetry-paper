"""Severity-table lookup. Reads configs/severity.yaml once on import."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SEVERITY_YAML = _PROJECT_ROOT / "configs" / "severity.yaml"

with open(_SEVERITY_YAML) as f:
    _TABLE: Dict[str, Dict[str, Dict[int, float]]] = yaml.safe_load(f)


VALID_IMAGE_TYPES = ("gaussian_noise", "gaussian_blur", "cutout")
VALID_TEXT_TYPES = ("mask", "shuffle", "replace")


def lookup(modality: str, corruption_type: str, severity: int) -> float:
    """Return the native parameter for a (modality, corruption_type, severity).

    severity 0 is "clean" and not stored in the table; callers should branch
    before calling this.
    """
    if modality not in _TABLE:
        raise KeyError(f"unknown modality {modality!r}; expected 'image' or 'text'")
    if corruption_type not in _TABLE[modality]:
        raise KeyError(
            f"unknown {modality} corruption {corruption_type!r}; "
            f"expected one of {list(_TABLE[modality].keys())}"
        )
    if severity == 0:
        raise ValueError("severity 0 is clean; do not call lookup for severity 0")
    if severity not in _TABLE[modality][corruption_type]:
        raise KeyError(
            f"severity {severity} not in {modality}.{corruption_type}; "
            f"available: {list(_TABLE[modality][corruption_type].keys())}"
        )
    return float(_TABLE[modality][corruption_type][severity])


def all_severities() -> tuple[int, ...]:
    return (1, 2, 3, 4, 5)
