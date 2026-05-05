# Severity table loads and is monotonic per corruption type.
from __future__ import annotations

import pytest
import yaml
from pathlib import Path

from src.corruptions.severity import lookup, VALID_IMAGE_TYPES, VALID_TEXT_TYPES


def test_yaml_loads():
    p = Path(__file__).resolve().parents[1] / "configs" / "severity.yaml"
    with open(p) as f:
        cfg = yaml.safe_load(f)
    assert "image" in cfg and "text" in cfg


@pytest.mark.parametrize("modality,types", [("image", VALID_IMAGE_TYPES), ("text", VALID_TEXT_TYPES)])
def test_severities_monotonic(modality, types):
    for ct in types:
        values = [lookup(modality, ct, sev) for sev in (1, 2, 3, 4, 5)]
        assert all(values[i] <= values[i + 1] for i in range(4)), f"non-monotonic: {modality}/{ct}={values}"


def test_lookup_returns_float():
    v = lookup("image", "gaussian_noise", 3)
    assert isinstance(v, float)
    v = lookup("text", "mask", 3)
    assert isinstance(v, float)


def test_lookup_rejects_severity_zero():
    with pytest.raises(ValueError):
        lookup("image", "gaussian_noise", 0)


def test_lookup_rejects_unknown_corruption():
    with pytest.raises(KeyError):
        lookup("image", "nonexistent_op", 3)
