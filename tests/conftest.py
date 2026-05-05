# Shared fixtures for the test suite.
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def rng():
    import random
    return random.Random(42)


@pytest.fixture
def numpy_rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_rgb():
    arr = (np.random.RandomState(0).rand(64, 64, 3) * 255).astype(np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def synthetic_manifest(tmp_path):
    """Tiny COCO-shaped manifest with 12 distinct items. Returns the path."""
    items = []
    for i in range(12):
        arr = (np.random.RandomState(i).rand(32, 32, 3) * 255).astype(np.uint8)
        from io import BytesIO
        buf = BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG")
        b = buf.getvalue()
        import hashlib
        items.append({
            "id": 1000 + i,
            "caption": f"a synthetic caption number {i}",
            "bytes_hex": b.hex(),
            "sha256": hashlib.sha256(b).hexdigest(),
        })
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(items))
    return p
