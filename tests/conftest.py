"""Shared test fixtures."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def small_rgb() -> Image.Image:
    """A 64 x 64 picture of random pixels."""
    return Image.fromarray((np.random.RandomState(0).rand(64, 64, 3) * 255).astype(np.uint8))
