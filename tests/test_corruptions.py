# Protects against silent non-determinism or shape changes in the 6 corruptors.
from __future__ import annotations

import random

import numpy as np
import pytest
from PIL import Image

from src.corruptions.image import apply_image_corruption
from src.corruptions.text import apply_text_corruption


CAPTION = "a small dog running on green grass in a sunny park"


@pytest.mark.parametrize("ctype", ["gaussian_noise", "gaussian_blur", "cutout"])
def test_image_corruption_deterministic_and_shape(small_rgb, ctype):
    a = apply_image_corruption(small_rgb, ctype, severity=3, rng=random.Random(7))
    b = apply_image_corruption(small_rgb, ctype, severity=3, rng=random.Random(7))
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert a.size == small_rgb.size
    assert a.mode == small_rgb.mode


@pytest.mark.parametrize("ctype", ["gaussian_noise", "gaussian_blur", "cutout"])
def test_image_corruption_severity_zero_is_noop(small_rgb, ctype):
    out = apply_image_corruption(small_rgb, ctype, severity=0, rng=random.Random(0))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(small_rgb))


@pytest.mark.parametrize("ctype", ["mask", "shuffle", "replace"])
def test_text_corruption_deterministic_and_str(ctype):
    a = apply_text_corruption(CAPTION, ctype, severity=3, rng=random.Random(11))
    b = apply_text_corruption(CAPTION, ctype, severity=3, rng=random.Random(11))
    assert isinstance(a, str)
    assert a == b


@pytest.mark.parametrize("ctype", ["mask", "shuffle", "replace"])
def test_text_corruption_severity_zero_is_noop(ctype):
    assert apply_text_corruption(CAPTION, ctype, severity=0, rng=random.Random(0)) == CAPTION


def test_shuffle_floor_on_short_caption():
    short = "tiny caption"
    out = apply_text_corruption(short, "shuffle", severity=1, rng=random.Random(0))
    assert isinstance(out, str)
    assert out  # non-empty
