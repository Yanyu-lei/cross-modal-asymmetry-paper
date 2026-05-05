# Damage rulers: bounds and clean-vs-clean identity behavior.
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.analysis.damage_metrics import image_damage, text_damage, PSNR_CAP_DB


def _img(shape=(64, 64, 3), seed=0):
    return Image.fromarray((np.random.RandomState(seed).rand(*shape) * 255).astype(np.uint8))


def test_image_damage_clean_vs_clean_is_max_similarity():
    img = _img()
    ssim, psnr = image_damage(img, img)
    np.testing.assert_allclose(ssim, 1.0, atol=1e-6)
    np.testing.assert_allclose(psnr, PSNR_CAP_DB, atol=1e-6)


def test_image_damage_bounds_under_corruption():
    a = _img(seed=0)
    b = _img(seed=1)
    ssim, psnr = image_damage(a, b)
    assert -1.0 <= ssim <= 1.0
    assert 0.0 <= psnr <= PSNR_CAP_DB


def test_text_damage_clean_vs_clean_is_max_similarity():
    s = "a small dog running on green grass"
    ed, bleu = text_damage(s, s)
    np.testing.assert_allclose(ed, 0.0, atol=1e-6)
    np.testing.assert_allclose(bleu, 1.0, atol=1e-6)


def test_text_damage_bounds():
    a = "a small dog running on green grass"
    b = "completely different text with other words entirely"
    ed, bleu = text_damage(a, b)
    assert 0.0 <= ed <= 1.0
    assert 0.0 <= bleu <= 1.0
