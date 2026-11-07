"""The damage measures stay in range and score an input against itself as undamaged, and
the damage of object cutout can be measured. BLEU-4 scores a caption shorter than four
words below 1 even against itself, and neither manifest has such a caption."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src import file_paths
from src.input_damage_metrics import damage_rows, image_damage, text_damage, PSNR_CAP_DB


def _img(shape=(64, 64, 3), seed=0):
    return Image.fromarray((np.random.RandomState(seed).rand(*shape) * 255).astype(np.uint8))


def test_image_damage_clean_vs_clean_is_max_similarity():
    img = _img()
    ssim, psnr = image_damage(img, img)
    np.testing.assert_allclose(ssim, 1.0, atol=1e-6)
    np.testing.assert_allclose(psnr, PSNR_CAP_DB, atol=1e-6)


def test_image_damage_bounds_for_unrelated_images():
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


@pytest.mark.skipif(not (file_paths.full_manifest("coco").exists() and file_paths.COCO_OBJECT_OUTLINES.exists()),
                    reason="needs the COCO manifest with images and the object outlines")
def test_object_cutout_damage_can_be_measured():
    rows = damage_rows("retrieval", "coco", 0, 2, ["object_cutout"], [])
    assert {r["corruption_type"] for r in rows} == {"object_cutout"}
    assert all(0 < r["ssim"] < 1 for r in rows)  # both pictures have objects, so every one is damaged
