"""Properties of the corruptions, such as repeatability, shapes, level 0, the typo rules,
the object outlines and the severity table."""
from __future__ import annotations

import random

import numpy as np
import pytest

from src import corruptions, settings
from src.corruptions import corrupt_caption, corrupt_image, object_mask, strength

CAPTION = "a small dog running on green grass in a sunny park"
SQUARE = [[[8, 8, 40, 8, 40, 40, 8, 40]]]  # one object, a polygon covering a 32 x 32 square


@pytest.mark.parametrize("kind", ["gaussian_noise", "gaussian_blur", "cutout", "object_cutout"])
def test_image_corruptions_repeat_and_keep_shape(small_rgb, kind):
    a = corrupt_image(small_rgb, kind, 3, rng=random.Random(7), objects=SQUARE)
    b = corrupt_image(small_rgb, kind, 3, rng=random.Random(7), objects=SQUARE)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert a.size == small_rgb.size and a.mode == small_rgb.mode
    np.testing.assert_array_equal(np.asarray(corrupt_image(small_rgb, kind, 0, rng=random.Random(0))), np.asarray(small_rgb))


@pytest.mark.parametrize("kind", ["mask", "shuffle", "replace", "typo"])
def test_text_corruptions_repeat(kind):
    assert corrupt_caption(CAPTION, kind, 3, rng=random.Random(11)) == corrupt_caption(CAPTION, kind, 3, rng=random.Random(11))
    assert corrupt_caption(CAPTION, kind, 0, rng=random.Random(0)) == CAPTION


def test_shuffle_keeps_the_words_of_a_short_caption():
    # Shuffle selects at least two words even at level 1, and may put them back in place.
    assert sorted(corrupt_caption("tiny caption", "shuffle", 1, rng=random.Random(1)).split()) == ["caption", "tiny"]


@pytest.mark.parametrize("severity", [1, 3, 5])
def test_typo_misspells_the_right_number_of_words_in_place(severity):
    words = CAPTION.split()
    for seed in range(20):
        out = corrupt_caption(CAPTION, "typo", severity, rng=random.Random(seed)).split()
        assert len(out) == len(words)
        changed = [(a, b) for a, b in zip(words, out) if a != b]
        assert len(changed) == max(1, round(strength("text", "typo", severity) * len(words)))
        # Each misspelling is one keyboard edit, so a word's length moves by at most one.
        assert all(abs(len(a) - len(b)) <= 1 for a, b in changed)


def test_a_word_without_letters_cannot_be_misspelled():
    assert corruptions._misspell("42", random.Random(0)) == "42"
    assert all(corruptions._misspell("kitchen", random.Random(s)) != "kitchen" for s in range(50))


def test_object_masks_from_polygons_and_run_lengths():
    # COCO's reference drawing gives this square exactly 32 x 32 pixels.
    assert object_mask(SQUARE[0], 64, 64).sum() == 32 * 32
    # Run lengths count down the columns, here 2 off, 3 on and the rest off, in a 4 x 2 image.
    rle = object_mask({"counts": [2, 3, 3], "size": [4, 2]}, 4, 2)
    # The same run lengths in COCO's compressed form give the same mask.
    from pycocotools import mask as coco_mask
    compressed = coco_mask.encode(np.asfortranarray(rle.astype(np.uint8)))
    assert (object_mask({"counts": compressed["counts"].decode(), "size": [4, 2]}, 4, 2) == rle).all()
    np.testing.assert_array_equal(rle, np.array([[0, 1], [0, 0], [1, 0], [1, 0]], dtype=bool))


def test_object_cutout_stops_once_every_object_is_gone(small_rgb):
    out = np.asarray(corrupt_image(small_rgb, "object_cutout", 5, rng=random.Random(0), objects=SQUARE))
    assert (out.sum(axis=2) == 0).mean() < 0.5  # the one object covers about a quarter of the picture


def test_severity_table_rises_with_level():
    for kind in settings.IMAGE_CORRUPTIONS + ("object_cutout",):
        values = [strength("image", kind, s) for s in settings.SEVERITIES]
        assert values == sorted(values), kind
    shares = [strength("text", "mask", s) for s in settings.SEVERITIES]
    assert shares == sorted(shares)
    with pytest.raises(ValueError):
        strength("image", "gaussian_noise", 0)


def test_retrieval_generators_do_not_depend_on_the_number_of_pairs():
    short = [r.random() for r in corruptions.retrieval_rngs(0, 3, 4, "image")]
    long = [r.random() for r in corruptions.retrieval_rngs(0, 3, 10, "image")]
    assert short == long[:4]
