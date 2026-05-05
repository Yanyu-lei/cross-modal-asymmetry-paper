"""Image corruptions, applied at the PIL level on raw RGB images.

Each function takes a PIL.Image, severity int, and an `rng` for deterministic
sampling (important for cutout coordinates). Returns a new PIL.Image. The
caller is responsible for whatever model-specific preprocessing comes next.
"""
from __future__ import annotations

import random
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter

from .severity import lookup


def gaussian_noise(img: Image.Image, severity: int, *, rng: random.Random) -> Image.Image:
    if severity == 0:
        return img.copy()
    sigma = lookup("image", "gaussian_noise", severity)
    arr = np.asarray(img, dtype=np.float32)
    # numpy RNG seeded from python rng for determinism
    seed = rng.randint(0, 2**32 - 1)
    nrng = np.random.default_rng(seed)
    noise = nrng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    out = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def gaussian_blur(img: Image.Image, severity: int, *, rng: random.Random) -> Image.Image:
    if severity == 0:
        return img.copy()
    radius = lookup("image", "gaussian_blur", severity)
    return img.filter(ImageFilter.GaussianBlur(radius=float(radius)))


def cutout(img: Image.Image, severity: int, *, rng: random.Random) -> Image.Image:
    if severity == 0:
        return img.copy()
    frac = lookup("image", "cutout", severity)
    w, h = img.size
    area = max(1, int(w * h * frac))
    side = int(np.sqrt(area))
    cw = max(1, min(w, side))
    ch = max(1, min(h, max(1, area // cw)))
    x0 = rng.randint(0, max(0, w - cw))
    y0 = rng.randint(0, max(0, h - ch))
    arr = np.asarray(img).copy()
    arr[y0 : y0 + ch, x0 : x0 + cw, :] = 0
    return Image.fromarray(arr)


CORRUPTORS = {
    "gaussian_noise": gaussian_noise,
    "gaussian_blur": gaussian_blur,
    "cutout": cutout,
}


def apply_image_corruption(
    img: Image.Image,
    corruption_type: str,
    severity: int,
    *,
    rng: random.Random,
) -> Image.Image:
    if corruption_type == "none" or severity == 0:
        return img.copy()
    if corruption_type not in CORRUPTORS:
        raise KeyError(f"unknown image corruption {corruption_type!r}")
    return CORRUPTORS[corruption_type](img, severity, rng=rng)
