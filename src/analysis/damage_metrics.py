"""Model-independent input-side damage metrics.

Image:
  ssim    — structural similarity, in [-1, 1] (typically [0, 1])
  psnr    — peak signal-to-noise ratio in dB; capped at 60 to avoid +inf
  damage_ssim = 1 - ssim
  damage_psnr = a normalized [0, 1] inverse-PSNR ranking; computed downstream after the table is built

Text:
  norm_edit_distance — token-level Levenshtein on whitespace-split words / max(len)
  bleu               — sentence-BLEU with method-1 smoothing (Chen & Cherry 2014)
  damage_bleu        = 1 - bleu
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as _ssim
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

PSNR_CAP_DB = 60.0
_BLEU_SMOOTH = SmoothingFunction().method1


def image_damage(clean: Image.Image, corrupt: Image.Image) -> Tuple[float, float]:
    """Return (ssim, psnr) for one (clean, corrupt) pair.

    Both images must be the same shape (corruption pipeline preserves shape).
    SSIM is computed channel-wise on uint8 RGB and averaged; PSNR uses MSE
    on float pixel values.
    """
    a = np.asarray(clean.convert("RGB"))
    b = np.asarray(corrupt.convert("RGB"))
    if a.shape != b.shape:
        raise ValueError(f"clean/corrupt size mismatch: {a.shape} vs {b.shape}")

    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    if mse <= 1e-12:
        psnr = PSNR_CAP_DB
    else:
        psnr = min(PSNR_CAP_DB, 20.0 * math.log10(255.0) - 10.0 * math.log10(mse))

    # SSIM with multichannel; data_range=255 because uint8.
    # Default win_size handles small/medium images cleanly.
    s = float(_ssim(a, b, channel_axis=-1, data_range=255))

    return s, psnr


def _word_edit_distance(a: List[str], b: List[str]) -> int:
    """Standard Levenshtein on token lists. O(len(a)*len(b))."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + cost, # substitution
            )
        prev, cur = cur, prev
    return prev[m]


def text_damage(clean: str, corrupt: str) -> Tuple[float, float]:
    """Return (normalized_token_edit_distance, bleu) for one (clean, corrupt) pair.

    Tokenization is whitespace-split (matches the corruption module's word-level
    operations). BLEU uses method-1 smoothing so high-corruption captions don't
    saturate at 0.
    """
    cw = clean.split()
    rw = corrupt.split()
    denom = max(len(cw), len(rw))
    if denom == 0:
        return 0.0, 1.0  # degenerate empty input
    ed = _word_edit_distance(cw, rw)
    norm_ed = ed / denom

    if len(rw) == 0:
        bleu = 0.0
    else:
        bleu = float(sentence_bleu([cw], rw, smoothing_function=_BLEU_SMOOTH))

    return float(norm_ed), bleu
