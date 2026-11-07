"""Replay a run's corruptions and measure image or caption damage without a model."""
from __future__ import annotations

import math

import numpy as np
from nltk.metrics.distance import edit_distance
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from PIL import Image
from skimage.metrics import structural_similarity

from src import settings
from src.corruptions import corrupt_caption, corrupt_image, pair_rngs
from src.load_data import fidelity_retention_pairs, retrieval_pairs

PSNR_CAP_DB = 60.0  # Identical images would otherwise have infinite PSNR.
_BLEU_SMOOTHING = SmoothingFunction().method1


def image_damage(clean: Image.Image, damaged: Image.Image) -> tuple[float, float]:
    """Return the channel-averaged SSIM and capped PSNR in dB for RGB uint8 images.

    Replacing this PSNR formula with scikit-image's can change the last digit.
    """
    a, b = np.asarray(clean.convert("RGB")), np.asarray(damaged.convert("RGB"))
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    psnr = PSNR_CAP_DB if mse <= 1e-12 else min(PSNR_CAP_DB, 20.0 * math.log10(255.0) - 10.0 * math.log10(mse))
    return float(structural_similarity(a, b, channel_axis=-1, data_range=255)), psnr


def text_damage(clean: str, damaged: str) -> tuple[float, float]:
    """Return word edit distance divided by the longer caption's length and smoothed BLEU-4.

    Method-1 smoothing distinguishes captions with zero higher-order n-gram overlap.
    """
    a, b = clean.split(), damaged.split()
    longer = max(len(a), len(b))
    if longer == 0:
        return 0.0, 1.0
    bleu = float(sentence_bleu([a], b, smoothing_function=_BLEU_SMOOTHING)) if b else 0.0
    return edit_distance(a, b) / longer, bleu


def _row(seed, pair_id, side, kind, severity, first, second) -> dict:
    image = side == "image"
    return {"seed": seed, "pair_id": pair_id, "modality": side, "corruption_type": kind, "severity": severity,
            "ssim": first if image else "", "psnr": second if image else "",
            "norm_edit_distance": "" if image else first, "bleu": "" if image else second}


def damage_rows(run_kind: str, dataset: str, seed: int, n_pairs: int, image_corruptions, text_corruptions) -> list[dict]:
    """Measure one seed's inputs using the same pairs and corruption streams as its run."""
    pairs = (retrieval_pairs if run_kind == "retrieval" else fidelity_retention_pairs)(dataset, seed, n_pairs)
    objects = {}
    if "object_cutout" in image_corruptions:
        from src.evaluate_model import object_outlines_by_image
        objects = object_outlines_by_image(dataset)

    def measure(side, kind, i, severity, rng):
        if side == "image":
            pair = pairs[i]
            return _row(seed, i, side, kind, severity,
                        *image_damage(pair.image, corrupt_image(pair.image, kind, severity, rng=rng, objects=objects.get(pair.image_id))))
        return _row(seed, i, side, kind, severity,
                    *text_damage(pairs[i].caption, corrupt_caption(pairs[i].caption, kind, severity, rng=rng)))

    rows = []
    for side, kinds in (("image", image_corruptions), ("text", text_corruptions)):
        for kind in kinds:
            # Restart each corruption's generators as the model run does.
            rngs = {s: pair_rngs(run_kind, seed, s, n_pairs, side) for s in settings.SEVERITIES}
            # Keep the row order fixed because floating-point sums depend on it.
            if run_kind == "retrieval":
                rows += [measure(side, kind, i, s, rngs[s][i]) for s in settings.SEVERITIES for i in range(n_pairs)]
            else:
                rows += [measure(side, kind, i, s, rngs[s][i]) for i in range(n_pairs) for s in settings.SEVERITIES]
    return rows
