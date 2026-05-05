"""Image-fidelity spoke.

For each image-encoder depth, compare patch-level cosine similarity between the
clean image and a corrupted image. The metric per (pair, depth) is the mean
cosine over patches.

Layer dimensionalities differ across depths (and across models), so cosine is
computed within each (model, depth) — never across depths or across models.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from ..models.base import ImageEncoding


_DEPTH_NAMES = ("early", "mid", "late")  # 3-tuple aligned with proportional_depths(...)


def patch_cosine_per_depth(
    clean: ImageEncoding,
    corrupted: ImageEncoding,
) -> List[float]:
    """Mean cosine similarity over patches at each depth. Returns one float per depth."""
    if len(clean.patch_states) != len(corrupted.patch_states):
        raise ValueError("clean and corrupted have different numbers of depths")
    out: List[float] = []
    for c, n in zip(clean.patch_states, corrupted.patch_states):
        # c, n: (1, n_patches, D). Cosine along D, then mean over patches.
        cos = F.cosine_similarity(c, n, dim=-1)  # (1, n_patches)
        out.append(float(cos.mean().item()))
    return out


def rows_for_pair(
    *,
    base_row: Dict,
    clean: ImageEncoding,
    corrupted: ImageEncoding,
    depth_layer_indices: List[int],
) -> List[Dict]:
    """Produce one CSV row per depth for one (pair, condition) measurement."""
    cosines = patch_cosine_per_depth(clean, corrupted)
    rows: List[Dict] = []
    for name, layer_idx, val in zip(_DEPTH_NAMES, depth_layer_indices, cosines):
        row = dict(base_row)
        row["spoke"] = "image_fidelity"
        row["depth"] = name
        row["depth_layer_index"] = int(layer_idx)
        row["metric"] = "patch_cosine_mean"
        row["value"] = float(val)
        rows.append(row)
    return rows
