"""Text-fidelity spoke.

For each text-encoder depth, compare the model's pooled-text representation
between the clean and corrupted captions via cosine similarity. Pooling
convention is per-model and is applied inside the adapter (argmax-of-EOT for
CLIP-family, last-position for SigLIP) so this module is symmetric with
image_fidelity and stays uniform across models.

Pooled text at very early layers may show artificially high similarity
regardless of corruption — what matters is where deviation begins, which
differs across models (set per-model from results/text_depth_diagnostic.csv).
"""
from __future__ import annotations

from typing import Dict, List

import torch.nn.functional as F

from ..models.base import TextEncoding


_DEPTH_NAMES = ("early", "mid", "late")


def pooled_cosine_per_depth(
    clean: TextEncoding,
    corrupted: TextEncoding,
) -> List[float]:
    if len(clean.pooled_states) != len(corrupted.pooled_states):
        raise ValueError("clean and corrupted have different numbers of depths")
    out: List[float] = []
    for c, n in zip(clean.pooled_states, corrupted.pooled_states):
        # c, n: (1, D_layer)
        cos = F.cosine_similarity(c, n, dim=-1)
        out.append(float(cos.item()))
    return out


def rows_for_pair(
    *,
    base_row: Dict,
    clean: TextEncoding,
    corrupted: TextEncoding,
    depth_layer_indices: List[int],
) -> List[Dict]:
    cosines = pooled_cosine_per_depth(clean, corrupted)
    rows: List[Dict] = []
    for name, layer_idx, val in zip(_DEPTH_NAMES, depth_layer_indices, cosines):
        row = dict(base_row)
        row["spoke"] = "text_fidelity"
        row["depth"] = name
        row["depth_layer_index"] = int(layer_idx)
        row["metric"] = "pooled_cosine"
        row["value"] = float(val)
        rows.append(row)
    return rows
