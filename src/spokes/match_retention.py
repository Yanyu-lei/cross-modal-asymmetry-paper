"""Match-retention spoke (paper-facing name: cross-modal coherence).

For each condition, compute:
    sim_match           = cos(corrupt-side embedding, clean-other-side embedding for the matched caption)
    sim_mismatch_mean   = mean cos(corrupt-side, clean-other-side) over the K-caption mismatch pool
    retention_margin    = sim_match - sim_mismatch_mean

The mismatch pool is sampled once per (model, seed) BEFORE any corruption is
applied, and is held disjoint from the eval pairs (configurable via
experiment.yaml: match_retention.pool_disjoint).

Every condition produces TWO directions as separate rows:
    "image_corrupted"   — corrupted image vs clean text(s)
    "text_corrupted"    — clean image vs corrupted text(s)

match_retention is closely related to retrieval (both derive from cosine
similarities over candidate sets); retrieval ranks against ALL candidates
while match_retention measures a margin against a fixed mismatched pool. This
is downstream-task translation, not independent validation.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F


def _cosine_to_pool(query: torch.Tensor, pool: torch.Tensor) -> float:
    """Mean cosine between a single query embedding (1, D) and a pool (P, D)."""
    q = F.normalize(query, dim=-1)
    p = F.normalize(pool, dim=-1)
    sims = (q @ p.T).squeeze(0)  # (P,)
    return float(sims.mean().item())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a, b, dim=-1).item())


def rows_for_pair(
    *,
    base_row: Dict,
    direction: str,                  # "image_corrupted" or "text_corrupted"
    img_corrupt_emb: torch.Tensor | None,
    txt_corrupt_emb: torch.Tensor | None,
    img_clean_emb: torch.Tensor,
    txt_clean_emb: torch.Tensor,
    pool_text_embs: torch.Tensor | None = None,   # used when direction == "image_corrupted"
    pool_img_embs: torch.Tensor | None = None,    # used when direction == "text_corrupted"
    k_pool: int,
) -> List[Dict]:
    """Three rows per call: sim_match, sim_mismatch_mean, retention_margin."""
    if direction == "image_corrupted":
        if img_corrupt_emb is None or pool_text_embs is None:
            raise ValueError("image_corrupted direction needs img_corrupt_emb and pool_text_embs")
        sim_match = cosine(img_corrupt_emb, txt_clean_emb)
        sim_mm = _cosine_to_pool(img_corrupt_emb, pool_text_embs)
    elif direction == "text_corrupted":
        if txt_corrupt_emb is None or pool_img_embs is None:
            raise ValueError("text_corrupted direction needs txt_corrupt_emb and pool_img_embs")
        sim_match = cosine(txt_corrupt_emb, img_clean_emb)
        sim_mm = _cosine_to_pool(txt_corrupt_emb, pool_img_embs)
    else:
        raise ValueError(f"unknown direction {direction!r}")

    margin = sim_match - sim_mm
    rows: List[Dict] = []
    for metric, val in (
        ("sim_match", sim_match),
        ("sim_mismatch_mean", sim_mm),
        ("retention_margin", margin),
    ):
        row = dict(base_row)
        row["spoke"] = "match_retention"
        row["match_retention_direction"] = direction
        row["metric"] = metric
        row["value"] = float(val)
        row["k_pool"] = int(k_pool)
        rows.append(row)
    return rows
