"""Compute encoder fidelity, match retention and retrieval metrics."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.load_models import ImageEncoding, TextEncoding

DEPTH_NAMES = ("early", "mid", "late")


def fidelity_rows(base: dict, side: str, clean: ImageEncoding | TextEncoding,
                  damaged: ImageEncoding | TextEncoding, layer_indices: Sequence[int]) -> list[dict]:
    """Return the cosine similarity between clean and damaged states at each requested depth.

    The image score averages over patches; each caption uses one pooled vector.
    """
    if side == "image":
        clean_states, damaged_states, metric = clean.patch_states, damaged.patch_states, "patch_cosine_mean"
    else:
        clean_states, damaged_states, metric = clean.pooled_states, damaged.pooled_states, "pooled_cosine"
    rows = []
    for depth, layer, c, d in zip(DEPTH_NAMES, layer_indices, clean_states, damaged_states):
        rows.append({**base, "spoke": f"{side}_fidelity", "depth": depth, "depth_layer_index": int(layer),
                     "metric": metric, "value": float(F.cosine_similarity(c, d, dim=-1).mean().item())})
    return rows


def retention_margin(query: torch.Tensor, partner: torch.Tensor, negatives: torch.Tensor) -> float:
    """Return the partner cosine similarity minus the mean similarity to the negatives.

    The query and partner have shape (1, D), and the negatives have shape (K, D).
    Raise an error if the result is not finite.
    """
    sim_partner = float(F.cosine_similarity(query, partner, dim=-1).item())
    sim_negatives = float((F.normalize(query, dim=-1) @ F.normalize(negatives, dim=-1).T).squeeze(0).mean().item())
    if not np.isfinite(sim_partner - sim_negatives):
        raise ValueError("The retention margin is not a finite number, so an embedding holds NaN or infinity.")
    return sim_partner - sim_negatives


def retention_row(base: dict, direction: str, margin: float, n_negatives: int) -> dict:
    return {**base, "spoke": "match_retention", "match_retention_direction": direction,
            "metric": "retention_margin", "value": margin, "n_negatives": int(n_negatives)}


def recall_hits(queries: torch.Tensor, candidates: torch.Tensor, ks: Sequence[int]) -> dict[int, np.ndarray]:
    """Return whether query i retrieves its paired candidate among the top k.

    A tie with the partner does not count against it. A similarity that is NaN or infinite
    raises an error. A NaN partner score would otherwise be counted as a hit.

    Returns:
        A dict from each k to an array with one hit (0 or 1) per query.
    """
    sims = F.normalize(queries, dim=-1) @ F.normalize(candidates, dim=-1).T
    finite = torch.isfinite(sims).all(dim=-1)
    if not finite.all():
        bad = (~finite).nonzero().flatten().tolist()
        raise ValueError(f"{len(bad)} queries, starting with {bad[:5]}, have similarities that are NaN or infinite.")
    ranks = (sims > sims.diag().unsqueeze(-1)).sum(dim=-1)
    return {k: (ranks < k).cpu().numpy().astype(int) for k in ks}


def retrieval_rows(base: dict, direction: str, queries: torch.Tensor, candidates: torch.Tensor, *,
                   gallery_sizes: Sequence[int], per_pair_gallery_sizes: Sequence[int],
                   ks: Sequence[int], clean: bool) -> list[dict]:
    """Score each nested gallery using the same prefix of queries and candidates.

    Clean and smaller-gallery averages use sweep rows; full-gallery damaged scores use
    retrieval rows. Requested galleries also receive per-query hit rows.
    """
    n = queries.shape[0]
    rows = []
    for size in sorted(s for s in gallery_sizes if s <= n):
        hits = recall_hits(queries[:size], candidates[:size], ks)
        if clean or size < n:
            rows += [{**base, "spoke": "retrieval_gallery_sweep", "metric": f"recall_at_{k}_{direction}_gallery{size}",
                      "value": float(hits[k].mean()), "n_eval": size} for k in ks]
        else:
            rows += [{**base, "spoke": "retrieval", "metric": f"recall_at_{k}_{direction}",
                      "value": float(hits[k].mean()), "n_eval": size} for k in ks]
        if size in per_pair_gallery_sizes:
            spoke, metric = ("clean_baseline_per_pair", "per_pair_clean_recall") if clean else ("retrieval_per_pair", "per_pair_recall")
            rows += [{**base, "pair_id": i, "spoke": spoke, "metric": f"{metric}_at_{k}_{direction}",
                      "value": int(hit), "n_eval": size}
                     for k in ks for i, hit in enumerate(hits[k])]
    return rows


def shared_space_similarity_rows(base: dict, direction: str, damaged: torch.Tensor, clean: torch.Tensor) -> list[dict]:
    """Compare damaged and clean shared-space embeddings with mean and per-query cosines."""
    cos = (F.normalize(damaged, dim=-1) * F.normalize(clean, dim=-1)).sum(dim=-1).cpu()
    n = int(cos.shape[0])
    row = {**base, "spoke": "retrieval_selfsim", "n_eval": n}
    return ([{**row, "metric": f"query_selfsim_{direction}", "value": float(cos.mean())}]
            + [{**row, "pair_id": i, "metric": f"per_pair_query_selfsim_{direction}", "value": float(v)}
               for i, v in enumerate(cos.tolist())])
