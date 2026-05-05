# Multi-positive-aware Recall@k. Wrapper around the scoring matrix used by
# src/spokes/retrieval.py. Asserts on duplicate query/candidate ids so the
# evaluator never silently treats two captions of the same image as
# independent examples.
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def multi_positive_recall_at_k(
    query_embs: torch.Tensor,    # (Q, D)
    cand_embs: torch.Tensor,     # (C, D)
    gt_indices: List[List[int]], # gt_indices[i] = candidate indices that count as a hit for query i
    ks: Sequence[int] = (1, 5, 10),
) -> Dict[int, float]:
    """Per query, hit if ANY ground-truth candidate falls within the top-k.
    Returns {k: recall_fraction}. Order of gt_indices matches query rows."""
    if len(gt_indices) != query_embs.shape[0]:
        raise ValueError(
            f"gt_indices length ({len(gt_indices)}) does not match number of queries ({query_embs.shape[0]})"
        )
    q = F.normalize(query_embs, dim=-1)
    c = F.normalize(cand_embs, dim=-1)
    sims = (q @ c.T).cpu().numpy()  # (Q, C)
    out: Dict[int, float] = {}
    for k in ks:
        topk_idx = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
        hits = 0
        for i, gts in enumerate(gt_indices):
            if not gts:
                continue
            if any(g in topk_idx[i] for g in gts):
                hits += 1
        out[k] = hits / len(gt_indices)
    return out


def assert_no_duplicate_ids(ids: Sequence) -> None:
    """Fail loudly if a manifest has duplicate id values that would cause
    silent multi-positive ambiguity downstream."""
    seen = set()
    dups = []
    for i in ids:
        if i in seen:
            dups.append(i)
        seen.add(i)
    if dups:
        raise ValueError(f"duplicate ids in manifest: {sorted(set(dups))[:10]} (showing first 10)")
