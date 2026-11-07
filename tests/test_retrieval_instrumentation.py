"""The retrieval rows, covering nested galleries, one row per query and shared-space similarity."""
from __future__ import annotations

import pytest
import torch

from src.evaluate_model import empty_row
from src.fidelity_retention_retrieval_metrics import (recall_hits, retention_margin, retrieval_rows,
                                                      shared_space_similarity_rows)


def _one_hot(n: int, dim: int = 16) -> torch.Tensor:
    embs = torch.zeros(n, dim)
    embs[torch.arange(n), torch.arange(n)] = 1.0
    return embs


def test_smaller_galleries_are_prefixes_of_larger_ones():
    # Query 5 points at candidate 6, so only galleries that contain query 5 lose a hit.
    q = _one_hot(8)
    q[5] = 0.0
    q[5, 6] = 1.0
    rows = retrieval_rows(empty_row(), "t2i", q, _one_hot(8), gallery_sizes=[4, 8],
                          per_pair_gallery_sizes=[4], ks=(1,), clean=False)
    values = {(r["spoke"], r["metric"], r["n_eval"]): r["value"] for r in rows}
    assert values[("retrieval_gallery_sweep", "recall_at_1_t2i_gallery4", 4)] == 1.0
    assert values[("retrieval", "recall_at_1_t2i", 8)] == 7 / 8
    per_pair = [r for r in rows if r["spoke"] == "retrieval_per_pair"]
    assert [r["value"] for r in per_pair] == [1, 1, 1, 1] and all(r["n_eval"] == 4 for r in per_pair)


def test_clean_queries_get_a_sweep_row_at_every_size():
    rows = retrieval_rows(empty_row(), "i2t", _one_hot(4), _one_hot(4), gallery_sizes=[2, 4],
                          per_pair_gallery_sizes=[4], ks=(1, 5), clean=True)
    assert {r["metric"] for r in rows if r["spoke"] == "retrieval_gallery_sweep"} == {
        "recall_at_1_i2t_gallery2", "recall_at_5_i2t_gallery2", "recall_at_1_i2t_gallery4", "recall_at_5_i2t_gallery4"}
    assert {r["spoke"] for r in rows} == {"retrieval_gallery_sweep", "clean_baseline_per_pair"}


def test_a_tie_with_the_partner_still_counts_as_a_hit():
    q = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    assert recall_hits(q, q.clone(), ks=(1,))[1].tolist() == [1, 1]


def test_shared_space_similarity_rows():
    clean = _one_hot(4, 8)
    damaged = clean.clone()
    damaged[0] = 0.0
    damaged[0, 7] = 1.0  # orthogonal to its clean embedding
    rows = shared_space_similarity_rows(empty_row(), "t2i", damaged, clean)
    assert rows[0]["metric"] == "query_selfsim_t2i" and abs(rows[0]["value"] - 0.75) < 1e-6
    assert [round(r["value"], 6) for r in rows[1:]] == [0.0, 1.0, 1.0, 1.0]


def test_retrieval_and_retention_refuse_vectors_that_are_not_finite():
    # Every comparison with NaN is false, so without a check a NaN query would rank its
    # partner first and count as a hit.
    for bad in (float("nan"), float("inf")):
        queries = _one_hot(4)
        queries[2, 0] = bad
        with pytest.raises(ValueError, match="NaN or infinite"):
            recall_hits(queries, _one_hot(4), (1, 5))
        with pytest.raises(ValueError, match="not a finite number"):
            retention_margin(queries[2:3], _one_hot(4)[2:3], _one_hot(4))
