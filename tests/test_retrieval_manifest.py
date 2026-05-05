# Multi-positive retrieval semantics + duplicate-id detection.
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.analysis.retrieval_eval import multi_positive_recall_at_k, assert_no_duplicate_ids


def _embeddings_with_known_topology():
    """3 query images, 5 candidate captions. Image 0 has captions 0 AND 3 as
    valid matches. Image 1: caption 1. Image 2: caption 2.
    Make captions 0 and 3 highly similar to image 0 (both should be top-ranked)."""
    rng = np.random.default_rng(0)
    D = 16
    img = rng.normal(0, 1, (3, D))
    cap = rng.normal(0, 1, (5, D))
    # Image 0 close to caption 0 and 3
    cap[0] = img[0] + 0.01 * rng.normal(0, 1, D)
    cap[3] = img[0] + 0.01 * rng.normal(0, 1, D)
    # Image 1 close to caption 1, image 2 close to caption 2
    cap[1] = img[1] + 0.01 * rng.normal(0, 1, D)
    cap[2] = img[2] + 0.01 * rng.normal(0, 1, D)
    return torch.tensor(img, dtype=torch.float32), torch.tensor(cap, dtype=torch.float32)


def test_multi_positive_hit_counts_either_match():
    img, cap = _embeddings_with_known_topology()
    gt = [[0, 3], [1], [2]]
    r = multi_positive_recall_at_k(img, cap, gt, ks=(1, 5))
    assert r[1] == 1.0  # all queries hit top-1


def test_single_positive_assumption_loses_a_hit():
    """If we ignore multi-positive structure and only label caption 4 (wrong) as
    image 0's match, R@1 should drop because the actual nearest captions are 0/3."""
    img, cap = _embeddings_with_known_topology()
    gt_single_wrong = [[4], [1], [2]]
    r = multi_positive_recall_at_k(img, cap, gt_single_wrong, ks=(1,))
    assert r[1] < 1.0


def test_duplicate_ids_raise():
    with pytest.raises(ValueError, match="duplicate"):
        assert_no_duplicate_ids([1, 2, 3, 2, 4])


def test_unique_ids_pass():
    assert_no_duplicate_ids([1, 2, 3, 4, 5])


def test_gt_length_mismatch_raises():
    img, cap = _embeddings_with_known_topology()
    with pytest.raises(ValueError, match="gt_indices length"):
        multi_positive_recall_at_k(img, cap, [[0]], ks=(1,))
