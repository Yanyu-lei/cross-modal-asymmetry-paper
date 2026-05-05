# load_pairs determinism + load_caption_pool exclusion behavior, on a synthetic manifest.
from __future__ import annotations

from src.data.coco import load_pairs, load_caption_pool


def test_load_pairs_deterministic_same_seed(synthetic_manifest):
    a = list(load_pairs(seed=3, n=5, manifest=synthetic_manifest))
    b = list(load_pairs(seed=3, n=5, manifest=synthetic_manifest))
    assert [t[2] for t in a] == [t[2] for t in b]
    assert [t[1] for t in a] == [t[1] for t in b]


def test_load_pairs_different_seed_different_order(synthetic_manifest):
    a = [t[2] for t in load_pairs(seed=0, n=5, manifest=synthetic_manifest)]
    b = [t[2] for t in load_pairs(seed=1, n=5, manifest=synthetic_manifest)]
    assert a != b


def test_caption_pool_excludes_eval_indices(synthetic_manifest):
    eval_idxs = {t[2] for t in load_pairs(seed=0, n=5, manifest=synthetic_manifest)}
    pool = load_caption_pool(seed=99, n=5, skip_indices=eval_idxs, manifest=synthetic_manifest)
    pool_idxs = {idx for _, idx in pool}
    assert pool_idxs.isdisjoint(eval_idxs)


def test_caption_pool_returns_requested_count(synthetic_manifest):
    pool = load_caption_pool(seed=0, n=4, manifest=synthetic_manifest)
    assert len(pool) == 4
