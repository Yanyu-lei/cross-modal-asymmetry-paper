# Hierarchical bootstrap basic behavior; no exact-CI assertions.
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.bootstrap import hier_boot_mean, hier_boot_diff, seed_mean_sd


def _balanced_df(n_seeds=3, n_pairs=50, mean=0.5, std=0.1, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_seeds):
        for p in range(n_pairs):
            rows.append({"seed": s, "pair_id": p, "value": float(rng.normal(mean, std))})
    return pd.DataFrame(rows)


def test_point_estimate_equals_raw_mean():
    df = _balanced_df()
    mean, lo, hi = hier_boot_mean(df, n_boot=500, rng_seed=1)
    np.testing.assert_allclose(mean, df["value"].mean(), atol=1e-12)
    assert lo <= hi


def test_reproducibility_with_fixed_seed():
    df = _balanced_df()
    a = hier_boot_mean(df, n_boot=500, rng_seed=7)
    b = hier_boot_mean(df, n_boot=500, rng_seed=7)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_paired_diff_point_estimate_paired_branch():
    df_a = _balanced_df(seed=10)
    df_b = df_a.copy()
    df_b["value"] = df_b["value"] + 0.1
    pt, lo, hi = hier_boot_diff(df_a, df_b, n_boot=500, paired=True)
    np.testing.assert_allclose(pt, df_a["value"].mean() - df_b["value"].mean(), atol=1e-12)
    assert lo <= hi


def test_paired_diff_point_estimate_unpaired_branch():
    df_a = _balanced_df(seed=20, n_pairs=40)
    df_b = _balanced_df(seed=21, n_pairs=60, mean=0.7)  # different N and mean
    pt, lo, hi = hier_boot_diff(df_a, df_b, n_boot=500, paired=False)
    np.testing.assert_allclose(pt, df_a["value"].mean() - df_b["value"].mean(), atol=1e-12)
    assert lo <= hi


def test_paired_branch_rejects_mismatched_pair_ids():
    import pytest as _pt
    df_a = _balanced_df(seed=30, n_pairs=50)
    df_b = _balanced_df(seed=31, n_pairs=50)
    df_b["pair_id"] = df_b["pair_id"] + 100  # disjoint pair_id sets
    with _pt.raises(ValueError, match="paired"):
        hier_boot_diff(df_a, df_b, n_boot=10, paired=True)


def test_degenerate_input_returns_nan():
    df = pd.DataFrame({"seed": [], "pair_id": [], "value": []})
    mean, lo, hi = hier_boot_mean(df, n_boot=10)
    assert np.isnan(mean) and np.isnan(lo) and np.isnan(hi)


def test_seed_mean_sd_against_raw():
    df = _balanced_df(n_seeds=3, n_pairs=50, seed=42)
    m, sd = seed_mean_sd(df)
    per_seed = df.groupby("seed")["value"].mean().to_numpy()
    np.testing.assert_allclose(m, per_seed.mean(), atol=1e-12)
    np.testing.assert_allclose(sd, per_seed.std(ddof=1), atol=1e-12)
