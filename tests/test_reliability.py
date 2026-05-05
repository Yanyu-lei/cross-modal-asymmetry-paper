# Cross-seed Spearman wrapper recovers a known correlation between two seeds.
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.reliability import per_cell_spearman


def test_recovers_known_high_correlation():
    rng = np.random.default_rng(0)
    n = 200
    seed_a_vals = rng.normal(0.5, 0.1, n)
    # Highly correlated with a tiny perturbation
    seed_b_vals = seed_a_vals + rng.normal(0.0, 0.005, n)
    df = pd.DataFrame({
        "seed": [0] * n + [1] * n,
        "pair_id": list(range(n)) * 2,
        "value": np.concatenate([seed_a_vals, seed_b_vals]),
    })
    out = per_cell_spearman(df)
    rho = out.iloc[0]["spearman_rho"]
    np.testing.assert_allclose(rho, 1.0, atol=0.02)


def test_zero_correlation_for_independent_seeds():
    rng = np.random.default_rng(1)
    n = 500
    a = rng.normal(0.5, 0.1, n); b = rng.normal(0.5, 0.1, n)
    df = pd.DataFrame({
        "seed": [0] * n + [1] * n,
        "pair_id": list(range(n)) * 2,
        "value": np.concatenate([a, b]),
    })
    rho = per_cell_spearman(df).iloc[0]["spearman_rho"]
    assert abs(rho) < 0.15  # ~0 for independent samples of n=500


def test_reports_all_seed_pairs():
    rng = np.random.default_rng(2)
    n = 100
    df = pd.DataFrame({
        "seed": [0] * n + [1] * n + [2] * n,
        "pair_id": list(range(n)) * 3,
        "value": rng.normal(0.5, 0.1, 3 * n),
    })
    out = per_cell_spearman(df)
    assert len(out) == 3  # (0,1), (0,2), (1,2)
