"""The bootstrap interval and the damage quintiles."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.bootstrap_and_quintiles import QUINTILES, bootstrap_ci, damage_quintile


def _rows(n_seeds=3, n_pairs=50, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame([{"seed": s, "pair_id": p, "value": float(rng.normal(0.5, 0.1))}
                         for s in range(n_seeds) for p in range(n_pairs)])


def test_bootstrap_point_is_the_mean_and_repeats():
    df = _rows()
    mean, lo, hi = bootstrap_ci(df, n_boot=500)
    assert abs(mean - df["value"].mean()) < 1e-12 and lo <= mean <= hi
    assert bootstrap_ci(df, n_boot=500) == (mean, lo, hi)


def test_bootstrap_moves_a_pair_with_all_its_rows():
    # Each pair has two identical rows, so resampling whole pairs gives the same interval as
    # resampling one row per pair.
    single = _rows(n_pairs=40, seed=3)
    doubled = pd.concat([single, single], ignore_index=True)
    assert bootstrap_ci(single, n_boot=300) == bootstrap_ci(doubled, n_boot=300)


def test_bootstrap_of_nothing_is_nan():
    assert all(np.isnan(v) for v in bootstrap_ci(pd.DataFrame({"seed": [], "pair_id": [], "value": []})))


def test_quintiles_split_the_range_in_fifths():
    q = np.asarray(damage_quintile(np.linspace(0, 1, 100)))
    assert (q[:20] == "Q1").all() and (q[80:] == "Q5").all()
    counts = pd.Series(np.asarray(damage_quintile(np.random.default_rng(0).uniform(0, 1, 1000)))).value_counts()
    assert all(180 <= counts[label] <= 220 for label in QUINTILES)


def test_quintiles_refuse_heavily_tied_damage():
    with pytest.raises(ValueError, match="tied"):
        damage_quintile(np.r_[np.zeros(50), np.linspace(0, 1, 10)])  # most values are exactly zero


def test_bootstrap_matches_explicit_resampling_with_unequal_seeds_and_repeats():
    # Seeds have different numbers of pairs and pairs different numbers of rows. Each draw
    # must average the rows of its resampled seeds and pairs together, like the point estimate.
    rng = np.random.default_rng(5)
    df = pd.DataFrame([{"seed": s, "pair_id": p, "value": float(rng.normal(s, 1.0))}
                       for s, n_pairs in enumerate((10, 25, 40)) for p in range(n_pairs)
                       for _ in range(1 + p % 3)])
    mean, lo, hi = bootstrap_ci(df, n_boot=200)

    draws_rng = np.random.default_rng(0)
    seeds = [df[df["seed"] == s] for s in sorted(df["seed"].unique())]
    pairs = [sorted(rows["pair_id"].unique()) for rows in seeds]
    draws = []
    for _ in range(200):
        picked_rows = []
        for s in draws_rng.integers(0, len(seeds), size=len(seeds)):
            for i in draws_rng.integers(0, len(pairs[s]), size=len(pairs[s])):
                picked_rows.append(seeds[s][seeds[s]["pair_id"] == pairs[s][i]]["value"])
        draws.append(pd.concat(picked_rows).mean())
    assert mean == pytest.approx(df["value"].mean(), abs=1e-12)
    assert (lo, hi) == pytest.approx((np.quantile(draws, 0.025), np.quantile(draws, 0.975)), abs=1e-12)
