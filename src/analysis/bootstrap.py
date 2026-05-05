"""Hierarchical bootstrap. ONE implementation; every figure/table imports from here.

Resample seeds (with replacement), then pairs within each sampled seed (with
replacement). For paired comparisons, use the same resampled pair indices
across both arms within each bootstrap draw.

Default n_boot=10000. Inputs accept long-format DataFrames keyed by seed_col +
pair_col + value_col. N is inferred from the data, not hardcoded.
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
def _per_seed_arrays(df: pd.DataFrame, value_col: str, seed_col: str, pair_col: str):
    """Return list of (seed, value_array, pair_id_array), one entry per seed.
    Allows different pair_id sets and different N per seed (retrieval uses
    different shuffles per seed, so pair_id 5 in seed 0 != pair_id 5 in seed 1)."""
    seeds = sorted(df[seed_col].unique())
    out = []
    for s in seeds:
        sub = df[df[seed_col] == s].sort_values(pair_col)
        out.append((s, sub[value_col].to_numpy(dtype=float), sub[pair_col].to_numpy()))
    return out


def _seed_pair_matrix(df: pd.DataFrame, value_col: str, seed_col: str, pair_col: str):
    """Return (n_seeds, n_pairs) matrix when pair_ids are balanced across seeds.
    Used by paired-diff for cases where a and b share (seed, pair_id) keys."""
    seeds = sorted(df[seed_col].unique())
    M_rows = []
    pair_ids = None
    for s in seeds:
        sub = df[df[seed_col] == s].sort_values(pair_col)
        ids = sub[pair_col].to_numpy()
        if pair_ids is None:
            pair_ids = ids
        else:
            if not np.array_equal(pair_ids, ids):
                raise ValueError(f"seed {s} has different pair_ids than seed {seeds[0]}")
        M_rows.append(sub[value_col].to_numpy(dtype=float))
    return np.stack(M_rows, axis=0), seeds, pair_ids


def _ci_quantiles(boots: np.ndarray, ci: float):
    a = (1 - ci) / 2
    return float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


# ---------------------------------------------------------------------------
def hier_boot_mean(
    df: pd.DataFrame,
    *,
    value_col: str = "value",
    seed_col: str = "seed",
    pair_col: str = "pair_id",
    n_boot: int = 10000,
    ci: float = 0.95,
    rng_seed: int = 0,
) -> Tuple[float, float, float]:
    """Hierarchical bootstrap mean. Returns (point_estimate, ci_low, ci_high).
    Allows different pair_id sets and different N across seeds (resamples each
    seed's pair list independently)."""
    arrays = _per_seed_arrays(df, value_col, seed_col, pair_col)
    if not arrays:
        return float("nan"), float("nan"), float("nan")
    seed_means = np.array([v.mean() for _, v, _ in arrays])
    point = float(np.concatenate([v for _, v, _ in arrays]).mean())  # over all observations equally weighted

    rng = np.random.default_rng(rng_seed)
    n_seeds = len(arrays)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        seed_idx = rng.integers(0, n_seeds, size=n_seeds)
        # for each chosen seed, resample its pairs with replacement and take the mean
        seed_resampled_means = np.empty(n_seeds)
        for k, si in enumerate(seed_idx):
            v = arrays[si][1]
            n = v.size
            if n == 0:
                seed_resampled_means[k] = np.nan
            else:
                idx = rng.integers(0, n, size=n)
                seed_resampled_means[k] = v[idx].mean()
        boots[b] = np.nanmean(seed_resampled_means)
    lo, hi = _ci_quantiles(boots, ci)
    return point, lo, hi


def hier_boot_diff(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    value_col: str = "value",
    seed_col: str = "seed",
    pair_col: str = "pair_id",
    n_boot: int = 10000,
    ci: float = 0.95,
    rng_seed: int = 0,
    paired: bool = True,
) -> Tuple[float, float, float]:
    """Bootstrap mean(a) - mean(b). If paired=True, both arms must share the same
    (seed, pair) index and we resample that index ONCE per bootstrap draw,
    preserving the within-pair comparison. If paired=False, resample each arm
    independently.
    """
    rng = np.random.default_rng(rng_seed)
    boots = np.empty(n_boot)
    if paired:
        M_a, seeds_a, pids_a = _seed_pair_matrix(df_a, value_col, seed_col, pair_col)
        M_b, seeds_b, pids_b = _seed_pair_matrix(df_b, value_col, seed_col, pair_col)
        if seeds_a != seeds_b or not np.array_equal(pids_a, pids_b):
            raise ValueError("paired=True requires identical seed × pair_id structure between a and b")
        n_seeds, n_pairs = M_a.shape
        for b in range(n_boot):
            seed_idx = rng.integers(0, n_seeds, size=n_seeds)
            pair_idx = rng.integers(0, n_pairs, size=(n_seeds, n_pairs))
            sample_a = M_a[seed_idx[:, None], pair_idx]
            sample_b = M_b[seed_idx[:, None], pair_idx]
            boots[b] = sample_a.mean() - sample_b.mean()
        point = float(M_a.mean() - M_b.mean())
    else:
        arr_a = _per_seed_arrays(df_a, value_col, seed_col, pair_col)
        arr_b = _per_seed_arrays(df_b, value_col, seed_col, pair_col)
        n_a, n_b = len(arr_a), len(arr_b)
        for b in range(n_boot):
            sm_a = np.empty(n_a); sm_b = np.empty(n_b)
            for k, si in enumerate(rng.integers(0, n_a, size=n_a)):
                v = arr_a[si][1]
                sm_a[k] = v[rng.integers(0, v.size, size=v.size)].mean() if v.size else np.nan
            for k, si in enumerate(rng.integers(0, n_b, size=n_b)):
                v = arr_b[si][1]
                sm_b[k] = v[rng.integers(0, v.size, size=v.size)].mean() if v.size else np.nan
            boots[b] = np.nanmean(sm_a) - np.nanmean(sm_b)
        point = float(np.concatenate([v for _, v, _ in arr_a]).mean()
                      - np.concatenate([v for _, v, _ in arr_b]).mean())
    lo, hi = _ci_quantiles(boots, ci)
    return point, lo, hi


def seed_mean_sd(
    df: pd.DataFrame,
    *,
    value_col: str = "value",
    seed_col: str = "seed",
) -> Tuple[float, float]:
    """Sanity check: mean and SD across the 3 seed-level means."""
    per_seed = df.groupby(seed_col)[value_col].mean().to_numpy(dtype=float)
    if per_seed.size < 1:
        return float("nan"), float("nan")
    return float(per_seed.mean()), float(per_seed.std(ddof=1)) if per_seed.size > 1 else 0.0


def fmt_ci(point: float, lo: float, hi: float, *, prec: int = 3) -> str:
    """Format (point, ci_lo, ci_hi) as 'X.XXX [Y.YYY, Z.ZZZ]' for tables."""
    if np.isnan(point):
        return "n/a"
    return f"{point:.{prec}f} [{lo:.{prec}f}, {hi:.{prec}f}]"


def fmt_seedcheck(mean: float, sd: float, *, prec: int = 3) -> str:
    """Format seed-level mean ± SD as 'X.XXX±Y.YYY'."""
    if np.isnan(mean):
        return "n/a"
    return f"{mean:.{prec}f}±{sd:.{prec}f}"
