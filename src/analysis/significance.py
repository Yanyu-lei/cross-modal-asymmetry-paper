"""Paired Wilcoxon signed-rank + Cohen's d for the i2t vs t2i Recall@1 comparisons."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Two-sided Wilcoxon signed-rank on paired arrays a and b. Returns (W, p)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size < 2:
        return float("nan"), float("nan")
    diffs = a - b
    nz = diffs[diffs != 0]
    if nz.size < 2:
        return float("nan"), 1.0
    res = stats.wilcoxon(a, b, zero_method="wilcox", correction=False, mode="auto")
    return float(res.statistic), float(res.pvalue)


def cohens_d(a: np.ndarray, b: np.ndarray, *, paired: bool = True) -> float:
    """Cohen's d (paired or independent). Returns nan on degenerate input."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    if paired:
        if a.shape != b.shape:
            raise ValueError("paired d requires equal-length arrays")
        diffs = a - b
        sd = diffs.std(ddof=1)
        if sd <= 0:
            return float("nan")
        return float(diffs.mean() / sd)
    pooled_var = (a.var(ddof=1) + b.var(ddof=1)) / 2.0
    if pooled_var <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / np.sqrt(pooled_var))


def per_pair_wilcoxon_d(
    pair_df_a: pd.DataFrame,
    pair_df_b: pd.DataFrame,
    *,
    value_col: str = "value",
    keys: tuple = ("seed", "pair_id"),
) -> Tuple[float, float, float]:
    """Aligns two per-pair frames by `keys`, returns (W, p, cohens_d_paired)."""
    a = pair_df_a.set_index(list(keys)).sort_index()[value_col].to_numpy(dtype=float)
    b = pair_df_b.set_index(list(keys)).sort_index()[value_col].to_numpy(dtype=float)
    W, p = paired_wilcoxon(a, b)
    d = cohens_d(a, b, paired=True)
    return W, p, d


def cohens_d_binary_indep(p1: float, p2: float) -> float:
    """Cohen's d for two independent binary samples given their means.
    sigma_pooled = sqrt((p1(1-p1) + p2(1-p2)) / 2). Returns nan on degenerate input."""
    pooled_var = (p1 * (1 - p1) + p2 * (1 - p2)) / 2.0
    if pooled_var <= 0:
        return float("nan")
    return float((p1 - p2) / pooled_var ** 0.5)
