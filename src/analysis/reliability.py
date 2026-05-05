"""Cross-seed Spearman reliability for the appendix supplementary."""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def per_cell_spearman(
    df: pd.DataFrame,
    *,
    value_col: str = "value",
    seed_col: str = "seed",
    pair_col: str = "pair_id",
    cell_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """For each unique cell (defined by `cell_cols`), compute Spearman correlation
    between every pair of seeds (across pair_id). Returns long-format DataFrame
    with columns: *cell_cols, seed_a, seed_b, spearman_rho, p, n.
    """
    seeds = sorted(df[seed_col].unique())
    rows = []
    cell_keys = list(cell_cols) if cell_cols else []
    if cell_keys:
        groups = df.groupby(cell_keys)
    else:
        groups = [(("all",), df)]
    for key, sub in groups:
        seed_pivot = sub.pivot_table(index=pair_col, columns=seed_col, values=value_col)
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                a, b = seed_pivot[seeds[i]], seed_pivot[seeds[j]]
                mask = a.notna() & b.notna()
                if mask.sum() < 3:
                    rho, p = float("nan"), float("nan")
                else:
                    rho, p = spearmanr(a[mask].to_numpy(), b[mask].to_numpy())
                row = {col: val for col, val in zip(cell_keys, key if isinstance(key, tuple) else (key,))} if cell_keys else {"cell": "all"}
                row.update({"seed_a": seeds[i], "seed_b": seeds[j], "spearman_rho": float(rho), "p": float(p), "n": int(mask.sum())})
                rows.append(row)
    return pd.DataFrame(rows)


def per_cell_mean_spearman(
    df: pd.DataFrame,
    *,
    value_col: str = "value",
    seed_col: str = "seed",
    cell_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """For retrieval per-pair data where pair_id is position-stable but
    item-different across seeds, compute Spearman across (cell mean) values
    rather than across pair_ids within a cell. For each cell defined by
    `cell_cols`, take the mean of `value_col` per seed; then for each seed
    pair, Spearman-correlate the cell-mean vectors across all cells. Returns
    long-format DataFrame with columns: seed_a, seed_b, spearman_rho, p, n_cells.
    """
    if not cell_cols:
        raise ValueError("cell_cols required")
    cell_keys = list(cell_cols)
    cell_means = df.groupby(cell_keys + [seed_col])[value_col].mean().reset_index()
    pivot = cell_means.pivot_table(index=cell_keys, columns=seed_col, values=value_col)
    seeds = sorted(pivot.columns)
    rows = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            a, b = pivot[seeds[i]], pivot[seeds[j]]
            mask = a.notna() & b.notna()
            if mask.sum() < 3:
                rho, p = float("nan"), float("nan")
            else:
                rho, p = spearmanr(a[mask].to_numpy(), b[mask].to_numpy())
            rows.append({
                "seed_a": int(seeds[i]),
                "seed_b": int(seeds[j]),
                "spearman_rho": float(rho),
                "p": float(p),
                "n_cells": int(mask.sum()),
            })
    return pd.DataFrame(rows)


def per_model_min_rho(reliability_df: pd.DataFrame, model_col: str = "model") -> pd.DataFrame:
    """Summarize reliability table to one row per model: min, median, mean rho."""
    if model_col not in reliability_df.columns:
        return pd.DataFrame()
    return (reliability_df.groupby(model_col)["spearman_rho"]
            .agg(min_rho="min", median_rho="median", mean_rho="mean")
            .reset_index())
