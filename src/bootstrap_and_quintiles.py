"""Compute the bootstrap intervals and damage quintiles."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from src import settings

QUINTILES = ("Q1", "Q2", "Q3", "Q4", "Q5")


def bootstrap_ci(df: pd.DataFrame, *, value_col: str = "value", n_boot: int = settings.BOOTSTRAP_DRAWS,
                 ci: float = 0.95, rng_seed: int = 0) -> tuple[float, float, float]:
    """Return the mean of the rows and the percentile bootstrap bounds.

    Resample the seeds, then whole pairs within each sampled seed, with replacement
    and equal weights for the sampled rows.
    """
    if df.empty:
        return float("nan"), float("nan"), float("nan")
    per_seed = []
    for seed in sorted(df["seed"].unique()):
        rows = df[df["seed"] == seed]
        by_pair = rows.groupby("pair_id", sort=True, dropna=False)[value_col]
        per_seed.append((rows.sort_values("pair_id")[value_col].to_numpy(dtype=float),
                         by_pair.sum().to_numpy(dtype=float), by_pair.count().to_numpy(dtype=float)))
    point = float(np.concatenate([values for values, _, _ in per_seed]).mean())

    rng = np.random.default_rng(rng_seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        total, count = 0.0, 0.0
        for s in rng.integers(0, len(per_seed), size=len(per_seed)):
            _, sums, counts = per_seed[s]
            picked = rng.integers(0, sums.size, size=sums.size)
            total += sums[picked].sum()
            count += counts[picked].sum()
        draws[b] = total / count
    tail = (1 - ci) / 2
    return point, float(np.quantile(draws, tail)), float(np.quantile(draws, 1 - tail))


def bootstrap_group_statistics(groups: dict[str, pd.DataFrame], statistics: dict[str, Callable[[dict[str, float]], float]],
                               *, value_col: str = "value", n_boot: int = settings.BOOTSTRAP_DRAWS,
                               ci: float = 0.95, rng_seed: int = 0) -> dict[str, tuple[float, float, float]]:
    """Return each statistic of the group means with its percentile bootstrap bounds.

    Resample the seeds, then whole pairs within each sampled seed, as bootstrap_ci does. Every
    group uses the same picks within a draw, so a ratio or a difference keeps the pairing of its
    groups. Group membership stays fixed, so quintile edges do not move between draws. A group
    that a draw leaves empty has a missing mean, and the statistic decides how to handle it.
    """
    names = list(groups)
    per_seed = []
    for seed in sorted(set().union(*(set(rows["seed"].unique()) for rows in groups.values()))):
        parts = [rows[rows["seed"] == seed].groupby("pair_id", sort=True, dropna=False)[value_col].agg(["sum", "count"])
                 for rows in groups.values()]
        pairs = sorted(set().union(*(set(part.index) for part in parts)))
        per_seed.append(tuple(np.column_stack([part[column].reindex(pairs, fill_value=0).to_numpy(dtype=float) for part in parts])
                              for column in ("sum", "count")))
    point = {name: float(groups[name][value_col].to_numpy(dtype=float).mean()) for name in names}

    rng = np.random.default_rng(rng_seed)
    draws = np.empty((n_boot, len(statistics)))
    for b in range(n_boot):
        total, count = np.zeros(len(names)), np.zeros(len(names))
        for s in rng.integers(0, len(per_seed), size=len(per_seed)):
            sums, counts = per_seed[s]
            picked = rng.integers(0, sums.shape[0], size=sums.shape[0])
            total += sums[picked].sum(axis=0)
            count += counts[picked].sum(axis=0)
        means = dict(zip(names, np.divide(total, count, out=np.full(len(names), np.nan), where=count > 0)))
        draws[b] = [statistic(means) for statistic in statistics.values()]
    tail = (1 - ci) / 2
    return {name: (float(statistic(point)), float(np.nanquantile(draws[:, i], tail)), float(np.nanquantile(draws[:, i], 1 - tail)))
            for i, (name, statistic) in enumerate(statistics.items())}


def damage_quintile(damage) -> pd.Categorical:
    """Assign the damage quintiles from Q1 (least damaged) to Q5 (most damaged).

    Tied values can make the bins unequal in size. Repeated boundaries raise a ValueError.
    """
    if len(np.unique(np.quantile(np.asarray(damage, dtype=float), np.linspace(0, 1, 6)))) < 6:
        raise ValueError("Too many damage values are tied to cut them into five quintiles.")
    return pd.qcut(damage, q=5, labels=list(QUINTILES))
