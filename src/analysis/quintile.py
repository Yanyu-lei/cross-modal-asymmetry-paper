# Within-modality damage quintile binning. Wrapper around pd.qcut so the
# binning rule has a single point of test coverage.
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

QUINTILE_LABELS = ("Q1", "Q2", "Q3", "Q4", "Q5")


def assign_quintile(damage: np.ndarray | pd.Series, *, labels: Sequence[str] = QUINTILE_LABELS) -> pd.Series:
    """Assign each damage value to one of 5 quintiles by within-array rank.
    Q1 = bottom 20% (least damaged), Q5 = top 20% (most damaged). When ties
    collapse some quintile boundaries, the resulting bin count may be < 5;
    bin 0 always maps to the first label (Q1) and increasing bins map to
    increasing labels."""
    series = pd.Series(damage).reset_index(drop=True)
    bin_idx = pd.qcut(series, q=5, labels=False, duplicates="drop")
    label_list = list(labels)
    return bin_idx.map(lambda i: label_list[int(i)] if pd.notna(i) else None).astype("category")
