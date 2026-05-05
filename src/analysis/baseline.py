# Strict clean-baseline accessor. Refuses to silently substitute sev=1 for
# missing sev=0 rows; protects against the F1/T2 caption-rounds bug class
# where "minimal damage" was treated as "clean".
from __future__ import annotations

from typing import Optional

import pandas as pd


CLEAN_VALUE = 0  # the canonical encoding for clean (no corruption) on each severity column


def get_clean_baseline(
    df: pd.DataFrame,
    *,
    model: str,
    metric: str,
    image_severity_col: str = "image_severity",
    text_severity_col: str = "text_severity",
    model_col: str = "model",
    metric_col: str = "metric",
    value_col: str = "value",
) -> float:
    """Mean of the clean (image_sev=0 AND text_sev=0) rows for the given model
    and metric. Raises ValueError if no such rows exist; never silently
    substitutes the mildest-corruption (sev=1) value as a fallback."""
    needed = {image_severity_col, text_severity_col, model_col, metric_col, value_col}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")

    sub = df[
        (df[model_col] == model)
        & (df[metric_col] == metric)
        & (pd.to_numeric(df[image_severity_col], errors="coerce").fillna(-1).astype(int) == CLEAN_VALUE)
        & (pd.to_numeric(df[text_severity_col], errors="coerce").fillna(-1).astype(int) == CLEAN_VALUE)
    ]
    if len(sub) == 0:
        raise ValueError(
            f"no clean baseline rows (image_severity={CLEAN_VALUE} AND text_severity={CLEAN_VALUE}) "
            f"for model={model!r}, metric={metric!r}; do not silently substitute sev=1 as a fallback"
        )
    return float(pd.to_numeric(sub[value_col]).mean())
