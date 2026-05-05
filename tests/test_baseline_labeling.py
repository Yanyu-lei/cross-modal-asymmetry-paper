# Protects against silent substitution of severity-1 for clean baseline.
from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.baseline import get_clean_baseline


def _df_with_clean_and_sev1(model="m1", clean_val=0.99, sev1_val=0.50):
    return pd.DataFrame([
        {"model": model, "metric": "recall_at_1_i2t", "image_severity": 0, "text_severity": 0, "value": clean_val},
        {"model": model, "metric": "recall_at_1_i2t", "image_severity": 1, "text_severity": 0, "value": sev1_val},
        {"model": model, "metric": "recall_at_1_i2t", "image_severity": 5, "text_severity": 0, "value": 0.10},
    ])


def test_clean_returned_not_sev1():
    df = _df_with_clean_and_sev1(clean_val=0.99, sev1_val=0.50)
    out = get_clean_baseline(df, model="m1", metric="recall_at_1_i2t")
    assert out == 0.99


def test_clean_and_sev1_are_distinct_columns():
    df = _df_with_clean_and_sev1(clean_val=0.99, sev1_val=0.50)
    clean_rows = df[(df["image_severity"] == 0) & (df["text_severity"] == 0)]
    sev1_rows = df[(df["image_severity"] == 1)]
    assert clean_rows["value"].iloc[0] != sev1_rows["value"].iloc[0]
    assert len(clean_rows) > 0 and len(sev1_rows) > 0


def test_missing_clean_raises_loudly():
    df = pd.DataFrame([
        {"model": "m1", "metric": "recall_at_1_i2t", "image_severity": 1, "text_severity": 0, "value": 0.50},
        {"model": "m1", "metric": "recall_at_1_i2t", "image_severity": 5, "text_severity": 0, "value": 0.10},
    ])
    with pytest.raises(ValueError, match="no clean baseline"):
        get_clean_baseline(df, model="m1", metric="recall_at_1_i2t")


def test_missing_required_columns_raises():
    df = pd.DataFrame([{"model": "m1", "value": 0.5}])
    with pytest.raises(KeyError):
        get_clean_baseline(df, model="m1", metric="anything")
