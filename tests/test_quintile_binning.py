# Within-modality quintile assignment correctness on synthetic damage values.
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.quintile import assign_quintile, QUINTILE_LABELS


def test_q1_contains_lowest_q5_contains_highest():
    damage = np.linspace(0, 1, 100)
    q = assign_quintile(damage)
    arr = np.asarray(q)
    assert (arr[:20] == "Q1").all()
    assert (arr[80:] == "Q5").all()


def test_quintiles_have_balanced_counts():
    damage = np.random.default_rng(0).uniform(0, 1, 1000)
    q = assign_quintile(damage)
    counts = pd.Series(q).value_counts().to_dict()
    for label in QUINTILE_LABELS:
        assert 180 <= counts[label] <= 220, f"unbalanced {label}: {counts[label]}"


def test_ties_handled_consistently():
    # Half the data is the same value; pd.qcut should not crash with duplicates='drop'
    damage = np.r_[np.zeros(50), np.linspace(0.01, 1, 50)]
    q = assign_quintile(damage)
    assert len(q) == 100


def test_labels_default_match_constant():
    damage = np.linspace(0, 1, 50)
    q = assign_quintile(damage)
    assert set(q.dropna().astype(str).unique()).issubset(set(QUINTILE_LABELS))
