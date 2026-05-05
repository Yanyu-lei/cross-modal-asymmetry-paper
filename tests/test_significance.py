# Wrappers in src.analysis.significance must agree with scipy / closed-form math.
from __future__ import annotations

import numpy as np
from scipy import stats

from src.analysis.significance import paired_wilcoxon, cohens_d


def test_paired_wilcoxon_matches_scipy():
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.1, 30)
    b = a + rng.normal(0.05, 0.02, 30)
    W, p = paired_wilcoxon(a, b)
    ref = stats.wilcoxon(a, b)
    np.testing.assert_allclose(W, ref.statistic, atol=1e-12)
    np.testing.assert_allclose(p, ref.pvalue, atol=1e-12)


def test_cohens_d_paired_matches_formula():
    a = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    b = a + 0.1
    diffs = a - b
    expected = diffs.mean() / diffs.std(ddof=1)
    np.testing.assert_allclose(cohens_d(a, b, paired=True), expected, atol=1e-12)


def test_cohens_d_independent_matches_formula():
    rng = np.random.default_rng(1)
    a = rng.normal(0.5, 0.1, 100)
    b = rng.normal(0.4, 0.1, 100)
    pooled = (a.var(ddof=1) + b.var(ddof=1)) / 2
    expected = (a.mean() - b.mean()) / np.sqrt(pooled)
    np.testing.assert_allclose(cohens_d(a, b, paired=False), expected, atol=1e-12)


def test_paired_wilcoxon_degenerate_returns_nan_and_p1():
    a = np.array([0.5, 0.5, 0.5])
    b = np.array([0.5, 0.5, 0.5])
    W, p = paired_wilcoxon(a, b)
    assert np.isnan(W) or W == 0
    np.testing.assert_allclose(p, 1.0, atol=1e-12)
