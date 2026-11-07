"""The early text depths in the settings follow the text-depth diagnostic's rule."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from src import file_paths, settings
from src.load_models import ModelAdapter

EARLY_TEXT_COSINE = 0.95
EARLY_TEXT_FALLBACK = 0.40


def test_early_text_depths_follow_the_diagnostic():
    if not file_paths.TEXT_DEPTH_DIAGNOSTIC.exists():
        pytest.skip("no text-depth diagnostic file")
    df = pd.read_csv(file_paths.TEXT_DEPTH_DIAGNOSTIC)
    if "captions" not in df.columns:
        pytest.skip("this diagnostic file does not use the released captions")
    for model, rows in df[df["captions"] == "both"].groupby("model"):
        n = int(rows["n_text_layers"].iloc[0])
        tower = SimpleNamespace(n_text_layers=n)
        early, middle, _ = ModelAdapter.proportional_depths(tower, settings.MODEL_SETTINGS[model]["text_depth_fractions"], "text")
        below = rows.sort_values("layer_index").query("mean_cosine_clean_vs_mask_sev2 < @EARLY_TEXT_COSINE")["layer_index"]
        first = int(below.iloc[0]) if len(below) else n
        expected = first if first < middle else ModelAdapter.proportional_depths(tower, [EARLY_TEXT_FALLBACK], "text")[0]
        assert early == expected, model
