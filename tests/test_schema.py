# Catches accidental drift in the long-format CSV schema.
from __future__ import annotations

from src.schema import COLUMNS, empty_row


EXPECTED_COLUMNS = {
    "run_tag", "timestamp", "model", "seed", "pair_id",
    "vision_corruption", "image_severity", "text_corruption", "text_severity",
    "spoke", "depth", "depth_layer_index", "match_retention_direction",
    "metric", "value", "k_pool", "n_eval", "notes",
}


def test_columns_set_matches_expected():
    assert set(COLUMNS) == EXPECTED_COLUMNS


def test_columns_order_is_stable_18_wide():
    assert len(COLUMNS) == 18


def test_empty_row_returns_all_columns_blank():
    row = empty_row()
    assert set(row.keys()) == EXPECTED_COLUMNS
    for v in row.values():
        assert v == ""
