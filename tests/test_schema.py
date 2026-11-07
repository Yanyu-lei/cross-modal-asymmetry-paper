"""The long-format output columns and their order stay fixed."""
from __future__ import annotations

from src.evaluate_model import COLUMNS, empty_row


EXPECTED_COLUMNS = (
    "run_tag", "timestamp", "model", "seed", "pair_id",
    "vision_corruption", "image_severity", "text_corruption", "text_severity",
    "spoke", "depth", "depth_layer_index", "match_retention_direction",
    "metric", "value", "n_negatives", "n_eval", "notes",
)


def test_columns_and_their_order_are_stable():
    assert tuple(COLUMNS) == EXPECTED_COLUMNS


def test_empty_row_returns_all_columns_blank():
    row = empty_row()
    assert set(row.keys()) == set(EXPECTED_COLUMNS)
    for v in row.values():
        assert v == ""
