"""Long-format CSV schema for experiment results.

One row = one (model, pair, condition, spoke, depth-or-aux, metric) measurement.
Many spokes don't fill every column; blanks are written as empty strings.

Spoke -> metric_name mapping:
    image_fidelity   : "patch_cosine_mean"   (per depth)
    text_fidelity    : "pooled_cosine"        (per depth)
    match_retention  : one of {"sim_match", "sim_mismatch_mean", "retention_margin"}
    retrieval        : one of {"recall_at_1_i2t", "recall_at_5_i2t", "recall_at_10_i2t",
                               "recall_at_1_t2i", "recall_at_5_t2i", "recall_at_10_t2i"}

Joint-grid rows (5x5 Gaussian-noise x text-mask) populate BOTH image_severity and
text_severity > 0; single-modality rows leave the other modality at severity 0.
"""
from __future__ import annotations

# Stable column order for the results CSV.
COLUMNS: tuple[str, ...] = (
    "run_tag",                  # str  – free-text label per run batch
    "timestamp",                # iso8601 string
    "model",                    # str  – key from configs/models.yaml
    "seed",                     # int  – one of {0, 1, 2}
    "pair_id",                  # int  – COCO pair index inside this seed's slice; "" for retrieval rollups
    "vision_corruption",        # "none"|"gaussian_noise"|"gaussian_blur"|"cutout"
    "image_severity",           # int 0..5; 0 = clean reference (clean reference rows are not written)
    "text_corruption",          # "none"|"mask"|"shuffle"|"replace"
    "text_severity",            # int 0..5
    "spoke",                    # "image_fidelity"|"text_fidelity"|"match_retention"|"retrieval"
    "depth",                    # "early"|"mid"|"late" (image- or text-encoder layer family); "" otherwise
    "depth_layer_index",        # int  – concrete block index for that depth; "" otherwise
    "match_retention_direction",# "image_corrupted"|"text_corrupted"|""  - direction of corruption
    "metric",                   # spoke-specific metric name (see module docstring)
    "value",                    # float
    "k_pool",                   # int  – match_retention pool size; "" otherwise
    "n_eval",                   # int  – retrieval eval-set size; "" otherwise
    "notes",                    # str  – freeform; usually ""
)


def empty_row() -> dict:
    """Return a dict with every column present and set to ''. Callers fill in
    what applies and write the row as-is."""
    return {c: "" for c in COLUMNS}
