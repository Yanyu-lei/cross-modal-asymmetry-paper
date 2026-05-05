"""Cross-seed reliability for retrieval per-pair Recall@1 indicators.

Reads:
    results/per_pair_retrieval_v2/per_pair_retrieval_v2.csv
    (regenerable via the inference pipeline: scripts/run_full_experiment.py
    with the per-pair retrieval log enabled, then aggregated to v2 path;
    NOT shipped with this code release because file size exceeds practical
    limits for code-repo distribution.)

Writes:
    results/reliability_cell_mean_summary.csv
    (one row per (seed_a, seed_b) pair: 0/1, 0/2, 1/2; Spearman correlation
    of cell-mean Recall@1 across the 150 (model x metric x corruption x
    severity) cells per seed.)

Why cell-mean rather than per-pair Spearman: retrieval uses non-overlapping
caption manifests per seed (pair_id 5 in seed 0 indexes a different COCO item
than pair_id 5 in seed 1), so per_cell_spearman applied to per-pair retrieval
rows correlates position-aligned vectors of different items and returns
rho ~ 0. Cell-mean Spearman correlates the cell-level mean R@1 across seeds,
which is the meaningful reliability quantity for retrieval data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analysis.reliability import per_cell_mean_spearman  # noqa: E402

INPUT_CSV = ROOT / "results/per_pair_retrieval_v2/per_pair_retrieval_v2.csv"
OUTPUT_CSV = ROOT / "results/reliability_cell_mean_summary.csv"


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"{INPUT_CSV} not found. Regenerate via the inference pipeline "
            "before running this script (see README, 'Re-running model inference')."
        )

    df = pd.read_csv(INPUT_CSV)
    sub = df[
        df["pair_id"].notna()
        & df["metric"].isin(["per_pair_recall_at_1_i2t", "per_pair_recall_at_1_t2i"])
    ].copy()

    out = per_cell_mean_spearman(
        sub,
        value_col="value",
        seed_col="seed",
        cell_cols=[
            "model",
            "metric",
            "vision_corruption",
            "image_severity",
            "text_corruption",
            "text_severity",
        ],
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
