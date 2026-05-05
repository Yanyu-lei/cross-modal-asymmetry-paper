"""Per-cell paired Wilcoxon and Cohen's d for the matched-quintile cross-modal
Recall@1 comparison.

Pairing structure: within each (model, quintile) cell, the i2t and t2i arms
contain different per-pair sets (quintile assignment is per-direction; pairs
land in different quintiles depending on which side's damage ruler is used).
The defensible paired unit is therefore (model, seed) cell-mean Recall@1:
each seed contributes one i2t-mean and one t2i-mean per (model, quintile)
cell. Per-cell paired Wilcoxon then operates over n=3 paired observations.

A separate Cohen's d for each cell is computed as an independent-samples d
on the per-pair binary indicators within the cell (different pair sets, so
"paired" doesn't apply at the per-pair level even though "paired" applies
at the seed level for the W test).

Headline row (model="ALL", quintile="Q5") aggregates across the 5 Q5 cells:
15 (model x seed) cell-mean pairs at Q5 give the headline paired Wilcoxon
W=0, p<0.001 reported in the paper. The headline mean per-cell Cohen's d
for Q5 (independent-samples on per-pair binary) is the value the paper
quotes as 1.15.

Reads:
    results/per_pair_retrieval_v2/per_pair_retrieval_v2.csv
    results/input_damage_retrieval.csv
    (per_pair_retrieval_v2 is regenerable via the inference pipeline; NOT
    shipped with this code release.)

Writes:
    results/paired_tests_summary.csv
    (25 cells = 5 models x 5 quintiles, plus one model="ALL", quintile="Q5"
    headline row.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INPUT_PP_CSV = ROOT / "results/per_pair_retrieval_v2/per_pair_retrieval_v2.csv"
INPUT_DMG_CSV = ROOT / "results/input_damage_retrieval.csv"
OUTPUT_CSV = ROOT / "results/paired_tests_summary.csv"
QUINTILE_LABELS = ["Q1", "Q2", "Q3", "Q4", "Q5"]


def _check_inputs() -> None:
    for p in (INPUT_PP_CSV, INPUT_DMG_CSV):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Regenerate via the inference pipeline "
                "before running this script (see README, 'Re-running model inference')."
            )


def _load_per_pair() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PP_CSV)
    pp = df[
        df["pair_id"].notna()
        & df["metric"].isin(["per_pair_recall_at_1_i2t", "per_pair_recall_at_1_t2i"])
    ].copy()
    pp["pair_id"] = pp["pair_id"].astype(int)
    pp["seed"] = pp["seed"].astype(int)
    pp["value"] = pd.to_numeric(pp["value"]).astype(int)
    return pp


def _join_with_damage(pp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dmg = pd.read_csv(INPUT_DMG_CSV)
    img_dmg = (
        dmg[dmg["modality"] == "image"][
            ["seed", "pair_id", "corruption_type", "severity", "damage_ssim"]
        ]
        .rename(columns={"severity": "image_severity", "corruption_type": "vision_corruption"})
        .astype({"seed": int, "pair_id": int, "image_severity": int})
    )
    txt_dmg = (
        dmg[dmg["modality"] == "text"][
            ["seed", "pair_id", "corruption_type", "severity", "damage_bleu"]
        ]
        .rename(columns={"severity": "text_severity", "corruption_type": "text_corruption"})
        .astype({"seed": int, "pair_id": int, "text_severity": int})
    )

    i2t = pp[pp["metric"] == "per_pair_recall_at_1_i2t"][
        ["seed", "model", "pair_id", "vision_corruption", "image_severity", "value"]
    ].copy()
    i2t["image_severity"] = i2t["image_severity"].astype(int)
    i2t = i2t.merge(
        img_dmg,
        on=["seed", "pair_id", "vision_corruption", "image_severity"],
        how="inner",
    )

    t2i = pp[pp["metric"] == "per_pair_recall_at_1_t2i"][
        ["seed", "model", "pair_id", "text_corruption", "text_severity", "value"]
    ].copy()
    t2i["text_severity"] = t2i["text_severity"].astype(int)
    t2i = t2i.merge(
        txt_dmg,
        on=["seed", "pair_id", "text_corruption", "text_severity"],
        how="inner",
    )

    return i2t, t2i


def _per_cell_row(m: str, q: str, i_cell: pd.DataFrame, t_cell: pd.DataFrame) -> tuple[dict, np.ndarray, np.ndarray]:
    i_seed = i_cell.groupby("seed")["value"].mean()
    t_seed = t_cell.groupby("seed")["value"].mean()
    common = sorted(set(i_seed.index) & set(t_seed.index))
    i_arr = i_seed[common].to_numpy(dtype=float)
    t_arr = t_seed[common].to_numpy(dtype=float)

    try:
        res = wilcoxon(i_arr, t_arr, zero_method="wilcox", alternative="two-sided")
        W = float(res.statistic)
        p_val = float(res.pvalue)
    except ValueError:
        W, p_val = float("nan"), float("nan")

    diffs = i_arr - t_arr
    d_paired_seed = (
        float(diffs.mean() / diffs.std(ddof=1))
        if len(diffs) >= 2 and diffs.std(ddof=1) > 0
        else float("nan")
    )

    p1 = float(i_cell["value"].mean())
    p2 = float(t_cell["value"].mean())
    pooled = ((p1 * (1 - p1) + p2 * (1 - p2)) / 2) ** 0.5
    d_indep_pair = (p1 - p2) / pooled if pooled > 0 else float("nan")

    row = {
        "model": m,
        "quintile": q,
        "n_i2t_pairs": len(i_cell),
        "n_t2i_pairs": len(t_cell),
        "n_seeds_paired": len(common),
        "mean_i2t": p1,
        "mean_t2i": p2,
        "paired_W_seed": W,
        "paired_p_seed": p_val,
        "cohens_d_paired_seed": d_paired_seed,
        "cohens_d_indep_pair": d_indep_pair,
    }
    return row, i_arr, t_arr


def main() -> None:
    _check_inputs()
    pp = _load_per_pair()
    i2t, t2i = _join_with_damage(pp)

    rows: list[dict] = []
    headline_q5_i: list[float] = []
    headline_q5_t: list[float] = []

    for m in sorted(pp["model"].unique()):
        i_sub = i2t[i2t["model"] == m].copy()
        t_sub = t2i[t2i["model"] == m].copy()
        i_sub["quintile"] = pd.qcut(i_sub["damage_ssim"], q=5, labels=QUINTILE_LABELS, duplicates="drop")
        t_sub["quintile"] = pd.qcut(t_sub["damage_bleu"], q=5, labels=QUINTILE_LABELS, duplicates="drop")
        for q in QUINTILE_LABELS:
            i_cell = i_sub[i_sub["quintile"] == q]
            t_cell = t_sub[t_sub["quintile"] == q]
            if len(i_cell) == 0 or len(t_cell) == 0:
                continue
            row, i_arr, t_arr = _per_cell_row(m, q, i_cell, t_cell)
            rows.append(row)
            if q == "Q5":
                headline_q5_i.extend(i_arr.tolist())
                headline_q5_t.extend(t_arr.tolist())

    # Headline row over the 15 Q5 (model x seed) cell-mean pairs
    h_i = np.array(headline_q5_i)
    h_t = np.array(headline_q5_t)
    res_h = wilcoxon(h_i, h_t, zero_method="wilcox", alternative="two-sided")
    h_diff = h_i - h_t
    headline_row = {
        "model": "ALL",
        "quintile": "Q5",
        "n_i2t_pairs": np.nan,
        "n_t2i_pairs": np.nan,
        "n_seeds_paired": len(h_i),
        "mean_i2t": float(h_i.mean()),
        "mean_t2i": float(h_t.mean()),
        "paired_W_seed": float(res_h.statistic),
        "paired_p_seed": float(res_h.pvalue),
        "cohens_d_paired_seed": float(h_diff.mean() / h_diff.std(ddof=1)),
        "cohens_d_indep_pair": np.nan,
    }

    out = pd.concat([pd.DataFrame(rows), pd.DataFrame([headline_row])], ignore_index=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)

    q5_indep = out[(out["quintile"] == "Q5") & (out["model"] != "ALL")]["cohens_d_indep_pair"]
    print(f"Saved {OUTPUT_CSV}  ({len(out)} rows)")
    print(
        f"Headline Q5: paired W = {res_h.statistic:.0f}, "
        f"p = {res_h.pvalue:.2e}, "
        f"mean per-cell Cohen's d (independent) = {q5_indep.mean():.4f}"
    )


if __name__ == "__main__":
    main()
