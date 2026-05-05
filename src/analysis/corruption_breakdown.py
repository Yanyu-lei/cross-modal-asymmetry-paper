"""Per-corruption-type quintile breakdown for F6.

The matched-quintile calibration test pools all 3 image corruptions and all 3
text corruptions within modality. F6 needs the same test broken out by
corruption type to show whether the asymmetry holds for each of the 6
corruptions individually.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .bootstrap import hier_boot_diff, hier_boot_mean


ROOT = Path(__file__).resolve().parents[2]
QUINTILES = ["Q1", "Q2", "Q3", "Q4", "Q5"]


def per_corruption_quintile_table(
    *,
    n_boot: int = 10000,
) -> pd.DataFrame:
    """Build a long-format table with one row per (model, corruption_type, quintile).
    Columns: model, modality, corruption_type, quintile, n, mean_R@1, ci_lo, ci_hi.
    """
    pp = pd.read_csv(ROOT / "results/per_pair_retrieval/per_pair_retrieval.csv", low_memory=False)
    pp = pp[pp["spoke"] == "retrieval_per_pair"].copy()
    for c in ("seed", "pair_id", "image_severity", "text_severity"):
        pp[c] = pd.to_numeric(pp[c], errors="coerce").astype("Int64")

    dmg = pd.read_csv(ROOT / "results/input_damage_retrieval.csv")
    dmg["seed"] = dmg["seed"].astype(int)
    dmg["pair_id"] = dmg["pair_id"].astype(int)
    dmg["severity"] = dmg["severity"].astype(int)

    rows = []

    # Image side: i2t Recall@1 vs damage_ssim quintile
    img_pp = pp[pp["metric"] == "per_pair_recall_at_1_i2t"][
        ["seed", "model", "pair_id", "vision_corruption", "image_severity", "value"]
    ].copy()
    img_pp["value"] = pd.to_numeric(img_pp["value"]).astype(int)
    img_dmg = dmg[dmg["modality"] == "image"][["seed", "pair_id", "corruption_type", "severity", "damage_ssim"]]
    img_dmg = img_dmg.rename(columns={"corruption_type": "vision_corruption", "severity": "image_severity"})
    img = img_pp.merge(img_dmg, on=["seed", "pair_id", "vision_corruption", "image_severity"], how="inner").rename(
        columns={"vision_corruption": "corruption_type"})

    for (m, ct), sub in img.groupby(["model", "corruption_type"]):
        sub = sub.copy()
        sub["q"] = pd.qcut(sub["damage_ssim"], q=5, labels=QUINTILES, duplicates="drop")
        for q in QUINTILES:
            sub_q = sub[sub["q"] == q]
            if len(sub_q) < 10:
                continue
            mean, lo, hi = hier_boot_mean(sub_q, n_boot=n_boot)
            rows.append({"model": m, "modality": "image", "corruption_type": ct, "quintile": q,
                         "n": len(sub_q), "mean": mean, "ci_lo": lo, "ci_hi": hi})

    # Text side: t2i Recall@1 vs damage_bleu quintile
    txt_pp = pp[pp["metric"] == "per_pair_recall_at_1_t2i"][
        ["seed", "model", "pair_id", "text_corruption", "text_severity", "value"]
    ].copy()
    txt_pp["value"] = pd.to_numeric(txt_pp["value"]).astype(int)
    txt_dmg = dmg[dmg["modality"] == "text"][["seed", "pair_id", "corruption_type", "severity", "damage_bleu"]]
    txt_dmg = txt_dmg.rename(columns={"corruption_type": "text_corruption", "severity": "text_severity"})
    txt = txt_pp.merge(txt_dmg, on=["seed", "pair_id", "text_corruption", "text_severity"], how="inner").rename(
        columns={"text_corruption": "corruption_type"})

    for (m, ct), sub in txt.groupby(["model", "corruption_type"]):
        sub = sub.copy()
        sub["q"] = pd.qcut(sub["damage_bleu"], q=5, labels=QUINTILES, duplicates="drop")
        for q in QUINTILES:
            sub_q = sub[sub["q"] == q]
            if len(sub_q) < 10:
                continue
            mean, lo, hi = hier_boot_mean(sub_q, n_boot=n_boot)
            rows.append({"model": m, "modality": "text", "corruption_type": ct, "quintile": q,
                         "n": len(sub_q), "mean": mean, "ci_lo": lo, "ci_hi": hi})

    return pd.DataFrame(rows)
