"""Upper-bound analysis: does the matched-quintile asymmetry survive
when we substitute mean-pool text retention for standard-pool text retention?

If yes, the bottleneck explains at most what mean-pool already closed (the
"drop reduction" measured in scripts/analyze_recall1_quintile.py); the rest
is text-encoder-specific.

Compare:
    image-side standard-pool retention vs damage_ssim quintile
    text-side  MEAN-POOL    retention vs damage_bleu quintile

both within matched within-modality damage quintiles. Show whether
image-side > text-side(mean-pool) holds quintile by quintile per model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.bootstrap import boot_mean_ci, boot_diff_ci

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2)
QUINTILES = ["Q1", "Q2", "Q3", "Q4", "Q5"]


def main() -> None:
    # Image-side data: text_corruption=none, vision_corruption ∈ {gaussian_noise,blur,cutout}
    seed_dfs = []
    for s in SEEDS:
        d = pd.read_csv(ROOT / f"results/seed{s}/seed{s}_results.csv", low_memory=False)
        d["seed"] = s
        seed_dfs.append(d)
    seed_all = pd.concat(seed_dfs, ignore_index=True)

    img_side = (
        seed_all[(seed_all["spoke"] == "match_retention")
                 & (seed_all["match_retention_direction"] == "image_corrupted")
                 & (seed_all["metric"] == "retention_margin")]
        [["seed", "model", "pair_id", "vision_corruption", "image_severity", "value"]]
        .rename(columns={"value": "retention_margin"}).copy()
    )
    for c in ("seed", "pair_id", "image_severity"):
        img_side[c] = pd.to_numeric(img_side[c]).astype(int)

    # Damage table for main-spoke pairs (matches the seed CSVs' pair indexing)
    dmg = pd.read_csv(ROOT / "results/input_damage.csv")
    img_dmg = dmg[dmg["modality"] == "image"][["seed", "pair_id", "corruption_type", "severity", "ssim", "damage_ssim"]]
    img_dmg = img_dmg.rename(columns={"corruption_type": "vision_corruption", "severity": "image_severity"})
    for c in ("seed", "pair_id", "image_severity"):
        img_dmg[c] = pd.to_numeric(img_dmg[c]).astype(int)
    img_side = img_side.merge(img_dmg, on=["seed", "pair_id", "vision_corruption", "image_severity"], how="inner")
    print(f"Image-side rows: {len(img_side):,}")

    # Text-side data: from probe v2, mean-pool only
    probe = pd.read_csv(ROOT / "results/pooling_probe/pooling_probe_v2.csv")
    probe["text_severity"] = pd.to_numeric(probe["text_severity"]).astype(int)
    probe["pair_id"] = pd.to_numeric(probe["pair_id"]).astype(int)
    probe["seed"] = pd.to_numeric(probe["seed"]).astype(int)
    txt_mean = probe[(probe["pool_type"] == "mean") & (probe["metric"] == "retention_margin")][
        ["seed", "model", "pair_id", "text_corruption", "text_severity", "value"]
    ].rename(columns={"value": "mean_pool_retention"}).copy()
    txt_std = probe[(probe["pool_type"] == "standard") & (probe["metric"] == "retention_margin")][
        ["seed", "model", "pair_id", "text_corruption", "text_severity", "value"]
    ].rename(columns={"value": "standard_pool_retention"}).copy()

    txt_dmg = dmg[dmg["modality"] == "text"][["seed", "pair_id", "corruption_type", "severity", "bleu", "damage_bleu"]]
    txt_dmg = txt_dmg.rename(columns={"corruption_type": "text_corruption", "severity": "text_severity"})
    for c in ("seed", "pair_id", "text_severity"):
        txt_dmg[c] = pd.to_numeric(txt_dmg[c]).astype(int)
    txt_mean = txt_mean.merge(txt_dmg, on=["seed", "pair_id", "text_corruption", "text_severity"], how="inner")
    txt_std = txt_std.merge(txt_dmg, on=["seed", "pair_id", "text_corruption", "text_severity"], how="inner")
    print(f"Text-side mean-pool rows: {len(txt_mean):,}")
    print(f"Text-side standard-pool rows: {len(txt_std):,}")

    print("\n" + "=" * 78)
    print("UPPER-BOUND TEST — quintile asymmetry with text=mean-pool")
    print("=" * 78)
    print("Compares image-side STANDARD-pool retention vs text-side MEAN-pool retention,")
    print("within matched within-modality input damage quintiles.\n")

    models = sorted(img_side["model"].unique())
    rows = []
    for m in models:
        i_sub = img_side[img_side["model"] == m].copy()
        t_sub = txt_mean[txt_mean["model"] == m].copy()
        i_sub["q"] = pd.qcut(i_sub["damage_ssim"], q=5, labels=QUINTILES, duplicates="drop")
        t_sub["q"] = pd.qcut(t_sub["damage_bleu"], q=5, labels=QUINTILES, duplicates="drop")
        for q in QUINTILES:
            iv = i_sub[i_sub["q"] == q]["retention_margin"].to_numpy(dtype=float)
            tv = t_sub[t_sub["q"] == q]["mean_pool_retention"].to_numpy(dtype=float)
            if len(iv) == 0 or len(tv) == 0:
                continue
            i_m, i_lo, i_hi = boot_mean_ci(iv)
            t_m, t_lo, t_hi = boot_mean_ci(tv)
            d_m, d_lo, d_hi = boot_diff_ci(iv, tv)
            rows.append({
                "model": m, "quintile": q,
                "img_n": len(iv), "img_mean": i_m,
                "txt_mean_pool_n": len(tv), "txt_mean_pool_mean": t_m,
                "diff_img_minus_txt": d_m, "diff_lo": d_lo, "diff_hi": d_hi,
                "img_higher": d_m > 0,
                "diff_significant": d_lo > 0,
            })
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/upper_bound_quintile_test.csv", index=False)
    print(out.round(4).to_string(index=False))

    print("\n--- HEADLINE: how often does image > text(mean-pool) within matched quintiles? ---")
    summary = (out.groupby("model")
               .agg(n_q=("img_higher", "size"), img_higher=("img_higher", "sum"),
                    sig=("diff_significant", "sum"))
               .reset_index())
    summary["all_q_img_higher"] = summary["img_higher"] == summary["n_q"]
    summary["all_q_sig"] = summary["sig"] == summary["n_q"]
    print(summary.to_string(index=False))

    # Compare to the original standard-pool result so we can quantify shrinkage
    print("\n" + "=" * 78)
    print("SHRINKAGE ANALYSIS — how much does mean-pool close the asymmetry?")
    print("=" * 78)
    print("For each (model, quintile), compare:")
    print("  delta_standard = image_std − text_std    (standard-pool asymmetry)")
    print("  delta_meanpool = image_std − text_meanpool (UPPER-BOUND test)")
    print("If delta_meanpool > 0: asymmetry persists; mean-pool didn't close it.")
    print("Magnitude shrinkage: 1 − delta_meanpool / delta_standard\n")

    # Compute standard-pool asymmetry from probe v2's standard-pool text retention,
    # so the comparison is apples-to-apples (same pair set as mean-pool).
    rows2 = []
    for m in models:
        i_sub = img_side[img_side["model"] == m].copy()
        t_std_sub = txt_std[txt_std["model"] == m].copy()
        t_mean_sub = txt_mean[txt_mean["model"] == m].copy()
        i_sub["q"] = pd.qcut(i_sub["damage_ssim"], q=5, labels=QUINTILES, duplicates="drop")
        t_std_sub["q"] = pd.qcut(t_std_sub["damage_bleu"], q=5, labels=QUINTILES, duplicates="drop")
        t_mean_sub["q"] = pd.qcut(t_mean_sub["damage_bleu"], q=5, labels=QUINTILES, duplicates="drop")
        for q in QUINTILES:
            iv = i_sub[i_sub["q"] == q]["retention_margin"].to_numpy(dtype=float).mean() if (i_sub["q"] == q).any() else np.nan
            tv_std = t_std_sub[t_std_sub["q"] == q]["standard_pool_retention"].to_numpy(dtype=float).mean() if (t_std_sub["q"] == q).any() else np.nan
            tv_mean = t_mean_sub[t_mean_sub["q"] == q]["mean_pool_retention"].to_numpy(dtype=float).mean() if (t_mean_sub["q"] == q).any() else np.nan
            ds = iv - tv_std
            dm = iv - tv_mean
            rows2.append({
                "model": m, "quintile": q,
                "delta_standard": ds, "delta_meanpool": dm,
                "shrinkage_pct": (1 - dm / ds) * 100 if ds > 0 else np.nan,
            })
    shr = pd.DataFrame(rows2).round(4)
    print(shr.to_string(index=False))

    print("\nMean shrinkage % per model (across the 5 quintiles):")
    print(shr.groupby("model")["shrinkage_pct"].mean().round(1).to_string())


if __name__ == "__main__":
    main()
