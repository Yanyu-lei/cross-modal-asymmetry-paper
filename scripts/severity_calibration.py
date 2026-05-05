"""Severity-calibration robustness check.

Implements the 6-part analysis from the user's spec:
  1. Per-(corruption, severity) damage table (model-independent), bootstrap CIs
  2. Within-modality damage curves (model performance vs damage)
  3. Percentile-binned (quintile) comparison — primary calibration test
  4. Crossover check — only reported if true
  5. CLIP embedding-fidelity calibration (model-internal diagnostic)
  6. Measured-damage dominance regression (parallel to ordinal version)

Outcome metric for analyses 3 and 6:
  Per-pair retention_margin (cosine-discriminability per pair). We use this
  rather than Recall@1 because the existing retrieval CSV is condition-level
  aggregated, while retention_margin is per-pair (4,500 obs per modality per
  model). retention_margin is a per-pair cosine discriminability score —
  conceptually the same family as Recall@1's matched-ranks-first signal.

  Aggregate Recall@1 is shown alongside, at the condition level, for the
  curve plots in analysis 2.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2)
QUINTILE_LABELS = ["Q1 (mildest)", "Q2", "Q3", "Q4", "Q5 (most damaged)"]


def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def bootstrap_ci(values: np.ndarray, *, n_boot: int = 1000, ci: float = 0.95, seed: int = 0) -> Tuple[float, float, float]:
    if len(values) < 2:
        return float(np.mean(values)) if len(values) else float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = values[idx].mean(axis=1)
    alpha = 1 - ci
    return float(values.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, *, n_boot: int = 1000, ci: float = 0.95, seed: int = 0) -> Tuple[float, float, float]:
    """Bootstrap CI for mean(a) - mean(b) using independent resampling within each."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, len(a), size=(n_boot, len(a)))
    idx_b = rng.integers(0, len(b), size=(n_boot, len(b)))
    boots = a[idx_a].mean(axis=1) - b[idx_b].mean(axis=1)
    alpha = 1 - ci
    return float(a.mean() - b.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


# =============================================================================
# Loaders
# =============================================================================
def load_damage() -> pd.DataFrame:
    return pd.read_csv(ROOT / "results" / "input_damage.csv")


def load_seed_results() -> pd.DataFrame:
    dfs = []
    for s in SEEDS:
        d = pd.read_csv(ROOT / f"results/seed{s}/seed{s}_results.csv", low_memory=False)
        d["seed"] = s
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


# =============================================================================
# Analysis 1: Damage table
# =============================================================================
def analysis_1_damage_table(damage: pd.DataFrame) -> None:
    header("1. INPUT-SIDE DAMAGE TABLE (model-independent)")
    print("Per (modality, corruption_type, severity): mean damage with bootstrap 95% CI, n pairs.\n")

    rows = []
    img = damage[damage["modality"] == "image"]
    for (ct, sev), g in img.groupby(["corruption_type", "severity"]):
        for metric in ("ssim", "psnr", "damage_ssim"):
            mean, lo, hi = bootstrap_ci(g[metric].to_numpy(dtype=float))
            rows.append({"modality": "image", "corruption_type": ct, "severity": sev,
                         "metric": metric, "mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(g)})
    txt = damage[damage["modality"] == "text"]
    for (ct, sev), g in txt.groupby(["corruption_type", "severity"]):
        for metric in ("norm_edit_distance", "bleu", "damage_bleu"):
            mean, lo, hi = bootstrap_ci(g[metric].to_numpy(dtype=float))
            rows.append({"modality": "text", "corruption_type": ct, "severity": sev,
                         "metric": metric, "mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(g)})

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "calibration_damage_table.csv", index=False)
    print(out.round(4).to_string(index=False))
    print(f"\n→ Saved: results/calibration_damage_table.csv")


# =============================================================================
# Helper: build per-pair (model, condition, damage, retention_margin) frame
# =============================================================================
def build_per_pair_frame(damage: pd.DataFrame, results: pd.DataFrame, models: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (image_side_df, text_side_df) at the (model × seed × pair × corruption × severity) level.
    Each row has the input damage and the per-pair retention_margin (image-corrupted or text-corrupted)."""
    mr = results[(results["spoke"] == "match_retention") & (results["metric"] == "retention_margin")]

    # Image side
    img_dmg = damage[damage["modality"] == "image"].rename(columns={"corruption_type": "vision_corruption", "severity": "image_severity"})
    img_dmg = img_dmg[["seed", "pair_id", "vision_corruption", "image_severity", "ssim", "psnr", "damage_ssim"]]

    img_mr = mr[mr["match_retention_direction"] == "image_corrupted"][["seed", "model", "pair_id", "vision_corruption", "image_severity", "value"]].rename(columns={"value": "retention_margin"})
    img_mr["pair_id"] = pd.to_numeric(img_mr["pair_id"])
    img_mr["seed"] = pd.to_numeric(img_mr["seed"])
    img_mr["image_severity"] = pd.to_numeric(img_mr["image_severity"])

    img_side = img_mr.merge(img_dmg, on=["seed", "pair_id", "vision_corruption", "image_severity"], how="inner")

    # Text side
    txt_dmg = damage[damage["modality"] == "text"].rename(columns={"corruption_type": "text_corruption", "severity": "text_severity"})
    txt_dmg = txt_dmg[["seed", "pair_id", "text_corruption", "text_severity", "norm_edit_distance", "bleu", "damage_bleu"]]

    txt_mr = mr[mr["match_retention_direction"] == "text_corrupted"][["seed", "model", "pair_id", "text_corruption", "text_severity", "value"]].rename(columns={"value": "retention_margin"})
    txt_mr["pair_id"] = pd.to_numeric(txt_mr["pair_id"])
    txt_mr["seed"] = pd.to_numeric(txt_mr["seed"])
    txt_mr["text_severity"] = pd.to_numeric(txt_mr["text_severity"])

    txt_side = txt_mr.merge(txt_dmg, on=["seed", "pair_id", "text_corruption", "text_severity"], how="inner")

    return img_side, txt_side


# =============================================================================
# Analysis 2: Within-modality damage curves
# =============================================================================
def analysis_2_within_modality_curves(damage: pd.DataFrame, results: pd.DataFrame, models: Sequence[str]) -> None:
    header("2. WITHIN-MODALITY DAMAGE CURVES (retention margin vs input damage)")
    print("Per-condition mean retention_margin and mean input damage; one curve per model per modality.\n")

    img_side, txt_side = build_per_pair_frame(damage, results, models)

    # Image-side curve: mean retention_margin vs mean damage_ssim, per (model, condition)
    img_curve = (img_side.groupby(["model", "vision_corruption", "image_severity"])
                 .agg(retention_margin=("retention_margin", "mean"),
                      damage_ssim=("damage_ssim", "mean"),
                      ssim=("ssim", "mean"),
                      psnr=("psnr", "mean"),
                      n=("retention_margin", "size")).reset_index())
    img_curve.to_csv(ROOT / "results" / "calibration_curve_image.csv", index=False)
    print("--- IMAGE-SIDE: retention_margin vs damage_ssim per condition ---")
    print(img_curve.sort_values(["model", "damage_ssim"]).round(4).to_string(index=False))

    # Text-side curve
    txt_curve = (txt_side.groupby(["model", "text_corruption", "text_severity"])
                 .agg(retention_margin=("retention_margin", "mean"),
                      damage_bleu=("damage_bleu", "mean"),
                      bleu=("bleu", "mean"),
                      norm_edit_distance=("norm_edit_distance", "mean"),
                      n=("retention_margin", "size")).reset_index())
    txt_curve.to_csv(ROOT / "results" / "calibration_curve_text.csv", index=False)
    print("\n--- TEXT-SIDE: retention_margin vs damage_bleu per condition ---")
    print(txt_curve.sort_values(["model", "damage_bleu"]).round(4).to_string(index=False))

    # Slope per model: linear regression of retention_margin on damage (within modality)
    print("\n--- Slope of retention_margin ~ damage, per model per modality ---")
    print("Steeper negative slope = retrieval-side is more sensitive to damage in this modality.")
    print("Note: image SSIM and text BLEU are NOT commensurate. Compare *shapes* across modalities, not raw slopes.")
    slopes = []
    for m in models:
        for modality, df, dam_col in (("image", img_side, "damage_ssim"), ("text", txt_side, "damage_bleu")):
            sub = df[df["model"] == m]
            x, y = sub[dam_col].to_numpy(dtype=float), sub["retention_margin"].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            ss_tot = ((y - y.mean()) ** 2).sum()
            ss_res = ((y - (slope * x + intercept)) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            slopes.append({"model": m, "modality": modality, "slope": slope, "intercept": intercept, "R2": r2, "n": len(sub)})
    print(pd.DataFrame(slopes).round(4).to_string(index=False))


# =============================================================================
# Analysis 3: Percentile-binned (quintile) comparison — primary calibration test
# =============================================================================
def analysis_3_quintile_calibration(damage: pd.DataFrame, results: pd.DataFrame, models: Sequence[str]) -> None:
    header("3. QUINTILE-BINNED CALIBRATION TEST (matched within-modality damage)")
    print("""\
For each modality, sort all (pair × corruption × severity) observations by input
damage, divide into 5 equal-N quintiles, and within each matched quintile compare
the mean retention_margin (image-side image-corrupted vs text-side text-corrupted).
The asymmetry headline survives if text-side retention_margin is consistently
LOWER than image-side retention_margin within matched within-modality damage
quintiles, i.e. (image - text) > 0 in matched quintiles.""")

    img_side, txt_side = build_per_pair_frame(damage, results, models)

    # Show condition-to-quintile mapping (sanity check)
    print("\n--- Condition-to-quintile mapping (sanity check, seed 0 only for compactness) ---")
    img_seed0 = img_side[img_side["seed"] == 0].copy()
    img_seed0["quintile"] = pd.qcut(img_seed0["damage_ssim"], q=5, labels=QUINTILE_LABELS)
    print("\nIMAGE-side: damage_ssim quintile distribution by condition:")
    print(img_seed0.groupby(["vision_corruption", "image_severity", "quintile"], observed=True).size().unstack(fill_value=0).to_string())

    txt_seed0 = txt_side[txt_side["seed"] == 0].copy()
    txt_seed0["quintile"] = pd.qcut(txt_seed0["damage_bleu"], q=5, labels=QUINTILE_LABELS)
    print("\nTEXT-side: damage_bleu quintile distribution by condition:")
    print(txt_seed0.groupby(["text_corruption", "text_severity", "quintile"], observed=True).size().unstack(fill_value=0).to_string())

    # Per-model quintile means
    print("\n--- Per-model retention_margin within matched quintiles (3-seed pooled) ---")
    print("Lower retention_margin = worse coherence under that level of damage.")
    rows = []
    for m in models:
        i_sub = img_side[img_side["model"] == m].copy()
        t_sub = txt_side[txt_side["model"] == m].copy()
        # Quintiles within each modality
        i_sub["quintile"] = pd.qcut(i_sub["damage_ssim"], q=5, labels=QUINTILE_LABELS)
        t_sub["quintile"] = pd.qcut(t_sub["damage_bleu"], q=5, labels=QUINTILE_LABELS)

        for q in QUINTILE_LABELS:
            i_vals = i_sub[i_sub["quintile"] == q]["retention_margin"].to_numpy(dtype=float)
            t_vals = t_sub[t_sub["quintile"] == q]["retention_margin"].to_numpy(dtype=float)
            i_mean, i_lo, i_hi = bootstrap_ci(i_vals)
            t_mean, t_lo, t_hi = bootstrap_ci(t_vals)
            d_mean, d_lo, d_hi = bootstrap_diff_ci(i_vals, t_vals)
            rows.append({
                "model": m, "quintile": q,
                "image_n": len(i_vals), "image_mean": i_mean, "image_ci_lo": i_lo, "image_ci_hi": i_hi,
                "text_n": len(t_vals), "text_mean": t_mean, "text_ci_lo": t_lo, "text_ci_hi": t_hi,
                "diff_image_minus_text": d_mean, "diff_ci_lo": d_lo, "diff_ci_hi": d_hi,
                "image_minus_text_>0": d_mean > 0,
                "diff_significant": d_lo > 0,  # CI excludes 0
            })
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "calibration_quintile_test.csv", index=False)
    print(out.round(4).to_string(index=False))
    print("\n→ Saved: results/calibration_quintile_test.csv")

    # Headline summary
    print("\n--- HEADLINE: how often does image_retention > text_retention within matched quintiles? ---")
    summary = out.groupby("model").agg(
        n_quintiles=("image_minus_text_>0", "size"),
        image_higher=("image_minus_text_>0", "sum"),
        sig_quintiles=("diff_significant", "sum"),
    ).reset_index()
    summary["all_quintiles_image_higher"] = summary["image_higher"] == summary["n_quintiles"]
    summary["all_quintiles_significant"] = summary["sig_quintiles"] == summary["n_quintiles"]
    print(summary.to_string(index=False))


# =============================================================================
# Analysis 4: Crossover check
# =============================================================================
def analysis_4_crossover(damage: pd.DataFrame, results: pd.DataFrame, models: Sequence[str]) -> None:
    header("4. CROSSOVER CHECK — does mildest text condition still outperform most-severe image condition?")
    img_side, txt_side = build_per_pair_frame(damage, results, models)

    rows = []
    for m in models:
        i_sub = img_side[img_side["model"] == m].copy()
        t_sub = txt_side[txt_side["model"] == m].copy()

        # Top image-damage quintile (most damaged image-side)
        i_sub["q"] = pd.qcut(i_sub["damage_ssim"], q=5, labels=False)
        i_top = i_sub[i_sub["q"] == 4]["retention_margin"].to_numpy(dtype=float)

        # Bottom text-damage quintile (mildest text-side)
        t_sub["q"] = pd.qcut(t_sub["damage_bleu"], q=5, labels=False)
        t_bottom = t_sub[t_sub["q"] == 0]["retention_margin"].to_numpy(dtype=float)

        i_mean, i_lo, i_hi = bootstrap_ci(i_top)
        t_mean, t_lo, t_hi = bootstrap_ci(t_bottom)
        rows.append({
            "model": m,
            "image_top_quintile_mean_retention": i_mean, "image_ci_lo": i_lo, "image_ci_hi": i_hi,
            "text_bottom_quintile_mean_retention": t_mean, "text_ci_lo": t_lo, "text_ci_hi": t_hi,
            "text_bottom > image_top?": t_mean > i_mean,
        })
    out = pd.DataFrame(rows)
    print(out.round(4).to_string(index=False))
    print("\nReport this in the paper only if 'text_bottom > image_top?' is False for every model")
    print("(i.e., the mildest text corruption leaves more retention than the most severe image corruption).")


# =============================================================================
# Analysis 5: Embedding-fidelity calibration (model-internal diagnostic)
# =============================================================================
def analysis_5_embedding_fidelity(results: pd.DataFrame, models: Sequence[str]) -> None:
    header("5. EMBEDDING-FIDELITY CALIBRATION (model-internal diagnostic)")
    print("Late-layer cosine fidelity per (model, severity), from existing image_fidelity / text_fidelity spokes.")
    print("This is NOT the primary calibration (uses CLIP's own representation), only a parallel diagnostic.\n")

    imf = results[(results["spoke"] == "image_fidelity") & (results["depth"] == "late")]
    tf = results[(results["spoke"] == "text_fidelity") & (results["depth"] == "late")]

    print("--- Image final-embedding fidelity (mean cosine over corruption types and pairs) ---")
    print(imf.groupby(["model", "image_severity"])["value"].mean().unstack().round(4).to_string())
    print("\n--- Text final-embedding fidelity ---")
    print(tf.groupby(["model", "text_severity"])["value"].mean().unstack().round(4).to_string())


# =============================================================================
# Analysis 6: Measured-damage dominance regression (parallel to ordinal version)
# =============================================================================
def analysis_6_measured_dominance(damage: pd.DataFrame, results: pd.DataFrame, models: Sequence[str]) -> None:
    header("6. MEASURED-DAMAGE DOMINANCE REGRESSION (parallel to ordinal-severity version)")
    print("Joint conditions: vision=gaussian_noise × text=mask, severities 1..5 each.")
    print("Replace ordinal severity with image_damage_percentile and text_damage_percentile.")
    print("Hypothesis: R²(more_damaged) > R²(less_damaged), mirroring the ordinal version.\n")

    # Build the percentile mapping per modality at the (corruption, severity) level
    img_dmg = damage[damage["modality"] == "image"].copy()
    txt_dmg = damage[damage["modality"] == "text"].copy()

    # Use within-condition mean damage_ssim and damage_bleu (across pairs, across seeds)
    # then assign each (corruption, severity) cell a percentile rank in its modality's
    # damage distribution.
    img_cell = img_dmg.groupby(["corruption_type", "severity"])["damage_ssim"].mean().reset_index()
    img_cell["rank"] = img_cell["damage_ssim"].rank() / len(img_cell)  # in (0, 1]
    img_pct = dict(zip(zip(img_cell["corruption_type"], img_cell["severity"]), img_cell["rank"]))

    txt_cell = txt_dmg.groupby(["corruption_type", "severity"])["damage_bleu"].mean().reset_index()
    txt_cell["rank"] = txt_cell["damage_bleu"].rank() / len(txt_cell)
    txt_pct = dict(zip(zip(txt_cell["corruption_type"], txt_cell["severity"]), txt_cell["rank"]))

    # Joint match_retention margin
    j = results[(results["spoke"] == "match_retention") & (results["match_retention_direction"] == "joint") & (results["metric"] == "retention_margin")].copy()
    j["pair_id"] = pd.to_numeric(j["pair_id"])
    j["image_severity"] = pd.to_numeric(j["image_severity"])
    j["text_severity"] = pd.to_numeric(j["text_severity"])

    # Joint corruption is gaussian_noise × mask
    j["image_dmg_pct"] = j["image_severity"].map(lambda s: img_pct[("gaussian_noise", s)])
    j["text_dmg_pct"] = j["text_severity"].map(lambda s: txt_pct[("mask", s)])
    j["more_damaged"] = j[["image_dmg_pct", "text_dmg_pct"]].max(axis=1)
    j["less_damaged"] = j[["image_dmg_pct", "text_dmg_pct"]].min(axis=1)

    rows = []
    for m in models:
        sub = j[j["model"] == m]
        y = sub["value"].to_numpy(dtype=float)
        for x_name in ("more_damaged", "less_damaged"):
            x = sub[x_name].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            ss_tot = ((y - y.mean()) ** 2).sum()
            ss_res = ((y - (slope * x + intercept)) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            rows.append({"model": m, "predictor": x_name, "slope": slope, "intercept": intercept, "R2": r2, "n": len(sub)})
    out = pd.DataFrame(rows)
    print(out.round(4).to_string(index=False))

    pivot = out.pivot_table(index="model", columns="predictor", values="R2").round(4)
    pivot["more_minus_less"] = pivot["more_damaged"] - pivot["less_damaged"]
    pivot["more_wins?"] = pivot["more_damaged"] > pivot["less_damaged"]
    print("\nMeasured-damage dominance summary:")
    print(pivot.to_string())

    print("\nFor reference, the ORIGINAL ordinal-severity version (3-seed mean):")
    print("(Already computed in scripts/review_3seeds.py; both should give the same direction.)")


def main():
    results = load_seed_results()
    damage = load_damage()
    models = sorted(results["model"].unique())

    analysis_1_damage_table(damage)
    analysis_2_within_modality_curves(damage, results, models)
    analysis_3_quintile_calibration(damage, results, models)
    analysis_4_crossover(damage, results, models)
    analysis_5_embedding_fidelity(results, models)
    analysis_6_measured_dominance(damage, results, models)

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()
