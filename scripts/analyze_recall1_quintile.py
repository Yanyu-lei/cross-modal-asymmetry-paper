"""Recall@1 quintile calibration analysis.

Inputs:
  results/input_damage.csv             # main-spoke damage (gaussian_noise, blur, cutout x mask, shuffle, replace)
  results/input_damage_retrieval.csv   # retrieval-pair damage (gaussian_noise + mask only)
  results/per_pair_retrieval/per_pair_retrieval.csv   # per-pair Recall@k indicators
  results/pooling_probe/pooling_probe.csv             # mean-pool vs argmax-pool retention margin

Analyses:
  A. Recall@1 quintile calibration (matched within-modality damage test of cross-modal asymmetry)
  B. EOT bottleneck mechanistic test (does mean-pool reduce text retention drop?)
  C. Sanity checks (does new aggregate retrieval match prior data?)

Output: stdout summary (redirect to results/recall1_quintile_report.txt) + supporting CSVs.
"""
from __future__ import annotations

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


def boot_ci(values: np.ndarray, *, n_boot: int = 1000, ci: float = 0.95, seed: int = 0):
    if len(values) < 2:
        return float(values.mean()) if len(values) else float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = values[idx].mean(axis=1)
    a = 1 - ci
    return float(values.mean()), float(np.quantile(boots, a / 2)), float(np.quantile(boots, 1 - a / 2))


def boot_diff_ci(a: np.ndarray, b: np.ndarray, *, n_boot: int = 1000, ci: float = 0.95, seed: int = 0):
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(n_boot, len(a)))
    ib = rng.integers(0, len(b), size=(n_boot, len(b)))
    boots = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    al = 1 - ci
    return float(a.mean() - b.mean()), float(np.quantile(boots, al / 2)), float(np.quantile(boots, 1 - al / 2))


# =============================================================================
# Load
# =============================================================================
def load_per_pair_retrieval() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "results/per_pair_retrieval/per_pair_retrieval.csv", low_memory=False)
    pp = df[df["spoke"] == "retrieval_per_pair"].copy()
    for col in ("seed", "pair_id", "image_severity", "text_severity"):
        pp[col] = pd.to_numeric(pp[col], errors="coerce").astype("Int64")
    return pp


def load_aggregate_retrieval() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "results/per_pair_retrieval/per_pair_retrieval.csv", low_memory=False)
    return df[df["spoke"] == "retrieval"].copy()


def load_retrieval_damage() -> pd.DataFrame:
    return pd.read_csv(ROOT / "results/input_damage_retrieval.csv")


def load_pooling_probe() -> pd.DataFrame:
    """Load the corrected (v2) probe data: pool_type ∈ {'standard', 'mean'}.
    Falls back to v1 (pool_type ∈ {'argmax', 'mean'}) if v2 isn't available."""
    v2 = ROOT / "results/pooling_probe/pooling_probe_v2.csv"
    if v2.exists():
        return pd.read_csv(v2)
    return pd.read_csv(ROOT / "results/pooling_probe/pooling_probe.csv")


# =============================================================================
# C. Sanity check: new aggregate matches old aggregate
# =============================================================================
def sanity_check_aggregate():
    header("C. SANITY CHECK — new aggregate retrieval vs original (seed 0)")
    new_agg = load_aggregate_retrieval()
    new_agg = new_agg[new_agg["seed"].astype(int) == 0]

    old_dfs = [pd.read_csv(ROOT / f"results/seed{s}/seed{s}_results.csv", low_memory=False) for s in SEEDS]
    old = pd.concat(old_dfs, ignore_index=True)
    old_seed0_ret = old[(old["spoke"] == "retrieval") & (pd.to_numeric(old["seed"]) == 0)]

    print("Aggregated Recall@1 means by model:")
    new_r1 = new_agg[new_agg["metric"].isin(["recall_at_1_i2t", "recall_at_1_t2i"])]
    print(new_r1.groupby(["model", "metric"])["value"].mean().round(4).unstack())

    print("\nOriginal (seed 0) Recall@1 means by model:")
    old_r1 = old_seed0_ret[old_seed0_ret["metric"].isin(["recall_at_1_i2t", "recall_at_1_t2i"])]
    print(old_r1.groupby(["model", "metric"])["value"].mean().round(4).unstack())


# =============================================================================
# A. Recall@1 quintile calibration
# =============================================================================
def recall1_quintile_calibration():
    header("A. RECALL@1 QUINTILE CALIBRATION (gold-standard)")
    pp = load_per_pair_retrieval()
    dmg = load_retrieval_damage()

    # Image side: i2t under gaussian_noise corruption, value = 0/1
    i2t = pp[pp["metric"] == "per_pair_recall_at_1_i2t"][["seed", "model", "pair_id", "image_severity", "value"]].copy()
    i2t["value"] = pd.to_numeric(i2t["value"]).astype(int)
    img_dmg = dmg[dmg["modality"] == "image"][["seed", "pair_id", "severity", "ssim", "damage_ssim"]].rename(columns={"severity": "image_severity"})
    img_dmg["seed"] = img_dmg["seed"].astype(int)
    img_dmg["pair_id"] = img_dmg["pair_id"].astype(int)
    img_dmg["image_severity"] = img_dmg["image_severity"].astype(int)
    i2t["seed"] = i2t["seed"].astype(int); i2t["pair_id"] = i2t["pair_id"].astype(int); i2t["image_severity"] = i2t["image_severity"].astype(int)
    i2t = i2t.merge(img_dmg, on=["seed", "pair_id", "image_severity"], how="inner")

    # Text side: t2i under mask corruption
    t2i = pp[pp["metric"] == "per_pair_recall_at_1_t2i"][["seed", "model", "pair_id", "text_severity", "value"]].copy()
    t2i["value"] = pd.to_numeric(t2i["value"]).astype(int)
    txt_dmg = dmg[dmg["modality"] == "text"][["seed", "pair_id", "severity", "bleu", "damage_bleu"]].rename(columns={"severity": "text_severity"})
    txt_dmg["seed"] = txt_dmg["seed"].astype(int)
    txt_dmg["pair_id"] = txt_dmg["pair_id"].astype(int)
    txt_dmg["text_severity"] = txt_dmg["text_severity"].astype(int)
    t2i["seed"] = t2i["seed"].astype(int); t2i["pair_id"] = t2i["pair_id"].astype(int); t2i["text_severity"] = t2i["text_severity"].astype(int)
    t2i = t2i.merge(txt_dmg, on=["seed", "pair_id", "text_severity"], how="inner")

    print(f"\nN observations: i2t = {len(i2t):,}, t2i = {len(t2i):,}")
    print(f"  expect 5 sev × 1000 pairs × 3 seeds = 15,000 each")

    # Per-model quintile binning within each modality
    print("\n--- Per-model R@1 within matched within-modality damage quintiles (3-seed pooled) ---")
    rows = []
    models = sorted(pp["model"].unique())
    for m in models:
        i_sub = i2t[i2t["model"] == m].copy()
        t_sub = t2i[t2i["model"] == m].copy()
        i_sub["quintile"] = pd.qcut(i_sub["damage_ssim"], q=5, labels=QUINTILE_LABELS, duplicates="drop")
        t_sub["quintile"] = pd.qcut(t_sub["damage_bleu"], q=5, labels=QUINTILE_LABELS, duplicates="drop")
        for q in QUINTILE_LABELS:
            i_vals = i_sub[i_sub["quintile"] == q]["value"].to_numpy(dtype=float)
            t_vals = t_sub[t_sub["quintile"] == q]["value"].to_numpy(dtype=float)
            if len(i_vals) == 0 or len(t_vals) == 0:
                continue
            i_mean, i_lo, i_hi = boot_ci(i_vals)
            t_mean, t_lo, t_hi = boot_ci(t_vals)
            d_mean, d_lo, d_hi = boot_diff_ci(i_vals, t_vals)
            rows.append({
                "model": m, "quintile": q,
                "i2t_n": len(i_vals), "i2t_R@1": i_mean, "i2t_lo": i_lo, "i2t_hi": i_hi,
                "t2i_n": len(t_vals), "t2i_R@1": t_mean, "t2i_lo": t_lo, "t2i_hi": t_hi,
                "diff_i2t_minus_t2i": d_mean, "diff_lo": d_lo, "diff_hi": d_hi,
                "i2t > t2i": d_mean > 0,
                "diff_significant": d_lo > 0,
            })
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "calibration_recall1_quintile.csv", index=False)
    print(out.round(4).to_string(index=False))
    print("\n→ Saved: results/calibration_recall1_quintile.csv")

    # Headline
    print("\n--- HEADLINE: how often is i2t R@1 > t2i R@1 within matched quintiles? ---")
    summary = out.groupby("model").agg(
        n_quintiles=("i2t > t2i", "size"),
        i2t_higher=("i2t > t2i", "sum"),
        sig_quintiles=("diff_significant", "sum"),
    ).reset_index()
    summary["all_quintiles_i2t_higher"] = summary["i2t_higher"] == summary["n_quintiles"]
    summary["all_quintiles_significant"] = summary["sig_quintiles"] == summary["n_quintiles"]
    print(summary.to_string(index=False))

    # Within-modality slopes (R^2 of R@1 ~ damage)
    print("\n--- Within-modality slope: R@1 ~ damage, per (model, modality) ---")
    print("Steeper |slope| within text means text R@1 is more damage-sensitive (within its modality).")
    sl = []
    for m in models:
        for modality, df, dam in (("image", i2t[i2t["model"] == m], "damage_ssim"),
                                   ("text", t2i[t2i["model"] == m], "damage_bleu")):
            x = df[dam].to_numpy(dtype=float)
            y = df["value"].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            ss_tot = ((y - y.mean()) ** 2).sum()
            ss_res = ((y - (slope * x + intercept)) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            sl.append({"model": m, "modality": modality, "slope": slope, "intercept": intercept, "R2": r2, "n": len(df)})
    print(pd.DataFrame(sl).round(4).to_string(index=False))


# =============================================================================
# B. EOT bottleneck mechanistic test
# =============================================================================
def bottleneck_test():
    header("B. EOT BOTTLENECK MECHANISTIC TEST")
    print("Hypothesis: if mean-pool retention drops LESS than argmax-pool under text corruption,")
    print("the asymmetry is at least partly driven by the single-token bottleneck of EOT pooling.\n")

    pp = load_pooling_probe()
    margin = pp[pp["metric"] == "retention_margin"].copy()
    margin["text_severity"] = pd.to_numeric(margin["text_severity"]).astype(int)

    # Per (model, text_corruption, severity, pool_type): mean retention margin across pairs and seeds
    print("--- Mean retention_margin per (model, text_corruption, severity, pool_type) ---")
    table = margin.groupby(["model", "text_corruption", "text_severity", "pool_type"])["value"].mean().unstack().round(4)
    print(table.to_string())

    # Per (model, text_corruption): drop (sev1) - (sev5) per pool type
    print("\n--- Drop sev1 - sev5 per (model, text_corruption, pool_type) ---")
    print("Bigger drop = more sensitive to corruption.")
    drop_rows = []
    for (m, tc, pt), g in margin.groupby(["model", "text_corruption", "pool_type"]):
        sev1 = g[g["text_severity"] == 1]["value"].mean()
        sev5 = g[g["text_severity"] == 5]["value"].mean()
        drop_rows.append({"model": m, "text_corruption": tc, "pool_type": pt, "drop": sev1 - sev5})
    drop = pd.DataFrame(drop_rows)
    drop_pivot = drop.pivot_table(index=["model", "text_corruption"], columns="pool_type", values="drop").round(4)
    drop_pivot["mean_minus_argmax"] = drop_pivot["mean"] - drop_pivot["standard"]
    drop_pivot["mean_more_robust?"] = drop_pivot["mean_minus_argmax"] < 0   # mean drops less = more robust
    print(drop_pivot.to_string())

    # Aggregate test: across all (model, text_corruption), how often does mean_pool reduce the drop?
    print("\n--- Aggregate: across (model × text_corruption), how often does mean reduce drop? ---")
    n_tested = len(drop_pivot)
    n_mean_better = drop_pivot["mean_more_robust?"].sum()
    print(f"Mean-pool more robust in {n_mean_better}/{n_tested} (model × corruption) cells")
    if n_mean_better == n_tested:
        print("→ EOT bottleneck hypothesis SUPPORTED across the board.")
    elif n_mean_better == 0:
        print("→ EOT bottleneck hypothesis REJECTED across the board. Mean-pool is NOT more robust.")
    else:
        print(f"→ Mixed result: {n_mean_better}/{n_tested} cells support, {n_tested - n_mean_better} reject.")

    # Per-pair paired test: at sev 5, is per-pair argmax retention < mean retention?
    print("\n--- Per-pair: paired difference (mean - argmax) at sev 5, per (model, text_corruption) ---")
    print("Positive = mean-pool retention higher = bottleneck supported.")
    paired_rows = []
    for (m, tc), g in margin[margin["text_severity"] == 5].groupby(["model", "text_corruption"]):
        g_a = g[g["pool_type"] == "standard"].set_index(["seed", "pair_id"])["value"]
        g_m = g[g["pool_type"] == "mean"].set_index(["seed", "pair_id"])["value"]
        diff = (g_m - g_a).dropna().to_numpy()
        m_mean, m_lo, m_hi = boot_ci(diff)
        paired_rows.append({"model": m, "text_corruption": tc, "n_pairs": len(diff),
                            "mean_minus_argmax": m_mean, "ci_lo": m_lo, "ci_hi": m_hi,
                            "ci_excludes_zero_above": m_lo > 0,
                            "ci_excludes_zero_below": m_hi < 0})
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(ROOT / "results" / "bottleneck_paired_sev5.csv", index=False)
    print(paired.round(4).to_string(index=False))

    n_supports = paired["ci_excludes_zero_above"].sum()
    n_rejects = paired["ci_excludes_zero_below"].sum()
    n_total = len(paired)
    print(f"\nAt sev 5 with paired-pair CIs: {n_supports}/{n_total} support bottleneck (CI > 0),")
    print(f"  {n_rejects}/{n_total} REJECT it (CI < 0), {n_total - n_supports - n_rejects}/{n_total} inconclusive (CI overlaps 0).")


def main():
    sanity_check_aggregate()
    recall1_quintile_calibration()
    bottleneck_test()
    print("\n" + "=" * 78); print("DONE"); print("=" * 78)


if __name__ == "__main__":
    main()
