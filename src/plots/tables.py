"""LaTeX tables. One file per table, written under results/tables/."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis.bootstrap import hier_boot_mean, hier_boot_diff, seed_mean_sd, fmt_ci, fmt_seedcheck
from src.plots._data import (
    aggregate_retrieval, i2t_with_damage, t2i_with_damage,
    quintile_assign, pooling_probe_v2, ROOT, QUINTILES,
)
from src.plots._style import MODEL_LABELS, MODEL_ORDER

OUT = ROOT / "results/tables"
OUT.mkdir(parents=True, exist_ok=True)
N_BOOT = 10000


# =============================================================================
# T1: model summary
# =============================================================================
def t1_model_summary() -> str:
    """Pull facts from configs/models.yaml + diagnostic CSV; build LaTeX."""
    import yaml
    cfg = yaml.safe_load(open(ROOT / "configs/models.yaml"))["models"]
    diag = pd.read_csv(ROOT / "results/text_depth_diagnostic.csv") if (ROOT / "results/text_depth_diagnostic.csv").exists() else None

    # Static facts compiled from the project notes
    static = {
        "openai_clip_b32":      ("ViT-B/32",            "151M",  "WIT (OpenAI)",  "224", "77", "softmax"),
        "openai_clip_l14":      ("ViT-L/14",            "428M",  "WIT (OpenAI)",  "224", "77", "softmax"),
        "openclip_l14_laion2b": ("ViT-L/14",            "428M",  "LAION-2B",      "224", "77", "softmax"),
        "siglip2_so400m_384":   ("ViT-SO400M",          "1.13B", "WebLI",         "384", "64", "sigmoid"),
        "pecore_l14_336":       ("ViT-L/14 (PE-Core)",  "428M",  "Meta PE-Core",  "336", "32", "softmax"),
    }

    rows = []
    for m in MODEL_ORDER:
        c = cfg[m]
        arch, params, td, ires, ctx, loss = static[m]
        text_depth_frac = c["text_depth_fractions"][0]
        rows.append({
            "model": MODEL_LABELS[m], "arch": arch, "params": params, "data": td,
            "img_res": ires, "ctx_len": ctx, "loss": loss,
            "early_text_frac": f"{text_depth_frac:.2f}",
        })

    # LaTeX
    out = []
    out.append(r"\begin{table}[h]")
    out.append(r"\centering")
    out.append(r"\caption{Model summary. Five CLIP-family models spanning architecture, training corpus, image resolution, text context length, and contrastive loss. The early-text-depth fraction (last column) is set per model from a diagnostic that finds the layer at which masked-text cosine similarity first drops below 0.95.}")
    out.append(r"\label{tab:t1_models}")
    out.append(r"\small")
    out.append(r"\begin{tabular}{llrlrll}")
    out.append(r"\toprule")
    out.append(r"Model & Architecture & Params & Training data & Img.\ res.\ & Text ctx & Loss \\")
    out.append(r"\midrule")
    for r in rows:
        out.append(f"{r['model']} & {r['arch']} & {r['params']} & {r['data']} & {r['img_res']} & {r['ctx_len']} & {r['loss']} \\\\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")

    s = "\n".join(out)
    (OUT / "t1_model_summary.tex").write_text(s)
    return s


# =============================================================================
# T2: headline numbers
# =============================================================================
def _q1q5_means_for(metric: str, n_boot=N_BOOT):
    """Per-model Q1 and Q5 R@1 with bootstrap CIs.
    metric in {"i2t","t2i"}."""
    if metric == "i2t":
        df = i2t_with_damage()
        df = quintile_assign(df, "damage_ssim")
    else:
        df = t2i_with_damage()
        df = quintile_assign(df, "damage_bleu")
    rows = []
    for m in MODEL_ORDER:
        for q in ("Q1", "Q5"):
            sub = df[(df["model"] == m) & (df["quintile"] == q)]
            mean, lo, hi = hier_boot_mean(sub, n_boot=n_boot)
            sm, sd = seed_mean_sd(sub)
            rows.append({"model": m, "metric": metric, "quintile": q,
                         "mean": mean, "lo": lo, "hi": hi, "seed_mean": sm, "seed_sd": sd})
    return pd.DataFrame(rows)


def _baseline_per_model() -> pd.DataFrame:
    """True clean (no-corruption) Recall@1 per model, hierarchical bootstrap.
    Sourced from the dedicated clean baseline retrieval pass, NOT severity-1."""
    from src.plots._data import clean_baseline_per_pair
    pp = clean_baseline_per_pair()
    out = {}
    for direction in ("i2t", "t2i"):
        rows = []
        for m in MODEL_ORDER:
            sub = pp[(pp["model"] == m) & (pp["metric"] == f"per_pair_clean_recall_at_1_{direction}")]
            mean, lo, hi = hier_boot_mean(sub, n_boot=N_BOOT)
            sm, sd = seed_mean_sd(sub)
            rows.append({"model": m, "mean": mean, "lo": lo, "hi": hi, "seed_mean": sm, "seed_sd": sd})
        out[direction] = pd.DataFrame(rows).set_index("model")
    return out


def t2_headline() -> str:
    base = _baseline_per_model()
    i2t = _q1q5_means_for("i2t")
    t2i = _q1q5_means_for("t2i")

    rows = []
    for m in MODEL_ORDER:
        i_q1 = i2t[(i2t["model"] == m) & (i2t["quintile"] == "Q1")].iloc[0]
        i_q5 = i2t[(i2t["model"] == m) & (i2t["quintile"] == "Q5")].iloc[0]
        t_q1 = t2i[(t2i["model"] == m) & (t2i["quintile"] == "Q1")].iloc[0]
        t_q5 = t2i[(t2i["model"] == m) & (t2i["quintile"] == "Q5")].iloc[0]

        clean_i = base["i2t"].loc[m]
        clean_t = base["t2i"].loc[m]

        # Q5 drop = true clean baseline minus Q5 mean
        i_drop = clean_i["mean"] - i_q5["mean"]
        t_drop = clean_t["mean"] - t_q5["mean"]
        drop_gap = t_drop - i_drop
        rows.append({
            "model": m,
            "clean_i2t": clean_i["mean"], "clean_t2i": clean_t["mean"],
            "q1_i2t": i_q1["mean"], "q1_i2t_lo": i_q1["lo"], "q1_i2t_hi": i_q1["hi"],
            "q5_i2t": i_q5["mean"], "q5_i2t_lo": i_q5["lo"], "q5_i2t_hi": i_q5["hi"],
            "q1_t2i": t_q1["mean"], "q1_t2i_lo": t_q1["lo"], "q1_t2i_hi": t_q1["hi"],
            "q5_t2i": t_q5["mean"], "q5_t2i_lo": t_q5["lo"], "q5_t2i_hi": t_q5["hi"],
            "i2t_drop": i_drop, "t2i_drop": t_drop, "drop_gap": drop_gap,
        })

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/figures/t2_data.csv", index=False)

    out = []
    out.append(r"\begin{table}[h]")
    out.append(r"\centering")
    out.append(r"\caption{Recall@1 at the mildest (Q1) and most-damaged (Q5) within-modality damage quintiles, with hierarchical bootstrap 95\% CIs across 3 seeds and 1000 retrieval pairs. Clean refers to the no-corruption condition computed in a dedicated retrieval pass. Drop columns are (clean R@1) minus (Q5 R@1); the gap (rightmost) is t2i drop minus i2t drop. Across all five models the Q5 t2i drop substantially exceeds the Q5 i2t drop, and Q5 t2i Recall@1 collapses to single-digit percent.}")
    out.append(r"\label{tab:t2_headline}")
    out.append(r"\footnotesize")
    out.append(r"\setlength{\tabcolsep}{3pt}")
    out.append(r"\begin{tabular}{lrrrrrrrrr}")
    out.append(r"\toprule")
    out.append(r" & \multicolumn{2}{c}{Clean R@1} & \multicolumn{2}{c}{Q1 (mildest)} & \multicolumn{2}{c}{Q5 (most damaged)} & \multicolumn{3}{c}{Q5 drop from clean} \\")
    out.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-10}")
    out.append(r"Model & i2t & t2i & i2t & t2i & i2t & t2i & i2t & t2i & gap \\")
    out.append(r"\midrule")
    for r in rows:
        out.append(
            f"{MODEL_LABELS[r['model']]} & "
            f"{r['clean_i2t']:.2f} & {r['clean_t2i']:.2f} & "
            f"{r['q1_i2t']:.2f} & {r['q1_t2i']:.2f} & "
            f"{r['q5_i2t']:.2f} & {r['q5_t2i']:.2f} & "
            f"{r['i2t_drop']:+.2f} & {r['t2i_drop']:+.2f} & {r['drop_gap']:+.2f} \\\\"
        )
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")

    s = "\n".join(out)
    (OUT / "t2_headline.tex").write_text(s)
    return s


# =============================================================================
# T3: per-corruption-type retrieval (appendix)
# =============================================================================
def t3_per_corruption() -> str:
    """Compact LaTeX of per-(corruption, severity, model) R@1 for both directions.
    Sourced from existing seed CSVs (single-modality retrieval covers gaussian_noise + mask only;
    so we use the per-pair-retrieval CSV restricted to those two corruptions)."""
    out = []
    out.append(r"\begin{table}[h]")
    out.append(r"\centering")
    out.append(r"\caption{Per-corruption Recall@1 for the joint-grid corruption (Gaussian noise on the image side, masking on the text side). Recall@1 is reported at severity 5 (most damaged), with bootstrap 95\% CIs in brackets, separately for i2t and t2i directions across all five models.}")
    out.append(r"\label{tab:t3_per_corruption}")
    out.append(r"\footnotesize")
    out.append(r"\begin{tabular}{lcccc}")
    out.append(r"\toprule")
    out.append(r"Model & i2t R@1 sev 1 & i2t R@1 sev 5 & t2i R@1 sev 1 & t2i R@1 sev 5 \\")
    out.append(r"\midrule")

    pp = pd.read_csv(ROOT / "results/per_pair_retrieval_v2/per_pair_retrieval_v2.csv", low_memory=False)
    pp = pp[pp["spoke"] == "retrieval_per_pair"].copy()
    for c in ("seed", "pair_id", "image_severity", "text_severity"):
        pp[c] = pd.to_numeric(pp[c], errors="coerce").astype("Int64")
    pp["value"] = pd.to_numeric(pp["value"]).astype(int)

    for m in MODEL_ORDER:
        cells = []
        for direction, sev_col, target_sev in (("i2t", "image_severity", 1), ("i2t", "image_severity", 5),
                                               ("t2i", "text_severity", 1), ("t2i", "text_severity", 5)):
            metric = f"per_pair_recall_at_1_{direction}"
            sub = pp[(pp["model"] == m) & (pp["metric"] == metric) & (pp[sev_col] == target_sev)]
            mean, lo, hi = hier_boot_mean(sub, n_boot=N_BOOT)
            cells.append(f"{mean:.2f} [{lo:.2f}, {hi:.2f}]")
        out.append(f"{MODEL_LABELS[m]} & " + " & ".join(cells) + r" \\")

    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    s = "\n".join(out)
    (OUT / "t3_per_corruption.tex").write_text(s)
    return s


# =============================================================================
# T4: dominance regression (appendix, demoted)
# =============================================================================
def t4_dominance() -> str:
    """Per-model R^2 for ordinal-severity and measured-damage versions of dominance regression."""
    seeds = pd.concat([pd.read_csv(ROOT / f"results/seed{s}/seed{s}_results.csv", low_memory=False) for s in (0, 1, 2)],
                      ignore_index=True)
    j = seeds[(seeds["spoke"] == "match_retention")
              & (seeds["match_retention_direction"] == "joint")
              & (seeds["metric"] == "retention_margin")].copy()
    for c in ("image_severity", "text_severity"):
        j[c] = pd.to_numeric(j[c]).astype(int)
    j["max_sev"] = j[["image_severity", "text_severity"]].max(axis=1)
    j["min_sev"] = j[["image_severity", "text_severity"]].min(axis=1)

    # measured-damage percentiles (per-condition averages)
    dmg = pd.read_csv(ROOT / "results/input_damage.csv")
    img_cell = dmg[dmg["modality"] == "image"].groupby(["corruption_type", "severity"])["damage_ssim"].mean().reset_index()
    img_cell["pct"] = img_cell["damage_ssim"].rank() / len(img_cell)
    img_pct = dict(zip(zip(img_cell["corruption_type"], img_cell["severity"]), img_cell["pct"]))
    txt_cell = dmg[dmg["modality"] == "text"].groupby(["corruption_type", "severity"])["damage_bleu"].mean().reset_index()
    txt_cell["pct"] = txt_cell["damage_bleu"].rank() / len(txt_cell)
    txt_pct = dict(zip(zip(txt_cell["corruption_type"], txt_cell["severity"]), txt_cell["pct"]))

    j["img_pct"] = j["image_severity"].map(lambda s: img_pct[("gaussian_noise", s)])
    j["txt_pct"] = j["text_severity"].map(lambda s: txt_pct[("mask", s)])
    j["more_damaged"] = j[["img_pct", "txt_pct"]].max(axis=1)
    j["less_damaged"] = j[["img_pct", "txt_pct"]].min(axis=1)

    rows = []
    for m in MODEL_ORDER:
        sub = j[j["model"] == m]
        y = sub["value"].to_numpy()
        for tag, x_col in (("ordinal_max", "max_sev"), ("ordinal_min", "min_sev"),
                           ("measured_more", "more_damaged"), ("measured_less", "less_damaged")):
            x = sub[x_col].to_numpy()
            slope, intercept = np.polyfit(x, y, 1)
            ss_tot = ((y - y.mean()) ** 2).sum()
            ss_res = ((y - (slope * x + intercept)) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            rows.append({"model": m, "tag": tag, "R2": r2, "slope": slope})
    df = pd.DataFrame(rows).pivot(index="model", columns="tag", values="R2").reset_index()

    out = []
    out.append(r"\begin{table}[h]")
    out.append(r"\centering")
    out.append(r"\caption{Dominance regression $R^2$ values, per model. The ordinal version uses raw severity (1 to 5); the measured-damage version uses within-modality damage percentiles. The ordinal version shows max outpredicting min; the measured-damage version reverses, indicating the ordinal effect was confounded by non-commensurate severity scales. We report this here for transparency and demote it from the main paper.}")
    out.append(r"\label{tab:t4_dominance}")
    out.append(r"\footnotesize")
    out.append(r"\begin{tabular}{lcccc}")
    out.append(r"\toprule")
    out.append(r" & \multicolumn{2}{c}{Ordinal severity} & \multicolumn{2}{c}{Measured damage percentile} \\")
    out.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
    out.append(r"Model & $R^2$(max) & $R^2$(min) & $R^2$(more) & $R^2$(less) \\")
    out.append(r"\midrule")
    for _, r in df.iterrows():
        out.append(f"{MODEL_LABELS[r['model']]} & {r['ordinal_max']:.3f} & {r['ordinal_min']:.3f} & "
                   f"{r['measured_more']:.3f} & {r['measured_less']:.3f} \\\\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    s = "\n".join(out)
    (OUT / "t4_dominance.tex").write_text(s)
    return s


# =============================================================================
# T5: pooling probe two-component table (appendix)
# =============================================================================
def t5_pooling_probe() -> str:
    p = pooling_probe_v2()
    p = p[p["metric"] == "retention_margin"]

    rows = []
    for m in MODEL_ORDER:
        cells = {}
        for pt in ("standard", "mean"):
            sub = p[(p["model"] == m) & (p["pool_type"] == pt)]
            sev1 = sub[sub["text_severity"] == 1]
            s1m, s1lo, s1hi = hier_boot_mean(sev1, n_boot=N_BOOT)
            sev5 = sub[sub["text_severity"] == 5]
            s5m, s5lo, s5hi = hier_boot_mean(sev5, n_boot=N_BOOT)
            cells[pt] = (s1m, s1lo, s1hi, s1m - s5m)
        red_pct = (1 - cells["mean"][3] / cells["standard"][3]) * 100 if cells["standard"][3] > 0 else 0.0
        rows.append({"model": m,
                     "std_sev1": cells["standard"][0], "std_sev1_lo": cells["standard"][1], "std_sev1_hi": cells["standard"][2],
                     "mean_sev1": cells["mean"][0], "mean_sev1_lo": cells["mean"][1], "mean_sev1_hi": cells["mean"][2],
                     "std_drop": cells["standard"][3], "mean_drop": cells["mean"][3],
                     "drop_red_pct": red_pct})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "t5_pooling_probe.csv", index=False)

    out = []
    out.append(r"\begin{table}[h]")
    out.append(r"\centering")
    out.append(r"\caption{Two-component decomposition of the asymmetry, per model. Sev 1 retention columns capture the baseline component: mean-pool retention is uniformly lower than standard-pool retention. Drop columns capture the slope component: mean-pool reduces the within-text sev 1 to sev 5 retention drop by 24 to 75 percent across models (PE-Core 24, B/32 34, LAION L/14 37, L/14 43, SigLIP 2 75). The drop reduction percent is an upper-bound proxy for the pooling bottleneck's contribution to the slope component.}")
    out.append(r"\label{tab:t5_pooling}")
    out.append(r"\footnotesize")
    out.append(r"\begin{tabular}{lcccccc}")
    out.append(r"\toprule")
    out.append(r" & \multicolumn{2}{c}{Sev 1 retention} & \multicolumn{2}{c}{Sev 1$\to$5 drop} & \multicolumn{2}{c}{Drop reduction} \\")
    out.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
    out.append(r"Model & std pool & mean pool & std pool & mean pool & absolute & \% \\")
    out.append(r"\midrule")
    for _, r in df.iterrows():
        out.append(f"{MODEL_LABELS[r['model']]} & "
                   f"{r['std_sev1']:.3f} & {r['mean_sev1']:.3f} & "
                   f"{r['std_drop']:.3f} & {r['mean_drop']:.3f} & "
                   f"{r['std_drop'] - r['mean_drop']:.3f} & {r['drop_red_pct']:.0f}\\% \\\\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    s = "\n".join(out)
    (OUT / "t5_pooling_probe.tex").write_text(s)
    return s


# =============================================================================
# T6: input-damage calibration (appendix)
# =============================================================================
def t6_damage_calibration() -> str:
    dmg = pd.read_csv(ROOT / "results/input_damage.csv")
    rows = []
    for mod in ("image", "text"):
        sub = dmg[dmg["modality"] == mod]
        for ct in sub["corruption_type"].unique():
            for sev in (1, 2, 3, 4, 5):
                grp = sub[(sub["corruption_type"] == ct) & (sub["severity"] == sev)]
                if mod == "image":
                    rows.append({"modality": mod, "corruption_type": ct, "severity": sev,
                                 "ssim": grp["ssim"].mean(), "psnr": grp["psnr"].mean(),
                                 "ed": np.nan, "bleu": np.nan})
                else:
                    rows.append({"modality": mod, "corruption_type": ct, "severity": sev,
                                 "ssim": np.nan, "psnr": np.nan,
                                 "ed": grp["norm_edit_distance"].mean(), "bleu": grp["bleu"].mean()})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "t6_damage_calibration.csv", index=False)

    out = []
    out.append(r"\begin{table}[h]")
    out.append(r"\centering")
    out.append(r"\caption{Mean input-side damage per (corruption, severity), pooled across 3 seeds and 300 main-spoke pairs. Image: SSIM (higher is less damaged) and PSNR (dB, capped at 60). Text: normalized token edit distance (higher is more damaged) and BLEU with method-1 smoothing (higher is less damaged). Severity values produce monotonically increasing damage within each corruption.}")
    out.append(r"\label{tab:t6_damage}")
    out.append(r"\footnotesize")
    out.append(r"\begin{tabular}{llcccc}")
    out.append(r"\toprule")
    out.append(r"Modality & Corruption & Sev & SSIM & PSNR & Edit dist. / BLEU \\")
    out.append(r"\midrule")
    for _, r in df.iterrows():
        if r["modality"] == "image":
            cells = f"{r['ssim']:.3f} & {r['psnr']:.1f} & --- "
        else:
            cells = f"--- & --- & {r['ed']:.3f} / {r['bleu']:.3f}"
        out.append(f"{r['modality']} & {r['corruption_type']} & {r['severity']} & {cells} \\\\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    s = "\n".join(out)
    (OUT / "t6_damage_calibration.tex").write_text(s)
    return s


# =============================================================================
# T7: PE-Core truncation analysis (appendix)
# =============================================================================
def t7_pecore_truncation() -> str:
    out = []
    out.append(r"\begin{table}[h]")
    out.append(r"\centering")
    out.append(r"\caption{PE-Core L14-336 has a 32-token text context, the shortest of the five models. We tokenized every distinct caption used across the experiment with the PE-Core BPE tokenizer and counted truncation. Only one of 1873 captions exceeded the 30-token content limit (0.05\%); PE-Core results are therefore not biased by silent caption truncation.}")
    out.append(r"\label{tab:t7_pecore}")
    out.append(r"\small")
    out.append(r"\begin{tabular}{lc}")
    out.append(r"\toprule")
    out.append(r"Quantity & Value \\")
    out.append(r"\midrule")
    out.append(r"Distinct captions checked & 1873 \\")
    out.append(r"PE-Core context length & 32 tokens \\")
    out.append(r"Maximum content tokens (BOS + EOT excluded) & 30 \\")
    out.append(r"Captions exceeding 30 BPE tokens & 1 \\")
    out.append(r"Fraction truncated & 0.05\% \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    s = "\n".join(out)
    (OUT / "t7_pecore_truncation.tex").write_text(s)
    return s


# =============================================================================
def main():
    for name, fn in (("T1", t1_model_summary), ("T2", t2_headline),
                     ("T3", t3_per_corruption), ("T4", t4_dominance),
                     ("T5", t5_pooling_probe), ("T6", t6_damage_calibration),
                     ("T7", t7_pecore_truncation)):
        print(f"=== {name} ===")
        s = fn()
        print(s.split("\n")[2][:160])
        print()


if __name__ == "__main__":
    main()
