"""F5 (appendix): two-component decomposition.

Panel A (slope component): per-model bars, sev1->sev5 retention drop under
standard pool vs mean pool, with drop-reduction percent annotated.

Panel B (baseline component): per-model bars, sev=1 retention for image
standard, text standard, text mean. Mean-pool baselines uniformly lower.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.plots._data import pooling_probe_v2, seeds_combined, ROOT
from src.plots._style import (
    MODEL_SHORT, MODEL_ORDER, COL_WIDTH_IN, POOL_COLORS, POOL_HATCH,
    apply_rc, style_axes, figpath, OKABE_ITO,
)


def _slope_data():
    """Drop sev1->sev5 per (model, pool_type), averaged across 3 text corruptions."""
    p = pooling_probe_v2()
    p = p[p["metric"] == "retention_margin"]
    rows = []
    for m in MODEL_ORDER:
        for pt in ("standard", "mean"):
            sub = p[(p["model"] == m) & (p["pool_type"] == pt)]
            sev1 = sub[sub["text_severity"] == 1]["value"].mean()
            sev5 = sub[sub["text_severity"] == 5]["value"].mean()
            rows.append({"model": m, "pool_type": pt, "drop": sev1 - sev5,
                         "sev1": sev1, "sev5": sev5})
    return pd.DataFrame(rows)


def _baseline_data():
    """Sev=1 retention for image-standard, text-standard, text-mean per model."""
    # Image-standard sev=1 retention margin (image_corrupted direction, sev=1)
    seeds = seeds_combined()
    img = seeds[(seeds["spoke"] == "match_retention")
                & (seeds["match_retention_direction"] == "image_corrupted")
                & (seeds["metric"] == "retention_margin")
                & (pd.to_numeric(seeds["image_severity"]).astype(int) == 1)]
    img_baseline = img.groupby("model")["value"].mean()

    # Text standard / mean from probe v2 sev=1
    p = pooling_probe_v2()
    p = p[(p["metric"] == "retention_margin") & (p["text_severity"] == 1)]
    txt_std = p[p["pool_type"] == "standard"].groupby("model")["value"].mean()
    txt_mean = p[p["pool_type"] == "mean"].groupby("model")["value"].mean()

    rows = []
    for m in MODEL_ORDER:
        rows.append({"model": m,
                     "img_std": img_baseline.get(m, np.nan),
                     "txt_std": txt_std.get(m, np.nan),
                     "txt_mean": txt_mean.get(m, np.nan)})
    return pd.DataFrame(rows)


def build():
    apply_rc()
    slope = _slope_data()
    base = _baseline_data()
    slope.to_csv(ROOT / "results/figures/f5_slope_data.csv", index=False)
    base.to_csv(ROOT / "results/figures/f5_baseline_data.csv", index=False)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(COL_WIDTH_IN * 1.3, 2.6),
                                     constrained_layout=True)

    # ---- Panel A: slope ----
    x = np.arange(len(MODEL_ORDER))
    width = 0.36
    std_drops = [slope[(slope["model"] == m) & (slope["pool_type"] == "standard")]["drop"].iloc[0] for m in MODEL_ORDER]
    mean_drops = [slope[(slope["model"] == m) & (slope["pool_type"] == "mean")]["drop"].iloc[0] for m in MODEL_ORDER]
    axA.bar(x - width/2, std_drops, width, color=POOL_COLORS["standard"],
            edgecolor="black", linewidth=0.4, label="standard pool")
    axA.bar(x + width/2, mean_drops, width, color=POOL_COLORS["mean"],
            edgecolor="black", linewidth=0.4, hatch=POOL_HATCH["mean"], label="mean pool")
    # annotate drop reduction percent
    for i, m in enumerate(MODEL_ORDER):
        s, mn = std_drops[i], mean_drops[i]
        pct = (1 - mn / s) * 100 if s > 0 else 0
        y = max(s, mn) + 0.005
        axA.text(x[i], y, f"{pct:.0f}%", ha="center", va="bottom", fontsize=6)
    axA.set_xticks(x)
    axA.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER], rotation=20, ha="right", fontsize=6)
    axA.set_ylabel("sev1 to sev5 retention drop")
    axA.set_title("A. Slope component (within-text drop)", pad=2)
    axA.legend(fontsize=6, frameon=False, loc="upper right")
    style_axes(axA)

    # ---- Panel B: baseline ----
    width = 0.26
    img_std = [base[base["model"] == m]["img_std"].iloc[0] for m in MODEL_ORDER]
    txt_std = [base[base["model"] == m]["txt_std"].iloc[0] for m in MODEL_ORDER]
    txt_mean = [base[base["model"] == m]["txt_mean"].iloc[0] for m in MODEL_ORDER]
    axB.bar(x - width, img_std, width, color="#0072B2",
            edgecolor="black", linewidth=0.4, label="image, std pool")
    axB.bar(x, txt_std, width, color="#D55E00",
            edgecolor="black", linewidth=0.4, label="text, std pool")
    axB.bar(x + width, txt_mean, width, color="#E69F00",
            edgecolor="black", linewidth=0.4, hatch="xx", label="text, mean pool")
    axB.set_xticks(x)
    axB.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER], rotation=20, ha="right", fontsize=6)
    axB.set_ylabel("Retention margin at sev 1")
    axB.set_title("B. Baseline component (sev 1 retention)", pad=2)
    axB.legend(fontsize=6, frameon=False, loc="upper right")
    style_axes(axB)

    out = figpath("f5_two_component_decomposition")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    build()
