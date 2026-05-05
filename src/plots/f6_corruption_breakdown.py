"""F6 (appendix): per-corruption Recall@1 at sev=5 + clean opposite-direction baseline.

For each corruption type, two bars per model:
  - corrupted-side bar: i2t (image corruption) or t2i (text corruption) at sev=5
  - clean opposite-direction baseline: true-clean R@1 in the opposite direction

The reference is true clean retrieval in the opposite direction with no corruption
applied; it is NOT a candidate-side corruption experiment.

Layout: 1 row x 5 model panels + shared legend. Vertical separator between
image-corruption group and text-corruption group.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis.bootstrap import hier_boot_mean
from src.plots._data import per_pair_retrieval, clean_baseline_per_pair, ROOT
from src.plots._style import (
    MODEL_SHORT, MODEL_ORDER, COL_WIDTH_IN, DIR_COLORS,
    apply_rc, style_axes, figpath,
)

IMG_CORRUPTIONS = ("gaussian_noise", "gaussian_blur", "cutout")
TXT_CORRUPTIONS = ("mask", "shuffle", "replace")
IMG_LABELS = ("noise", "blur", "cutout")
TXT_LABELS = ("mask", "shuffle", "replace")
SEV = 5
N_BOOT = 10000


def _summarize() -> pd.DataFrame:
    pp = per_pair_retrieval()
    cln = clean_baseline_per_pair()
    rows = []
    for m in MODEL_ORDER:
        for ct in IMG_CORRUPTIONS:
            sub = pp[(pp["model"] == m) & (pp["metric"] == "per_pair_recall_at_1_i2t")
                     & (pp["vision_corruption"] == ct) & (pp["image_severity"] == SEV)]
            mean, lo, hi = hier_boot_mean(sub, n_boot=N_BOOT)
            rows.append({"model": m, "side": "image", "corruption_type": ct,
                         "kind": "corrupted", "mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(sub)})
            ref = cln[(cln["model"] == m) & (cln["metric"] == "per_pair_clean_recall_at_1_t2i")]
            rmean, rlo, rhi = hier_boot_mean(ref, n_boot=N_BOOT)
            rows.append({"model": m, "side": "image", "corruption_type": ct,
                         "kind": "clean_opposite", "mean": rmean, "ci_lo": rlo, "ci_hi": rhi, "n": len(ref)})
        for ct in TXT_CORRUPTIONS:
            sub = pp[(pp["model"] == m) & (pp["metric"] == "per_pair_recall_at_1_t2i")
                     & (pp["text_corruption"] == ct) & (pp["text_severity"] == SEV)]
            mean, lo, hi = hier_boot_mean(sub, n_boot=N_BOOT)
            rows.append({"model": m, "side": "text", "corruption_type": ct,
                         "kind": "corrupted", "mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(sub)})
            ref = cln[(cln["model"] == m) & (cln["metric"] == "per_pair_clean_recall_at_1_i2t")]
            rmean, rlo, rhi = hier_boot_mean(ref, n_boot=N_BOOT)
            rows.append({"model": m, "side": "text", "corruption_type": ct,
                         "kind": "clean_opposite", "mean": rmean, "ci_lo": rlo, "ci_hi": rhi, "n": len(ref)})
    return pd.DataFrame(rows)


def build():
    apply_rc()
    df = _summarize()
    df.to_csv(ROOT / "results/figures/f6_data.csv", index=False)

    fig, axes = plt.subplots(1, 5, figsize=(COL_WIDTH_IN * 1.4, 2.6), sharey=True,
                             constrained_layout=True)
    bar_w = 0.4

    for i, m in enumerate(MODEL_ORDER):
        ax = axes[i]
        sub = df[df["model"] == m]
        x_img = np.arange(3)
        x_txt = np.arange(3) + 4

        for j, ct in enumerate(IMG_CORRUPTIONS):
            corr = sub[(sub["side"] == "image") & (sub["corruption_type"] == ct) & (sub["kind"] == "corrupted")].iloc[0]
            ref  = sub[(sub["side"] == "image") & (sub["corruption_type"] == ct) & (sub["kind"] == "clean_opposite")].iloc[0]
            ax.bar(x_img[j] - bar_w/2, corr["mean"], bar_w,
                   color=DIR_COLORS["i2t"], edgecolor="black", linewidth=0.4,
                   yerr=([corr["mean"] - corr["ci_lo"]], [corr["ci_hi"] - corr["mean"]]),
                   error_kw={"elinewidth": 0.5, "capsize": 1.0})
            ax.bar(x_img[j] + bar_w/2, ref["mean"], bar_w,
                   color="lightgray", edgecolor="black", linewidth=0.4, hatch="//",
                   yerr=([ref["mean"] - ref["ci_lo"]], [ref["ci_hi"] - ref["mean"]]),
                   error_kw={"elinewidth": 0.5, "capsize": 1.0})
        for j, ct in enumerate(TXT_CORRUPTIONS):
            corr = sub[(sub["side"] == "text") & (sub["corruption_type"] == ct) & (sub["kind"] == "corrupted")].iloc[0]
            ref  = sub[(sub["side"] == "text") & (sub["corruption_type"] == ct) & (sub["kind"] == "clean_opposite")].iloc[0]
            ax.bar(x_txt[j] - bar_w/2, corr["mean"], bar_w,
                   color=DIR_COLORS["t2i"], edgecolor="black", linewidth=0.4,
                   yerr=([corr["mean"] - corr["ci_lo"]], [corr["ci_hi"] - corr["mean"]]),
                   error_kw={"elinewidth": 0.5, "capsize": 1.0})
            ax.bar(x_txt[j] + bar_w/2, ref["mean"], bar_w,
                   color="lightgray", edgecolor="black", linewidth=0.4, hatch="//",
                   yerr=([ref["mean"] - ref["ci_lo"]], [ref["ci_hi"] - ref["mean"]]),
                   error_kw={"elinewidth": 0.5, "capsize": 1.0})

        ax.set_xticks(list(x_img) + list(x_txt))
        ax.set_xticklabels(list(IMG_LABELS) + list(TXT_LABELS), fontsize=6, rotation=30, ha="right")
        ax.axvline(3.0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_ylim(0, 1.0)
        if i == 0:
            ax.set_ylabel("Recall@1 at sev 5")
        ax.set_title(MODEL_SHORT[m], pad=2)
        style_axes(ax)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=DIR_COLORS["i2t"], edgecolor="black", linewidth=0.4),
        plt.Rectangle((0, 0), 1, 1, facecolor=DIR_COLORS["t2i"], edgecolor="black", linewidth=0.4),
        plt.Rectangle((0, 0), 1, 1, facecolor="lightgray", edgecolor="black", linewidth=0.4, hatch="//"),
    ]
    labels = [
        "i2t (image corrupted)",
        "t2i (text corrupted)",
        "clean opposite-direction baseline",
    ]
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.04))

    out = figpath("f6_corruption_breakdown")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")

    print("\nText-side corrupted-bar means at sev=5 (sanity check):")
    text_corr = df[(df["side"] == "text") & (df["kind"] == "corrupted")]
    pivot = text_corr.pivot_table(index="model", columns="corruption_type", values="mean").round(4)
    print(pivot.to_string())
    print(f"\nshuffle/mask R@1 ratio per model: {(pivot['shuffle'] / pivot['mask']).round(2).to_dict()}")


if __name__ == "__main__":
    build()
