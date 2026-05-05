"""PSNR cross-check for F1/F2 image-side quintile binning (diagnostic only).

Re-runs F1 and F2 with PSNR (inverted to damage_psnr = -psnr, since lower PSNR = more
damage) as the image-side ruler. Text side keeps BLEU. Outputs:

  results/figures/f1_psnr.pdf
  results/figures/f2_psnr.pdf
  results/diagnostics/f1_psnr_data.csv
  results/diagnostics/f2_psnr_data.csv

T2 stays SSIM-canonical; this is a robustness check on the F1/F2 binning, not a
parallel main-table analysis.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.bootstrap import hier_boot_mean
from src.plots._data import (
    per_pair_retrieval, retrieval_damage, t2i_with_damage, quintile_assign,
    ROOT as PROOT, QUINTILES,
)
from src.plots._style import (
    DIR_COLORS, DIR_HATCH, MODEL_SHORT, MODEL_ORDER, COL_WIDTH_IN,
    apply_rc, style_axes, figpath,
)


def i2t_with_psnr() -> pd.DataFrame:
    pp = per_pair_retrieval()
    i2t = pp[pp["metric"] == "per_pair_recall_at_1_i2t"][
        ["seed", "model", "pair_id", "vision_corruption", "image_severity", "value"]
    ].copy()
    dmg = retrieval_damage()
    img = dmg[dmg["modality"] == "image"][
        ["seed", "pair_id", "corruption_type", "severity", "psnr"]
    ].rename(columns={"corruption_type": "vision_corruption", "severity": "image_severity"})
    return i2t.merge(img, on=["seed", "pair_id", "vision_corruption", "image_severity"], how="inner")


def _summary(n_boot: int = 10000) -> pd.DataFrame:
    i2t = i2t_with_psnr().copy()
    i2t["damage_psnr"] = -i2t["psnr"]  # higher = more damage
    t2i = t2i_with_damage()
    rows = []
    for m in MODEL_ORDER:
        i_sub = quintile_assign(i2t[i2t["model"] == m], "damage_psnr")
        t_sub = quintile_assign(t2i[t2i["model"] == m], "damage_bleu")
        for q in QUINTILES:
            iq = i_sub[i_sub["quintile"] == q]
            tq = t_sub[t_sub["quintile"] == q]
            i_m, i_lo, i_hi = hier_boot_mean(iq, n_boot=n_boot)
            t_m, t_lo, t_hi = hier_boot_mean(tq, n_boot=n_boot)
            rows.append({"model": m, "quintile": q,
                         "i2t": i_m, "i2t_lo": i_lo, "i2t_hi": i_hi,
                         "t2i": t_m, "t2i_lo": t_lo, "t2i_hi": t_hi,
                         "i2t_n": len(iq), "t2i_n": len(tq)})
    return pd.DataFrame(rows)


def build_f1(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(COL_WIDTH_IN, 3.4), sharey=True,
                             constrained_layout=True)
    flat = axes.flatten()
    for i, m in enumerate(MODEL_ORDER):
        ax = flat[i]
        sub = df[df["model"] == m].sort_values("quintile")
        x = np.arange(5); width = 0.4
        i_h = sub["i2t"].to_numpy(); t_h = sub["t2i"].to_numpy()
        i_err = np.array([sub["i2t"] - sub["i2t_lo"], sub["i2t_hi"] - sub["i2t"]])
        t_err = np.array([sub["t2i"] - sub["t2i_lo"], sub["t2i_hi"] - sub["t2i"]])
        ax.bar(x - width/2, i_h, width, yerr=i_err, color=DIR_COLORS["i2t"],
               edgecolor="black", linewidth=0.4,
               error_kw={"elinewidth": 0.5, "capsize": 1.5})
        ax.bar(x + width/2, t_h, width, yerr=t_err, color=DIR_COLORS["t2i"],
               edgecolor="black", linewidth=0.4, hatch=DIR_HATCH["t2i"],
               error_kw={"elinewidth": 0.5, "capsize": 1.5})
        ax.set_xticks(x); ax.set_xticklabels(QUINTILES); ax.set_ylim(0, 1.0)
        ax.set_title(MODEL_SHORT[m], pad=2)
        if i % 3 == 0: ax.set_ylabel("Recall@1")
        if i >= 3: ax.set_xlabel("Damage quintile (PSNR)")
        style_axes(ax)
    flat[5].axis("off")
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=DIR_COLORS["i2t"], edgecolor="black", linewidth=0.4),
        plt.Rectangle((0, 0), 1, 1, facecolor=DIR_COLORS["t2i"], edgecolor="black", linewidth=0.4, hatch=DIR_HATCH["t2i"]),
    ]
    flat[5].legend(handles, ["i2t Recall@1 (PSNR-binned)", "t2i Recall@1 (BLEU-binned)"],
                   loc="center", fontsize=7, frameon=False, handlelength=1.5)
    out = PROOT / "results/figures/f1_psnr.pdf"
    fig.savefig(out); plt.close(fig)
    print(f"Saved {out}")


def build_f2(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(COL_WIDTH_IN, 3.4), sharey=True,
                             constrained_layout=True)
    flat = axes.flatten()
    for i, m in enumerate(MODEL_ORDER):
        ax = flat[i]
        sub = df[df["model"] == m].sort_values("quintile")
        x = np.arange(5)
        i_h = sub["i2t"].to_numpy(); t_h = sub["t2i"].to_numpy()
        ax.fill_between(x, sub["i2t_lo"], sub["i2t_hi"], color=DIR_COLORS["i2t"], alpha=0.2, linewidth=0)
        ax.fill_between(x, sub["t2i_lo"], sub["t2i_hi"], color=DIR_COLORS["t2i"], alpha=0.2, linewidth=0)
        ax.plot(x, i_h, marker="o", color=DIR_COLORS["i2t"], linewidth=1.4)
        ax.plot(x, t_h, marker="s", color=DIR_COLORS["t2i"], linewidth=1.4, linestyle="--")
        ax.set_xticks(x); ax.set_xticklabels(QUINTILES); ax.set_ylim(0, 1.0)
        if i % 3 == 0: ax.set_ylabel("Recall@1")
        if i >= 3: ax.set_xlabel("Damage quintile (PSNR)")
        ax.set_title(MODEL_SHORT[m], pad=2)
        style_axes(ax)
    flat[5].axis("off")
    handles = [
        plt.Line2D([0], [0], color=DIR_COLORS["i2t"], marker="o", linewidth=1.4),
        plt.Line2D([0], [0], color=DIR_COLORS["t2i"], marker="s", linewidth=1.4, linestyle="--"),
    ]
    flat[5].legend(handles, ["i2t Recall@1 (PSNR-binned)", "t2i Recall@1 (BLEU-binned)"],
                   loc="center", fontsize=7, frameon=False, handlelength=2.0)
    out = PROOT / "results/figures/f2_psnr.pdf"
    fig.savefig(out); plt.close(fig)
    print(f"Saved {out}")


def main():
    apply_rc()
    df = _summary(n_boot=10000)
    (PROOT / "results/diagnostics").mkdir(exist_ok=True, parents=True)
    df.to_csv(PROOT / "results/diagnostics/f1_psnr_data.csv", index=False)
    df.to_csv(PROOT / "results/diagnostics/f2_psnr_data.csv", index=False)
    build_f1(df)
    build_f2(df)

    q5 = df[df["quintile"] == "Q5"].copy()
    q5["i2t_t2i_ratio"] = (q5["i2t"] / q5["t2i"]).round(2)
    print("\nPSNR cross-check Q5 means and i2t/t2i ratios:")
    print(q5[["model", "i2t", "t2i", "i2t_t2i_ratio", "i2t_n", "t2i_n"]].to_string(index=False))

    for m in MODEL_ORDER:
        sub = df[df["model"] == m].sort_values("quintile")
        i_q4 = sub[sub["quintile"] == "Q4"]["i2t"].iloc[0]
        i_q5 = sub[sub["quintile"] == "Q5"]["i2t"].iloc[0]
        bump = i_q5 - i_q4
        print(f"  {MODEL_SHORT[m]:12s}  i2t Q4={i_q4:.3f}  Q5={i_q5:.3f}  bump={bump:+.3f}")


if __name__ == "__main__":
    main()
