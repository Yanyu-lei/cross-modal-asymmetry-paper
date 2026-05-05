"""F1: Retrieval asymmetry at matched within-modality damage quintiles."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis.bootstrap import hier_boot_mean
from src.plots._data import (
    i2t_with_damage, t2i_with_damage, quintile_assign, clean_baseline_per_pair,
    ROOT, QUINTILES,
)
from src.plots._style import (
    DIR_COLORS, DIR_HATCH, MODEL_SHORT, MODEL_ORDER, COL_WIDTH_IN,
    apply_rc, style_axes, figpath,
)


def _quintile_summary(n_boot: int = 10000) -> pd.DataFrame:
    i2t = i2t_with_damage()
    t2i = t2i_with_damage()
    rows = []
    for m in MODEL_ORDER:
        i_sub = quintile_assign(i2t[i2t["model"] == m], "damage_ssim")
        t_sub = quintile_assign(t2i[t2i["model"] == m], "damage_bleu")
        for q in QUINTILES:
            iq = i_sub[i_sub["quintile"] == q]
            tq = t_sub[t_sub["quintile"] == q]
            i_mean, i_lo, i_hi = hier_boot_mean(iq, n_boot=n_boot)
            t_mean, t_lo, t_hi = hier_boot_mean(tq, n_boot=n_boot)
            rows.append({"model": m, "quintile": q,
                         "i2t": i_mean, "i2t_lo": i_lo, "i2t_hi": i_hi,
                         "t2i": t_mean, "t2i_lo": t_lo, "t2i_hi": t_hi,
                         "i2t_n": len(iq), "t2i_n": len(tq)})
    return pd.DataFrame(rows)


def _clean_overlay() -> pd.DataFrame:
    """True clean (no-corruption) Recall@1 per model, hierarchical bootstrap."""
    pp = clean_baseline_per_pair()
    rows = []
    for m in MODEL_ORDER:
        for direction in ("i2t", "t2i"):
            sub = pp[(pp["model"] == m) & (pp["metric"] == f"per_pair_clean_recall_at_1_{direction}")]
            mean, lo, hi = hier_boot_mean(sub, n_boot=10000)
            rows.append({"model": m, "direction": direction,
                         "mean": mean, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def _draw_panel(ax, sub, baseline, *, show_xlabel: bool):
    x = np.arange(5)
    width = 0.4
    i2t_h = sub["i2t"].to_numpy()
    t2i_h = sub["t2i"].to_numpy()
    i2t_err = np.array([sub["i2t"] - sub["i2t_lo"], sub["i2t_hi"] - sub["i2t"]])
    t2i_err = np.array([sub["t2i"] - sub["t2i_lo"], sub["t2i_hi"] - sub["t2i"]])
    ax.bar(x - width/2, i2t_h, width, yerr=i2t_err, color=DIR_COLORS["i2t"],
           edgecolor="black", linewidth=0.4,
           error_kw={"elinewidth": 0.5, "capsize": 1.5})
    ax.bar(x + width/2, t2i_h, width, yerr=t2i_err, color=DIR_COLORS["t2i"],
           edgecolor="black", linewidth=0.4, hatch=DIR_HATCH["t2i"],
           error_kw={"elinewidth": 0.5, "capsize": 1.5})
    if baseline is not None:
        b_i = baseline[baseline["direction"] == "i2t"]["mean"].iloc[0]
        b_t = baseline[baseline["direction"] == "t2i"]["mean"].iloc[0]
        ax.axhline(b_i, color=DIR_COLORS["i2t"], linewidth=1.2, linestyle=(0, (4, 2)), alpha=1.0)
        ax.axhline(b_t, color=DIR_COLORS["t2i"], linewidth=1.2, linestyle=(0, (4, 2)), alpha=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(QUINTILES)
    ax.set_ylim(0, 1.0)
    ax.set_title(MODEL_SHORT[sub["model"].iloc[0]], pad=2)
    if show_xlabel:
        ax.set_xlabel("Damage quintile")
    style_axes(ax)


def build():
    apply_rc()
    summary = _quintile_summary(n_boot=10000)
    overlay = _clean_overlay()
    summary.to_csv(ROOT / "results/figures/f1_data.csv", index=False)
    overlay.to_csv(ROOT / "results/figures/f1_clean_overlay.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(COL_WIDTH_IN, 3.4), sharey=True,
                             constrained_layout=True)
    flat = axes.flatten()
    for i, m in enumerate(MODEL_ORDER):
        ax = flat[i]
        sub = summary[summary["model"] == m].sort_values("quintile")
        b = overlay[overlay["model"] == m]
        _draw_panel(ax, sub, b, show_xlabel=(i >= 3))
        if i % 3 == 0:
            ax.set_ylabel("Recall@1")

    legend_ax = flat[5]
    legend_ax.axis("off")
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=DIR_COLORS["i2t"], edgecolor="black", linewidth=0.4),
        plt.Rectangle((0, 0), 1, 1, facecolor=DIR_COLORS["t2i"], edgecolor="black", linewidth=0.4, hatch=DIR_HATCH["t2i"]),
        plt.Line2D([0], [0], color=DIR_COLORS["i2t"], linestyle=(0, (4, 2)), linewidth=1.2),
        plt.Line2D([0], [0], color=DIR_COLORS["t2i"], linestyle=(0, (4, 2)), linewidth=1.2),
    ]
    legend_ax.legend(handles,
                     ["i2t Recall@1", "t2i Recall@1", "i2t clean baseline", "t2i clean baseline"],
                     loc="center", fontsize=7, frameon=False, handlelength=2.0)

    out = figpath("f1_retrieval_asymmetry")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")

    q1_gap = summary[summary["quintile"] == "Q1"].apply(lambda r: r["i2t"] - r["t2i"], axis=1).mean()
    q5_gap = summary[summary["quintile"] == "Q5"].apply(lambda r: r["i2t"] - r["t2i"], axis=1).mean()
    q5_ratios = summary[summary["quintile"] == "Q5"].apply(
        lambda r: r["i2t"] / r["t2i"] if r["t2i"] > 0 else np.nan, axis=1)
    print(f"Mean Q1 gap (i2t-t2i): {q1_gap:.3f}")
    print(f"Mean Q5 gap (i2t-t2i): {q5_gap:.3f}")
    print(f"Q5 i2t/t2i ratio per model: {q5_ratios.tolist()}")


if __name__ == "__main__":
    build()
