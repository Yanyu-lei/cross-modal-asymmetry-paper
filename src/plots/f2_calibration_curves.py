"""F2: Recall@1 vs within-modality damage quintile, two curves per panel, 2x3 layout."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis.bootstrap import hier_boot_mean
from src.plots._data import i2t_with_damage, t2i_with_damage, quintile_assign, ROOT, QUINTILES
from src.plots._style import (
    DIR_COLORS, MODEL_SHORT, MODEL_ORDER, COL_WIDTH_IN,
    apply_rc, style_axes, figpath,
)


def _summarize(n_boot: int = 10000) -> pd.DataFrame:
    i2t = i2t_with_damage()
    t2i = t2i_with_damage()
    rows = []
    for m in MODEL_ORDER:
        i_sub = quintile_assign(i2t[i2t["model"] == m], "damage_ssim")
        t_sub = quintile_assign(t2i[t2i["model"] == m], "damage_bleu")
        for q in QUINTILES:
            iq = i_sub[i_sub["quintile"] == q]
            tq = t_sub[t_sub["quintile"] == q]
            i_m, i_lo, i_hi = hier_boot_mean(iq, n_boot=n_boot)
            t_m, t_lo, t_hi = hier_boot_mean(tq, n_boot=n_boot)
            rows.append({"model": m, "quintile": q,
                         "i2t": i_m, "i2t_lo": i_lo, "i2t_hi": i_hi,
                         "t2i": t_m, "t2i_lo": t_lo, "t2i_hi": t_hi})
    return pd.DataFrame(rows)


def build():
    apply_rc()
    df = _summarize(n_boot=10000)
    df.to_csv(ROOT / "results/figures/f2_data.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(COL_WIDTH_IN, 3.4), sharey=True,
                             constrained_layout=True)
    flat = axes.flatten()
    for i, m in enumerate(MODEL_ORDER):
        ax = flat[i]
        sub = df[df["model"] == m].sort_values("quintile")
        x = np.arange(5)
        i_h = sub["i2t"].to_numpy(); t_h = sub["t2i"].to_numpy()
        i_lo = sub["i2t_lo"].to_numpy(); i_hi = sub["i2t_hi"].to_numpy()
        t_lo = sub["t2i_lo"].to_numpy(); t_hi = sub["t2i_hi"].to_numpy()
        ax.fill_between(x, i_lo, i_hi, color=DIR_COLORS["i2t"], alpha=0.2, linewidth=0)
        ax.fill_between(x, t_lo, t_hi, color=DIR_COLORS["t2i"], alpha=0.2, linewidth=0)
        ax.plot(x, i_h, marker="o", color=DIR_COLORS["i2t"], linewidth=1.4)
        ax.plot(x, t_h, marker="s", color=DIR_COLORS["t2i"], linewidth=1.4, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(QUINTILES)
        ax.set_ylim(0, 1.0)
        if i % 3 == 0:
            ax.set_ylabel("Recall@1")
        if i >= 3:
            ax.set_xlabel("Damage quintile")
        ax.set_title(MODEL_SHORT[m], pad=2)
        style_axes(ax)

    legend_ax = flat[5]
    legend_ax.axis("off")
    handles = [
        plt.Line2D([0], [0], color=DIR_COLORS["i2t"], marker="o", linewidth=1.4),
        plt.Line2D([0], [0], color=DIR_COLORS["t2i"], marker="s", linewidth=1.4, linestyle="--"),
    ]
    legend_ax.legend(handles, ["i2t Recall@1", "t2i Recall@1"],
                     loc="center", fontsize=7, frameon=False, handlelength=2.0)

    out = figpath("f2_calibration_curves")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    build()
