"""F4 (appendix): Image and text layer-wise fidelity per model.

Layout: 2 rows by 5 columns. Top = image fidelity, bottom = text fidelity.
One column per model. Lines = severity 1..5; x = depth (early/mid/late).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.plots._data import seeds_combined, ROOT
from src.plots._style import (
    MODEL_SHORT, MODEL_ORDER, COL_WIDTH_IN,
    apply_rc, style_axes, figpath, OKABE_ITO,
)

DEPTHS = ("early", "mid", "late")
SEVERITY_COLORS = ("#deebf7", "#9ecae1", "#4292c6", "#2171b5", "#08306b")  # 5 shades


def _fidelity_table(spoke: str, sev_col: str):
    df = seeds_combined()
    sub = df[df["spoke"] == spoke].copy()
    sub[sev_col] = pd.to_numeric(sub[sev_col]).astype(int)
    return sub.groupby(["model", sev_col, "depth"])["value"].mean().reset_index()


def build():
    apply_rc()
    img = _fidelity_table("image_fidelity", "image_severity")
    txt = _fidelity_table("text_fidelity", "text_severity")

    fig, axes = plt.subplots(2, 5, figsize=(COL_WIDTH_IN * 1.4, 4.0), sharey=True,
                             constrained_layout=True)
    x = np.arange(len(DEPTHS))

    for i, m in enumerate(MODEL_ORDER):
        ax = axes[0][i]
        for sev in (1, 2, 3, 4, 5):
            row = img[(img["model"] == m) & (img["image_severity"] == sev)]
            y = [row[row["depth"] == d]["value"].mean() for d in DEPTHS]
            ax.plot(x, y, marker="o", color=SEVERITY_COLORS[sev - 1], linewidth=1.2,
                    label=f"sev {sev}" if i == 0 else None)
        ax.set_xticks(x); ax.set_xticklabels(DEPTHS, fontsize=6)
        ax.set_ylim(0.4, 1.0)
        if i == 0:
            ax.set_ylabel("Image patch cosine")
        ax.set_title(MODEL_SHORT[m], pad=2)
        style_axes(ax)
        if i == 0:
            ax.legend(fontsize=6, loc="lower left", frameon=False, handlelength=1.0, ncol=1)

        ax = axes[1][i]
        for sev in (1, 2, 3, 4, 5):
            row = txt[(txt["model"] == m) & (txt["text_severity"] == sev)]
            y = [row[row["depth"] == d]["value"].mean() for d in DEPTHS]
            ax.plot(x, y, marker="o", color=SEVERITY_COLORS[sev - 1], linewidth=1.2)
        ax.set_xticks(x); ax.set_xticklabels(DEPTHS, fontsize=6)
        ax.set_ylim(0.4, 1.0)
        if i == 0:
            ax.set_ylabel("Text pooled cosine")
        ax.set_xlabel("Depth")
        style_axes(ax)

    out = figpath("f4_layer_fidelity")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    build()
