"""F3: 5x5 joint corruption retention margin heatmaps in 2x3 layout (5 panels + colorbar)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.plots._data import seeds_combined, ROOT
from src.plots._style import MODEL_SHORT, MODEL_ORDER, COL_WIDTH_IN, apply_rc, figpath


def _joint_grid() -> pd.DataFrame:
    df = seeds_combined()
    j = df[(df["spoke"] == "match_retention")
           & (df["match_retention_direction"] == "joint")
           & (df["metric"] == "retention_margin")].copy()
    for c in ("image_severity", "text_severity"):
        j[c] = pd.to_numeric(j[c]).astype(int)
    return j.groupby(["model", "image_severity", "text_severity"])["value"].mean().reset_index()


def build():
    apply_rc()
    j = _joint_grid()
    j.to_csv(ROOT / "results/figures/f3_data.csv", index=False)

    vmax = float(j["value"].max())
    vmin = float(j["value"].min())

    fig, axes = plt.subplots(2, 3, figsize=(COL_WIDTH_IN, 3.6), constrained_layout=True)
    flat = axes.flatten()
    last_im = None

    for i, m in enumerate(MODEL_ORDER):
        ax = flat[i]
        sub = j[j["model"] == m]
        M = sub.pivot(index="image_severity", columns="text_severity", values="value").to_numpy()
        last_im = ax.imshow(M, origin="lower", aspect="equal", cmap="viridis_r",
                            vmin=vmin, vmax=vmax, extent=[0.5, 5.5, 0.5, 5.5])
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.tick_params(labelsize=6, length=2)
        if i % 3 == 0:
            ax.set_ylabel("Image severity")
        if i >= 3:
            ax.set_xlabel("Text severity")
        ax.set_title(MODEL_SHORT[m], pad=2)

    cbar_ax = flat[5]
    cbar_ax.axis("off")
    # Vertical colorbar inside the 6th panel slot, sized to match heatmap height
    cbar = fig.colorbar(last_im, ax=cbar_ax, fraction=0.5, pad=0.05, aspect=14, shrink=0.85)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("retention margin", fontsize=7)

    out = figpath("f3_joint_heatmaps")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    build()
