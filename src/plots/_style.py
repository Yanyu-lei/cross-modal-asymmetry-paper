"""Shared matplotlib style for all paper figures."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette (8 colors, distinguishable in grayscale via lightness)
OKABE_ITO = (
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
)

# Per-model color map (5 models, distinct hues + grayscale-distinguishable)
MODEL_COLORS = {
    "openai_clip_b32":      "#0072B2",  # blue
    "openai_clip_l14":      "#56B4E9",  # sky
    "openclip_l14_laion2b": "#009E73",  # green
    "siglip2_so400m_384":   "#D55E00",  # vermillion
    "pecore_l14_336":       "#CC79A7",  # purple
}

# Per-model display label (paper-facing). Full names appear ONLY in T1 and figure
# captions, NEVER in panel titles. Use MODEL_SHORT for panel titles.
MODEL_LABELS = {
    "openai_clip_b32":      "CLIP B/32",
    "openai_clip_l14":      "CLIP L/14",
    "openclip_l14_laion2b": "OpenCLIP L/14 (LAION)",
    "siglip2_so400m_384":   "SigLIP 2 SO400M",
    "pecore_l14_336":       "PE-Core L/14",
}

# Short panel-title labels per v5 spec layout rules
MODEL_SHORT = {
    "openai_clip_b32":      "B/32",
    "openai_clip_l14":      "L/14",
    "openclip_l14_laion2b": "LAION L/14",
    "siglip2_so400m_384":   "SigLIP 2",
    "pecore_l14_336":       "PE-Core",
}

MODEL_ORDER = (
    "openai_clip_b32",
    "openai_clip_l14",
    "openclip_l14_laion2b",
    "siglip2_so400m_384",
    "pecore_l14_336",
)

# Direction colors (i2t vs t2i)
DIR_COLORS = {"i2t": "#0072B2", "t2i": "#D55E00"}
DIR_HATCH = {"i2t": "", "t2i": "//"}
POOL_COLORS = {"standard": "#0072B2", "mean": "#E69F00"}
POOL_HATCH = {"standard": "", "mean": "xx"}

COL_WIDTH_IN = 5.5
TWO_COL_IN = 5.5


def apply_rc():
    mpl.rcParams.update({
        "font.family":          "DejaVu Sans",
        "font.size":            8,
        "axes.titlesize":       9,
        "axes.labelsize":       8,
        "xtick.labelsize":      7,
        "ytick.labelsize":      7,
        "legend.fontsize":      7,
        "figure.titlesize":     9,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.linewidth":       0.6,
        "xtick.major.width":    0.6,
        "ytick.major.width":    0.6,
        "xtick.major.size":     2.5,
        "ytick.major.size":     2.5,
        "lines.linewidth":      1.2,
        "lines.markersize":     4,
        "grid.linewidth":       0.4,
        "grid.alpha":           0.30,
        "savefig.dpi":          200,
        "savefig.bbox":         "tight",
        "pdf.fonttype":         42,    # TrueType, not Type-3
        "ps.fonttype":          42,
    })


def style_axes(ax, *, grid: bool = True):
    ax.tick_params(direction="out")
    if grid:
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)


def figpath(name: str, *, ext: str = "pdf") -> Path:
    out = Path(__file__).resolve().parents[2] / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{name}.{ext}"
