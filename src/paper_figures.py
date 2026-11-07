"""Generate the paper's figures and their source-data CSVs."""
from __future__ import annotations

from functools import lru_cache

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator, PercentFormatter
import numpy as np
import pandas as pd

from src import file_paths, settings
from src.bootstrap_and_quintiles import QUINTILES, bootstrap_ci, bootstrap_group_statistics
from src.load_results import (clean_baseline_per_pair, datasets_with_300_pair_run, datasets_with_headline_rows,
                              fidelity_retention_rows, per_pair_retrieval, quintiles_by_model, recall_with_damage,
                              text_pooling_ablation)

MODELS = settings.MODELS
MODEL_LABELS = dict(zip(MODELS, ("CLIP B/32", "CLIP L/14", "OpenCLIP L/14", "SigLIP 2", "PE-Core")))  # Figures and tables share these.
MODEL_COLORS = dict(zip(MODELS, ("#3D7D80", "#755291", "#4F8665", "#A35E78", "#9B8042")))
MODEL_FILLS = dict(zip(MODELS, ("#7BB9C2", "#AF96C7", "#8BB69C", "#D49AAD", "#D5BD7F")))
MODEL_MARKERS = dict(zip(MODELS, ("s", "D", "^", "v", "P")))
DATASET_TITLES = {"coco": "COCO", "flickr30k": "Flickr30K"}

# Crosshatching identifies corrupted retrieval conditions in both directions.
DIRECTION_COLORS = {"i2t": "#0072B2", "t2i": "#D55E00"}
DIRECTION_HATCH = {"i2t": "xx", "t2i": "xx"}
CLEAN_COLORS = {"i2t": "#2687BE", "t2i": "#DB7626"}
POOLING_COLORS = {"standard": "#D55E00", "mean": "#E69F00"}
SEVERITY_COLORS = plt.get_cmap("Blues")(np.linspace(0.38, 0.90, 5))
ERROR_BARS = {"elinewidth": 0.6, "capsize": 1.2, "capthick": 0.6, "ecolor": "#202020"}
BAR_EDGE = {"edgecolor": "#333333", "linewidth": 0.45}
WIDTH_IN = 5.5  # The text width of the NeurIPS template, so a figure placed at the line width keeps these font sizes.
DEPTHS = ("early", "mid", "late")
SHORT_CORRUPTION_LABELS = {"gaussian_noise": "Noise", "gaussian_blur": "Blur", "cutout": "Rect. cutout", "mask": "Deletion",
                           "shuffle": "Shuffle", "replace": "Replacement", "typo": "Typo"}
IMAGE_RULERS = {"ssim": "damage_ssim", "psnr": "damage_psnr"}
DIRECTION_LEGEND = {"i2t": "Image-to-text", "t2i": "Text-to-image"}
CLEAN_THEN_SEVERITIES = np.array([0, 1.4, 2.4, 3.4, 4.4, 5.4])  # Severity 1 is already corrupted, so clean stands apart.

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5, "figure.titlesize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.pad": 2, "ytick.major.pad": 2, "lines.linewidth": 1.2, "lines.markersize": 4,
    "grid.linewidth": 0.4, "grid.alpha": 0.80, "hatch.linewidth": 0.45,  # Thin crosses keep the bar heights readable.
    "savefig.dpi": 240, "savefig.bbox": "standard",  # A fixed page size keeps each figure exactly WIDTH_IN wide.
    "pdf.fonttype": 42, "ps.fonttype": 42,  # Embed TrueType fonts.
})


def _style(ax, title: str | None = None) -> None:
    ax.tick_params(direction="out")
    ax.grid(True, axis="y", color="#D9D9D9")
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, pad=6)


def _save(fig, path) -> None:
    """Save the figure without a creation timestamp so the PDF is reproducible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"CreationDate": None})
    plt.close(fig)
    print(f"saved {path.relative_to(file_paths.ROOT)}")


def _percent_axis(ax) -> None:
    """Put a Recall@1 axis on the shared 0 to 100% scale."""
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0, symbol=""))


def _clean_then_severities_axis(ax) -> None:
    """Label an x axis with clean and each severity, leaving room between neighboring panels."""
    ax.set_xticks(CLEAN_THEN_SEVERITIES, ["Clean"] + [str(s) for s in settings.SEVERITIES])
    ax.set_xlim(-0.6, 5.7)


def _direction_handles() -> list:
    """Return the legend entries for corrupted bars and clean reference lines."""
    bars = [plt.Rectangle((0, 0), 1, 1, facecolor=DIRECTION_COLORS[d], hatch=DIRECTION_HATCH[d], **BAR_EDGE) for d in ("i2t", "t2i")]
    lines = [plt.Line2D([0], [0], color=CLEAN_COLORS[d], linestyle=":", linewidth=1.4) for d in ("i2t", "t2i")]
    return bars + lines


def _label_rows(fig, axes, row_labels: list[str], x: float = 0.004) -> None:
    """Name each row of panels at the left edge of the figure."""
    for row, label in enumerate(row_labels):
        box = axes[row, 0].get_position()
        fig.text(x, (box.y0 + box.y1) / 2, label, rotation=90, va="center", ha="left", fontsize=8, fontweight="bold")


def _with_dataset_column(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join one table per dataset, with the dataset as the first column."""
    joined = pd.concat([table.assign(dataset=dataset) for dataset, table in tables.items()], ignore_index=True)
    return joined[["dataset"] + [column for column in joined.columns if column != "dataset"]]


# Figures and tables share these summaries.
@lru_cache(maxsize=4)
def quintile_rows(image_ruler: str, dataset: str = "coco") -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Return each model's image and caption hits with their damage quintiles at the headline gallery.

    Rank the images by SSIM or PSNR, and the captions by BLEU.
    """
    ruler = IMAGE_RULERS[image_ruler]
    return (quintiles_by_model(recall_with_damage("i2t", ruler, dataset), ruler),
            quintiles_by_model(recall_with_damage("t2i", "damage_bleu", dataset), "damage_bleu"))


@lru_cache(maxsize=4)
def quintile_summary(image_ruler: str, dataset: str = "coco") -> pd.DataFrame:
    """Summarize Recall@1 and its 95% interval for each model, direction and damage quintile."""
    images, captions = quintile_rows(image_ruler, dataset)
    rows = []
    for m in MODELS:
        for q in QUINTILES:
            i_cell = images[m][images[m]["quintile"] == q]
            t_cell = captions[m][captions[m]["quintile"] == q]
            (i_mean, i_lo, i_hi), (t_mean, t_lo, t_hi) = bootstrap_ci(i_cell), bootstrap_ci(t_cell)
            rows.append({"model": m, "quintile": q, "i2t": i_mean, "i2t_lo": i_lo, "i2t_hi": i_hi,
                         "t2i": t_mean, "t2i_lo": t_lo, "t2i_hi": t_hi, "i2t_n": len(i_cell), "t2i_n": len(t_cell)})
    return pd.DataFrame(rows)


@lru_cache(maxsize=2)
def clean_recall(dataset: str = "coco") -> pd.DataFrame:
    """Summarize the clean Recall@1 and its 95% interval for each model and direction."""
    clean = clean_baseline_per_pair(dataset)
    rows = []
    for m in MODELS:
        for direction in ("i2t", "t2i"):
            cell = clean[(clean["model"] == m) & (clean["metric"] == f"per_pair_clean_recall_at_1_{direction}")]
            mean, lo, hi = bootstrap_ci(cell)
            rows.append({"model": m, "direction": direction, "mean": mean, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def _pooling_drop(means: dict[str, float], pooling: str, start: str = "sev1") -> float:
    return means[f"{pooling}_{start}"] - means[f"{pooling}_sev5"]


def _pooling_relative_drop(means: dict[str, float], pooling: str, start: str = "sev1") -> float:
    """Divide a pooling method's drop by its own starting margin."""
    return 100 * _pooling_drop(means, pooling, start) / means[f"{pooling}_{start}"]


def _drop_reduction_pct(means: dict[str, float], start: str = "sev1") -> float:
    """Express the difference in absolute drops as a share of the standard-pooling drop."""
    return 100 * (1 - _pooling_drop(means, "mean", start) / _pooling_drop(means, "standard", start))


POOLING_STATISTICS = {
    "standard_drop": lambda g: _pooling_drop(g, "standard"), "mean_drop": lambda g: _pooling_drop(g, "mean"),
    "drop_reduction_pct": _drop_reduction_pct,
    "standard_rel_drop_pct": lambda g: _pooling_relative_drop(g, "standard"),
    "mean_rel_drop_pct": lambda g: _pooling_relative_drop(g, "mean"),
    "rel_drop_mean_minus_standard": lambda g: _pooling_relative_drop(g, "mean") - _pooling_relative_drop(g, "standard"),
    "standard_drop_from_clean": lambda g: _pooling_drop(g, "standard", "clean"),
    "mean_drop_from_clean": lambda g: _pooling_drop(g, "mean", "clean"),
    "drop_reduction_from_clean_pct": lambda g: _drop_reduction_pct(g, "clean"),
    "standard_rel_drop_from_clean_pct": lambda g: _pooling_relative_drop(g, "standard", "clean"),
    "mean_rel_drop_from_clean_pct": lambda g: _pooling_relative_drop(g, "mean", "clean"),
}


@lru_cache(maxsize=2)
def pooling_summary(dataset: str = "coco") -> pd.DataFrame:
    """Summarize each model's clean, severity-1 and severity-5 margins and drops under both text poolings.

    Severity 1 is a corrupted condition, so the drops are reported from severity 1 and from clean.
    Each margin has its own 95% interval. The drops and their comparisons share one joint bootstrap,
    which resamples seeds and then whole pairs, because every caption is scored under both poolings.
    """
    ablation = text_pooling_ablation(dataset)
    margins = _pooling_margins_by_level(dataset).set_index(["model", "text_pooling", "level"])
    rows = []
    for m in MODELS:
        row, groups = {"model": m}, {}
        for pooling in ("standard", "mean"):
            sub = ablation[(ablation["model"] == m) & (ablation["text_pooling"] == pooling)]
            for name, level in (("clean", 0), ("sev1", 1), ("sev5", 5)):
                groups[f"{pooling}_{name}"] = sub[sub["text_severity"] == level]
                margin = margins.loc[(m, pooling, level)]
                row.update({f"{pooling}_{name}{suffix}": margin[f"margin{suffix}"] for suffix in ("", "_lo", "_hi")})
        for name, (point, lo, hi) in bootstrap_group_statistics(groups, POOLING_STATISTICS).items():
            row.update({name: point, f"{name}_lo": lo, f"{name}_hi": hi})
        rows.append(row)
    return pd.DataFrame(rows)


# Compare the retrieval directions within each damage quintile.
def _draw_quintiles(summaries: dict[str, pd.DataFrame], clean: dict[str, pd.DataFrame] | None, path, xlabel: str) -> None:
    """Draw one row per dataset and one column per model on one Recall@1 scale.

    Panel placement tells the datasets apart, so both keep the direction colors and the crosshatching.
    """
    height = 0.86 + 1.25 * len(summaries)
    fig, axes = plt.subplots(len(summaries), len(MODELS), figsize=(WIDTH_IN, height), sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.112, right=0.995, top=1 - 0.42 / height, bottom=0.44 / height, hspace=0.08, wspace=0.08)
    x = np.arange(len(QUINTILES))
    for row, (dataset, summary) in enumerate(summaries.items()):
        for col, m in enumerate(MODELS):
            ax, cells = axes[row, col], summary[summary["model"] == m].sort_values("quintile")
            for shift, direction in ((-0.22, "i2t"), (0.22, "t2i")):
                err = np.array([cells[direction] - cells[f"{direction}_lo"], cells[f"{direction}_hi"] - cells[direction]])
                ax.bar(x + shift, cells[direction].to_numpy(), 0.44, yerr=err, color=DIRECTION_COLORS[direction],
                       hatch=DIRECTION_HATCH[direction], error_kw=ERROR_BARS, **BAR_EDGE)
                if clean is not None:
                    level = clean[dataset][(clean[dataset]["model"] == m) & (clean[dataset]["direction"] == direction)]["mean"].iloc[0]
                    ax.axhline(level, color=CLEAN_COLORS[direction], linewidth=1.3, linestyle=":", zorder=3)
            ax.set_xticks(x, QUINTILES)
            _percent_axis(ax)
            _style(ax, MODEL_LABELS[m] if row == 0 else None)
        axes[row, 0].set_ylabel("Recall@1 (%)", labelpad=2)
    _label_rows(fig, axes, [DATASET_TITLES[dataset] for dataset in summaries])
    handles, labels = _direction_handles(), list(DIRECTION_LEGEND.values()) + ["Clean i2t", "Clean t2i"]
    keep = len(labels) if clean is not None else 2
    fig.legend(handles[:keep], labels[:keep], loc="upper center", bbox_to_anchor=(0.55, 1.0), ncol=keep, frameon=False,
               handlelength=2.0, columnspacing=1.4)
    fig.text(0.55, 0.012, xlabel, ha="center", fontsize=7.5, linespacing=1.25)
    _save(fig, path)


def build_recall_at_1_by_damage_quintile() -> None:
    datasets = datasets_with_headline_rows()
    summaries = {d: quintile_summary("ssim", d) for d in datasets}
    clean = {d: clean_recall(d) for d in datasets}
    _with_dataset_column(summaries).to_csv(file_paths.FIG_RECALL_BY_DAMAGE_QUINTILE_DATA, index=False)
    _with_dataset_column(clean).to_csv(file_paths.FIG_RECALL_BY_DAMAGE_QUINTILE_CLEAN, index=False)
    _draw_quintiles(summaries, clean, file_paths.FIG_RECALL_BY_DAMAGE_QUINTILE,
                    "Damage quintile within each modality and dataset (Q1 least, Q5 most)\n"
                    f"Gallery of {settings.HEADLINE_GALLERY_SIZE:,}. Images ranked by 1 - SSIM, captions by 1 - BLEU")


def build_recall_at_1_psnr_binned_crosscheck() -> None:
    """Plot the quintile check with PSNR-ranked images and BLEU-ranked captions."""
    summaries = {d: quintile_summary("psnr", d) for d in datasets_with_headline_rows()}
    _with_dataset_column(summaries).to_csv(file_paths.FIG_PSNR_CROSSCHECK_DATA, index=False)
    _draw_quintiles(summaries, None, file_paths.FIG_PSNR_CROSSCHECK,
                    "Damage quintile within each modality and dataset (Q1 least, Q5 most)\n"
                    f"Gallery of {settings.HEADLINE_GALLERY_SIZE:,}. Images ranked by PSNR, captions by 1 - BLEU")


# Plot joint Gaussian-noise and word-deletion damage.
def _joint_retention_rows(dataset: str) -> pd.DataFrame:
    """Load the retention margins of the grid that damages both the image and the caption."""
    rows = fidelity_retention_rows(dataset)
    joint = rows[(rows["spoke"] == "match_retention") & (rows["match_retention_direction"] == "joint")
                 & (rows["metric"] == "retention_margin")].copy()
    for c in ("image_severity", "text_severity"):
        joint[c] = pd.to_numeric(joint[c]).astype(int)
    return joint


def build_retention_joint_noise_mask_heatmaps() -> None:
    """Plot one joint margin per grid cell, with one color scale across models and datasets."""
    grids = {d: _joint_retention_rows(d).groupby(["model", "image_severity", "text_severity"])["value"].mean().reset_index()
             for d in datasets_with_300_pair_run()}
    table = _with_dataset_column(grids)
    table.to_csv(file_paths.FIG_JOINT_HEATMAPS_DATA, index=False)

    height = 0.48 + 1.0 * len(grids)
    fig, axes = plt.subplots(len(grids), len(MODELS), figsize=(WIDTH_IN, height), sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.105, right=0.875, top=1 - 0.2 / height, bottom=0.34 / height, hspace=0.05, wspace=0.12)
    low, high = float(table["value"].min()), float(table["value"].max())
    for row, (dataset, grid) in enumerate(grids.items()):
        for col, m in enumerate(MODELS):
            ax = axes[row, col]
            values = grid[grid["model"] == m].pivot(index="image_severity", columns="text_severity", values="value")
            image = ax.imshow(values.to_numpy(), origin="lower", aspect="equal", cmap="viridis_r", vmin=low, vmax=high,
                              extent=[0.5, 5.5, 0.5, 5.5], interpolation="nearest")
            ax.set_xticks(settings.SEVERITIES)
            ax.set_yticks(settings.SEVERITIES)
            if row == 0:
                ax.set_title(MODEL_LABELS[m], pad=3)
    fig.canvas.draw()
    _label_rows(fig, axes, [DATASET_TITLES[dataset] for dataset in grids])
    top, bottom = axes[0, 0].get_position().y1, axes[-1, 0].get_position().y0
    bar = fig.colorbar(image, cax=fig.add_axes([0.893, bottom, 0.018, top - bottom]))
    bar.ax.yaxis.set_major_locator(MaxNLocator(5))
    bar.set_label("Retention margin", labelpad=3)
    fig.text(0.49, 0.02, "Word-deletion severity", ha="center", fontsize=8)
    fig.text(0.04, (top + bottom) / 2, "Gaussian-noise severity", rotation=90, va="center", fontsize=8)
    _save(fig, file_paths.FIG_JOINT_HEATMAPS)


# Compare fidelity at the selected encoder depths.
def _fidelity_rows(dataset: str) -> pd.DataFrame:
    """Load the image and text fidelity rows, with each side's severity in one numeric column."""
    rows = fidelity_retention_rows(dataset)
    rows = rows[rows["spoke"].isin(["image_fidelity", "text_fidelity"])].copy()
    image = rows["spoke"] == "image_fidelity"
    rows["severity"] = pd.to_numeric(rows["image_severity"].where(image, rows["text_severity"])).astype(int)
    rows[["seed", "pair_id"]] = rows[["seed", "pair_id"]].apply(pd.to_numeric)
    return rows


def _fidelity_by_depth(dataset: str) -> dict[str, pd.DataFrame]:
    """Average the fidelity by model, severity and depth for each side of a dataset.

    The block column counts encoder blocks from 1, while depth_layer_index counts from 0.
    """
    rows = _fidelity_rows(dataset)
    tables = {}
    for side in ("image", "text"):
        sub = rows[rows["spoke"] == f"{side}_fidelity"]
        means = sub.groupby(["model", "severity", "depth"])["value"].mean().reset_index().rename(columns={"severity": f"{side}_severity"})
        blocks = (pd.to_numeric(sub["depth_layer_index"]) + 1).astype(int).groupby([sub["model"], sub["depth"]]).first()
        means["block"] = [blocks[key] for key in zip(means["model"], means["depth"])]
        tables[side] = means
    return tables


def build_fidelity_by_layer_depth() -> None:
    """Plot fidelity at each model's three sampled blocks, averaging each severity over corruptions, pairs and seeds."""
    by_dataset = {d: _fidelity_by_depth(d) for d in datasets_with_300_pair_run()}
    pd.concat([t[s].rename(columns={f"{s}_severity": "severity"}).assign(dataset=d, side=s)
               for d, t in by_dataset.items() for s in ("image", "text")])[
        ["dataset", "side", "model", "severity", "depth", "value", "block"]].to_csv(file_paths.FIG_FIDELITY_BY_DEPTH_DATA, index=False)

    # Extend the shared scale when fidelity falls below 0.4.
    lowest = min(t["value"].min() for tables in by_dataset.values() for t in tables.values())
    bottom = min(0.4, np.floor(10 * lowest) / 10)
    columns = [(dataset, side) for dataset in by_dataset for side in ("image", "text")]
    fig, axes = plt.subplots(len(MODELS), len(columns), figsize=(WIDTH_IN, 7.3), sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.065, top=0.858, hspace=0.42, wspace=0.11)
    x = np.arange(len(DEPTHS))
    for row, m in enumerate(MODELS):
        for col, (dataset, side) in enumerate(columns):
            table, ax = by_dataset[dataset][side], axes[row, col]
            for sev in settings.SEVERITIES:
                cells = table[(table["model"] == m) & (table[f"{side}_severity"] == sev)]
                ax.plot(x, [cells[cells["depth"] == d]["value"].mean() for d in DEPTHS], marker="o", markersize=2.8,
                        linewidth=1.0, color=SEVERITY_COLORS[sev - 1])
            ax.set_xticks(x, [str(table[(table["model"] == m) & (table["depth"] == d)]["block"].iloc[0]) for d in DEPTHS])
            ax.set_ylim(bottom, 1.0)
            ax.set_yticks(np.arange(1.0, bottom - 1e-9, -0.2))
            ax.set_xlim(-0.25, len(DEPTHS) - 0.75)
            _style(ax)
        axes[row, 0].set_title(MODEL_LABELS[m], loc="left", pad=3)
    for i, dataset in enumerate(by_dataset):
        left, right = axes[0, 2 * i].get_position(), axes[0, 2 * i + 1].get_position()
        fig.text((left.x0 + right.x1) / 2, 0.925, DATASET_TITLES[dataset], ha="center", fontsize=8, fontweight="bold")
    for col, (_, side) in enumerate(columns):
        box = axes[0, col].get_position()
        heading = "Image fidelity\n(mean patch cosine)" if side == "image" else "Text fidelity\n(pooled-vector cosine)"
        fig.text((box.x0 + box.x1) / 2, 0.92, heading, ha="center", va="top", fontsize=7, linespacing=1.15)
    handles = [plt.Line2D([], [], color=SEVERITY_COLORS[sev - 1], marker="o", markersize=3, label=str(sev)) for sev in settings.SEVERITIES]
    fig.legend(handles=handles, title="Corruption severity", loc="upper center", bbox_to_anchor=(0.54, 1.0), ncol=len(handles),
               frameon=False, handlelength=1.5, columnspacing=1.2, title_fontsize=7.5)
    fig.text(0.54, 0.012, "Sampled encoder block, early, middle and last, counted from 1", ha="center", fontsize=8)
    fig.text(0.012, 0.47, "Cosine similarity to clean hidden states", rotation=90, va="center", fontsize=8)
    _save(fig, file_paths.FIG_FIDELITY_BY_DEPTH)


# Compare the two text poolings at clean and at each severity.
@lru_cache(maxsize=2)
def _pooling_margins_by_level(dataset: str) -> pd.DataFrame:
    """Summarize the margins and their 95% intervals once for the pooling figures and tables."""
    ablation = text_pooling_ablation(dataset)
    rows = []
    for m in MODELS:
        for pooling in ("standard", "mean"):
            sub = ablation[(ablation["model"] == m) & (ablation["text_pooling"] == pooling)]
            for level in (0,) + settings.SEVERITIES:
                mean, lo, hi = bootstrap_ci(sub[sub["text_severity"] == level])
                rows.append({"model": m, "text_pooling": pooling, "level": level, "margin": mean, "margin_lo": lo, "margin_hi": hi})
    return pd.DataFrame(rows)


def _draw_text_pooling(dataset: str, levels: pd.DataFrame, summary: pd.DataFrame, path, top: float) -> None:
    """Draw one dataset's margins by level for each model, then the absolute and relative drops from severity 1.

    Clean stands apart from the severity line because severity 1 is already a corrupted condition.
    """
    height = 3.9
    fig = plt.figure(figsize=(WIDTH_IN, height))
    upper = fig.add_gridspec(1, len(MODELS), left=0.085, right=0.995, top=1 - 0.62 / height, bottom=1 - 1.95 / height, wspace=0.11)
    lower = fig.add_gridspec(1, 2, left=0.085, right=0.995, top=1 - 2.57 / height, bottom=0.34 / height, wspace=0.17)
    x = CLEAN_THEN_SEVERITIES
    bars = {**ERROR_BARS, "capsize": 1.0}
    for col, m in enumerate(MODELS):
        ax = fig.add_subplot(upper[0, col])
        for pooling, style in (("standard", "-"), ("mean", "--")):
            cells = levels[(levels["model"] == m) & (levels["text_pooling"] == pooling)].sort_values("level")
            err = np.array([cells["margin"] - cells["margin_lo"], cells["margin_hi"] - cells["margin"]])
            ax.errorbar(x[0], cells["margin"].iloc[0], yerr=err[:, :1], marker="o", markersize=2.6, color=POOLING_COLORS[pooling],
                        linestyle="none", **bars)
            ax.errorbar(x[1:], cells["margin"].iloc[1:], yerr=err[:, 1:], marker="o", markersize=2.4, color=POOLING_COLORS[pooling],
                        linestyle=style, linewidth=1.1, **bars)
        _clean_then_severities_axis(ax)
        ax.set_ylim(0, top)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        if col:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("Retention margin", labelpad=2)
        _style(ax, MODEL_LABELS[m])
    model_positions = np.arange(len(MODELS))
    names = [MODEL_LABELS[m].replace(" L/14", "\nL/14").replace(" B/32", "\nB/32") for m in MODELS]
    cells = summary.set_index("model").loc[list(MODELS)]
    panels = ((0, "Margin drop", "{}_drop", "(b) Absolute margin drop, severity 1 to 5"),
              (1, "Relative drop (%)", "{}_rel_drop_pct", "(c) Relative drop, severity 1 to 5"))
    for col, ylabel, column, title in panels:
        ax = fig.add_subplot(lower[0, col])
        for shift, pooling in ((-0.18, "standard"), (0.18, "mean")):
            name = column.format(pooling)
            err = np.array([cells[name] - cells[f"{name}_lo"], cells[f"{name}_hi"] - cells[name]])
            ax.bar(model_positions + shift, cells[name].to_numpy(), 0.36, yerr=err, color=POOLING_COLORS[pooling], error_kw=ERROR_BARS, **BAR_EDGE)
        ax.set_xticks(model_positions, names)
        ax.set_xlim(-0.55, len(MODELS) - 0.45)
        ax.set_ylabel(ylabel, labelpad=2)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        _style(ax)
        ax.set_title(title, loc="left", pad=3)
    fig.text(0.085, 1 - 0.41 / height, f"(a) {DATASET_TITLES[dataset]} retention margin on clean captions and at each severity", fontsize=8, va="bottom")
    fig.text(0.54, 1 - 2.2 / height, "Caption severity, averaged over word deletion, shuffle and replacement", ha="center", va="center", fontsize=7.5)
    handles = [plt.Line2D([0], [0], color=POOLING_COLORS["standard"], marker="o", markersize=3, linestyle="-"),
               plt.Line2D([0], [0], color=POOLING_COLORS["mean"], marker="o", markersize=3, linestyle="--")]
    fig.legend(handles, ["Standard pooling", "Mean pooling"], loc="upper center", bbox_to_anchor=(0.54, 1.005), ncol=2, frameon=False,
               handlelength=2.4, columnspacing=1.6)
    _save(fig, path)


def build_retention_by_text_pooling() -> None:
    """Plot each dataset's margins and drops under both text poolings, for the main text and the appendix.

    Both figures share one margin scale, so they compare directly.
    """
    datasets = datasets_with_300_pair_run()
    levels = {d: _pooling_margins_by_level(d) for d in datasets}
    _with_dataset_column(levels).to_csv(file_paths.FIG_RETENTION_BY_TEXT_POOLING_DATA, index=False)
    top = max(table["margin_hi"].max() for table in levels.values()) * 1.08
    for dataset in datasets:
        _draw_text_pooling(dataset, levels[dataset], pooling_summary(dataset), file_paths.fig_retention_by_text_pooling(dataset), top)


# Compare individual corruptions at severity 5.
def build_recall_at_1_by_corruption_sev5() -> None:
    """Plot each corruption at severity 5 beside the clean Recall@1 of its own direction, on COCO.

    Typo is included as a separate bar. It stays out of the damage quintiles, and this bar uses the
    headline gallery while the typo curves use the full gallery.
    """
    hits, clean = per_pair_retrieval("coco"), clean_recall("coco")
    sides = (("image", "i2t", "vision_corruption", "image_severity", settings.IMAGE_CORRUPTIONS),
             ("text", "t2i", "text_corruption", "text_severity", settings.TEXT_CORRUPTIONS))
    clean_rows = clean_baseline_per_pair("coco")
    scope = {"dataset": "coco", "gallery_size": settings.HEADLINE_GALLERY_SIZE}
    rows = []
    for m in MODELS:
        for side, direction, kind_col, severity_col, kinds in sides:
            own = clean[(clean["model"] == m) & (clean["direction"] == direction)].iloc[0]
            n_clean = int(((clean_rows["model"] == m) & (clean_rows["metric"] == f"per_pair_clean_recall_at_1_{direction}")).sum())
            rows.append({**scope, "model": m, "side": side, "direction": direction, "corruption_type": "none", "kind": "clean",
                         "mean": own["mean"], "ci_lo": own["lo"], "ci_hi": own["hi"], "n": n_clean})
            for kind in kinds:
                cell = hits[(hits["model"] == m) & (hits["metric"] == f"per_pair_recall_at_1_{direction}")
                            & (hits[kind_col] == kind) & (hits[severity_col] == 5)]
                mean, lo, hi = bootstrap_ci(cell)
                rows.append({**scope, "model": m, "side": side, "direction": direction, "corruption_type": kind, "kind": "corrupted",
                             "mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(cell)})
    df = pd.DataFrame(rows)
    df.to_csv(file_paths.FIG_RECALL_BY_CORRUPTION_SEV5_DATA, index=False)

    # A small gap separates typo from the binned text corruptions.
    positions = dict(zip(settings.IMAGE_CORRUPTIONS + settings.TEXT_CORRUPTIONS, (0, 1, 2, 3.6, 4.6, 5.6, 6.9)))
    spans = {"i2t": (-0.55, 2.55), "t2i": (3.05, 7.45)}
    height = 2.45
    fig, axes = plt.subplots(1, len(MODELS), figsize=(WIDTH_IN, height), sharey=True)
    fig.subplots_adjust(left=0.098, right=0.995, top=1 - 0.40 / height, bottom=0.68 / height, wspace=0.08)
    for ax, m in zip(axes, MODELS):
        cells = df[df["model"] == m]
        for r in cells[cells["kind"] == "corrupted"].itertuples():
            ax.bar(positions[r.corruption_type], r.mean, 0.76, color=DIRECTION_COLORS[r.direction], hatch=DIRECTION_HATCH[r.direction],
                   yerr=([r.mean - r.ci_lo], [r.ci_hi - r.mean]), error_kw=ERROR_BARS, **BAR_EDGE)
        for r in cells[cells["kind"] == "clean"].itertuples():
            ax.plot(spans[r.direction], [r.mean, r.mean], color=CLEAN_COLORS[r.direction], linewidth=1.3, linestyle=":", zorder=3)
        ax.set_xticks(list(positions.values()), [SHORT_CORRUPTION_LABELS[kind] for kind in positions], rotation=90)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.75, 7.65)
        _percent_axis(ax)
        _style(ax, MODEL_LABELS[m])
    axes[0].set_ylabel(f"Recall@1 at severity 5 (%)\nCOCO, gallery of {settings.HEADLINE_GALLERY_SIZE:,}", labelpad=2, linespacing=1.35)
    labels = list(DIRECTION_LEGEND.values()) + ["Clean i2t", "Clean t2i"]
    fig.legend(_direction_handles(), labels, loc="upper center", bbox_to_anchor=(0.54, 1.0), ncol=4, frameon=False, handlelength=2.0, columnspacing=1.4)
    _save(fig, file_paths.FIG_RECALL_BY_CORRUPTION_SEV5)


# Plot typo and object-cutout curves separately from the headline bins.
def draw_typo_recall_by_level(table: pd.DataFrame, path) -> None:
    """Plot typo Recall@1 for each model at the full gallery, with one panel per dataset.

    Column "0" holds the clean recall, and columns "1" through "5" hold the corrupted recall.
    """
    datasets = list(dict.fromkeys(table["dataset"]))
    levels = [0] + list(settings.SEVERITIES)
    height = 2.85
    fig, axes = plt.subplots(1, len(datasets), figsize=(WIDTH_IN, height), sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.105, right=0.985, top=1 - 0.39 / height, bottom=0.62 / height, wspace=0.12)
    for ax, dataset in zip(axes[0], datasets):
        for m in MODELS:
            row = table[(table["dataset"] == dataset) & (table["model"] == m)]
            if not row.empty:
                ax.plot(levels, [row[str(level)].iloc[0] for level in levels], marker=MODEL_MARKERS[m], color=MODEL_COLORS[m],
                        markerfacecolor=MODEL_FILLS[m], markeredgecolor=MODEL_COLORS[m], linewidth=1.3, markersize=4.5)
        ax.set_xticks(levels, ["Clean"] + [str(s) for s in settings.SEVERITIES])
        ax.set_xlabel("Typo severity")
        _percent_axis(ax)
        _style(ax, DATASET_TITLES[dataset])
    axes[0][0].set_ylabel("Text-to-image Recall@1 (%)")
    handles = [plt.Line2D([], [], color=MODEL_COLORS[m], marker=MODEL_MARKERS[m], markerfacecolor=MODEL_FILLS[m],
                          markeredgecolor=MODEL_COLORS[m], label=MODEL_LABELS[m]) for m in MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=len(MODELS), frameon=False, bbox_to_anchor=(0.5, 0.0), columnspacing=1.2)
    fig.suptitle(f"Typo robustness at gallery size {max(settings.GALLERY_SIZES):,}", y=0.985)
    _save(fig, path)


def draw_object_and_rectangle_cutout(table: pd.DataFrame, path, xlabel: str) -> None:
    """Plot object and rectangle cutout with one panel per model, because ten lines on one axis hide each other.

    Column "0" holds the clean recall, which stands apart from the severity line as in the text-pooling figure.
    """
    styles = (("object_cutout", "-", "Object cutout"), ("cutout", "--", "Rectangle cutout"))
    height = 2.35
    fig, axes = plt.subplots(1, len(MODELS), figsize=(WIDTH_IN, height), sharey=True)
    fig.subplots_adjust(left=0.098, right=0.995, top=1 - 0.40 / height, bottom=0.46 / height, wspace=0.11)
    for ax, m in zip(axes, MODELS):
        marks = {"marker": MODEL_MARKERS[m], "color": MODEL_COLORS[m], "markerfacecolor": MODEL_FILLS[m],
                 "markeredgecolor": MODEL_COLORS[m], "markersize": 3.6}
        for kind, style, _ in styles:
            row = table[(table["model"] == m) & (table["corruption"] == kind)].iloc[0]
            values = [row[str(level)] for level in (0,) + settings.SEVERITIES]
            ax.plot(CLEAN_THEN_SEVERITIES[:1], values[:1], linestyle="none", **marks)
            ax.plot(CLEAN_THEN_SEVERITIES[1:], values[1:], linestyle=style, linewidth=1.2, **marks)
        _clean_then_severities_axis(ax)
        _percent_axis(ax)
        _style(ax, MODEL_LABELS[m])
    axes[0].set_ylabel(f"Image-to-text Recall@1 (%)\nCOCO, gallery of {max(settings.GALLERY_SIZES):,}", labelpad=2, linespacing=1.35)
    handles = [plt.Line2D([], [], color="#555555", linestyle=style, label=label) for _, style, label in styles]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 1.0), ncol=len(styles), frameon=False, handlelength=2.4, columnspacing=1.6)
    fig.text(0.54, 0.03, xlabel, ha="center", fontsize=8)
    _save(fig, path)
