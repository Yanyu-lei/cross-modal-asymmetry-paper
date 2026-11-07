"""Typeset the appendix tables from the numerical files that the other paper steps write."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src import file_paths, settings
from src.load_results import note_skipped
from src.paper_figures import DATASET_TITLES, MODEL_LABELS, MODELS, SHORT_CORRUPTION_LABELS
from src.paper_numbers import MIN_QUERIES_FOR_CORRUPTION_RATE
from src.paper_tables import CORRUPTION_LABELS, dataset_heading, latex_table

MODEL_COLUMNS = " & ".join(MODEL_LABELS[m] for m in MODELS)
RULER_LABELS = {"ssim": "SSIM", "psnr": "PSNR", "bleu": "BLEU", "edit_distance": "Word edit distance"}
GALLERY = f"{settings.HEADLINE_GALLERY_SIZE:,}"
FULL_GALLERY = f"{max(settings.GALLERY_SIZES):,}"
COLUMN_GROUP_GAP = r"@{\hspace{12pt}}"  # A wider space between groups of columns in the widest tables.


def _cell(value, form: str = "{:.3f}", scale: float = 1.0) -> str:
    """Format a number, or print a dash where a measurement is missing."""
    return "---" if pd.isna(value) else form.format(scale * value)


def _rows_by_model(table: pd.DataFrame) -> pd.DataFrame:
    """Index a table by model in the paper's order."""
    return table.set_index("model").loc[list(MODELS)]


# Report PE-Core's caption truncation and the Q5 ratio by caption length.
def pecore_truncation_table() -> str:
    """Print the truncation counts and the BLEU-ranked PE-Core Q5 ratio with and without truncated captions."""
    counts = {r["dataset"]: r for r in pd.read_csv(file_paths.TAB_PECORE_TRUNCATION_DATA).to_dict("records")}
    ratios = pd.read_csv(file_paths.TAB_PECORE_TRUNCATION_ROBUSTNESS)
    ratios = ratios[ratios["text_ruler"] == "bleu"].set_index(["dataset", "subset"])["ratio"]
    coco, flickr = counts["coco"], counts["flickr30k"]
    return latex_table(
        file_paths.TAB_PECORE_TRUNCATION,
        f"PE-Core caption truncation and its associated Q5 i2t/t2i Recall@1 ratios are shown for both datasets at "
        f"a gallery of {FULL_GALLERY}.",
        "tab:t7_pecore", r"\small", "lcc",
        [r"Quantity & COCO & Flickr30K \\"],
        [f"Distinct captions & {coco['distinct_captions']:,} & {flickr['distinct_captions']:,} \\\\",
         f"Maximum content-token count & {coco['longest_caption_tokens']} & {flickr['longest_caption_tokens']} \\\\",
         f"Captions exceeding the 30-content-token limit & {coco['over_30_tokens']} & {flickr['over_30_tokens']} \\\\",
         f"Share cut short & {coco['percent_over']:.2f}\\% & {flickr['percent_over']:.2f}\\% \\\\",
         f"PE-Core Q5 ratio, all pairs & {ratios[('coco', 'all pairs')]:.2f} & {ratios[('flickr30k', 'all pairs')]:.2f} \\\\",
         f"PE-Core Q5 ratio, without truncated captions & {ratios[('coco', 'without truncated captions')]:.2f} & "
         f"{ratios[('flickr30k', 'without truncated captions')]:.2f} \\\\"])


def caption_length_table() -> str:
    """Print the BLEU-ranked Q5 ratio in each caption word-count bin, for each model and over all models."""
    per_model = pd.read_csv(file_paths.TAB_CAPTION_LENGTH_PER_MODEL)
    pooled = pd.read_csv(file_paths.TAB_CAPTION_LENGTH)
    per_model, pooled = per_model[per_model["text_ruler"] == "bleu"], pooled[pooled["text_ruler"] == "bleu"]
    datasets = list(dict.fromkeys(per_model["dataset"]))

    def bins(dataset: str) -> list[tuple[int, int]]:
        cells = per_model[per_model["dataset"] == dataset]
        return sorted(set(zip(cells["caption_words_min"], cells["caption_words_max"])))

    def label(low: int, high: int) -> str:
        return str(low) if low == high else f"{low}--{high}"

    widths = [len(bins(d)) for d in datasets]
    titles = " & ".join(rf"\multicolumn{{{w}}}{{c}}{{{DATASET_TITLES[d]}, caption words}}" for d, w in zip(datasets, widths))
    edges = np.cumsum([1] + widths)
    rules = "".join(rf"\cmidrule(lr){{{start + 1}-{end}}}" for start, end in zip(edges[:-1], edges[1:]))
    words = " & ".join(label(*b) for d in datasets for b in bins(d))
    body = []
    for m in MODELS:
        cells = []
        for d in datasets:
            sub = per_model[(per_model["dataset"] == d) & (per_model["model"] == m)].set_index(["caption_words_min", "caption_words_max"])["ratio"]
            cells += [_cell(sub.get(b), "{:.2f}") for b in bins(d)]
        body.append(f"{MODEL_LABELS[m]} & " + " & ".join(cells) + r" \\")
    cells = []
    for d in datasets:
        sub = pooled[pooled["dataset"] == d].set_index(["caption_words_min", "caption_words_max"])["ratio"]
        cells += [_cell(sub.get(b), "{:.2f}") for b in bins(d)]
    body += [r"\midrule", "All models & " + " & ".join(cells) + r" \\"]
    return latex_table(
        file_paths.TAB_CAPTION_LENGTH_TEX,
        f"Q5 i2t/t2i Recall@1 ratios are shown by caption word count at a gallery of {FULL_GALLERY}, using SSIM and BLEU "
        "with observations pooled across three seeds.",
        "tab:caption_length", r"\footnotesize", "l" + "r" * sum(widths),
        [rf" & {titles} \\", rules, rf"Model & {words} \\"], body, column_space="4pt")


# Compare the Q5 ratio across rulers, galleries and datasets.
def q5_ratio_by_protocol_table() -> str:
    table = pd.read_csv(file_paths.TAB_Q5_RATIO_BY_PROTOCOL)
    body = []
    for position, dataset in enumerate(dict.fromkeys(table["dataset"])):
        if position:
            body.append(r"\midrule")
        sub = table[table["dataset"] == dataset]
        for gallery in sorted(sub["gallery_size"].unique()):
            for image_ruler in ("ssim", "psnr"):
                for text_ruler in ("bleu", "edit_distance"):
                    for ties in ("lower bin", "Q5"):
                        cell = sub[(sub["gallery_size"] == gallery) & (sub["image_ruler"] == image_ruler)
                                   & (sub["text_ruler"] == text_ruler) & (sub["caption_ties_at_q5_edge"] == ties)]
                        if cell.empty:
                            continue
                        ratios = _rows_by_model(cell)["ratio"]
                        ruler = RULER_LABELS[text_ruler] + (r", ties in Q5" if ties == "Q5" else "")
                        body.append(f"{DATASET_TITLES[dataset]} & {gallery:,} & {RULER_LABELS[image_ruler]} & {ruler} & "
                                    + " & ".join(f"{ratios[m]:.2f}" for m in MODELS) + r" \\")
    return latex_table(
        file_paths.TAB_Q5_RATIO_BY_PROTOCOL_TEX,
        "Q5 i2t/t2i Recall@1 ratios are compared across damage rulers and gallery sizes on both datasets, "
        "with observations pooled across three seeds.",
        "tab:q5_ratio_by_protocol", r"\footnotesize", "lrll" + "r" * len(MODELS),
        [rf"Dataset & Gallery & Images & Captions & {MODEL_COLUMNS} \\"], body, column_space="3pt")


# Print the full Recall@1/5/10 values with each seed.
def _recall_k_rows(cells: pd.DataFrame, direction: str, kind_col: str, level_col: str, kinds: tuple[str, ...]) -> list[str]:
    """Return one row per level, with seeds 0 to 2, the mean and the seed deviation for each k, in percent."""
    def values(rows: pd.DataFrame) -> str:
        parts = []
        for k in settings.RECALL_KS:
            r = rows[rows["metric"] == f"recall_at_{k}_{direction}"].iloc[0]
            parts += [f"{100 * r[column]:.1f}" for column in ("seed_0", "seed_1", "seed_2", "mean", "seed_std")]
        return " & ".join(parts)

    other_col = "text_corruption" if kind_col == "vision_corruption" else "vision_corruption"
    clean = cells[(cells["vision_corruption"] == "none") & (cells["text_corruption"] == "none")]
    lines = [f"Clean & --- & {values(clean)} \\\\"]
    for kind in kinds:
        for level in settings.SEVERITIES:
            rows = cells[(cells[kind_col] == kind) & (cells[other_col] == "none") & (cells[level_col] == level)]
            lines.append(f"{CORRUPTION_LABELS[kind] if level == settings.SEVERITIES[0] else ''} & {level} & {values(rows)} \\\\")
    return lines


def recall_at_k_tables() -> None:
    """Write one file per dataset, with one table per model that lists every corruption, severity and seed."""
    table = pd.read_csv(file_paths.TAB_RECALL_AT_K, keep_default_na=False, na_values=[""])
    n_columns = 2 + 5 * len(settings.RECALL_KS)
    spans = " & ".join(rf"\multicolumn{{5}}{{c}}{{Recall@{k} (\%)}}" for k in settings.RECALL_KS)
    rules = "".join(rf"\cmidrule(lr){{{3 + 5 * i}-{7 + 5 * i}}}" for i in range(len(settings.RECALL_KS)))
    columns = " & ".join(["S0 & S1 & S2 & Mean & SD"] * len(settings.RECALL_KS))
    for dataset in dict.fromkeys(table["dataset"]):
        for position, m in enumerate(MODELS):
            cells = table[(table["dataset"] == dataset) & (table["model"] == m)]
            has_objects = (cells["vision_corruption"] == "object_cutout").any()
            images = settings.IMAGE_CORRUPTIONS + (("object_cutout",) if has_objects else ())
            body = [rf"\multicolumn{{{n_columns}}}{{l}}{{\textit{{Image-to-text, damaged image query}}}} \\"]
            body += _recall_k_rows(cells, "i2t", "vision_corruption", "image_severity", images)
            body += [r"\midrule", rf"\multicolumn{{{n_columns}}}{{l}}{{\textit{{Text-to-image, damaged caption query}}}} \\"]
            body += _recall_k_rows(cells, "t2i", "text_corruption", "text_severity", settings.TEXT_CORRUPTIONS)
            latex_table(
                file_paths.tab_recall_at_k(dataset),
                f"Recall@1, 5 and 10 (\\%) are shown for {MODEL_LABELS[m]} on {DATASET_TITLES[dataset]} at a gallery of "
                f"{GALLERY}, with individual seeds, their mean and sample standard deviation.",
                f"tab:recall_k_{dataset}_{m}", r"\scriptsize", "lr" + (COLUMN_GROUP_GAP + "rrrrr") * len(settings.RECALL_KS),
                [rf" & & {spans} \\", rules, rf"Corruption & Sev. & {columns} \\"], body, column_space="3pt", placement="p", append=position > 0)


# Print shared-space similarity by corruption, severity and model.
def similarity_tables() -> None:
    table = pd.read_csv(file_paths.TAB_SIMILARITY_BY_CORRUPTION)
    columns = " & ".join(r"\shortstack{" + r"\\".join(label.rsplit(" ", 1)) + "}" if " " in label else label
                         for label in MODEL_LABELS.values())
    for dataset in dict.fromkeys(table["dataset"]):
        sub = table[table["dataset"] == dataset]
        body = []
        for side, direction, kinds in (("Image", "i2t", settings.IMAGE_CORRUPTIONS), ("Text", "t2i", settings.TEXT_CORRUPTIONS)):
            if side == "Text":
                body.append(r"\midrule")
            for kind in kinds:
                for severity in settings.SEVERITIES:
                    cell = _rows_by_model(sub[(sub["direction"] == direction) & (sub["corruption_type"] == kind) & (sub["severity"] == severity)])
                    label = f"{CORRUPTION_LABELS[kind]}" if severity == settings.SEVERITIES[0] else ""
                    body.append(f"{label} & {severity} & " + " & ".join(f"{cell.loc[m, 'selfsim']:.3f}" for m in MODELS) + " & "
                                + " & ".join(f"{100 * cell.loc[m, 'r_at_1']:.1f}" for m in MODELS) + r" \\")
        latex_table(
            file_paths.tab_similarity(dataset),
            f"Shared-space cosine similarity to the clean query and Recall@1 are shown by corruption and severity on "
            f"{DATASET_TITLES[dataset]} at a gallery of {FULL_GALLERY}.",
            f"tab:similarity_{dataset}", r"\scriptsize", "lr" + (COLUMN_GROUP_GAP + "r" * len(MODELS)) * 2,
            [rf" & & \multicolumn{{{len(MODELS)}}}{{c}}{{Shared-space similarity}} & \multicolumn{{{len(MODELS)}}}{{c}}{{Recall@1 (\%)}} \\",
             rf"\cmidrule(lr){{3-{2 + len(MODELS)}}}\cmidrule(lr){{{3 + len(MODELS)}-{2 + 2 * len(MODELS)}}}",
             rf"Corruption & Sev. & {columns} & {columns} \\"],
            body, column_space="3pt", placement="p")


# Compare retrieval within similarity bands and within finer per-query bins.
def similarity_band_table() -> str:
    table = pd.read_csv(file_paths.TAB_RECALL_BY_SIMILARITY_BAND)
    datasets = list(dict.fromkeys(table["dataset"]))
    bands = list(dict.fromkeys(table["band"]))
    titles = " & ".join(rf"\multicolumn{{4}}{{c}}{{{DATASET_TITLES[d]}}}" for d in datasets)
    rules = "".join(rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(datasets)))
    body = []
    for band in bands:
        cells = []
        for d in datasets:
            r = table[(table["dataset"] == d) & (table["band"] == band)].iloc[0]
            cells += [_cell(r["mean_r1_i2t"], "{:.1f}", 100), _cell(r["n_cells_i2t"], "{:.0f}"), _cell(r["mean_r1_t2i"], "{:.1f}", 100), _cell(r["n_cells_t2i"], "{:.0f}")]
        label = band.replace("<0.60", "up to 0.60").replace(">0.90", "above 0.90").replace("-", "--")
        body.append(f"{label} & " + " & ".join(cells) + r" \\")
    return latex_table(
        file_paths.TAB_RECALL_BY_SIMILARITY_BAND_TEX,
        f"Recall@1 (\\%) is averaged across model-corruption-severity cells within shared-space similarity bands on "
        f"both datasets at a gallery of {FULL_GALLERY}.",
        "tab:similarity_bands", r"\footnotesize", "l" + "rrrr" * len(datasets),
        [rf" & {titles} \\", rules, "Similarity & " + " & ".join(["i2t & Cells & t2i & Cells"] * len(datasets)) + r" \\"], body, column_space="4pt")


def _bin_label(interval: str) -> str:
    """Turn a stored interval such as (0.3, 0.35] into a label for a table row."""
    low, high = (float(v) for v in re.findall(r"-?\d+\.?\d*", interval))
    return f"up to {high:.2f}" if low < 0 else f"{low:.2f}--{high:.2f}"


def _bin_start(interval: str) -> float:
    return float(re.findall(r"-?\d+\.?\d*", interval)[0])


def displacement_bin_table() -> str:
    """Print the pooled per-query Recall@1 and query counts in each similarity bin."""
    table = pd.read_csv(file_paths.FIG_RECALL_BY_DISPLACEMENT_BIN)
    datasets = list(dict.fromkeys(table["dataset"]))
    titles = " & ".join(rf"\multicolumn{{4}}{{c}}{{{DATASET_TITLES[d]}}}" for d in datasets)
    rules = "".join(rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(datasets)))
    body = []
    for interval in sorted(table["bin"].unique(), key=_bin_start):
        cells = []
        for d in datasets:
            for direction in ("i2t", "t2i"):
                r = table[(table["dataset"] == d) & (table["bin"] == interval) & (table["direction"] == direction)]
                cells += ["---", "0"] if r.empty else [f"{100 * r['rate'].iloc[0]:.1f}", f"{int(r['n'].iloc[0]):,}"]
        body.append(f"{_bin_label(interval)} & " + " & ".join(cells) + r" \\")
    return latex_table(
        file_paths.TAB_DISPLACEMENT_BINS_TEX,
        f"Pooled per-query Recall@1 (\\%) and observation counts are shown within shared-space similarity bins on "
        f"both datasets at a gallery of {FULL_GALLERY}.",
        "tab:displacement_bins", r"\footnotesize", "l" + "rrrr" * len(datasets),
        [rf" & {titles} \\", rules, "Similarity & " + " & ".join(["i2t & n & t2i & n"] * len(datasets)) + r" \\"], body, column_space="3pt")


def displacement_corruption_table() -> str:
    """Print Recall@1 for each corruption in each similarity bin that holds enough queries."""
    table = pd.read_csv(file_paths.TAB_DISPLACEMENT_BY_CORRUPTION)
    kinds = settings.IMAGE_CORRUPTIONS + settings.TEXT_CORRUPTIONS
    body = []
    for dataset in dict.fromkeys(table["dataset"]):
        body.append(dataset_heading(dataset, len(kinds) + 1))
        sub = table[(table["dataset"] == dataset) & (table["n"] >= MIN_QUERIES_FOR_CORRUPTION_RATE)]
        for interval in sorted(sub["bin"].unique(), key=_bin_start):
            rates = sub[sub["bin"] == interval].set_index("corruption")["rate"]
            body.append(f"{_bin_label(interval)} & " + " & ".join(_cell(rates.get(kind), "{:.1f}", 100) for kind in kinds) + r" \\")
    return latex_table(
        file_paths.TAB_DISPLACEMENT_BY_CORRUPTION_TEX,
        f"Recall@1 (\\%) is compared across corruptions within shared-space similarity bins on both datasets at a "
        f"gallery of {FULL_GALLERY}.",
        "tab:displacement_corruption", r"\footnotesize", "l" + "r" * len(kinds),
        [" Similarity & " + " & ".join(SHORT_CORRUPTION_LABELS[kind] for kind in kinds) + r" \\"], body, column_space="4pt")


def _signed_points(value: float) -> str:
    """Print a proportion in percentage points with a true plus or minus sign."""
    return f"{'$-$' if value < 0 else '$+$'}{abs(100 * value):.1f}"


def displacement_gap_table() -> str:
    """Print each model's direction gap within the same similarity bins."""
    table = pd.read_csv(file_paths.TAB_DISPLACEMENT_RESIDUAL)
    datasets = list(dict.fromkeys(table["dataset"]))
    titles = " & ".join(rf"{DATASET_TITLES[d]}" for d in datasets)
    body = []
    for m in MODELS:
        cells = []
        for d in datasets:
            r = table[(table["dataset"] == d) & (table["model"] == m)].iloc[0]
            cells.append(f"{_signed_points(r['direction_gap'])} [{_signed_points(r['direction_gap_lo'])}, {_signed_points(r['direction_gap_hi'])}]")
        body.append(f"{MODEL_LABELS[m]} & " + " & ".join(cells) + r" \\")
    return latex_table(
        file_paths.TAB_DISPLACEMENT_RESIDUAL_TEX,
        f"The i2t minus t2i Recall@1 gap is shown within shared-space similarity bins at a gallery of "
        f"{FULL_GALLERY}, in percentage points with 95\\% intervals.",
        "tab:displacement_gap", r"\small", "l" + "c" * len(datasets),
        [rf"Model & {titles} \\"], body)


# Print the gallery-size sweep.
def gallery_sweep_table() -> str:
    table = pd.read_csv(file_paths.TAB_GALLERY_SIZE_SWEEP)
    datasets = list(dict.fromkeys(table["dataset"]))
    sizes = sorted(table["gallery_size"].unique())
    titles = " & ".join(rf"\multicolumn{{{len(sizes)}}}{{c}}{{{DATASET_TITLES[d]}, gallery size}}" for d in datasets)
    rules = "".join(rf"\cmidrule(lr){{{2 + len(sizes) * i}-{1 + len(sizes) * (i + 1)}}}" for i in range(len(datasets)))
    body = []
    for m in MODELS:
        cells = []
        for d in datasets:
            sub = table[(table["dataset"] == d) & (table["model"] == m)].set_index("gallery_size")["asym_ratio"]
            cells += [f"{sub[size]:.2f}" for size in sizes]
        body.append(f"{MODEL_LABELS[m]} & " + " & ".join(cells) + r" \\")
    return latex_table(
        file_paths.TAB_GALLERY_SIZE_SWEEP_TEX,
        "The i2t/t2i Recall@1 ratio compares severity-5 Gaussian noise with severity-5 word deletion across "
        "gallery sizes on both datasets.",
        "tab:gallery_sweep", r"\footnotesize", "l" + "r" * (len(sizes) * len(datasets)),
        [rf" & {titles} \\", rules, "Model & " + " & ".join(f"{size:,}" for _ in datasets for size in sizes) + r" \\"], body, column_space="4pt")


# Print the object cutout areas.
def object_cutout_area_table() -> str | None:
    """Print the achieved areas, unless the object cutout step left them out and has already said so."""
    if not file_paths.TAB_OBJECT_CUTOUT_DAMAGE_AND_AREA.exists():
        return None
    table = pd.read_csv(file_paths.TAB_OBJECT_CUTOUT_DAMAGE_AND_AREA)
    if table["mean_area"].isna().all():
        note_skipped("no achieved object cutout areas were measured, so the area table prints dashes")
    body = [f"{int(r['severity'])} & {100 * r['target_area']:.0f} & {_cell(r['mean_area'], '{:.1f}', 100)} & {_cell(r['median_area'], '{:.1f}', 100)} & "
            f"{_cell(r['area_p10'], '{:.1f}', 100)} & {_cell(r['area_p90'], '{:.1f}', 100)} & {_cell(r['share_below_target'], '{:.1f}', 100)} & "
            f"{r['median_damage_ssim']:.3f} & {r['damage_percentile']:.1f} \\\\" for r in table.to_dict("records")]
    return latex_table(
        file_paths.TAB_OBJECT_CUTOUT_DAMAGE_AND_AREA_TEX,
        "Achieved object-cutout areas and input damage are summarized on COCO across three seeds of 2,000 pairs.",
        "tab:object_cutout_area", r"\footnotesize", "rrrrrrrrr",
        [r" & \multicolumn{5}{c}{Image area (\%)} & & \multicolumn{2}{c}{Damage} \\", r"\cmidrule(lr){2-6}\cmidrule(lr){8-9}",
         r"Severity & Target & Mean & Median & P10 & P90 & \shortstack{Images below\\target (\%)} & \shortstack{Median\\1 - SSIM} & Percentile \\"], body, column_space="4pt")


def build_appendix_tables() -> None:
    """Write the tables that read the numerical files, so this step runs after those steps.

    Check every input first, so a missing file stops the step before it writes any table.
    """
    inputs = (file_paths.TAB_PECORE_TRUNCATION_DATA, file_paths.TAB_PECORE_TRUNCATION_ROBUSTNESS, file_paths.TAB_CAPTION_LENGTH_PER_MODEL,
              file_paths.TAB_CAPTION_LENGTH, file_paths.TAB_Q5_RATIO_BY_PROTOCOL, file_paths.TAB_RECALL_AT_K, file_paths.TAB_SIMILARITY_BY_CORRUPTION,
              file_paths.TAB_RECALL_BY_SIMILARITY_BAND, file_paths.FIG_RECALL_BY_DISPLACEMENT_BIN, file_paths.TAB_DISPLACEMENT_BY_CORRUPTION,
              file_paths.TAB_DISPLACEMENT_RESIDUAL, file_paths.TAB_GALLERY_SIZE_SWEEP)
    missing = [path.name for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"appendix_tables reads files that the other steps write, and these are missing: {', '.join(missing)}")
    for build in (pecore_truncation_table, caption_length_table, q5_ratio_by_protocol_table, recall_at_k_tables, similarity_tables,
                  similarity_band_table, displacement_bin_table, displacement_corruption_table, displacement_gap_table,
                  gallery_sweep_table, object_cutout_area_table):
        build()
        print(f"built {build.__name__}")
