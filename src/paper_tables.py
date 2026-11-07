"""Generate the LaTeX tables that the saved runs supply directly, and their numerical summaries."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import file_paths, settings
from src.bootstrap_and_quintiles import bootstrap_group_statistics
from src.corruptions import strength
from src.load_results import datasets_with_300_pair_run, datasets_with_headline_rows, fidelity_retention_damage
from src.paper_figures import (DATASET_TITLES, MODEL_LABELS, MODELS, _joint_retention_rows, clean_recall, pooling_summary,
                               quintile_rows, quintile_summary)

CORRUPTION_LABELS = {"gaussian_noise": "Gaussian noise", "gaussian_blur": "Gaussian blur", "cutout": "Rectangle cutout",
                     "object_cutout": "Object cutout", "mask": "Word deletion", "shuffle": "Word shuffle",
                     "replace": "Word replacement", "typo": "Typo"}


def latex_table(path, caption: str, label: str, size: str, columns: str, header: list[str], body: list[str],
                column_space: str | None = None, placement: str = "h", append: bool = False) -> str:
    """Write one booktabs table, or add it to a file of related tables, and return its LaTeX."""
    lines = [rf"\begin{{table}}[{placement}]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", size]
    if column_space:
        lines.append(rf"\setlength{{\tabcolsep}}{{{column_space}}}")
    lines += [rf"\begin{{tabular}}{{{columns}}}", r"\toprule", *header, r"\midrule", *body,
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    if append:
        with open(path, "a") as f:
            f.write("\n\n" + text)
    else:
        path.write_text(text)
    return text


def dataset_heading(dataset: str, n_columns: int) -> str:
    """Return a row that opens one dataset's block in a table covering both datasets."""
    return rf"\multicolumn{{{n_columns}}}{{l}}{{\textit{{{DATASET_TITLES[dataset]}}}}} \\"


# Summarize the model specifications.
# The parameter counts include both towers. PE-Core has 317M + 354M; SigLIP 2 has 428M + 708M.
MODEL_FACTS = {
    "openai_clip_b32":      ("151",   "512",   "224 / 32", "77", "EOT token",     "InfoNCE", "WebImageText"),
    "openai_clip_l14":      ("428",   "768",   "224 / 14", "77", "EOT token",     "InfoNCE", "WebImageText"),
    "openclip_l14_laion2b": ("428",   "768",   "224 / 14", "77", "EOT token",     "InfoNCE", "LAION-2B-en"),
    "siglip2_so400m_384":   ("1,136", "1,152", "384 / 16", "64", "last position", "sigmoid", "WebLI"),
    "pecore_l14_336":       ("671",   "1,024", "336 / 14", "32", "EOT token",     "InfoNCE", "MetaCLIP-curated, video"),
}


def t1_model_summary() -> str:
    return latex_table(
        file_paths.TAB_SPECS,
        "The five evaluated models are summarized by their size, shared-space embedding, input limits, text "
        "pooling, contrastive objective and training data.",
        "tab:t1_models", r"\footnotesize", "lrrcrlll",
        [r" & Params & Embed. & Image size / & Max. text & Text & Contrastive & Training \\",
         r"Model & (M) & dim. & patch (px) & tokens & pooling & loss & data \\"],
        [f"{MODEL_LABELS[m]} & {' & '.join(MODEL_FACTS[m])} \\\\" for m in MODELS],
        column_space="3.5pt")


# State what each corruption does and why it is in the suite.
CORRUPTION_ROLES = (
    ("gaussian_noise", "Image", "Adds Gaussian pixel noise", "Scenario", "Low-light sensor noise"),
    ("gaussian_blur", "Image", "Applies Gaussian blur", "Scenario", "Defocus, and low-resolution sources upscaled at ingestion"),
    ("cutout", "Image", "Blacks out one rectangle", "Scenario", "Occlusion, cropping and watermarks"),
    ("object_cutout", "Image", "Blacks out whole annotated objects in random order until a target area is reached or the objects run out", "Probe",
     "Removes image content at object boundaries, on COCO only and outside the damage quintiles"),
    ("mask", "Text", "Deletes the selected words", "Both", "Dropped words in speech transcription or OCR, and a test of content loss at heavy levels"),
    ("replace", "Text", "Swaps the selected words for random words", "Both",
     "Misrecognition, a wrong word in place of the right one, and mostly unrelated words at heavy levels"),
    ("shuffle", "Text", "Moves the selected words among their positions", "Probe",
     "Holds lexical content fixed and destroys order, which separates content loss from order loss"),
    ("typo", "Text", "Misspells each selected word once", "Both",
     "Keyboard errors in typed queries, and a test of widespread misspelling at heavy levels, reported outside the damage quintiles"),
)


def corruption_roles_table() -> str:
    """Print each corruption's role, with rows ended by a command that still works after ragged-right text."""
    return latex_table(
        file_paths.TAB_CORRUPTION_ROLES,
        "The corruption suite combines practical degradation scenarios with controlled probes of content and word "
        "order.",
        "tab:corruption_roles", r"\footnotesize", r"llp{3.6cm}lp{5.1cm}",
        [r"Corruption & Side & What it does & Role & Scenario or probe function \\"],
        [rf"{CORRUPTION_LABELS[kind]} & {side} & \raggedright {action} & {role} & \raggedright {function} \tabularnewline"
         for kind, side, action, role, function in CORRUPTION_ROLES],
        column_space="3pt")


def severity_settings_table() -> str:
    """Print the strength of each corruption at levels 1 to 5, read from the settings file."""
    def percent(value: float) -> str:
        return f"{100 * value:.0f}\\%"

    rows = (("gaussian_noise", "image", "Standard deviation on the 0 to 255 pixel scale", lambda v: f"{v:g}"),
            ("gaussian_blur", "image", "PIL GaussianBlur radius in pixels", lambda v: f"{v:g}"),
            ("cutout", "image", "Share of the stored image blacked out", percent),
            ("object_cutout", "image", "Target share of the stored image", percent))
    body = [f"{CORRUPTION_LABELS[kind]} & {meaning} & " + " & ".join(show(strength(side, kind, level)) for level in settings.SEVERITIES) + r" \\"
            for kind, side, meaning, show in rows]
    body.append("Word deletion, shuffle, replacement and typo & Nominal share of a caption's words selected & "
                + " & ".join(percent(strength("text", "mask", level)) for level in settings.SEVERITIES) + r" \\")
    return latex_table(
        file_paths.TAB_SEVERITY_SETTINGS,
        "The nominal corruption strengths are listed for severity levels 1 to 5.",
        "tab:severity_settings", r"\footnotesize", "p{3.2cm}p{4.2cm}rrrrr",
        [r"Corruption & Setting & 1 & 2 & 3 & 4 & 5 \\"], body, column_space="3pt")


# Compare the headline Q1 and Q5 retrieval scores against the clean queries.
def _ratio_statistics(images: pd.DataFrame, captions: pd.DataFrame) -> tuple[float, float, float]:
    """Return the i2t to t2i ratio with an interval that resamples both directions' pairs together."""
    return bootstrap_group_statistics({"i2t": images, "t2i": captions}, {"ratio": lambda g: g["i2t"] / g["t2i"]})["ratio"]


def _headline_rows(dataset: str) -> list[dict]:
    quintiles, clean = quintile_summary("ssim", dataset), clean_recall(dataset)
    images, captions = quintile_rows("ssim", dataset)
    rows = []
    for m in MODELS:
        q = {k: quintiles[(quintiles["model"] == m) & (quintiles["quintile"] == k.upper())].iloc[0] for k in ("q1", "q5")}
        c = {d: clean[(clean["model"] == m) & (clean["direction"] == d)]["mean"].iloc[0] for d in ("i2t", "t2i")}
        row = {"dataset": dataset, "model": m, "clean_i2t": c["i2t"], "clean_t2i": c["t2i"]}
        for d in ("i2t", "t2i"):
            for k in ("q1", "q5"):
                row.update({f"{k}_{d}": q[k][d], f"{k}_{d}_lo": q[k][f"{d}_lo"], f"{k}_{d}_hi": q[k][f"{d}_hi"]})
        row.update({"i2t_drop": c["i2t"] - q["q5"]["i2t"], "t2i_drop": c["t2i"] - q["q5"]["t2i"]})
        row["drop_gap"] = row["t2i_drop"] - row["i2t_drop"]
        for k in ("q1", "q5"):
            ratio, lo, hi = _ratio_statistics(images[m][images[m]["quintile"] == k.upper()], captions[m][captions[m]["quintile"] == k.upper()])
            row.update({f"{k}_ratio": ratio, f"{k}_ratio_lo": lo, f"{k}_ratio_hi": hi})
        rows.append(row)
    return rows


def t2_headline() -> str:
    """Print COCO's headline table, while the CSV holds the same columns for both datasets."""
    by_dataset = {d: _headline_rows(d) for d in datasets_with_headline_rows()}
    pd.DataFrame([row for rows in by_dataset.values() for row in rows]).to_csv(file_paths.TAB_Q1_Q5_VS_CLEAN_DATA, index=False)
    return latex_table(
        file_paths.TAB_Q1_Q5_VS_CLEAN,
        "COCO Recall@1 (\\%) is shown for clean queries and the least and most damaged quintiles at a gallery of 1,000, "
        "with observations pooled across three seeds.",
        "tab:t2_headline", r"\footnotesize", "lrrrrrrrcrrr",
        [r" & \multicolumn{2}{c}{Clean} & \multicolumn{3}{c}{Q1} & \multicolumn{3}{c}{Q5} & \multicolumn{3}{c}{Drop to Q5 (pp)} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}",
         r"Model & i2t & t2i & i2t & t2i & Ratio & i2t & t2i & Ratio & i2t & t2i & \shortstack{Excess\\text} \\"],
        [f"{MODEL_LABELS[r['model']]} & {100 * r['clean_i2t']:.1f} & {100 * r['clean_t2i']:.1f} & "
         f"{100 * r['q1_i2t']:.1f} & {100 * r['q1_t2i']:.1f} & {r['q1_ratio']:.2f} & "
         f"{100 * r['q5_i2t']:.1f} & {100 * r['q5_t2i']:.1f} & {r['q5_ratio']:.2f} [{r['q5_ratio_lo']:.2f}, {r['q5_ratio_hi']:.2f}] & "
         f"{100 * r['i2t_drop']:.1f} & {100 * r['t2i_drop']:.1f} & {100 * r['drop_gap']:.1f} \\\\"
         for r in by_dataset["coco"]],
        column_space="2.5pt")


# Compare damage predictors on the joint noise-mask grid.
def _r_squared(x: np.ndarray, y: np.ndarray) -> float:
    slope, intercept = np.polyfit(x, y, 1)
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - ((y - (slope * x + intercept)) ** 2).sum() / ss_tot if ss_tot > 0 else float("nan")


def _dominance_rows(dataset: str) -> list[dict]:
    """Fit the joint retention margin on the max, the min and each side alone, under both rulers."""
    joint = _joint_retention_rows(dataset)
    dmg = fidelity_retention_damage(dataset)
    percentile = {}
    for side, column, kind in (("image", "damage_ssim", settings.NOISE_MASK_GRID["image_corruption"]),
                               ("text", "damage_bleu", settings.NOISE_MASK_GRID["text_corruption"])):
        cells = dmg[dmg["modality"] == side].groupby(["corruption_type", "severity"])[column].mean().reset_index()
        cells["pct"] = cells[column].rank() / len(cells)
        mine = cells[cells["corruption_type"] == kind]
        percentile[side] = dict(zip(mine["severity"], mine["pct"]))
    image_pct = joint["image_severity"].map(percentile["image"])
    text_pct = joint["text_severity"].map(percentile["text"])
    predictors = {"ordinal_max": joint[["image_severity", "text_severity"]].max(axis=1),
                  "ordinal_min": joint[["image_severity", "text_severity"]].min(axis=1),
                  "percentile_max": np.maximum(image_pct, text_pct), "percentile_min": np.minimum(image_pct, text_pct),
                  "noise_severity_alone": joint["image_severity"], "deletion_severity_alone": joint["text_severity"]}
    return [{"dataset": dataset, "model": m,
             **{name: _r_squared(x[joint["model"] == m].to_numpy(dtype=float), joint[joint["model"] == m]["value"].to_numpy())
                for name, x in predictors.items()}} for m in MODELS]


def t4_dominance() -> str:
    """Compare the max/min predictors of joint retention under two damage rulers, on both datasets.

    Use ordinal severity or percentiles of condition-mean damage, and fit each side alone as well.
    """
    by_dataset = {d: _dominance_rows(d) for d in datasets_with_300_pair_run()}
    pd.DataFrame([row for rows in by_dataset.values() for row in rows]).to_csv(file_paths.TAB_DOMINANCE_DATA, index=False)
    body = []
    for dataset, rows in by_dataset.items():
        body.append(dataset_heading(dataset, 7))
        body += [f"{MODEL_LABELS[r['model']]} & {r['ordinal_max']:.3f} & {r['ordinal_min']:.3f} & {r['percentile_max']:.3f} & "
                 f"{r['percentile_min']:.3f} & {r['noise_severity_alone']:.3f} & {r['deletion_severity_alone']:.3f} \\\\" for r in rows]
    return latex_table(
        file_paths.TAB_DOMINANCE,
        "Univariate regression $R^2$ compares predictors of the joint match-retention margin under ordinal "
        "severity and condition-damage percentiles on both datasets.",
        "tab:t4_dominance", r"\footnotesize", "lcccccc",
        [r" & \multicolumn{2}{c}{Ordinal severity} & \multicolumn{2}{c}{Condition-damage percentile} & \multicolumn{2}{c}{One side alone} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"Model & Max & Min & Max & Min & Noise & Deletion \\"], body, column_space="4pt")


# Compare standard and mean text pooling.
def t5_text_pooling_ablation() -> str:
    """Print both datasets' margins and drops, with the clean and severity-1 starting points named."""
    by_dataset = {d: pooling_summary(d) for d in datasets_with_300_pair_run()}
    table = pd.concat([rows.assign(dataset=d) for d, rows in by_dataset.items()])
    table[["dataset"] + [c for c in table.columns if c != "dataset"]].to_csv(file_paths.TAB_STANDARD_VS_MEAN_POOLING_DATA, index=False)
    body = []
    for dataset, rows in by_dataset.items():
        body.append(dataset_heading(dataset, 10))
        for r in rows.to_dict("records"):
            body.append(f"{MODEL_LABELS[r['model']]} & {r['standard_clean']:.3f} & {r['mean_clean']:.3f} & {r['standard_sev1']:.3f} & {r['mean_sev1']:.3f} & "
                        f"{r['standard_drop']:.3f} & {r['mean_drop']:.3f} & {r['drop_reduction_pct']:.1f} & "
                        f"{r['standard_rel_drop_pct']:.1f} & {r['mean_rel_drop_pct']:.1f} \\\\")
    return latex_table(
        file_paths.TAB_STANDARD_VS_MEAN_POOLING,
        "The inference-only text-pooling ablation compares clean margins and severity-1-to-5 drops, and its drop "
        "reduction is neither a demonstrated upper bound nor a causal estimate.",
        "tab:t5_pooling", r"\footnotesize", "lccccccccc",
        [r" & \multicolumn{2}{c}{Clean margin} & \multicolumn{2}{c}{Severity-1 margin} & \multicolumn{2}{c}{Absolute drop} & Drop & \multicolumn{2}{c}{Relative drop (\%)} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){9-10}",
         r"Model & Standard & Mean & Standard & Mean & Standard & Mean & reduction (\%) & Standard & Mean \\"], body, column_space="2.5pt")


# Summarize input damage by corruption and severity.
def _input_damage_rows(dataset: str) -> list[dict]:
    dmg = fidelity_retention_damage(dataset)
    rows = []
    for side in ("image", "text"):
        sub = dmg[dmg["modality"] == side]
        for kind in sub["corruption_type"].unique():
            for severity in settings.SEVERITIES:
                cell = sub[(sub["corruption_type"] == kind) & (sub["severity"] == severity)]
                image = side == "image"
                rows.append({"dataset": dataset, "modality": side, "corruption_type": kind, "severity": severity,
                             "ssim": cell["ssim"].mean() if image else np.nan, "psnr": cell["psnr"].mean() if image else np.nan,
                             "ed": np.nan if image else cell["norm_edit_distance"].mean(),
                             "bleu": np.nan if image else cell["bleu"].mean()})
    return rows


def t6_input_damage() -> str:
    """Print both datasets' input damage side by side, with one block for each modality's measures."""
    datasets = datasets_with_300_pair_run()
    df = pd.DataFrame([row for d in datasets for row in _input_damage_rows(d)])
    df.to_csv(file_paths.TAB_INPUT_DAMAGE_DATA, index=False)
    titles = " & ".join(rf"\multicolumn{{2}}{{c}}{{{DATASET_TITLES[d]}}}" for d in datasets)
    rules = "".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(datasets)))
    body = []
    for side, first, second, label, measures in (("image", "ssim", "psnr", "Image corruption", r"SSIM & PSNR (dB)"),
                                                 ("text", "ed", "bleu", "Text corruption", r"Word edit distance & BLEU-4")):
        if side == "text":
            body += [r"\midrule", rf"{label} & Severity & " + " & ".join([measures] * len(datasets)) + r" \\", r"\midrule"]
        sub = df[df["modality"] == side]
        for kind in sub["corruption_type"].unique():
            for severity in settings.SEVERITIES:
                cells = [sub[(sub["dataset"] == d) & (sub["corruption_type"] == kind) & (sub["severity"] == severity)].iloc[0] for d in datasets]
                formats = ("{:.3f}", "{:.1f}") if side == "image" else ("{:.3f}", "{:.3f}")
                values = " & ".join(f"{formats[0].format(r[first])} & {formats[1].format(r[second])}" for r in cells)
                body.append(f"{CORRUPTION_LABELS[kind] if severity == settings.SEVERITIES[0] else ''} & {severity} & {values} \\\\")
    return latex_table(
        file_paths.TAB_INPUT_DAMAGE,
        "Mean input-damage measurements are shown for each corruption and severity on both datasets, using 300 "
        "pairs per seed across three seeds.",
        "tab:t6_damage", r"\footnotesize", "lr" + "rr" * len(datasets),
        [rf" & & {titles} \\", rules, r"Image corruption & Severity & " + " & ".join(["SSIM & PSNR (dB)"] * len(datasets)) + r" \\"],
        body, column_space="5pt")


# Count the captions that exceed PE-Core's context limit.
PE_CORE_CONTENT_TOKENS = 30  # Reserve two of the 32 positions for start and end tokens.


def pecore_content_tokens(captions: list[str]) -> np.ndarray:
    """Count content tokens using PE-Core's tokenizer with a 128-token context."""
    from core.vision_encoder.tokenizer import SimpleTokenizer

    tokenizer = SimpleTokenizer(context_length=128)
    # The end token has the highest id, so its position is the argmax.
    return np.array([int(np.argmax(tokenizer([c])[0].tolist())) - 1 for c in captions])


def pecore_truncation_counts() -> pd.DataFrame:
    """Count the distinct captions that exceed PE-Core's content limit in each released sample."""
    rows = []
    for dataset in file_paths.DATASETS:
        with open(file_paths.manifest(dataset)) as f:
            captions = sorted({entry["caption"] for entry in json.load(f)})
        lengths = pecore_content_tokens(captions)
        over = int((lengths > PE_CORE_CONTENT_TOKENS).sum())
        rows.append({"dataset": dataset, "distinct_captions": len(captions), "over_30_tokens": over,
                     "percent_over": 100 * over / len(captions), "longest_caption_tokens": int(lengths.max())})
    df = pd.DataFrame(rows)
    df.to_csv(file_paths.TAB_PECORE_TRUNCATION_DATA, index=False)
    return df


def build_tables() -> None:
    """Write the tables computed directly from the saved runs. The other tables follow the numerical steps."""
    for build in (t1_model_summary, corruption_roles_table, severity_settings_table, t2_headline, t4_dominance,
                  t5_text_pooling_ablation, t6_input_damage, pecore_truncation_counts):
        build()
        print(f"built {build.__name__}")
