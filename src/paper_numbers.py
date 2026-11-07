"""Generate statistical summaries, robustness analyses and supporting plots."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from src import file_paths, settings
from src.bootstrap_and_quintiles import QUINTILES, bootstrap_ci, bootstrap_group_statistics, damage_quintile
from src.corruptions import strength
from src.load_data import retrieval_positions
from src.load_results import (datasets_with_300_pair_run, datasets_with_headline_rows, fidelity_retention_damage, fidelity_retention_rows,
                              full_gallery_damage, full_gallery_rows, headline_damage, note_skipped,
                              object_cutout_per_pair_at_headline_gallery, per_pair_retrieval, quintiles_by_model,
                              recall_with_damage, typo_fidelity_retention_rows)
from src.paper_figures import (IMAGE_RULERS, MODELS, _fidelity_rows, _joint_retention_rows, _with_dataset_column, clean_recall,
                               draw_object_and_rectangle_cutout, draw_typo_recall_by_level, quintile_rows, quintile_summary)
from src.paper_tables import PE_CORE_CONTENT_TOKENS, pecore_content_tokens

DATASETS = file_paths.DATASETS
# Captions are ranked by 1 - BLEU, the paper's ruler. Word edit distance is the check.
TEXT_RULERS = {"bleu": "damage_bleu", "edit_distance": "norm_edit_distance"}


def _sort_model_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Keep the paper's model order within each dataset and caption ruler."""
    order = {model: i for i, model in enumerate(MODELS)}
    keys = [column for column in ("dataset", "text_ruler", "model") if column in table.columns]
    return table.sort_values(keys, key=lambda values: values.map(order) if values.name == "model" else values,
                             kind="stable")


def _ratio_rows(table: pd.DataFrame) -> pd.DataFrame:
    return table.assign(ratio=table["i2t"] / table["t2i"])


def _retrieval_conditions(frame: pd.DataFrame) -> pd.DataFrame:
    """Name the direction, query corruption and severity of each retrieval measurement."""
    frame = frame.copy()
    frame["direction"] = frame["metric"].str[-3:]
    frame["corruption_type"] = np.where(frame["direction"] == "i2t", frame["vision_corruption"], frame["text_corruption"])
    frame["severity"] = np.where(frame["direction"] == "i2t", frame["image_severity"], frame["text_severity"])
    return frame


# Compare the retrieval directions within each damage quintile.
def build_i2t_vs_t2i_paired_tests() -> None:
    """Report per-quintile Cohen's d and one Wilcoxon test on 15 model-seed Q5 pairs for each dataset.

    Cohen's d treats the directions as independent and uses their mean Bernoulli variance,
    rather than a count-weighted pooled sample variance. The Wilcoxon pairs share
    pictures and corruptions, so they are not independent replications. Three seeds
    cannot yield a significant two-sided per-cell Wilcoxon test; none is run.
    """
    rows = []
    for dataset in datasets_with_headline_rows():
        for r in quintile_summary("ssim", dataset).itertuples():
            spread = np.sqrt((r.i2t * (1 - r.i2t) + r.t2i * (1 - r.t2i)) / 2)
            rows.append({"dataset": dataset, "model": r.model, "quintile": r.quintile, "mean_i2t": r.i2t, "mean_t2i": r.t2i,
                         "cohens_d": (r.i2t - r.t2i) / spread if spread > 0 else np.nan})
        images, captions = quintile_rows("ssim", dataset)
        i2t_means, t2i_means = [], []
        for m in MODELS:
            i_seed = images[m][images[m]["quintile"] == "Q5"].groupby("seed")["value"].mean()
            t_seed = captions[m][captions[m]["quintile"] == "Q5"].groupby("seed")["value"].mean()
            i2t_means += i_seed.tolist()
            t2i_means += t_seed.tolist()
        test = wilcoxon(i2t_means, t2i_means, zero_method="wilcox", alternative="two-sided")
        rows.append({"dataset": dataset, "model": "ALL", "quintile": "Q5", "mean_i2t": float(np.mean(i2t_means)),
                     "mean_t2i": float(np.mean(t2i_means)), "wilcoxon_W": float(test.statistic), "wilcoxon_p": float(test.pvalue),
                     "n_model_seed_pairs": len(i2t_means)})
    pd.DataFrame(rows).to_csv(file_paths.NUM_PAIRED_TESTS, index=False)
    print(f"saved {file_paths.NUM_PAIRED_TESTS.relative_to(file_paths.ROOT)}")


# Measure agreement between seeds.
def build_cross_seed_spearman() -> None:
    """Correlate seed-level Recall@1 means across model, direction, corruption and severity.

    Exclude the typo corruption. Compare the condition means because the seeds select
    different, overlapping sets of pairs.
    """
    rows = []
    for dataset in datasets_with_headline_rows():
        hits = per_pair_retrieval(dataset)
        hits = hits[hits["metric"].isin(["per_pair_recall_at_1_i2t", "per_pair_recall_at_1_t2i"])
                    & hits["text_corruption"].isin(("none",) + settings.BINNED_TEXT_CORRUPTIONS)]
        cells = ["model", "metric", "vision_corruption", "image_severity", "text_corruption", "text_severity"]
        means = hits.groupby(cells + ["seed"])["value"].mean().reset_index().pivot_table(index=cells, columns="seed", values="value")
        seeds = sorted(means.columns)
        for i, a in enumerate(seeds):
            for b in seeds[i + 1:]:
                both = means[a].notna() & means[b].notna()
                rho, p = spearmanr(means[a][both].to_numpy(), means[b][both].to_numpy())
                rows.append({"dataset": dataset, "seed_a": int(a), "seed_b": int(b), "spearman_rho": float(rho), "p": float(p),
                             "n_cells": int(both.sum())})
    pd.DataFrame(rows).to_csv(file_paths.NUM_CROSS_SEED_SPEARMAN, index=False)
    print(f"saved {file_paths.NUM_CROSS_SEED_SPEARMAN.relative_to(file_paths.ROOT)}")


# Bin match retention by input damage.
def _retention_quintile_rows(dataset: str) -> pd.DataFrame:
    rows = fidelity_retention_rows(dataset)
    margin = rows[(rows["spoke"] == "match_retention") & (rows["metric"] == "retention_margin")]
    dmg = fidelity_retention_damage(dataset)
    sides = {}
    for side, kind_col, severity_col, ruler in (("image", "vision_corruption", "image_severity", "damage_ssim"),
                                                ("text", "text_corruption", "text_severity", "damage_bleu")):
        sub = margin[margin["match_retention_direction"] == f"{side}_corrupted"][
            ["seed", "model", "pair_id", kind_col, severity_col, "value"]].copy()
        for c in ("seed", "pair_id", severity_col):
            sub[c] = pd.to_numeric(sub[c])
        side_dmg = dmg[dmg["modality"] == side][["seed", "pair_id", "corruption_type", "severity", ruler]]
        side_dmg = side_dmg.rename(columns={"corruption_type": kind_col, "severity": severity_col})
        sides[side] = quintiles_by_model(sub.merge(side_dmg, on=["seed", "pair_id", kind_col, severity_col]), ruler)
    out = []
    for m in MODELS:
        for q in QUINTILES:
            (i_mean, i_lo, i_hi), (t_mean, t_lo, t_hi) = (
                bootstrap_ci(sides[s][m][sides[s][m]["quintile"] == q]) for s in ("image", "text"))
            out.append({"model": m, "quintile": q, "image_mean": i_mean, "image_lo": i_lo, "image_hi": i_hi,
                        "text_mean": t_mean, "text_lo": t_lo, "text_hi": t_hi, "image_minus_text": i_mean - t_mean})
    return pd.DataFrame(out)


def build_retention_by_damage_quintile() -> None:
    """Summarize the match retention and its 95% interval within each model's damage quintiles.

    Rank the damaged images by SSIM and the damaged captions by BLEU, within each dataset.
    """
    table = _with_dataset_column({d: _retention_quintile_rows(d) for d in datasets_with_300_pair_run()})
    table.to_csv(file_paths.NUM_RETENTION_BY_DAMAGE_QUINTILE, index=False)
    print(f"saved {file_paths.NUM_RETENTION_BY_DAMAGE_QUINTILE.relative_to(file_paths.ROOT)}")


def build_retention_image_minus_caption_by_level() -> None:
    """Compare image and caption match retention at clean and at each severity, with a paired interval.

    The clean level shows whether a gap exists before any damage. Each severity averages that
    side's binned corruptions, and both sides are scored on the same pairs.
    """
    rows = []
    statistics = {"image_margin": lambda g: g["image"], "caption_margin": lambda g: g["text"],
                  "image_minus_caption": lambda g: g["image"] - g["text"]}
    for dataset in datasets_with_300_pair_run():
        table = fidelity_retention_rows(dataset)
        margin = table[(table["spoke"] == "match_retention") & (table["metric"] == "retention_margin")].copy()
        for c in ("seed", "pair_id", "image_severity", "text_severity"):
            margin[c] = pd.to_numeric(margin[c])
        for m in MODELS:
            cells = margin[margin["model"] == m]
            direction = cells["match_retention_direction"]
            levels = {0: (cells[direction == "clean_image_query"], cells[direction == "clean_caption_query"])}
            for level in settings.SEVERITIES:
                levels[level] = (cells[(direction == "image_corrupted") & (cells["image_severity"] == level)],
                                 cells[(direction == "text_corrupted") & (cells["text_severity"] == level)])
            for level, (image, text) in levels.items():
                row = {"dataset": dataset, "model": m, "level": level}
                for name, (point, lo, hi) in bootstrap_group_statistics({"image": image, "text": text}, statistics).items():
                    row.update({name: point, f"{name}_lo": lo, f"{name}_hi": hi})
                rows.append(row)
    pd.DataFrame(rows).to_csv(file_paths.NUM_RETENTION_IMAGE_MINUS_CAPTION, index=False)
    print(f"saved {file_paths.NUM_RETENTION_IMAGE_MINUS_CAPTION.relative_to(file_paths.ROOT)}")


def build_typo_fidelity_and_retention() -> None:
    """Report typo's caption fidelity at each sampled depth and its match retention, by model and severity.

    The typo rows stay out of the fidelity and retention readers, so this summary holds them for the release.
    """
    rows = []
    for dataset in datasets_with_300_pair_run():
        typo = typo_fidelity_retention_rows(dataset).copy()
        typo["text_severity"] = pd.to_numeric(typo["text_severity"]).astype(int)
        fidelity = typo[typo["spoke"] == "text_fidelity"].groupby(["model", "text_severity", "depth"])["value"].mean().unstack("depth")
        retention = typo[(typo["spoke"] == "match_retention") & (typo["metric"] == "retention_margin")].groupby(
            ["model", "text_severity"])["value"].mean()
        for m in MODELS:
            for severity in settings.SEVERITIES:
                rows.append({"dataset": dataset, "model": m, "severity": severity, "retention_margin": retention[(m, severity)],
                             **{f"fidelity_{depth}": fidelity.loc[(m, severity), depth] for depth in ("early", "mid", "late")}})
    pd.DataFrame(rows).to_csv(file_paths.NUM_TYPO_FIDELITY_RETENTION, index=False)
    print(f"saved {file_paths.NUM_TYPO_FIDELITY_RETENTION.relative_to(file_paths.ROOT)}")


# Check robustness on the 2,000-pair runs.
def _per_pair_with_quintiles(hits: pd.DataFrame, dmg: pd.DataFrame, text_kinds, text_ruler: str,
                             image_part: pd.DataFrame | None = None, image_ruler: str = "damage_ssim") -> pd.DataFrame:
    """Join the Recall@1 hits to damage quintiles, pooling models because they share the input damage."""
    def side(direction, modality, kind_col, severity_col, kinds, ruler):
        d = dmg[(dmg["modality"] == modality) & dmg["corruption_type"].isin(kinds)].copy()
        d["damage"] = d[ruler]
        d["quintile"] = np.asarray(damage_quintile(d["damage"].to_numpy()))
        h = hits[hits["metric"] == f"per_pair_recall_at_1_{direction}"].rename(
            columns={kind_col: "corruption_type", severity_col: "severity"})
        h = h[h["corruption_type"].isin(kinds)].merge(d[["seed", "pair_id", "corruption_type", "severity", "damage", "quintile"]],
                                                      on=["seed", "pair_id", "corruption_type", "severity"])
        return h.assign(direction=direction)

    images = image_part if image_part is not None else side(
        "i2t", "image", "vision_corruption", "image_severity", settings.IMAGE_CORRUPTIONS, image_ruler)
    captions = side("t2i", "text", "text_corruption", "text_severity", text_kinds, text_ruler)
    keep = ["model", "seed", "pair_id", "corruption_type", "severity", "damage", "quintile", "direction", "value"]
    return pd.concat([images[keep], captions[keep]], ignore_index=True)


def _quintile_table(per_pair: pd.DataFrame) -> pd.DataFrame:
    table = per_pair.groupby(["model", "quintile", "direction"], observed=True)["value"].mean().unstack("direction").reset_index()
    return _ratio_rows(table)


def _flatten(table: pd.DataFrame) -> pd.DataFrame:
    """Flatten the column names, turning ('mean', 'i2t') into 'mean_i2t' and ('ratio', '') into 'ratio'."""
    table = table.copy()
    table.columns = ["_".join(str(part) for part in col if part != "") for col in table.columns]
    return table.reset_index()


def _caption_word_bounds(table: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    """Replace caption-length intervals with inclusive integer word bounds."""
    position = table.columns.get_loc("len_bin")
    codes = table["len_bin"].cat.codes.to_numpy()
    lower = np.floor(edges[:-1]).astype(int) + 1
    lower[0] = int(np.ceil(edges[0]))
    upper = np.floor(edges[1:]).astype(int)
    table = table.drop(columns="len_bin")
    table.insert(position, "caption_words_min", lower[codes])
    table.insert(position + 1, "caption_words_max", upper[codes])
    return table


def _retrieval_pair_lookup(dataset: str, values: list) -> dict[tuple[int, int], object]:
    """Map (seed, pair_id) to a per-picture value using that seed's retrieval draw."""
    out = {}
    for seed in settings.SEEDS:
        for pair_id, position in enumerate(retrieval_positions(len(values), seed, max(settings.GALLERY_SIZES))):
            out[(seed, pair_id)] = values[position]
    return out


def recall_by_level(df: pd.DataFrame, direction: str, kind: str) -> pd.DataFrame:
    """Summarize full-gallery Recall@1 by model and level for one corruption.

    Use the rows from full_gallery_rows. Column "0" holds the clean recall, and columns
    "1" through "5" hold the corrupted recall. Use i2t for image corruptions and t2i for text.
    """
    kind_col, level_col = ("vision_corruption", "image_severity") if direction == "i2t" else ("text_corruption", "text_severity")
    damaged = df[(df["spoke"] == "retrieval") & (df["metric"] == f"recall_at_1_{direction}") & (df[kind_col] == kind)]
    table = damaged.groupby(["model", level_col])["value"].mean().unstack()
    table.columns = [str(int(c)) for c in table.columns]
    clean = df[(df["spoke"] == "retrieval_gallery_sweep") & (df["vision_corruption"] == "none") & (df["text_corruption"] == "none")
               & (df["metric"] == f"recall_at_1_{direction}_gallery{int(df['n_eval'].max())}")]
    table.insert(0, "0", clean.groupby("model")["value"].mean())
    return table.reset_index().assign(corruption=kind)[["corruption", "model"] + [str(s) for s in (0,) + tuple(settings.SEVERITIES)]]


def analyze_robustness(dataset: str) -> dict:
    """Build the robustness tables for a dataset, keyed by their output files."""
    df, dmg = full_gallery_rows(dataset), full_gallery_damage(dataset)
    hits = df[df["spoke"] == "retrieval_per_pair"]
    whole = df[df["spoke"] == "retrieval"]
    binned = settings.BINNED_TEXT_CORRUPTIONS
    tables = {}

    # Compare the matched quintiles under each caption ruler, without and with typo.
    image_part = None
    per_pair, with_typo, placement = {}, [], []
    for ruler, column in TEXT_RULERS.items():
        per_pair[ruler] = _per_pair_with_quintiles(hits, dmg, binned, column, image_part)
        image_part = per_pair[ruler][per_pair[ruler]["direction"] == "i2t"]
        with_typo.append(_quintile_table(_per_pair_with_quintiles(hits, dmg, binned + ("typo",), column, image_part))
                         .assign(text_ruler=ruler))
        # Rank typo's median damage against the binned corruptions.
        binned_damage = np.sort(dmg[(dmg["modality"] == "text") & dmg["corruption_type"].isin(binned)][column].to_numpy())
        for severity, group in dmg[(dmg["modality"] == "text") & (dmg["corruption_type"] == "typo")].groupby("severity"):
            median = group[column].median()
            placement.append({"text_ruler": ruler, "severity": severity, "median_damage": median,
                              "percentile": 100.0 * np.searchsorted(binned_damage, median) / len(binned_damage)})
    tables[file_paths.TAB_QUINTILES_GALLERY2000] = pd.concat(
        [_quintile_table(per_pair[r]).assign(text_ruler=r) for r in TEXT_RULERS], ignore_index=True)
    tables[file_paths.TAB_QUINTILES_WITH_TYPO] = pd.concat(with_typo, ignore_index=True)
    tables[file_paths.TAB_TYPO_DAMAGE_PERCENTILE] = pd.DataFrame(placement)
    tables[file_paths.FIG_RECALL_BY_TYPO_SEVERITY_DATA] = recall_by_level(df, "t2i", "typo")

    # Full-gallery damaged rows need the same metric suffix as the gallery-sweep rows.
    full = whole.assign(metric=whole["metric"] + "_gallery" + whole["n_eval"].astype(int).astype(str))
    sweep = pd.concat([df[df["spoke"] == "retrieval_gallery_sweep"], full], ignore_index=True)
    sweep["gallery_size"] = sweep["metric"].str.extract(r"_gallery(\d+)$")[0].astype(int)
    sweep["base"] = sweep["metric"].str.replace(r"_gallery\d+$", "", regex=True)
    sizes = []
    for (model, gallery_size), g in sweep.groupby(["model", "gallery_size"]):
        def cell(base, image_kind, text_kind, image_sev, text_sev):
            return g[(g["base"] == base) & (g["vision_corruption"] == image_kind) & (g["text_corruption"] == text_kind)
                     & (g["image_severity"] == image_sev) & (g["text_severity"] == text_sev)]["value"].mean()
        sizes.append({"model": model, "gallery_size": gallery_size, "clean_i2t": cell("recall_at_1_i2t", "none", "none", 0, 0),
                      "clean_t2i": cell("recall_at_1_t2i", "none", "none", 0, 0),
                      "noise5_i2t": cell("recall_at_1_i2t", "gaussian_noise", "none", 5, 0),
                      "mask5_t2i": cell("recall_at_1_t2i", "none", "mask", 0, 5)})
    sizes = pd.DataFrame(sizes).sort_values(["model", "gallery_size"])
    sizes["asym_ratio"] = sizes["noise5_i2t"] / sizes["mask5_t2i"]
    tables[file_paths.TAB_GALLERY_SIZE_SWEEP] = sizes

    # Summarize the shared-space similarity by condition and Recall@1 within similarity bins.
    keys = ["model", "direction", "corruption_type", "severity"]
    similarity = _retrieval_conditions(df[(df["spoke"] == "retrieval_selfsim") & df["metric"].str.startswith("query_selfsim")])
    cells = similarity.groupby(keys)["value"].mean().rename("selfsim").reset_index()
    recall = _retrieval_conditions(whole[whole["metric"].isin(["recall_at_1_i2t", "recall_at_1_t2i"])])
    cells = cells.merge(recall.groupby(keys)["value"].mean().rename("r_at_1").reset_index(), on=keys)
    tables[file_paths.TAB_SIMILARITY_BY_CORRUPTION] = cells
    bands = pd.cut(cells["selfsim"], [0, 0.6, 0.7, 0.8, 0.9, 1.0], labels=["<0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", ">0.90"])
    band_table = (cells.assign(band=bands).groupby(["band", "direction"], observed=True)
                  .agg(mean_r1=("r_at_1", "mean"), n_cells=("r_at_1", "size")).reset_index().pivot(index="band", columns="direction"))
    band_table = _flatten(band_table)
    counts = [column for column in band_table.columns if column.startswith("n_cells_")]
    band_table[counts] = band_table[counts].astype("Int64")
    tables[file_paths.TAB_RECALL_BY_SIMILARITY_BAND] = band_table

    # Compare the Q5 asymmetry across caption-length bins.
    with open(file_paths.manifest(dataset)) as f:
        captions = [entry["caption"] for entry in json.load(f)]
    words = _retrieval_pair_lookup(dataset, [len(c.split()) for c in captions])
    edges = np.unique(np.quantile(list(words.values()), [0, 0.25, 0.5, 0.75, 1.0]))
    all_models, per_model = [], []
    for ruler in TEXT_RULERS:
        q5 = per_pair[ruler][per_pair[ruler]["quintile"] == "Q5"].copy()
        q5["words"] = [words.get((int(s), int(p))) for s, p in zip(q5["seed"], q5["pair_id"])]
        q5["len_bin"] = pd.cut(q5["words"], edges, include_lowest=True)
        lengths = q5.groupby(["len_bin", "direction"], observed=True)["value"].agg(["mean", "size"]).unstack("direction")
        lengths[("ratio", "")] = lengths[("mean", "i2t")] / lengths[("mean", "t2i")]
        all_models.append(_caption_word_bounds(_flatten(lengths), edges).assign(text_ruler=ruler))
        models = q5.groupby(["model", "len_bin", "direction"], observed=True)["value"].mean().unstack("direction")
        per_model.append(_caption_word_bounds(_ratio_rows(models).reset_index(), edges).assign(text_ruler=ruler))
    tables[file_paths.TAB_CAPTION_LENGTH] = pd.concat(all_models, ignore_index=True)
    tables[file_paths.TAB_CAPTION_LENGTH_PER_MODEL] = pd.concat(per_model, ignore_index=True)

    # Compare PE-Core's Q5 asymmetry with and without captions that exceed its 32-token context.
    cut_short = _retrieval_pair_lookup(dataset, (pecore_content_tokens(captions) > PE_CORE_CONTENT_TOKENS).tolist())
    truncation = []
    for ruler in TEXT_RULERS:
        pe = per_pair[ruler][per_pair[ruler]["model"] == "pecore_l14_336"].copy()
        pe["truncated"] = [cut_short.get((int(s), int(p)), False) for s, p in zip(pe["seed"], pe["pair_id"])]
        for subset, rows in (("all pairs", pe), ("without truncated captions", pe[~pe["truncated"]])):
            means = rows[rows["quintile"] == "Q5"].groupby("direction")["value"].mean()
            truncation.append({"text_ruler": ruler, "subset": subset, "q5_i2t": means.get("i2t"), "q5_t2i": means.get("t2i"),
                               "ratio": means.get("i2t") / means.get("t2i"), "q5_rows": int((rows["quintile"] == "Q5").sum())})
    tables[file_paths.TAB_PECORE_TRUNCATION_ROBUSTNESS] = pd.DataFrame(truncation)
    return tables


def build_robustness_checks() -> None:
    """Write robustness tables for both datasets and plot typo curves."""
    merged: dict = {}
    for dataset in DATASETS:
        for path, table in analyze_robustness(dataset).items():
            merged.setdefault(path, []).append(table.assign(dataset=dataset))
    for path, frames in merged.items():
        table = pd.concat(frames, ignore_index=True)
        if "model" in table.columns:
            table = _sort_model_rows(table)
        first = ["dataset"] + (["text_ruler"] if "text_ruler" in table.columns else [])
        table[first + [c for c in table.columns if c not in first]].to_csv(path, index=False)
        print(f"saved {path.relative_to(file_paths.ROOT)}")
    draw_typo_recall_by_level(pd.concat(merged[file_paths.FIG_RECALL_BY_TYPO_SEVERITY_DATA]), file_paths.FIG_RECALL_BY_TYPO_SEVERITY)


def build_quintiles_at_headline_gallery() -> None:
    """Write both datasets' damage quintiles at the headline gallery, so they compare like for like."""
    tables = []
    for dataset in DATASETS:
        hits = per_pair_retrieval(dataset)
        if hits.empty:
            note_skipped(f"no {dataset} per-query rows at the headline gallery, so its quintiles there are left out")
            continue
        dmg, image_part = headline_damage(dataset), None
        for ruler, column in TEXT_RULERS.items():
            per_pair = _per_pair_with_quintiles(hits, dmg, settings.BINNED_TEXT_CORRUPTIONS, column, image_part)
            image_part = per_pair[per_pair["direction"] == "i2t"]
            tables.append(_quintile_table(per_pair).assign(dataset=dataset, text_ruler=ruler))
    table = _sort_model_rows(pd.concat(tables, ignore_index=True))
    first = ["dataset", "text_ruler"]
    table[first + [c for c in table.columns if c not in first]].to_csv(file_paths.TAB_QUINTILES_GALLERY1000, index=False)
    print(f"saved {file_paths.TAB_QUINTILES_GALLERY1000.relative_to(file_paths.ROOT)}")


def _q5_rows(per_pair: pd.DataFrame, **labels) -> list[dict]:
    """Return each model's Q5 means and ratio, with the labels that name the protocol."""
    means = per_pair[per_pair["quintile"] == "Q5"].groupby(["model", "direction"], observed=True)["value"].mean().unstack("direction")
    return [{**labels, "model": m, "q5_i2t": means.loc[m, "i2t"], "q5_t2i": means.loc[m, "t2i"],
             "ratio": means.loc[m, "i2t"] / means.loc[m, "t2i"]} for m in MODELS]


def build_q5_ratio_by_protocol() -> None:
    """Report the Q5 i2t to t2i ratio at each gallery under each pair of image and caption rulers, on both datasets.

    pandas puts rows tied at a quintile edge in the lower bin. Captions tied at the Q5 edge can be
    many, so one extra row per dataset shows the headline protocol with those captions in Q5.
    """
    rows = []
    for dataset in DATASETS:
        whole = full_gallery_rows(dataset)
        whole = whole[whole["spoke"] == "retrieval_per_pair"]
        galleries = ((settings.HEADLINE_GALLERY_SIZE, per_pair_retrieval(dataset), headline_damage(dataset)),
                     (max(settings.GALLERY_SIZES), whole, full_gallery_damage(dataset)))
        for gallery, hits, dmg in galleries:
            for image_name, image_ruler in IMAGE_RULERS.items():
                image_part = None
                for text_name, text_ruler in TEXT_RULERS.items():
                    per_pair = _per_pair_with_quintiles(hits, dmg, settings.BINNED_TEXT_CORRUPTIONS, text_ruler, image_part, image_ruler)
                    image_part = per_pair[per_pair["direction"] == "i2t"]
                    labels = {"dataset": dataset, "gallery_size": gallery, "image_ruler": image_name, "text_ruler": text_name}
                    rows += _q5_rows(per_pair, **labels, caption_ties_at_q5_edge="lower bin")
                    if gallery == settings.HEADLINE_GALLERY_SIZE and image_name == "ssim" and text_name == "bleu":
                        text = dmg[(dmg["modality"] == "text") & dmg["corruption_type"].isin(settings.BINNED_TEXT_CORRUPTIONS)]
                        edge = text[text_ruler].quantile(0.8)
                        tied = per_pair.assign(quintile=per_pair["quintile"].astype(object))
                        tied.loc[(tied["direction"] == "t2i") & (tied["damage"] == edge), "quintile"] = "Q5"
                        rows += _q5_rows(tied, **labels, caption_ties_at_q5_edge="Q5")
    pd.DataFrame(rows).to_csv(file_paths.TAB_Q5_RATIO_BY_PROTOCOL, index=False)
    print(f"saved {file_paths.TAB_Q5_RATIO_BY_PROTOCOL.relative_to(file_paths.ROOT)}")


# Report object cutout separately from the headline bins.
OBJECT_CUTOUT_RUN = "coco_2000_pairs_retrieval_object_cutout"


def _object_cutout_areas() -> pd.DataFrame:
    """Return the area that object cutout blacked out in each image at each severity.

    The areas need the images, so replay them from the run's seeds and the COCO image manifest when it
    exists and save them. A rebuild without that manifest reads the saved areas.
    """
    if not file_paths.full_manifest("coco").exists():
        if file_paths.OBJECT_CUTOUT_AREAS.exists():
            return pd.read_csv(file_paths.OBJECT_CUTOUT_AREAS, float_precision="round_trip")
        note_skipped("no COCO manifest with images and no saved areas, so object cutout's blacked-out areas are left out")
        return pd.DataFrame(columns=["seed", "pair_id", "severity", "area"])

    from src.corruptions import object_cutout_mask, pair_rngs
    from src.evaluate_model import object_outlines_by_image
    from src.load_data import retrieval_pairs

    n_pairs = settings.RUNS[OBJECT_CUTOUT_RUN]["pairs"]
    objects = object_outlines_by_image("coco")
    rows = []
    for seed in settings.SEEDS:
        pairs = retrieval_pairs("coco", seed, n_pairs)
        for severity in settings.SEVERITIES:
            for pair_id, (pair, rng) in enumerate(zip(pairs, pair_rngs("retrieval", seed, severity, n_pairs, "image"))):
                mask = object_cutout_mask(*pair.image.size, severity, rng=rng, objects=objects.get(pair.image_id, []))
                rows.append({"seed": seed, "pair_id": pair_id, "severity": severity, "area": float(mask.mean())})
    areas = pd.DataFrame(rows)
    areas.to_csv(file_paths.OBJECT_CUTOUT_AREAS, index=False)
    print(f"saved {file_paths.OBJECT_CUTOUT_AREAS.relative_to(file_paths.ROOT)}")
    return areas


def build_object_cutout() -> None:
    """Compare object and rectangle cutout at the full gallery, with damage and area summaries.

    Whole-object masks may overshoot the target area or exhaust the objects below it.
    """
    if not (file_paths.run_output(OBJECT_CUTOUT_RUN).exists() and file_paths.damage_output(OBJECT_CUTOUT_RUN).exists()):
        note_skipped(f"no {OBJECT_CUTOUT_RUN} output saved, so object cutout is skipped")
        return
    recall = pd.concat([recall_by_level(full_gallery_rows("coco", OBJECT_CUTOUT_RUN), "i2t", "object_cutout"),
                        recall_by_level(full_gallery_rows("coco"), "i2t", "cutout")], ignore_index=True)
    recall.insert(0, "dataset", "coco")
    recall = _sort_model_rows(recall)
    recall.to_csv(file_paths.FIG_RECALL_BY_OBJECT_CUTOUT_SEVERITY_DATA, index=False)
    print(f"saved {file_paths.FIG_RECALL_BY_OBJECT_CUTOUT_SEVERITY_DATA.relative_to(file_paths.ROOT)}")
    targets = ", ".join(f"{100 * strength('image', 'object_cutout', severity):.0f}" for severity in settings.SEVERITIES)
    draw_object_and_rectangle_cutout(recall, file_paths.FIG_RECALL_BY_OBJECT_CUTOUT_SEVERITY, f"Cutout severity (target area {targets}% of the image)")

    binned = full_gallery_damage("coco")
    binned_damage = np.sort(binned[(binned["modality"] == "image")
                                   & binned["corruption_type"].isin(settings.IMAGE_CORRUPTIONS)]["damage_ssim"].to_numpy())
    objects = full_gallery_damage("coco", OBJECT_CUTOUT_RUN)
    areas = _object_cutout_areas()
    rows = []
    for severity, group in objects[objects["corruption_type"] == "object_cutout"].groupby("severity"):
        median = group["damage_ssim"].median()
        area = areas[areas["severity"] == severity]["area"]
        target = strength("image", "object_cutout", int(severity))
        rows.append({"dataset": "coco", "severity": severity, "median_damage_ssim": median,
                     "damage_percentile": 100.0 * np.searchsorted(binned_damage, median) / len(binned_damage),
                     "target_area": target, "mean_area": area.mean(), "median_area": area.median(),
                     "area_p10": area.quantile(0.1), "area_p90": area.quantile(0.9),
                     "share_below_target": (area < target).mean() if len(area) else np.nan})
    pd.DataFrame(rows).to_csv(file_paths.TAB_OBJECT_CUTOUT_DAMAGE_AND_AREA, index=False)
    print(f"saved {file_paths.TAB_OBJECT_CUTOUT_DAMAGE_AND_AREA.relative_to(file_paths.ROOT)}")
    _object_cutout_against_captions(areas)


def _object_cutout_against_captions(areas: pd.DataFrame) -> None:
    """Compare object cutout with damaged captions at the headline gallery, level by level and on the Q5 edge.

    Areas and word shares are different rulers, and achieved areas do not match their targets, so
    neither comparison is a matched-damage result. The edge comparison keeps only the object cutout
    rows beyond the headline image Q5 edge, under SSIM or PSNR, and divides by the caption Q5 Recall@1.
    """
    objects = object_cutout_per_pair_at_headline_gallery(OBJECT_CUTOUT_RUN)
    hits = per_pair_retrieval("coco")
    rectangle = hits[(hits["metric"] == "per_pair_recall_at_1_i2t") & (hits["vision_corruption"] == "cutout")]
    captions = hits[hits["metric"] == "per_pair_recall_at_1_t2i"]
    level_rows = []
    for m in MODELS:
        for severity in settings.SEVERITIES:
            row = {"dataset": "coco", "gallery_size": settings.HEADLINE_GALLERY_SIZE, "model": m, "severity": severity,
                   "object_cutout_i2t": objects[(objects["model"] == m) & (objects["image_severity"] == severity)]["value"].mean(),
                   "rectangle_cutout_i2t": rectangle[(rectangle["model"] == m) & (rectangle["image_severity"] == severity)]["value"].mean()}
            for kind in ("mask", "replace"):
                row[f"{kind}_t2i"] = captions[(captions["model"] == m) & (captions["text_corruption"] == kind)
                                              & (captions["text_severity"] == severity)]["value"].mean()
                row[f"object_over_{kind}"] = row["object_cutout_i2t"] / row[f"{kind}_t2i"]
            level_rows.append(row)
    pd.DataFrame(level_rows).to_csv(file_paths.NUM_OBJECT_CUTOUT_BY_LEVEL, index=False)
    print(f"saved {file_paths.NUM_OBJECT_CUTOUT_BY_LEVEL.relative_to(file_paths.ROOT)}")

    binned = headline_damage("coco")
    binned = binned[(binned["modality"] == "image") & binned["corruption_type"].isin(settings.IMAGE_CORRUPTIONS)]
    damage = full_gallery_damage("coco", OBJECT_CUTOUT_RUN)
    damage = damage[(damage["corruption_type"] == "object_cutout") & (damage["pair_id"] < settings.HEADLINE_GALLERY_SIZE)]
    joined = objects.merge(damage, left_on=["seed", "pair_id", "image_severity"], right_on=["seed", "pair_id", "severity"]).merge(
        areas.astype({"seed": int, "pair_id": int, "severity": int}), on=["seed", "pair_id", "severity"], how="left")
    q5_t2i = quintile_summary("ssim", "coco").set_index(["model", "quintile"])["t2i"]
    edge_rows = []
    for image_ruler, column in IMAGE_RULERS.items():
        edge = binned[column].quantile(0.8)
        for m in MODELS:
            cell = joined[(joined["model"] == m) & (joined[column] > edge)]
            total = int((joined["model"] == m).sum())
            edge_rows.append({"dataset": "coco", "gallery_size": settings.HEADLINE_GALLERY_SIZE, "image_ruler": image_ruler, "model": m,
                              "q5_edge": edge, "object_rows_beyond_edge": len(cell), "object_rows": total, "object_cutout_i2t": cell["value"].mean(),
                              "caption_q5_t2i": q5_t2i[(m, "Q5")], "ratio": cell["value"].mean() / q5_t2i[(m, "Q5")],
                              "mean_area_beyond_edge": cell["area"].mean()})
    pd.DataFrame(edge_rows).to_csv(file_paths.NUM_OBJECT_CUTOUT_ON_QUINTILE_EDGE, index=False)
    print(f"saved {file_paths.NUM_OBJECT_CUTOUT_ON_QUINTILE_EDGE.relative_to(file_paths.ROOT)}")


# Compare retrieval within shared-space similarity bins.
MIN_QUERIES_FOR_CORRUPTION_RATE = 500  # Rates from fewer queries in a bin are too noisy to compare.


def _direction_gap_rows(joined: pd.DataFrame, dataset: str) -> list[dict]:
    """Measure each model's i2t minus t2i Recall@1 within the same similarity bins.

    Weight each bin that holds both directions by the smaller of its two query counts. The interval
    resamples seeds and then pairs with those weights fixed, and skips any bin a draw leaves empty.
    """
    rows = []
    for m in MODELS:
        cell = joined[joined["model"] == m]
        counts = cell.groupby(["bin", "direction"], observed=True).size().unstack("direction").dropna()
        weights = np.minimum(counts["i2t"], counts["t2i"]).to_numpy(dtype=float)
        groups = {}
        for i, interval in enumerate(counts.index):
            for direction in ("i2t", "t2i"):
                groups[f"{direction}{i}"] = cell[(cell["bin"] == interval) & (cell["direction"] == direction)]

        def weighted_gap(means: dict[str, float]) -> float:
            gaps = np.array([means[f"i2t{i}"] - means[f"t2i{i}"] for i in range(len(weights))])
            usable = ~np.isnan(gaps)
            return float(np.average(gaps[usable], weights=weights[usable])) if usable.any() else float("nan")

        gap, lo, hi = bootstrap_group_statistics(groups, {"gap": weighted_gap}, value_col="success")["gap"]
        # Bins with few queries in one direction can show very large gaps, so the range leaves them out.
        by_bin = [groups[f"i2t{i}"]["success"].mean() - groups[f"t2i{i}"]["success"].mean()
                  for i in range(len(weights)) if weights[i] >= MIN_QUERIES_FOR_CORRUPTION_RATE]
        rows.append({"dataset": dataset, "model": m, "direction_gap": gap, "direction_gap_lo": lo, "direction_gap_hi": hi,
                     "bins_with_both_directions": len(weights), "bins_for_range": len(by_bin),
                     "smallest_bin_gap": min(by_bin), "largest_bin_gap": max(by_bin), "queries": len(cell)})
    return rows


def build_recall_at_1_by_displacement_bin() -> None:
    """Report Recall@1 within shared-space similarity bins for both datasets.

    Save the pooled rates, the rates for each corruption, the rates for each model and corruption,
    and each model's direction gap within those bins, along with their counts.
    """
    tables, by_corruption, detailed, gaps = [], [], [], []
    for dataset in DATASETS:
        df = full_gallery_rows(dataset)
        keys = ["model", "seed", "pair_id", "direction", "corruption", "severity"]

        def per_query(frame, name):
            frame = _retrieval_conditions(frame).rename(columns={"corruption_type": "corruption"})
            return frame[keys + ["value"]].rename(columns={"value": name})
        hits = per_query(df[(df["spoke"] == "retrieval_per_pair")
                            & df["metric"].isin(["per_pair_recall_at_1_i2t", "per_pair_recall_at_1_t2i"])], "success")
        similarity = per_query(df[(df["spoke"] == "retrieval_selfsim") & df["metric"].str.startswith("per_pair_query_selfsim")], "selfsim")
        joined = hits.merge(similarity, on=keys)
        # Round edges to retain exact 1.0 similarities. A wider first bin includes values below 0.30.
        joined["bin"] = pd.cut(joined["selfsim"], [-1.0] + list(np.round(np.arange(0.30, 1.0001, 0.05), 2)))
        table = joined.groupby(["bin", "direction"], observed=True).agg(rate=("success", "mean"), n=("success", "size")).reset_index()
        tables.append(table.assign(dataset=dataset)[["dataset", "bin", "direction", "rate", "n"]])
        split = ["direction", "corruption", "bin"]
        table = joined.groupby(split, observed=True).agg(rate=("success", "mean"), n=("success", "size")).reset_index()
        by_corruption.append(table.assign(dataset=dataset)[["dataset"] + split + ["rate", "n"]])
        split = ["model", "direction", "corruption", "bin"]
        table = joined.groupby(split, observed=True).agg(rate=("success", "mean"), n=("success", "size")).reset_index()
        detailed.append(_sort_model_rows(table.assign(dataset=dataset))[["dataset"] + split + ["rate", "n"]])
        gaps += _direction_gap_rows(joined, dataset)
    for path, frames in ((file_paths.FIG_RECALL_BY_DISPLACEMENT_BIN, tables),
                         (file_paths.TAB_DISPLACEMENT_BY_CORRUPTION, by_corruption),
                         (file_paths.FIG_RECALL_BY_DISPLACEMENT_BIN_BY_MODEL, detailed)):
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        print(f"saved {path.relative_to(file_paths.ROOT)}")
    pd.DataFrame(gaps).to_csv(file_paths.TAB_DISPLACEMENT_RESIDUAL, index=False)
    print(f"saved {file_paths.TAB_DISPLACEMENT_RESIDUAL.relative_to(file_paths.ROOT)}")


# Measure the intervals of the means that some figures draw without error bars.
UNDRAWN_INTERVAL_DRAWS = 2000  # Captions state only a rough size, so fewer draws keep this step short.


def _largest_half_width(cells: list[pd.DataFrame]) -> float:
    """Return the largest distance from a mean to either end of its 95% interval."""
    intervals = [bootstrap_ci(cell, n_boot=UNDRAWN_INTERVAL_DRAWS) for cell in cells]
    return max(max(mean - lo, hi - mean) for mean, lo, hi in intervals)


def build_undrawn_interval_sizes() -> None:
    """Report the largest 95% half-width behind each figure's means that carry no error bar, for its caption."""
    rows = []
    for dataset in datasets_with_300_pair_run():
        fidelity = _fidelity_rows(dataset)
        cells = [cell for _, cell in fidelity.groupby(["spoke", "model", "severity", "depth"])]
        rows.append({"output": "fidelity_by_layer_depth", "dataset": dataset, "unit": "cosine", "means": len(cells), "largest_half_width": _largest_half_width(cells)})
        joint = _joint_retention_rows(dataset)
        joint[["seed", "pair_id"]] = joint[["seed", "pair_id"]].apply(pd.to_numeric)
        cells = [cell for _, cell in joint.groupby(["model", "image_severity", "text_severity"])]
        rows.append({"output": "retention_joint_noise_mask_heatmaps", "dataset": dataset, "unit": "retention margin", "means": len(cells),
                     "largest_half_width": _largest_half_width(cells)})
    for dataset in datasets_with_headline_rows():
        clean = clean_recall(dataset)
        rows.append({"output": "recall_at_1_by_damage_quintile clean lines", "dataset": dataset, "unit": "Recall@1", "means": len(clean),
                     "largest_half_width": float(np.maximum(clean["mean"] - clean["lo"], clean["hi"] - clean["mean"]).max())})
    for dataset in DATASETS:
        whole = full_gallery_rows(dataset)
        typo = whole[(whole["spoke"] == "retrieval_per_pair") & (whole["metric"] == "per_pair_recall_at_1_t2i") & (whole["text_corruption"] == "typo")]
        clean = whole[(whole["spoke"] == "clean_baseline_per_pair") & (whole["metric"] == "per_pair_clean_recall_at_1_t2i")]
        cells = [cell for _, cell in typo.groupby(["model", "text_severity"])] + [cell for _, cell in clean.groupby("model")]
        rows.append({"output": "recall_at_1_by_typo_severity", "dataset": dataset, "unit": "Recall@1", "means": len(cells), "largest_half_width": _largest_half_width(cells)})
    if file_paths.run_output(OBJECT_CUTOUT_RUN).exists():
        whole = full_gallery_rows("coco")
        rectangle = whole[(whole["spoke"] == "retrieval_per_pair") & (whole["metric"] == "per_pair_recall_at_1_i2t") & (whole["vision_corruption"] == "cutout")]
        clean = whole[(whole["spoke"] == "clean_baseline_per_pair") & (whole["metric"] == "per_pair_clean_recall_at_1_i2t")]
        objects = full_gallery_rows("coco", OBJECT_CUTOUT_RUN)
        objects = objects[(objects["spoke"] == "retrieval_per_pair") & (objects["metric"] == "per_pair_recall_at_1_i2t")]
        cells = [cell for frame in (rectangle, objects) for _, cell in frame.groupby(["model", "image_severity"])] + [cell for _, cell in clean.groupby("model")]
        rows.append({"output": "recall_at_1_by_object_cutout_severity", "dataset": "coco", "unit": "Recall@1", "means": len(cells), "largest_half_width": _largest_half_width(cells)})
    else:
        note_skipped(f"no {OBJECT_CUTOUT_RUN} output saved, so its interval size is left out")
    pd.DataFrame(rows).to_csv(file_paths.NUM_UNDRAWN_INTERVAL_SIZES, index=False)
    print(f"saved {file_paths.NUM_UNDRAWN_INTERVAL_SIZES.relative_to(file_paths.ROOT)}")


# Generate supporting retrieval and damage summaries.
def build_recall_at_k_table() -> None:
    """Report headline Recall@1/5/10 for both datasets, including clean queries, typo and COCO's object cutout.

    Each row identifies the gallery size, model, direction, corruption and severity,
    with per-seed values, their mean and their standard deviation across seeds.
    """
    suffix = f"_gallery{settings.HEADLINE_GALLERY_SIZE}"
    cells = ["model", "vision_corruption", "text_corruption", "image_severity", "text_severity", "metric"]
    runs = [(dataset, f"{dataset}_2000_pairs_retrieval") for dataset in DATASETS]
    if file_paths.run_output(OBJECT_CUTOUT_RUN).exists():
        runs.append(("coco", OBJECT_CUTOUT_RUN))
    tables = []
    for dataset, run in runs:
        df = pd.read_csv(file_paths.run_output(run), low_memory=False,
                         usecols=["model", "seed", "vision_corruption", "image_severity", "text_corruption",
                                  "text_severity", "spoke", "metric", "value"])
        rows = df[(df["spoke"] == "retrieval_gallery_sweep") & df["metric"].str.endswith(suffix)].copy()
        if run == OBJECT_CUTOUT_RUN:
            rows = rows[rows["vision_corruption"] == "object_cutout"]  # Its clean rows repeat those of the main run.
        rows["metric"] = rows["metric"].str.removesuffix(suffix)
        rows = rows.sort_values("seed", kind="stable")  # Fix seed order because floating-point sums depend on it.
        per_seed = rows.pivot_table(index=cells, columns="seed", values="value")
        per_seed.columns = [f"seed_{int(seed)}" for seed in per_seed.columns]
        summary = per_seed.join(rows.groupby(cells)["value"].agg(mean="mean", seed_std="std")).reset_index()
        tables.append(summary.assign(dataset=dataset, gallery_size=settings.HEADLINE_GALLERY_SIZE))
    table = _sort_model_rows(pd.concat(tables, ignore_index=True))
    first = ["dataset", "gallery_size"]
    table[first + [c for c in table.columns if c not in first]].to_csv(file_paths.TAB_RECALL_AT_K, index=False)
    print(f"saved {file_paths.TAB_RECALL_AT_K.relative_to(file_paths.ROOT)}")


def build_ssim_quintile_composition() -> None:
    """Count image-corruption shares in headline SSIM Q4 and Q5 across all models' rows, for each dataset."""
    rows = []
    for dataset in datasets_with_headline_rows():
        images = recall_with_damage("i2t", "damage_ssim", dataset)
        images["quintile"] = damage_quintile(images["damage_ssim"])
        for kind in settings.IMAGE_CORRUPTIONS:
            for severity in settings.SEVERITIES:
                row = {"dataset": dataset, "corruption_type": kind, "severity": severity}
                for q in ("Q4", "Q5"):
                    in_q = images[images["quintile"] == q]
                    count = int(((in_q["vision_corruption"] == kind) & (in_q["image_severity"] == severity)).sum())
                    row.update({f"{q}_count": count, f"{q}_fraction": round(count / len(in_q), 3)})
                rows.append(row)
    pd.DataFrame(rows).to_csv(file_paths.NUM_SSIM_Q4_Q5_COMPOSITION, index=False)
    print(f"saved {file_paths.NUM_SSIM_Q4_Q5_COMPOSITION.relative_to(file_paths.ROOT)}")


def build_caption_quintile_composition() -> None:
    """Count each caption corruption's rows in each BLEU quintile, and the rows it left unchanged.

    A corruption can leave a caption unchanged, for example when shuffle puts the words back, so a
    share of Q1 holds undamaged captions. Use each dataset's headline gallery, with typo left out.
    """
    rows = []
    for dataset in datasets_with_headline_rows():
        dmg = headline_damage(dataset)
        text = dmg[dmg["modality"] == "text"].copy()
        text["quintile"] = damage_quintile(text["damage_bleu"])
        totals = text.groupby("quintile", observed=True).size()
        for (quintile, kind, severity), group in text.groupby(["quintile", "corruption_type", "severity"], observed=True):
            unchanged = int((group["norm_edit_distance"] == 0).sum())
            rows.append({"dataset": dataset, "quintile": quintile, "corruption_type": kind, "severity": severity, "rows": len(group),
                         "unchanged_rows": unchanged, "share_of_quintile": len(group) / totals[quintile],
                         "unchanged_share_of_quintile": unchanged / totals[quintile]})
    pd.DataFrame(rows).to_csv(file_paths.NUM_CAPTION_QUINTILE_COMPOSITION, index=False)
    print(f"saved {file_paths.NUM_CAPTION_QUINTILE_COMPOSITION.relative_to(file_paths.ROOT)}")


def build_noise5_mask5_at_full_gallery() -> None:
    """Report the mean Recall@1 across seeds for severity-5 noise (i2t) and mask (t2i).

    Both use the full gallery of 2,000 pairs on each dataset.
    """
    tables = []
    for dataset in DATASETS:
        whole = full_gallery_rows(dataset)
        whole = whole[whole["spoke"] == "retrieval"]
        noise = whole[(whole["metric"] == "recall_at_1_i2t") & (whole["vision_corruption"] == "gaussian_noise") & (whole["image_severity"] == 5)]
        mask = whole[(whole["metric"] == "recall_at_1_t2i") & (whole["text_corruption"] == "mask") & (whole["text_severity"] == 5)]
        tables.append(pd.DataFrame({"mask5_t2i_2000": mask.groupby("model")["value"].mean().round(4),
                                    "noise5_i2t_2000": noise.groupby("model")["value"].mean().round(4)}).reset_index().assign(dataset=dataset))
    table = _sort_model_rows(pd.concat(tables, ignore_index=True))
    table[["dataset", "model", "mask5_t2i_2000", "noise5_i2t_2000"]].to_csv(file_paths.NUM_NOISE5_VS_MASK5_GALLERY2000, index=False)
    print(f"saved {file_paths.NUM_NOISE5_VS_MASK5_GALLERY2000.relative_to(file_paths.ROOT)}")
