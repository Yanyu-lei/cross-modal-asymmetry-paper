"""Load saved runs at the headline or full gallery, with the required corruption filters."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from src import file_paths, settings
from src.bootstrap_and_quintiles import damage_quintile

PER_QUERY_COLUMNS = ["model", "seed", "pair_id", "vision_corruption", "image_severity", "text_corruption",
                     "text_severity", "spoke", "metric", "value", "n_eval"]


# Missing inputs make the final build fail when --require-all is set.
SKIPPED: list[str] = []


def note_skipped(message: str) -> None:
    """Print each reason for a skipped output once and keep it for the final summary."""
    if message not in SKIPPED:
        SKIPPED.append(message)
        print(message)


def _with_damage(dmg: pd.DataFrame) -> pd.DataFrame:
    """Add the damage columns, where lower PSNR means more damage."""
    return dmg.assign(damage_ssim=1.0 - dmg["ssim"], damage_psnr=-dmg["psnr"], damage_bleu=1.0 - dmg["bleu"])


# Headline retrieval uses a gallery of 1,000 on both datasets. The main-text tables show COCO.
@lru_cache(maxsize=2)
def _headline_per_query_rows(dataset: str = "coco") -> pd.DataFrame:
    df = pd.read_csv(file_paths.run_output(f"{dataset}_2000_pairs_retrieval"), low_memory=False, usecols=PER_QUERY_COLUMNS)
    keep = (df["spoke"].isin(["retrieval_per_pair", "clean_baseline_per_pair"])
            & (pd.to_numeric(df["n_eval"]) == settings.HEADLINE_GALLERY_SIZE))
    return df[keep]


def datasets_with_headline_rows() -> list[str]:
    """Return the datasets with per-query rows at the headline gallery and record any omissions."""
    done = []
    for dataset in file_paths.DATASETS:
        if _headline_per_query_rows(dataset).empty:
            note_skipped(f"no {dataset} per-query rows at the headline gallery, so its rows are left out")
        else:
            done.append(dataset)
    return done


def per_pair_retrieval(dataset: str = "coco") -> pd.DataFrame:
    """Load the Recall@k hits (0 or 1) for damaged queries at the headline gallery size."""
    df = _headline_per_query_rows(dataset)
    pp = df[df["spoke"] == "retrieval_per_pair"].copy()
    for c in ("seed", "pair_id", "image_severity", "text_severity"):
        pp[c] = pd.to_numeric(pp[c], errors="coerce").astype("Int64")
    pp["value"] = pd.to_numeric(pp["value"]).astype(int)
    return pp


def clean_baseline_per_pair(dataset: str = "coco") -> pd.DataFrame:
    """Load the Recall@k hits for clean queries at the headline gallery size."""
    df = _headline_per_query_rows(dataset)
    pp = df[df["spoke"] == "clean_baseline_per_pair"].copy()
    for c in ("seed", "pair_id"):
        pp[c] = pd.to_numeric(pp[c], errors="coerce").astype("Int64")
    pp["value"] = pd.to_numeric(pp["value"]).astype(int)
    return pp


@lru_cache(maxsize=2)
def headline_damage(dataset: str = "coco") -> pd.DataFrame:
    """Load damage for each seed's first 1,000 pairs, excluding unbinned corruptions."""
    dmg = pd.read_csv(file_paths.damage_output(f"{dataset}_2000_pairs_retrieval"))
    keep = (dmg["pair_id"] < settings.HEADLINE_GALLERY_SIZE) & (dmg["corruption_type"] != "typo")
    dmg = _with_damage(dmg[keep].reset_index(drop=True))
    for c in ("seed", "pair_id", "severity"):
        dmg[c] = dmg[c].astype(int)
    return dmg


def recall_with_damage(direction: str, damage_col: str, dataset: str = "coco") -> pd.DataFrame:
    """Join each headline query's Recall@1 hit to its input damage for the requested direction."""
    side, kind_col, severity_col = (("image", "vision_corruption", "image_severity") if direction == "i2t"
                                    else ("text", "text_corruption", "text_severity"))
    pp = per_pair_retrieval(dataset)
    hits = pp[pp["metric"] == f"per_pair_recall_at_1_{direction}"][
        ["seed", "model", "pair_id", kind_col, severity_col, "value"]].copy()
    dmg = headline_damage(dataset)
    dmg = dmg[dmg["modality"] == side][["seed", "pair_id", "corruption_type", "severity", damage_col]]
    dmg = dmg.rename(columns={"corruption_type": kind_col, "severity": severity_col})
    return hits.merge(dmg, on=["seed", "pair_id", kind_col, severity_col], how="inner")


def quintiles_by_model(df: pd.DataFrame, damage_col: str) -> dict[str, pd.DataFrame]:
    """Group the rows by model and assign damage quintiles within each group."""
    out = {}
    for model in settings.MODELS:
        part = df[df["model"] == model].copy()
        part["quintile"] = damage_quintile(part[damage_col])
        out[model] = part
    return out


# The 300-pair readers exclude unbinned corruptions.
def datasets_with_300_pair_run() -> list[str]:
    """Return the datasets with all 300-pair outputs and record any omissions for the build summary."""
    done = []
    for dataset in file_paths.DATASETS:
        files = [file_paths.run_output(f"{dataset}_300_pairs_fidelity_retention", s) for s in settings.SEEDS]
        if all(f.exists() for f in files + [file_paths.text_pooling_ablation_output(dataset)]):
            done.append(dataset)
        else:
            note_skipped(f"no {dataset} 300-pair run saved, so its rows are left out")
    return done


@lru_cache(maxsize=2)
def _all_fidelity_retention_rows(dataset: str) -> pd.DataFrame:
    return pd.concat([pd.read_csv(file_paths.run_output(f"{dataset}_300_pairs_fidelity_retention", s), low_memory=False)
                      for s in settings.SEEDS], ignore_index=True)


def fidelity_retention_rows(dataset: str = "coco") -> pd.DataFrame:
    """Load all seeds of a dataset's 300-pair runs, excluding typo but keeping joint and clean rows."""
    df = _all_fidelity_retention_rows(dataset)
    return df[df["text_corruption"] != "typo"]


def typo_fidelity_retention_rows(dataset: str) -> pd.DataFrame:
    """Load the typo rows of a dataset's 300-pair runs, which the headline readers leave out."""
    df = _all_fidelity_retention_rows(dataset)
    return df[df["text_corruption"] == "typo"]


@lru_cache(maxsize=2)
def fidelity_retention_damage(dataset: str = "coco") -> pd.DataFrame:
    """Load the input damage from a dataset's 300-pair runs, keeping only the binned corruptions."""
    dmg = pd.read_csv(file_paths.damage_output(f"{dataset}_300_pairs_fidelity_retention"))
    return _with_damage(dmg[dmg["corruption_type"] != "typo"].reset_index(drop=True))


@lru_cache(maxsize=2)
def text_pooling_ablation(dataset: str = "coco") -> pd.DataFrame:
    """Load the retention margins for each caption and text-pooling method, with clean captions at severity 0."""
    df = pd.read_csv(file_paths.text_pooling_ablation_output(dataset))
    df = df[df["metric"] == "retention_margin"]
    return df.astype({"seed": int, "pair_id": int, "text_severity": int})


# Robustness analyses use the full gallery on both datasets.
def full_gallery_rows(dataset: str, run: str | None = None) -> pd.DataFrame:
    """Load a retrieval run, excluding smaller-gallery per-query rows."""
    df = pd.read_csv(file_paths.run_output(run or f"{dataset}_2000_pairs_retrieval"), low_memory=False,
                     usecols=PER_QUERY_COLUMNS)
    for c in ("seed", "image_severity", "text_severity", "pair_id", "value", "n_eval"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    smaller = df["spoke"].isin(["retrieval_per_pair", "clean_baseline_per_pair"]) & (df["n_eval"] != df["n_eval"].max())
    return df[~smaller]


def object_cutout_per_pair_at_headline_gallery(run: str) -> pd.DataFrame:
    """Load the object cutout run's Recall@1 hits at the headline gallery size."""
    df = pd.read_csv(file_paths.run_output(run), low_memory=False, usecols=PER_QUERY_COLUMNS)
    keep = ((df["spoke"] == "retrieval_per_pair") & (df["metric"] == "per_pair_recall_at_1_i2t")
            & (pd.to_numeric(df["n_eval"]) == settings.HEADLINE_GALLERY_SIZE))
    hits = df[keep].copy()
    for c in ("seed", "pair_id", "image_severity", "value"):
        hits[c] = pd.to_numeric(hits[c]).astype(int)
    return hits


def full_gallery_damage(dataset: str, run: str | None = None) -> pd.DataFrame:
    return _with_damage(pd.read_csv(file_paths.damage_output(run or f"{dataset}_2000_pairs_retrieval")))
