"""Shared data loaders for paper figures and tables. Loaded once, used everywhere."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (0, 1, 2)
QUINTILES = ["Q1", "Q2", "Q3", "Q4", "Q5"]


_V2_PATH = ROOT / "results/per_pair_retrieval_v2/per_pair_retrieval_v2.csv"
_CLEAN_PATH = ROOT / "results/clean_baseline_retrieval.csv"


def per_pair_retrieval() -> pd.DataFrame:
    pp = pd.read_csv(_V2_PATH, low_memory=False)
    pp = pp[pp["spoke"] == "retrieval_per_pair"].copy()
    for c in ("seed", "pair_id", "image_severity", "text_severity"):
        pp[c] = pd.to_numeric(pp[c], errors="coerce").astype("Int64")
    pp["value"] = pd.to_numeric(pp["value"]).astype(int)
    return pp


def aggregate_retrieval() -> pd.DataFrame:
    df = pd.read_csv(_V2_PATH, low_memory=False)
    return df[df["spoke"] == "retrieval"].copy()


def clean_baseline_retrieval() -> pd.DataFrame:
    """Aggregate clean (no-corruption) Recall@k per (model, seed, direction, k)."""
    df = pd.read_csv(_CLEAN_PATH, low_memory=False)
    agg = df[df["spoke"] == "clean_baseline"].copy()
    agg["seed"] = pd.to_numeric(agg["seed"]).astype(int)
    return agg


def clean_baseline_per_pair() -> pd.DataFrame:
    """Per-pair clean Recall@k indicators (for hierarchical bootstrap)."""
    df = pd.read_csv(_CLEAN_PATH, low_memory=False)
    pp = df[df["spoke"] == "clean_baseline_per_pair"].copy()
    for c in ("seed", "pair_id"):
        pp[c] = pd.to_numeric(pp[c], errors="coerce").astype("Int64")
    pp["value"] = pd.to_numeric(pp["value"]).astype(int)
    return pp


def retrieval_damage() -> pd.DataFrame:
    dmg = pd.read_csv(ROOT / "results/input_damage_retrieval.csv")
    for c in ("seed", "pair_id", "severity"):
        dmg[c] = dmg[c].astype(int)
    return dmg


def main_damage() -> pd.DataFrame:
    dmg = pd.read_csv(ROOT / "results/input_damage.csv")
    for c in ("seed", "pair_id", "severity"):
        dmg[c] = dmg[c].astype(int)
    return dmg


def seeds_combined() -> pd.DataFrame:
    return pd.concat([
        pd.read_csv(ROOT / f"results/seed{s}/seed{s}_results.csv", low_memory=False) for s in SEEDS
    ], ignore_index=True)


def pooling_probe_v2() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "results/pooling_probe/pooling_probe_v2.csv")
    df["seed"] = df["seed"].astype(int)
    df["pair_id"] = df["pair_id"].astype(int)
    df["text_severity"] = df["text_severity"].astype(int)
    return df


def i2t_with_damage() -> pd.DataFrame:
    """Per-pair i2t Recall@1 indicator joined to image-side damage.
    Joins on (seed, pair_id, vision_corruption, image_severity) so each corruption
    type gets its own damage values (not a noise-damage proxy)."""
    pp = per_pair_retrieval()
    i2t = pp[pp["metric"] == "per_pair_recall_at_1_i2t"][
        ["seed", "model", "pair_id", "vision_corruption", "image_severity", "value"]
    ].copy()
    dmg = retrieval_damage()
    img_dmg = dmg[dmg["modality"] == "image"][
        ["seed", "pair_id", "corruption_type", "severity", "ssim", "damage_ssim"]
    ].rename(columns={"corruption_type": "vision_corruption", "severity": "image_severity"})
    return i2t.merge(img_dmg, on=["seed", "pair_id", "vision_corruption", "image_severity"], how="inner")


def t2i_with_damage() -> pd.DataFrame:
    pp = per_pair_retrieval()
    t2i = pp[pp["metric"] == "per_pair_recall_at_1_t2i"][
        ["seed", "model", "pair_id", "text_corruption", "text_severity", "value"]
    ].copy()
    dmg = retrieval_damage()
    txt_dmg = dmg[dmg["modality"] == "text"][
        ["seed", "pair_id", "corruption_type", "severity", "bleu", "damage_bleu"]
    ].rename(columns={"corruption_type": "text_corruption", "severity": "text_severity"})
    return t2i.merge(txt_dmg, on=["seed", "pair_id", "text_corruption", "text_severity"], how="inner")


def quintile_assign(df: pd.DataFrame, damage_col: str) -> pd.DataFrame:
    out = df.copy()
    out["quintile"] = pd.qcut(out[damage_col], q=5, labels=QUINTILES, duplicates="drop")
    return out
