"""Load the shared experiment, model and run settings once when this module is imported."""
from __future__ import annotations

import yaml

from src import file_paths


def _read(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


_EXPERIMENT = _read(file_paths.EXPERIMENT_SETTINGS)

SEEDS: tuple[int, ...] = tuple(_EXPERIMENT["seeds"])
SEVERITIES: tuple[int, ...] = tuple(_EXPERIMENT["severities"])
IMAGE_CORRUPTIONS: tuple[str, ...] = tuple(_EXPERIMENT["image_corruptions"])
TEXT_CORRUPTIONS: tuple[str, ...] = tuple(_EXPERIMENT["text_corruptions"])
# Keep the typo corruption out of the headline damage quintiles.
BINNED_TEXT_CORRUPTIONS: tuple[str, ...] = tuple(_EXPERIMENT["binned_text_corruptions"])
NOISE_MASK_GRID: dict = _EXPERIMENT["noise_mask_grid"]
N_NEGATIVES: int = _EXPERIMENT["n_negatives"]
GALLERY_SIZES: tuple[int, ...] = tuple(_EXPERIMENT["gallery_sizes"])
PER_PAIR_GALLERY_SIZES: tuple[int, ...] = tuple(_EXPERIMENT["per_pair_gallery_sizes"])
HEADLINE_GALLERY_SIZE: int = _EXPERIMENT["headline_gallery_size"]
RECALL_KS: tuple[int, ...] = tuple(_EXPERIMENT["recall_ks"])
IMAGE_BATCH_SIZE: int = _EXPERIMENT["batch_size"]["images"]
CAPTION_BATCH_SIZE: int = _EXPERIMENT["batch_size"]["captions"]
NEGATIVE_CAPTION_BATCH_SIZE: int = _EXPERIMENT["batch_size"]["negative_captions"]
BOOTSTRAP_DRAWS: int = _EXPERIMENT["bootstrap_draws"]

MODEL_SETTINGS: dict[str, dict] = _read(file_paths.MODEL_SETTINGS)["models"]
MODELS: tuple[str, ...] = tuple(MODEL_SETTINGS)

RUNS: dict[str, dict] = _read(file_paths.PAPER_RUNS)["runs"]
