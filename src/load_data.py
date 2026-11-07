"""Load image-caption pairs using fixed sampling rules for each seed.

Samples from different seeds can overlap. Full-size retrieval uses every pair.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO

from PIL import Image

from src import file_paths

# These offsets are part of the sampling protocol.
RETRIEVAL_SEED_OFFSET = 50_000
NEGATIVES_SEED_OFFSET = 10_000


@dataclass(frozen=True)
class Pair:
    image: Image.Image
    caption: str
    position: int   # This indexes the manifest.
    image_id: int   # This identifies the image in its dataset.


@lru_cache(maxsize=2)
def read_manifest(dataset: str) -> tuple[dict, ...]:
    """Read the full manifest, keeping the image bytes hex-encoded."""
    with open(file_paths.full_manifest(dataset)) as f:
        return tuple(json.load(f))


def _shuffled_positions(n_items: int, shuffle_seed: int) -> list[int]:
    order = list(range(n_items))
    random.Random(shuffle_seed).shuffle(order)
    return order


def fidelity_retention_positions(n_items: int, seed: int, n: int) -> list[int]:
    """Take the seed's slice from its own shuffle, so slices from different seeds can overlap."""
    start = seed * n
    if start + n > n_items:
        raise ValueError(f"seed {seed} needs positions up to {start + n}, but the manifest has {n_items}")
    return _shuffled_positions(n_items, seed)[start:start + n]


def retrieval_positions(n_items: int, seed: int, n: int) -> list[int]:
    if n > n_items:
        raise ValueError(f"{n} retrieval pairs requested, but the manifest has {n_items}")
    return _shuffled_positions(n_items, seed + RETRIEVAL_SEED_OFFSET)[:n]


def negative_positions(n_items: int, seed: int, k: int, exclude: set[int]) -> list[int]:
    order = [p for p in _shuffled_positions(n_items, seed + NEGATIVES_SEED_OFFSET) if p not in exclude]
    if len(order) < k:
        raise ValueError(f"only {len(order)} positions are left for {k} negatives")
    return order[:k]


def _pairs(dataset: str, positions: list[int]) -> list[Pair]:
    items = read_manifest(dataset)
    return [Pair(image=Image.open(BytesIO(bytes.fromhex(items[p]["bytes_hex"]))).convert("RGB"),
                 caption=items[p]["caption"], position=p, image_id=int(items[p]["id"]))
            for p in positions]


def fidelity_retention_pairs(dataset: str, seed: int, n: int) -> list[Pair]:
    return _pairs(dataset, fidelity_retention_positions(len(read_manifest(dataset)), seed, n))


def retrieval_pairs(dataset: str, seed: int, n: int) -> list[Pair]:
    return _pairs(dataset, retrieval_positions(len(read_manifest(dataset)), seed, n))


def negative_pairs(dataset: str, seed: int, k: int, exclude: set[int]) -> list[Pair]:
    return _pairs(dataset, negative_positions(len(read_manifest(dataset)), seed, k, exclude))
