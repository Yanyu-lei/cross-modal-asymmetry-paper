"""COCO data loader.

Streams COCO 2017 captions via Hugging Face datasets and caches each fetched
image's bytes + caption + sha256 into a manifest JSON for deterministic replay.
Once the manifest contains enough entries, all later runs replay from disk
(no network) and yield (PIL.Image, caption) pairs in a seeded shuffle order.

Single-source-of-truth manifest path:
    configs/experiment.yaml -> manifest_path

Slicing semantics:
    load_pairs(seed, n, offset)
        returns the slice [offset, offset + n) of the deterministically-shuffled
        manifest (shuffle seed = `seed`). Use distinct (seed, offset) for
        disjoint subsets.
"""
from __future__ import annotations

import hashlib
import json
import random
from io import BytesIO
from pathlib import Path
from typing import Iterator, List, Tuple

import requests
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _http_get(url: str, timeout: float = 20.0) -> bytes:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def manifest_path(experiment_yaml_path: Path | str | None = None) -> Path:
    """Resolve the manifest path from configs/experiment.yaml."""
    import yaml
    p = Path(experiment_yaml_path) if experiment_yaml_path else (_PROJECT_ROOT / "configs" / "experiment.yaml")
    with open(p) as f:
        cfg = yaml.safe_load(f)
    raw = cfg["manifest_path"]
    return (_PROJECT_ROOT / raw) if not Path(raw).is_absolute() else Path(raw)


def _load_manifest(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _save_manifest(path: Path, items: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(items, f)


def build_manifest(
    target_size: int,
    *,
    manifest: Path | None = None,
    hf_seed: int = 0,
    save_every: int = 50,
    progress: bool = True,
) -> int:
    """Stream COCO 2017 train captions until the manifest has at least `target_size` items.

    Saves to disk every `save_every` new items so progress isn't lost on
    interrupt. Idempotent: re-running with a larger target appends without
    duplicating (sha256 dedup).
    """
    from datasets import load_dataset

    path = manifest or manifest_path()
    items = _load_manifest(path)
    seen = {it["sha256"] for it in items}
    if len(items) >= target_size:
        return len(items)

    last_saved = len(items)
    ds = load_dataset("phiyodr/coco2017", split="train", streaming=True)
    for ex in ds.shuffle(seed=hf_seed):
        if len(items) >= target_size:
            break
        try:
            img_bytes = _http_get(ex["coco_url"])
        except Exception:
            continue
        sha = hashlib.sha256(img_bytes).hexdigest()
        if sha in seen:
            continue
        items.append(
            {
                "id": int(ex["image_id"]),
                "caption": ex["captions"][0],
                "bytes_hex": img_bytes.hex(),
                "sha256": sha,
            }
        )
        seen.add(sha)

        if len(items) - last_saved >= save_every:
            _save_manifest(path, items)
            last_saved = len(items)
            if progress:
                print(f"[manifest] {len(items)} / {target_size}", flush=True)

    _save_manifest(path, items)
    return len(items)


def load_pairs(
    seed: int,
    n: int,
    *,
    offset: int = 0,
    manifest: Path | None = None,
) -> Iterator[Tuple[Image.Image, str, int]]:
    """Yield `n` (image, caption, manifest_index) triples deterministically.

    Manifest is shuffled with `seed`, then sliced [offset, offset+n). The
    third element is the original index into the manifest (useful for
    cross-referencing to the COCO image_id).
    """
    path = manifest or manifest_path()
    items = _load_manifest(path)
    if len(items) < offset + n:
        raise RuntimeError(
            f"manifest at {path} has {len(items)} items but {offset + n} were requested. "
            f"Run build_manifest({offset + n}) first."
        )
    rng = random.Random(seed)
    order = list(range(len(items)))
    rng.shuffle(order)
    for idx in order[offset : offset + n]:
        entry = items[idx]
        img = Image.open(BytesIO(bytes.fromhex(entry["bytes_hex"]))).convert("RGB")
        yield img, entry["caption"], idx


def load_caption_pool(
    seed: int,
    n: int,
    *,
    skip_indices: set[int] | None = None,
    manifest: Path | None = None,
) -> List[Tuple[str, int]]:
    """Sample `n` captions deterministically from the manifest, optionally
    excluding a set of manifest indices (used to keep the mismatch pool disjoint
    from the eval pairs per experiment.yaml: match_retention.pool_disjoint)."""
    path = manifest or manifest_path()
    items = _load_manifest(path)
    skip = skip_indices or set()

    rng = random.Random(seed)
    order = list(range(len(items)))
    rng.shuffle(order)
    out: List[Tuple[str, int]] = []
    for idx in order:
        if idx in skip:
            continue
        out.append((items[idx]["caption"], idx))
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(
            f"only {len(out)} disjoint captions available in manifest (needed {n}). "
            f"Grow the manifest with build_manifest()."
        )
    return out
