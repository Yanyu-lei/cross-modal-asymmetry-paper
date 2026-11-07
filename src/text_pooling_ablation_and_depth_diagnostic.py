"""Measure text-pooling retention and each text layer's response to word deletion."""
from __future__ import annotations

import datetime as _dt
import json
import random

import numpy as np
import torch
import torch.nn.functional as F

from src import file_paths, settings
from src.corruptions import corrupt_caption, mask_text, single_text_rng
from src.load_data import fidelity_retention_pairs, negative_pairs
from src.load_models import encode_images, load_model

TEXT_POOLINGS = ("standard", "mean")


def _pooled_captions(adapter, captions: list[str], batch_size: int) -> dict[str, torch.Tensor]:
    """Return the (N, D) embeddings for "standard" and "mean" pooling."""
    parts = {p: [] for p in TEXT_POOLINGS}
    for i in range(0, len(captions), batch_size):
        out = adapter.encode_text_both_poolings(captions[i:i + batch_size])
        for p in TEXT_POOLINGS:
            parts[p].append(out[p])
    return {p: torch.cat(parts[p]) for p in TEXT_POOLINGS}


def _retention_margins(captions: torch.Tensor, images: torch.Tensor, negatives: torch.Tensor):
    """Return each caption's partner cosine similarity minus its mean similarity to image negatives.

    Keep this batched calculation because a per-pair rewrite can change the sixth decimal.
    """
    t = F.normalize(captions, dim=-1)
    sim_partner = (t * F.normalize(images, dim=-1)).sum(dim=-1).cpu().numpy()
    sim_negatives = (t @ F.normalize(negatives, dim=-1).T).cpu().numpy().mean(axis=1)
    return sim_partner - sim_negatives


def text_pooling_ablation_rows(run: str, adapter, dataset: str, seed: int, n_pairs: int) -> list[dict]:
    """Compare standard and mean pooling on the same damaged captions and image negatives.

    Mean pooling averages through the end of the caption and uses the unchanged projection,
    which was trained for standard pooling. The pair selection and corruption streams match
    the fidelity-retention run.
    """
    pairs = fidelity_retention_pairs(dataset, seed, n_pairs)
    negatives = negative_pairs(dataset, seed, settings.N_NEGATIVES, exclude={p.position for p in pairs})
    clean_images = encode_images(adapter, [p.image for p in pairs])
    negative_images = encode_images(adapter, [n.image for n in negatives])
    timestamp = _dt.datetime.now().isoformat(timespec="seconds")

    rows = []
    for kind in settings.BINNED_TEXT_CORRUPTIONS:
        for severity in settings.SEVERITIES:
            damaged = [corrupt_caption(p.caption, kind, severity, rng=single_text_rng(seed, i, severity))
                       for i, p in enumerate(pairs)]
            pooled = _pooled_captions(adapter, damaged, settings.CAPTION_BATCH_SIZE)
            for pooling in TEXT_POOLINGS:
                margins = _retention_margins(pooled[pooling], clean_images, negative_images)
                rows += [{"run_tag": run, "timestamp": timestamp, "model": adapter.name, "seed": seed, "pair_id": i,
                          "text_corruption": kind, "text_severity": severity, "text_pooling": pooling,
                          "metric": "retention_margin", "value": float(m)} for i, m in enumerate(margins)]
    return rows


def clean_caption_pooling_rows(run: str, adapter, dataset: str, seed: int, n_pairs: int) -> list[dict]:
    """Score the clean captions under both poolings at severity 0, encoded as in the ablation."""
    pairs = fidelity_retention_pairs(dataset, seed, n_pairs)
    negatives = negative_pairs(dataset, seed, settings.N_NEGATIVES, exclude={p.position for p in pairs})
    clean_images = encode_images(adapter, [p.image for p in pairs])
    negative_images = encode_images(adapter, [n.image for n in negatives])
    timestamp = _dt.datetime.now().isoformat(timespec="seconds")
    pooled = _pooled_captions(adapter, [p.caption for p in pairs], settings.CAPTION_BATCH_SIZE)
    return [{"run_tag": run, "timestamp": timestamp, "model": adapter.name, "seed": seed, "pair_id": i,
             "text_corruption": "none", "text_severity": 0, "text_pooling": pooling,
             "metric": "retention_margin", "value": float(m)}
            for pooling in TEXT_POOLINGS
            for i, m in enumerate(_retention_margins(pooled[pooling], clean_images, negative_images))]


DIAGNOSTIC_SEVERITY = 2


def released_captions(dataset: str) -> list[str]:
    """Return the captions of one dataset's released pair list, which holds no images."""
    with open(file_paths.manifest(dataset)) as f:
        return [entry["caption"] for entry in json.load(f)]


def layer_cosines(adapter, captions: list[str], seed: int = 0) -> np.ndarray:
    """Return the cosine between each clean and masked caption's pooled vector at every text layer.

    The result has one row per caption and one column per layer. The captions use the
    configured batch size, and every tokenizer pads to a fixed length.
    """
    master = random.Random(seed)
    masked = [mask_text(c, DIAGNOSTIC_SEVERITY, rng=random.Random(master.randint(0, 2**31))) for c in captions]
    layers = list(range(adapter.n_text_layers))
    parts = []
    with torch.no_grad():
        for start in range(0, len(captions), settings.CAPTION_BATCH_SIZE):
            batch = slice(start, start + settings.CAPTION_BATCH_SIZE)
            clean = adapter.encode_text(captions[batch], layers).pooled_states
            damaged = adapter.encode_text(masked[batch], layers).pooled_states
            parts.append(torch.stack([F.cosine_similarity(a, b, dim=-1).cpu().double() for a, b in zip(clean, damaged)], dim=1))
    return torch.cat(parts).numpy()


def text_depth_rows(model: str) -> list[dict]:
    """Return per-layer cosine statistics for each dataset's captions and for both pooled.

    Each model's early text depth in configs/model_settings.yaml is set by hand from the pooled
    rows. It is the first layer with a mean cosine below 0.95, or a depth fraction of 0.40 when
    no layer before the middle one falls that low. This function only reports the statistics.
    """
    adapter = load_model(model)
    cosines = {d: layer_cosines(adapter, released_captions(d)) for d in file_paths.DATASETS}
    cosines["both"] = np.vstack(list(cosines.values()))
    n_layers = adapter.n_text_layers
    return [{"model": model, "captions": name, "n_text_layers": n_layers, "layer_index": layer,
             "layer_fraction": (layer + 1) / n_layers, "n_captions": len(values),
             "mean_cosine_clean_vs_mask_sev2": float(values[:, layer].mean()),
             "min_cosine": float(values[:, layer].min()), "max_cosine": float(values[:, layer].max())}
            for name, values in cosines.items() for layer in range(n_layers)]
