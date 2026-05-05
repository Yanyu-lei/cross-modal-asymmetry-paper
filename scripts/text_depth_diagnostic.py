"""Text-depth diagnostic.

For each model, capture the pooled text representation at EVERY text-transformer
layer for clean and severity-2 masked versions of N captions. Compute cosine
similarity per layer, average over captions. Find the layer where the average
cosine first drops below a threshold (default 0.95) — that's where the pooled
token starts carrying real sequence information.

The recommended layer is then used to set each model's "early" text depth in
configs/models.yaml. If the cosine never crosses the threshold, fall back to
the configured default (40%).

Output:
- results/text_depth_diagnostic.csv (one row per model × layer)
- prints recommended early-depth fraction per model

Usage:
    .venv/bin/python -m scripts.text_depth_diagnostic
    .venv/bin/python -m scripts.text_depth_diagnostic --threshold 0.95 --n-captions 24
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.corruptions.text import mask_text
from src.models.registry import list_models, load_model, pick_device


# A small fixed set of caption-shaped strings, hand-picked to roughly match
# COCO style (concrete subject + activity + setting). Used here so the
# diagnostic doesn't depend on the COCO data layer (Phase 3).
DIAGNOSTIC_CAPTIONS: List[str] = [
    "a small dog running on green grass in a sunny park",
    "two children playing with a red ball on the beach",
    "a man riding a bicycle on a narrow mountain road",
    "a wooden table with a bowl of fresh fruit and a glass of water",
    "a black cat sleeping on a soft blue blanket",
    "an old fisherman casting a line into a calm lake at dawn",
    "a yellow taxi parked outside a busy city restaurant",
    "a young woman reading a book under a large oak tree",
    "a chef tossing pasta in a steel pan over a gas stove",
    "a herd of elephants walking across a dry savanna at sunset",
    "a brown horse standing in a wide green field with a wooden fence",
    "a couple walking hand in hand along a snowy forest path",
    "a baker pulling a tray of croissants from a hot brick oven",
    "a sailboat with white sails crossing a calm blue bay",
    "a child building a tall sandcastle near the ocean shore",
    "a violinist performing on a small stage with golden lighting",
    "a hiker climbing a steep trail with a heavy backpack and walking poles",
    "a market stall stacked high with bright fresh tomatoes and peppers",
    "a vintage red car parked on a cobblestone street in a quiet town",
    "a librarian shelving books in a dimly lit aisle of an old library",
    "a black-and-white photograph of three musicians playing in a smoky cafe",
    "a tiger drinking from a clear stream in a dense bamboo forest",
    "a couple eating slices of pizza on a checkered red tablecloth",
    "a runner crossing a finish line at the end of a long marathon race",
]


def diagnose_one(model_name: str, captions: List[str], severity: int, seed: int) -> List[dict]:
    """For one model, compute clean-vs-corrupted cosine at every text layer."""
    device = pick_device()
    print(f"\n=== {model_name} ===")
    adapter = load_model(model_name, device=device)
    n_layers = adapter.n_text_layers
    print(f"  n_text_layers={n_layers}  context_length={adapter.context_length}")

    # Apply severity-{severity} mask deterministically per caption
    rng = random.Random(seed)
    pairs = []
    for c in captions:
        # Each caption gets its OWN seeded rng so corruption is consistent
        # across model runs (we want to compare apples to apples).
        sub_rng = random.Random(rng.randint(0, 2**31))
        corrupted = mask_text(c, severity=severity, rng=sub_rng)
        pairs.append((c, corrupted))

    # Build a per-caption cosine vector across all layers
    all_layers = list(range(n_layers))
    per_layer_cos = [[] for _ in all_layers]

    for clean, corrupted in pairs:
        clean_enc = adapter.encode_text([clean], all_layers)
        corr_enc = adapter.encode_text([corrupted], all_layers)
        for li in all_layers:
            cos = F.cosine_similarity(
                clean_enc.pooled_states[li],
                corr_enc.pooled_states[li],
                dim=-1,
            ).item()
            per_layer_cos[li].append(cos)

    # Free the model before the next one is loaded
    del adapter
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rows = []
    for li in all_layers:
        vals = per_layer_cos[li]
        rows.append(
            {
                "model": model_name,
                "n_text_layers": n_layers,
                "layer_index": li,
                "layer_fraction": (li + 1) / n_layers,
                "n_captions": len(vals),
                "mean_cosine_clean_vs_mask_sev2": sum(vals) / len(vals),
                "min_cosine": min(vals),
                "max_cosine": max(vals),
            }
        )
    return rows


def recommend_early_depth(rows: list[dict], threshold: float, fallback_fraction: float) -> float:
    """Find the smallest layer fraction where mean cosine first drops below threshold.

    If never drops below threshold (model is essentially indifferent to masking,
    or the diagnostic is inconclusive), return the fallback fraction.
    """
    rows = sorted(rows, key=lambda r: r["layer_index"])
    for r in rows:
        if r["mean_cosine_clean_vs_mask_sev2"] < threshold:
            return r["layer_fraction"]
    return fallback_fraction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--severity", type=int, default=2,
                    help="severity of mask corruption (default 2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.95,
                    help="layer fraction where mean cos first drops below this is the recommended 'early' depth")
    ap.add_argument("--fallback-fraction", type=float, default=0.40,
                    help="if cosine never crosses the threshold, fall back to this fraction (default 0.40)")
    ap.add_argument("--n-captions", type=int, default=len(DIAGNOSTIC_CAPTIONS),
                    help=f"how many captions to use (max {len(DIAGNOSTIC_CAPTIONS)})")
    ap.add_argument("--models", nargs="*", default=None,
                    help="restrict to specific models (default: all)")
    ap.add_argument("--out", type=str, default="results/text_depth_diagnostic.csv")
    args = ap.parse_args()

    captions = DIAGNOSTIC_CAPTIONS[: args.n_captions]
    models = args.models or list_models()

    all_rows = []
    recommendations = {}
    for name in models:
        try:
            rows = diagnose_one(name, captions, args.severity, args.seed)
            all_rows.extend(rows)
            rec = recommend_early_depth(rows, args.threshold, args.fallback_fraction)
            recommendations[name] = rec
            # Print compact summary
            short = [(r["layer_index"], round(r["mean_cosine_clean_vs_mask_sev2"], 3))
                     for r in rows if r["layer_index"] in {0, len(rows)//4, len(rows)//2, 3*len(rows)//4, len(rows)-1}]
            print(f"  layer cosine (clean vs sev{args.severity}-mask), sample: {short}")
            print(f"  → recommended early depth fraction: {rec:.3f}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAIL: {type(e).__name__}: {e}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {out}")

    print("\n=== Recommended text early-depth fractions ===")
    for name, frac in recommendations.items():
        print(f"  {name:30s}  {frac:.3f}")
    print(f"\n(threshold={args.threshold}, fallback={args.fallback_fraction})")


if __name__ == "__main__":
    main()
