"""Compute model-independent input-side damage table.

For each (seed, pair_id, modality, corruption_type, severity), replay the EXACT
corruption that the experiment runner used (matching RNG seeding) and compute:

  image:  ssim, psnr, damage_ssim = 1 - ssim
  text:   norm_edit_distance, bleu, damage_bleu = 1 - bleu

Output: results/input_damage.csv

The replay is deterministic — corruption modules use seeded random.Random
instances and the runner's seeding pattern is reproduced here.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.damage_metrics import image_damage, text_damage
from src.corruptions.image import apply_image_corruption
from src.corruptions.severity import VALID_IMAGE_TYPES, VALID_TEXT_TYPES
from src.corruptions.text import apply_text_corruption
from src.data.coco import load_pairs


# Mirrors the seed mixing used in src/runner.py for image and text corruption.
def _img_rng(seed: int, pair_id: int, severity: int) -> random.Random:
    return random.Random((seed << 16) ^ (pair_id << 8) ^ severity)


def _txt_rng(seed: int, pair_id: int, severity: int) -> random.Random:
    return random.Random((seed << 16) ^ (pair_id << 8) ^ severity ^ 0xCAFE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=300, help="match experiment.yaml n_pairs_main")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--out", type=str, default="results/input_damage.csv")
    ap.add_argument("--severities", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    args = ap.parse_args()

    rows = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    for seed in args.seeds:
        offset = seed * args.n_pairs
        print(f"\n=== seed {seed}, offset {offset}, n {args.n_pairs} ===", flush=True)
        triples = list(load_pairs(seed=seed, n=args.n_pairs, offset=offset))

        # Image corruptions
        for vc in VALID_IMAGE_TYPES:
            t0 = time.time()
            for pair_id, (img, _, _) in enumerate(triples):
                for sev in args.severities:
                    rng = _img_rng(seed, pair_id, sev)
                    corrupted = apply_image_corruption(img, vc, sev, rng=rng)
                    s, p = image_damage(img, corrupted)
                    rows.append({
                        "seed": seed,
                        "pair_id": pair_id,
                        "modality": "image",
                        "corruption_type": vc,
                        "severity": sev,
                        "ssim": s,
                        "psnr": p,
                        "damage_ssim": 1.0 - s,
                        "norm_edit_distance": "",
                        "bleu": "",
                        "damage_bleu": "",
                    })
            dt = time.time() - t0
            print(f"  image/{vc:14s}: {len(triples) * len(args.severities):5d} rows in {dt:5.1f}s", flush=True)

        # Text corruptions
        for tc in VALID_TEXT_TYPES:
            t0 = time.time()
            for pair_id, (_, cap, _) in enumerate(triples):
                for sev in args.severities:
                    rng = _txt_rng(seed, pair_id, sev)
                    corrupted = apply_text_corruption(cap, tc, sev, rng=rng)
                    ed, bleu = text_damage(cap, corrupted)
                    rows.append({
                        "seed": seed,
                        "pair_id": pair_id,
                        "modality": "text",
                        "corruption_type": tc,
                        "severity": sev,
                        "ssim": "",
                        "psnr": "",
                        "damage_ssim": "",
                        "norm_edit_distance": ed,
                        "bleu": bleu,
                        "damage_bleu": 1.0 - bleu,
                    })
            dt = time.time() - t0
            print(f"  text /{tc:14s}: {len(triples) * len(args.severities):5d} rows in {dt:5.1f}s", flush=True)

    cols = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    dt = time.time() - t_start
    print(f"\nWrote {len(rows):,} rows to {out_path} in {dt:.1f}s")


if __name__ == "__main__":
    main()
