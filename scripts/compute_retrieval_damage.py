"""Compute input-side damage for the RETRIEVAL pair set (different shuffle from main spokes).

Retrieval uses load_pairs(seed=seed+50000, n=n_pairs_retrieval, offset=0). Inside
retrieval.py the corruption RNG is seeded as
    rng_master = random.Random(seed * 31 + severity)   # i2t (image side)
    rng_master = random.Random(seed * 53 + severity)   # t2i (text side)
    for each input: sub_rng = random.Random(rng_master.randint(0, 2**31))

We replay that exact RNG to get damage values for the exact corrupted inputs
the retrieval pipeline saw. Only joint_grid corruptions are needed:
gaussian_noise (image) and mask (text).

Output: results/input_damage_retrieval.csv  (per (seed, pair_id, modality, severity))
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.analysis.damage_metrics import image_damage, text_damage
from src.corruptions.image import apply_image_corruption
from src.corruptions.severity import VALID_IMAGE_TYPES, VALID_TEXT_TYPES
from src.corruptions.text import apply_text_corruption
from src.data.coco import load_pairs

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--out", type=str, default="results/input_damage_retrieval.csv")
    ap.add_argument("--severities", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    args = ap.parse_args()

    cfg = yaml.safe_load(open(PROJECT_ROOT / "configs" / "experiment.yaml"))
    n_pairs = cfg["n_pairs_retrieval"]
    image_types = list(VALID_IMAGE_TYPES)
    text_types = list(VALID_TEXT_TYPES)

    rows = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    for seed in args.seeds:
        # Match retrieval's pair selection: load_pairs(seed=seed+50000, n=n_pairs, offset=0)
        retrieval_seed = seed + 50_000
        triples = list(load_pairs(seed=retrieval_seed, n=n_pairs, offset=0))
        print(f"\n=== seed {seed} (retrieval seed {retrieval_seed}, n={n_pairs}) ===", flush=True)

        # IMAGE side: all 3 image corruptions
        for vc in image_types:
            for sev in args.severities:
                t0 = time.time()
                rng_master = random.Random(seed * 31 + sev)    # mirrors retrieval.py
                sub_rngs = [random.Random(rng_master.randint(0, 2**31)) for _ in triples]
                for pair_id, ((img, _, _), sub_rng) in enumerate(zip(triples, sub_rngs)):
                    corrupted = apply_image_corruption(img, vc, sev, rng=sub_rng)
                    s, p = image_damage(img, corrupted)
                    rows.append({
                        "seed": seed, "pair_id": pair_id, "modality": "image",
                        "corruption_type": vc, "severity": sev,
                        "ssim": s, "psnr": p, "damage_ssim": 1.0 - s,
                        "norm_edit_distance": "", "bleu": "", "damage_bleu": "",
                    })
                print(f"  image/{vc:14s} sev={sev}: {len(triples)} rows in {time.time()-t0:.1f}s", flush=True)

        # TEXT side: all 3 text corruptions
        for tc in text_types:
            for sev in args.severities:
                t0 = time.time()
                rng_master = random.Random(seed * 53 + sev)    # mirrors retrieval.py
                sub_rngs = [random.Random(rng_master.randint(0, 2**31)) for _ in triples]
                for pair_id, ((_, cap, _), sub_rng) in enumerate(zip(triples, sub_rngs)):
                    corrupted = apply_text_corruption(cap, tc, sev, rng=sub_rng)
                    ed, bleu = text_damage(cap, corrupted)
                    rows.append({
                        "seed": seed, "pair_id": pair_id, "modality": "text",
                        "corruption_type": tc, "severity": sev,
                        "ssim": "", "psnr": "", "damage_ssim": "",
                        "norm_edit_distance": ed, "bleu": bleu, "damage_bleu": 1.0 - bleu,
                    })
                print(f"  text /{tc:14s} sev={sev}: {len(triples)} rows in {time.time()-t0:.1f}s", flush=True)

    cols = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows):,} rows to {out_path} in {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
