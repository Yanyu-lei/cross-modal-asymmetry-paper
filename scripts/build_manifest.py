"""Build/extend the COCO manifest used by all experiment runs.

We need:
  - 3 seeds × 300 pairs (main spokes)        = 900 distinct pairs (slices may overlap; we pre-fetch enough)
  - 1 seed × 1000 pairs (retrieval, optional) = 1000
  - K=64 mismatch-pool captions per seed; pool stays disjoint from eval pairs

Pre-fetching ~2000 unique items keeps every later (seed, offset) slice satisfied.

Usage:
    .venv/bin/python -m scripts.build_manifest --target 2000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.coco import build_manifest, manifest_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=2000,
                    help="manifest size to ensure (default 2000)")
    ap.add_argument("--hf-seed", type=int, default=0,
                    help="HF dataset shuffle seed for the underlying COCO stream")
    args = ap.parse_args()

    path = manifest_path()
    print(f"Manifest path: {path}")
    final = build_manifest(args.target, hf_seed=args.hf_seed)
    print(f"Final manifest size: {final}")


if __name__ == "__main__":
    main()
