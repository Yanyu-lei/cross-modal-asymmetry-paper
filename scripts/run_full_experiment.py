"""Full multi-seed experiment driver.

For each (seed, model) the runner appends:
  - all 6 single-modality corruptions x 5 severities x 300 pairs
       -> image_fidelity (3 depths) + match_retention (image_corrupted) per vision row
       -> text_fidelity  (3 depths) + match_retention (text_corrupted)  per text row
  - the 5x5 joint grid (vision_corruption x text_corruption from experiment.yaml)
  - retrieval at experiment.yaml.n_pairs_retrieval (with timing-based fallback)

Order: seed-major, model-minor — so partial CSVs are coherent if interrupted.
The wrapper writes per-(seed, model) command lines and shells out to
run_experiment.py so each subprocess has a fresh Python and a fresh GPU/MPS
allocator state. Append-mode CSV.

Usage:
    .venv/bin/python -m scripts.run_full_experiment
    .venv/bin/python -m scripts.run_full_experiment --models openai_clip_b32 --seeds 0
    .venv/bin/python -m scripts.run_full_experiment --skip-retrieval
"""
from __future__ import annotations

import argparse
import datetime as _dt
import subprocess
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_experiment_yaml() -> dict:
    with open(PROJECT_ROOT / "configs" / "experiment.yaml") as f:
        return yaml.safe_load(f)


def _all_models() -> list[str]:
    with open(PROJECT_ROOT / "configs" / "models.yaml") as f:
        return list(yaml.safe_load(f)["models"].keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="restrict to specific models (default: all 5)")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="restrict to specific seeds (default: experiment.yaml.seeds)")
    ap.add_argument("--pairs", type=int, default=None,
                    help="override n_pairs_main (smoke testing)")
    ap.add_argument("--retrieval-pairs", type=int, default=None,
                    help="override n_pairs_retrieval")
    ap.add_argument("--skip-retrieval", action="store_true")
    ap.add_argument("--skip-joint", action="store_true")
    ap.add_argument("--skip-single", action="store_true")
    ap.add_argument("--retrieval-per-pair", action="store_true",
                    help="emit per-pair Recall@k indicators alongside aggregate retrieval (Job A: severity calibration on R@1)")
    ap.add_argument("--retrieval-all-corruptions", action="store_true",
                    help="run retrieval over all 3 image + 3 text corruption types (paper-grade: consistent Recall@1 metric across F1/F2/F6)")
    ap.add_argument("--out", default=None,
                    help="results CSV path (default: results/full_<timestamp>.csv)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the planned commands and exit")
    args = ap.parse_args()

    cfg = _load_experiment_yaml()
    models = args.models or _all_models()
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    pairs = args.pairs if args.pairs is not None else cfg["n_pairs_main"]

    out = Path(args.out) if args.out else (PROJECT_ROOT / "results" / f"full_{_dt.datetime.now():%Y%m%d_%H%M%S}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    plan: list[list[str]] = []
    for seed in seeds:
        offset = seed * pairs  # disjoint manifest slices per seed
        for model in models:
            cmd = [
                sys.executable, "-m", "scripts.run_experiment",
                "--model", model,
                "--seed", str(seed),
                "--pairs", str(pairs),
                "--offset", str(offset),
                "--out", str(out),
                "--run-tag", f"full_seed{seed}",
            ]
            if not args.skip_single:
                cmd.append("--all-single")
            if not args.skip_joint:
                cmd.append("--joint")
            if not args.skip_retrieval:
                cmd.append("--retrieval")
                if args.retrieval_pairs is not None:
                    cmd.extend(["--retrieval-pairs", str(args.retrieval_pairs)])
                if args.retrieval_per_pair:
                    cmd.append("--retrieval-per-pair")
                if args.retrieval_all_corruptions:
                    cmd.append("--retrieval-all-corruptions")
            plan.append(cmd)

    print(f"Planned {len(plan)} subprocess invocations:")
    for c in plan:
        print(" ", " ".join(c))
    print(f"Output CSV: {out}")
    if args.dry_run:
        return

    t_total = time.time()
    for i, cmd in enumerate(plan, 1):
        t0 = time.time()
        print(f"\n[{i}/{len(plan)}] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        dt = time.time() - t0
        print(f"[{i}/{len(plan)}] returncode={result.returncode}  elapsed={dt:.1f}s")
        if result.returncode != 0:
            print("  *** FAILURE ***  (continuing with remaining jobs; partial CSV is preserved)")

    print(f"\nAll done in {time.time() - t_total:.1f}s. CSV: {out}")


if __name__ == "__main__":
    main()
