"""Top-level experiment driver.

Examples:
    # Single-modality vision sweep, one model, one seed
    python -m scripts.run_experiment --model openai_clip_b32 --seed 0 \
        --pairs 300 --vision gaussian_noise --out results/run.csv

    # All single-modality corruptions for one model+seed
    python -m scripts.run_experiment --model openai_clip_b32 --seed 0 --pairs 300 --all-single

    # 5x5 joint grid
    python -m scripts.run_experiment --model openai_clip_b32 --seed 0 --pairs 300 --joint

    # Tiny integration smoke
    python -m scripts.run_experiment --model openai_clip_b32 --seed 0 --pairs 5 \
        --vision gaussian_noise --out results/smoke.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.corruptions.severity import VALID_IMAGE_TYPES, VALID_TEXT_TYPES, all_severities
from src.models.registry import get_model_config, load_model, pick_device
from src.runner import Runner, append_rows


def _load_experiment_yaml() -> dict:
    p = Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs", type=int, default=None,
                    help="number of eval pairs (default = experiment.yaml.n_pairs_main)")
    ap.add_argument("--offset", type=int, default=0,
                    help="offset into the seeded shuffle (use distinct values for disjoint slices across seeds)")
    ap.add_argument("--vision", default=None, choices=list(VALID_IMAGE_TYPES) + [None],
                    help="single vision corruption to run")
    ap.add_argument("--text", default=None, choices=list(VALID_TEXT_TYPES) + [None],
                    help="single text corruption to run")
    ap.add_argument("--all-single", action="store_true",
                    help="run all 6 single-modality corruptions")
    ap.add_argument("--joint", action="store_true",
                    help="run the 5x5 joint grid (vision_corr x text_corr from experiment.yaml.joint_grid)")
    ap.add_argument("--retrieval", action="store_true",
                    help="run the retrieval sanity check (uses experiment.yaml.n_pairs_retrieval and joint_grid corruptions)")
    ap.add_argument("--retrieval-pairs", type=int, default=None,
                    help="override n_pairs_retrieval (useful for smoke tests)")
    ap.add_argument("--retrieval-per-pair", action="store_true",
                    help="emit per-pair Recall@k indicators (one row per query) alongside aggregate retrieval")
    ap.add_argument("--retrieval-all-corruptions", action="store_true",
                    help="when --retrieval is set, run over ALL 3 image + 3 text corruption types instead of just the joint_grid pair")
    ap.add_argument("--severities", type=str, default=None,
                    help="comma-separated severities (default 1,2,3,4,5)")
    ap.add_argument("--out", default="results/results.csv")
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    cfg_exp = _load_experiment_yaml()
    cfg_model = get_model_config(args.model)
    n_pairs = args.pairs if args.pairs is not None else cfg_exp["n_pairs_main"]
    severities = [int(s) for s in (args.severities.split(",") if args.severities else all_severities())]
    run_tag = args.run_tag or f"{args.model}+seed{args.seed}"

    device = pick_device()
    logging.info("Loading %s on %s", args.model, device)
    t0 = time.time()
    adapter = load_model(args.model, device=device)
    logging.info("Loaded in %.1fs", time.time() - t0)

    runner = Runner(
        adapter=adapter,
        image_depth_fractions=cfg_model["image_depth_fractions"],
        text_depth_fractions=cfg_model["text_depth_fractions"],
        k_pool=cfg_exp["match_retention"]["k_pool"],
        seed=args.seed,
        run_tag=run_tag,
    )

    t0 = time.time()
    runner.precompute(n_pairs, offset=args.offset)
    logging.info("Precompute done in %.1fs", time.time() - t0)

    out = Path(args.out)
    total_rows = 0

    def _flush(rows):
        nonlocal total_rows
        if not rows:
            return
        append_rows(rows, out)
        total_rows += len(rows)
        logging.info("Wrote %d rows (total %d) -> %s", len(rows), total_rows, out)

    if args.all_single:
        for v in VALID_IMAGE_TYPES:
            _flush(runner.run_image_corruption(v, severities))
        for t in VALID_TEXT_TYPES:
            _flush(runner.run_text_corruption(t, severities))
    else:
        if args.vision:
            _flush(runner.run_image_corruption(args.vision, severities))
        if args.text:
            _flush(runner.run_text_corruption(args.text, severities))

    if args.joint:
        jg = cfg_exp["joint_grid"]
        _flush(runner.run_joint_grid(jg["vision_corruption"], jg["text_corruption"], jg["severities"]))

    if args.retrieval:
        n_ret = args.retrieval_pairs if args.retrieval_pairs is not None else cfg_exp["n_pairs_retrieval"]
        if args.retrieval_all_corruptions:
            _flush(runner.run_retrieval_all_corruptions(
                n_pairs=n_ret,
                severities=severities,
                per_pair_log=args.retrieval_per_pair,
            ))
        else:
            jg = cfg_exp["joint_grid"]
            _flush(runner.run_retrieval(
                n_pairs=n_ret,
                vision_corruption=jg["vision_corruption"],
                text_corruption=jg["text_corruption"],
                severities=severities,
                per_pair_log=args.retrieval_per_pair,
            ))

    logging.info("Done. Total rows: %d  ->  %s", total_rows, out.resolve())


if __name__ == "__main__":
    main()
