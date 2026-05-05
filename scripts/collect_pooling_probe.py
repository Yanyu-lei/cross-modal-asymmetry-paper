"""Mechanistic probe — EOT bottleneck hypothesis.

For each (model, seed, pair, text_corruption, severity, pool_type ∈ {argmax, mean}):
  Compute text_corrupted retention_margin where the *text* side is pooled with
  pool_type, but the rest of the pipeline is identical to the main experiment.

    sim_match           = cos(corrupted_text_pool, clean_image_emb)
    sim_mismatch_mean   = mean cos(corrupted_text_pool, K-image-pool_embs)
    retention_margin    = sim_match - sim_mismatch_mean

The image side always uses the model's standard pooled image embedding — only
the text-pool varies. Both argmax (CLIP convention) and mean are computed from
the SAME forward pass via captured hidden states at the last text-encoder
layer, so the comparison is causal: same model, same input, same hidden
representation, only the pool changes.

Hypothesis: if mean-pool retention drops *less* than argmax-pool retention
under text corruption, the asymmetry is at least partly driven by the
single-token bottleneck of EOT-style pooling.

Output: results/pooling_probe.csv

Usage:
  python -m scripts.collect_pooling_probe                            # all 5 models × all 3 seeds
  python -m scripts.collect_pooling_probe --models openai_clip_b32   # one model
  python -m scripts.collect_pooling_probe --seeds 0 --n-pairs 5      # smoke
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import logging
import random
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.corruptions.severity import VALID_TEXT_TYPES, all_severities
from src.corruptions.text import apply_text_corruption
from src.data.coco import _load_manifest, load_caption_pool, load_pairs, manifest_path
from src.models.registry import list_models, load_model, pick_device

LOG = logging.getLogger("pooling_probe")
DEFAULT_SEEDS = (0, 1, 2)
# "standard" resolves per-adapter to the model's native pool (argmax for CLIP-family,
# last for SigLIP). "mean" is the experimental alternative.
POOL_TYPES = ("standard", "mean")


def _txt_rng(seed: int, pair_id: int, severity: int) -> random.Random:
    return random.Random((seed << 16) ^ (pair_id << 8) ^ severity ^ 0xCAFE)


@torch.no_grad()
def _encode_clean_image_embs(adapter, images: List[Image.Image], batch_size: int = 16) -> torch.Tensor:
    out = []
    for i in range(0, len(images), batch_size):
        out.append(
            adapter.encode_image(list(images[i : i + batch_size]), [adapter.n_image_layers - 1]).pooled.detach()
        )
    return torch.cat(out, dim=0)


@torch.no_grad()
def _encode_text_alt_pools_batched(
    adapter,
    texts: List[str],
    pool_types,
    batch_size: int = 32,
) -> Dict[str, torch.Tensor]:
    """Returns {pool_type: (N, D_proj)} — projected text embeddings per pool_type."""
    accum = {pt: [] for pt in pool_types}
    for i in range(0, len(texts), batch_size):
        out = adapter.encode_text_alt_pools(list(texts[i : i + batch_size]), pool_types=pool_types)
        for pt in pool_types:
            accum[pt].append(out[pt])
    return {pt: torch.cat(accum[pt], dim=0) for pt in pool_types}


def _per_pair_retention(
    text_pooled: torch.Tensor,    # (N, D)  corrupted-text embeddings (one pool_type)
    clean_img_embs: torch.Tensor, # (N, D)  per-pair clean image embeddings
    pool_img_embs: torch.Tensor,  # (K, D)  mismatch-pool clean image embeddings
):
    """Returns three numpy arrays of shape (N,): sim_match, sim_mismatch_mean, retention_margin."""
    t = F.normalize(text_pooled, dim=-1)
    img = F.normalize(clean_img_embs, dim=-1)
    pool_img = F.normalize(pool_img_embs, dim=-1)

    sim_match = (t * img).sum(dim=-1).cpu().numpy()             # (N,)
    pool_sims = (t @ pool_img.T).cpu().numpy()                  # (N, K)
    sim_mismatch = pool_sims.mean(axis=1)                       # (N,)
    margin = sim_match - sim_mismatch
    return sim_match, sim_mismatch, margin


def run_for_model_seed(adapter, *, seed: int, n_pairs: int, k_pool: int, run_tag: str) -> List[Dict]:
    LOG.info("loading %d pairs (seed=%d, offset=%d)", n_pairs, seed, seed * n_pairs)
    triples = list(load_pairs(seed=seed, n=n_pairs, offset=seed * n_pairs))
    eval_idxs = {idx for _, _, idx in triples}
    eval_imgs = [t[0] for t in triples]
    eval_caps = [t[1] for t in triples]

    LOG.info("loading mismatch pool (K=%d, disjoint from eval)", k_pool)
    pool_pairs = load_caption_pool(seed=seed + 10_000, n=k_pool, skip_indices=eval_idxs)
    items = _load_manifest(manifest_path())
    pool_imgs = [Image.open(BytesIO(bytes.fromhex(items[midx]["bytes_hex"]))).convert("RGB")
                 for _, midx in pool_pairs]

    LOG.info("encoding clean image embeddings (eval + pool)")
    clean_img_embs = _encode_clean_image_embs(adapter, eval_imgs)
    pool_img_embs = _encode_clean_image_embs(adapter, pool_imgs)

    timestamp = _dt.datetime.now().isoformat(timespec="seconds")
    rows: List[Dict] = []

    for tc in VALID_TEXT_TYPES:
        for severity in all_severities():
            t0 = time.time()
            corrupted_caps = []
            for pair_id, cap in enumerate(eval_caps):
                rng = _txt_rng(seed, pair_id, severity)
                corrupted_caps.append(apply_text_corruption(cap, tc, severity, rng=rng))

            corr_pools = _encode_text_alt_pools_batched(adapter, corrupted_caps, POOL_TYPES)

            for pt in POOL_TYPES:
                sim_m, sim_mm, margin = _per_pair_retention(
                    corr_pools[pt], clean_img_embs, pool_img_embs
                )
                for pair_id in range(n_pairs):
                    base = {
                        "run_tag": run_tag,
                        "timestamp": timestamp,
                        "model": adapter.name,
                        "seed": seed,
                        "pair_id": pair_id,
                        "text_corruption": tc,
                        "text_severity": severity,
                        "pool_type": pt,
                    }
                    rows.append({**base, "metric": "sim_match", "value": float(sim_m[pair_id])})
                    rows.append({**base, "metric": "sim_mismatch_mean", "value": float(sim_mm[pair_id])})
                    rows.append({**base, "metric": "retention_margin", "value": float(margin[pair_id])})
            LOG.info("  %s sev=%d done (%.1fs)", tc, severity, time.time() - t0)

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="restrict to specific models (default: all 5)")
    ap.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    ap.add_argument("--n-pairs", type=int, default=300)
    ap.add_argument("--k-pool", type=int, default=64)
    ap.add_argument("--out", type=str, default="results/pooling_probe.csv")
    ap.add_argument("--run-tag", type=str, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    models = args.models or list_models()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_tag", "timestamp", "model", "seed", "pair_id",
        "text_corruption", "text_severity", "pool_type",
        "metric", "value",
    ]
    if not out_path.exists() or out_path.stat().st_size == 0:
        with out_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    for m in models:
        for s in args.seeds:
            run_tag = args.run_tag or f"pooling_probe_{m}_seed{s}"
            t0 = time.time()
            LOG.info("=== %s seed=%d ===", m, s)
            adapter = load_model(m, device=pick_device())
            rows = run_for_model_seed(adapter, seed=s, n_pairs=args.n_pairs,
                                      k_pool=args.k_pool, run_tag=run_tag)
            with out_path.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                for r in rows:
                    w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
            LOG.info("=== %s seed=%d  %d rows  %.1fs ===", m, s, len(rows), time.time() - t0)
            del adapter
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
