"""Compute clean (no corruption) Recall@k for all 5 models × 3 seeds × n retrieval pairs.

Per (model, seed): encode the same 1000 retrieval pairs that the corrupted runs use
(seed + 50000 offset, matching `Runner.run_retrieval_all_corruptions`), compute
diagonal Recall@k for i2t and t2i. Logs both aggregate and per-pair indicators
so bootstrap CIs can be hierarchical.

Output: results/clean_baseline_retrieval.csv
Schema: run_tag,timestamp,model,seed,pair_id,vision_corruption,image_severity,
        text_corruption,text_severity,spoke,depth,depth_layer_index,
        match_retention_direction,metric,value,k_pool,n_eval,notes

Spoke values used: 'clean_baseline' (aggregate rows, pair_id="") and
'clean_baseline_per_pair' (per-pair rows). Metrics: clean_recall_at_{k}_{i2t|t2i}
and per_pair_clean_recall_at_{k}_{i2t|t2i}.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.coco import load_pairs
from src.models.registry import list_models, load_model, pick_device
from src.schema import COLUMNS, empty_row

LOG = logging.getLogger("clean_baseline")
DEFAULT_SEEDS = (0, 1, 2)
KS = (1, 5, 10)


@torch.no_grad()
def _batched_encode_images(adapter, images, batch_size: int = 16) -> torch.Tensor:
    out = []
    for i in range(0, len(images), batch_size):
        out.append(adapter.encode_image(list(images[i : i + batch_size]), [adapter.n_image_layers - 1]).pooled.detach())
    return torch.cat(out, dim=0)


@torch.no_grad()
def _batched_encode_texts(adapter, texts, batch_size: int = 32) -> torch.Tensor:
    out = []
    for i in range(0, len(texts), batch_size):
        out.append(adapter.encode_text(list(texts[i : i + batch_size]), [adapter.n_text_layers - 1]).pooled.detach())
    return torch.cat(out, dim=0)


def _diagonal_recall_at_k(query_embs: torch.Tensor, cand_embs: torch.Tensor, ks):
    q = F.normalize(query_embs, dim=-1)
    c = F.normalize(cand_embs, dim=-1)
    sims = q @ c.T
    diag = sims.diag().unsqueeze(-1)
    ranks = (sims > diag).sum(dim=-1)
    aggregate: Dict[int, float] = {}
    per_query: Dict[int, list] = {}
    for k in ks:
        in_topk = (ranks < k).cpu().numpy().astype(int)
        aggregate[k] = float(in_topk.mean())
        per_query[k] = in_topk.tolist()
    return aggregate, per_query


def run_for_model_seed(adapter, *, seed: int, n_pairs: int, run_tag: str) -> List[Dict]:
    LOG.info("loading %d retrieval pairs (seed=%d+50000)", n_pairs, seed)
    triples = list(load_pairs(seed=seed + 50_000, n=n_pairs, offset=0))
    images = [t[0] for t in triples]
    captions = [t[1] for t in triples]

    t0 = time.time()
    LOG.info("encoding %d clean images", n_pairs)
    img_embs = _batched_encode_images(adapter, images)
    LOG.info("encoding %d clean captions", n_pairs)
    txt_embs = _batched_encode_texts(adapter, captions)

    LOG.info("computing diagonal Recall@k (i2t)")
    i2t_agg, i2t_per = _diagonal_recall_at_k(img_embs, txt_embs, KS)
    LOG.info("computing diagonal Recall@k (t2i)")
    t2i_agg, t2i_per = _diagonal_recall_at_k(txt_embs, img_embs, KS)

    LOG.info("clean baseline encode+score in %.1fs", time.time() - t0)
    LOG.info("  i2t R@1=%.4f R@5=%.4f R@10=%.4f", i2t_agg[1], i2t_agg[5], i2t_agg[10])
    LOG.info("  t2i R@1=%.4f R@5=%.4f R@10=%.4f", t2i_agg[1], t2i_agg[5], t2i_agg[10])

    timestamp = _dt.datetime.now().isoformat(timespec="seconds")
    rows: List[Dict] = []

    def _base() -> Dict:
        r = empty_row()
        r["run_tag"] = run_tag
        r["timestamp"] = timestamp
        r["model"] = adapter.name
        r["seed"] = seed
        r["vision_corruption"] = "none"
        r["text_corruption"] = "none"
        r["image_severity"] = 0
        r["text_severity"] = 0
        r["n_eval"] = n_pairs
        return r

    # Aggregate rows
    for direction, agg in (("i2t", i2t_agg), ("t2i", t2i_agg)):
        for k, v in agg.items():
            r = _base()
            r["pair_id"] = ""
            r["spoke"] = "clean_baseline"
            r["metric"] = f"clean_recall_at_{k}_{direction}"
            r["value"] = float(v)
            rows.append(r)

    # Per-pair rows
    for direction, per in (("i2t", i2t_per), ("t2i", t2i_per)):
        for k in KS:
            for pair_idx, ind in enumerate(per[k]):
                r = _base()
                r["pair_id"] = pair_idx
                r["spoke"] = "clean_baseline_per_pair"
                r["metric"] = f"per_pair_clean_recall_at_{k}_{direction}"
                r["value"] = int(ind)
                rows.append(r)

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="restrict to specific models (default: all 5)")
    ap.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    ap.add_argument("--n-pairs", type=int, default=1000,
                    help="must match retrieval n_pairs (1000 in experiment.yaml)")
    ap.add_argument("--out", type=str, default="results/clean_baseline_retrieval.csv")
    ap.add_argument("--run-tag", type=str, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    models = args.models or list_models()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists() or out_path.stat().st_size == 0:
        with out_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=list(COLUMNS)).writeheader()

    t_total = time.time()
    for m in models:
        for s in args.seeds:
            run_tag = args.run_tag or f"clean_baseline_{m}_seed{s}"
            t0 = time.time()
            LOG.info("=== %s seed=%d ===", m, s)
            adapter = load_model(m, device=pick_device())
            rows = run_for_model_seed(adapter, seed=s, n_pairs=args.n_pairs, run_tag=run_tag)
            with out_path.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(COLUMNS), extrasaction="ignore")
                for r in rows:
                    w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
            LOG.info("=== %s seed=%d  %d rows  %.1fs ===", m, s, len(rows), time.time() - t0)
            del adapter
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    LOG.info("\nAll done in %.1fs. CSV: %s", time.time() - t_total, out_path)


if __name__ == "__main__":
    main()
