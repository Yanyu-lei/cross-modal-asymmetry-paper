"""Retrieval sanity check (downstream task translation, not independent validation).

For each (corruption_type, severity), we corrupt the QUERY side, encode in batches,
and rank against the CLEAN candidate set on the other side. Aggregate to
Recall@1/@5/@10 across all queries. One CSV row per (severity, direction, k).

Both retrieval and match_retention derive from cosine similarities — retrieval
ranks against ALL candidates; match_retention measures a margin against a fixed
mismatched pool. The two answer related but not identical questions.
"""
from __future__ import annotations

import logging
import random
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from PIL import Image

from ..models.base import ModelAdapter

LOG = logging.getLogger("retrieval")


@torch.no_grad()
def _batched_encode_images(adapter: ModelAdapter, images: Sequence[Image.Image], batch_size: int = 16) -> torch.Tensor:
    out = []
    for i in range(0, len(images), batch_size):
        batch = list(images[i : i + batch_size])
        # Pass [late depth] only — we only need the pooled embedding, but
        # capture_block_outputs needs at least one depth. Use the last block.
        enc = adapter.encode_image(batch, [adapter.n_image_layers - 1])
        out.append(enc.pooled.detach())
    return torch.cat(out, dim=0)


@torch.no_grad()
def _batched_encode_texts(adapter: ModelAdapter, texts: Sequence[str], batch_size: int = 32) -> torch.Tensor:
    out = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i : i + batch_size])
        enc = adapter.encode_text(batch, [adapter.n_text_layers - 1])
        out.append(enc.pooled.detach())
    return torch.cat(out, dim=0)


def _recall_at_k(query_embs: torch.Tensor, cand_embs: torch.Tensor, ks: Sequence[int]):
    """For each row i in query, the matched candidate is row i. Compute Recall@k.

    Returns: (aggregate: Dict[int, float], per_query: Dict[int, list[int]]).
    The per_query dict has 0/1 indicators per query (matched-was-in-top-k).
    """
    q = F.normalize(query_embs, dim=-1)
    c = F.normalize(cand_embs, dim=-1)
    sims = q @ c.T  # (Q, C); Q == C and matched is the diagonal
    diag = sims.diag().unsqueeze(-1)  # (Q, 1)
    ranks = (sims > diag).sum(dim=-1)  # 0 means matched is the best
    aggregate: Dict[int, float] = {}
    per_query: Dict[int, list] = {}
    for k in ks:
        in_topk = (ranks < k).cpu().numpy().astype(int)
        aggregate[k] = float(in_topk.mean())
        per_query[k] = in_topk.tolist()
    return aggregate, per_query


def run_retrieval(
    adapter: ModelAdapter,
    *,
    images: List[Image.Image],
    captions: List[str],
    base_row: Dict,
    image_corruptors: dict | None = None,  # {corruption_name: callable(img, severity, rng) -> Image}
    text_corruptors: dict | None = None,   # {corruption_name: callable(text, severity, rng) -> str}
    severities: Sequence[int] = (1, 2, 3, 4, 5),
    seed: int = 0,
    image_batch_size: int = 16,
    text_batch_size: int = 32,
    ks: Sequence[int] = (1, 5, 10),
    per_pair_log: bool = False,
    # Legacy single-corruption args (backward compat):
    image_corruptor=None,
    text_corruptor=None,
    image_corruption_name: str | None = None,
    text_corruption_name: str | None = None,
) -> List[Dict]:
    """Compute Recall@k under one or more corruption setups, sharing the clean-candidate encoding.

    Provide either:
      - dicts of corruptors keyed by corruption name (preferred for multi), or
      - single image_corruptor/text_corruptor with their corruption_name (legacy).
    """
    if image_corruptors is None and image_corruptor is not None:
        image_corruptors = {(image_corruption_name or "image"): image_corruptor}
    if text_corruptors is None and text_corruptor is not None:
        text_corruptors = {(text_corruption_name or "text"): text_corruptor}

    n = len(images)
    assert len(captions) == n

    LOG.info("retrieval: encoding %d clean images + %d clean captions", n, n)
    clean_img_embs = _batched_encode_images(adapter, images, batch_size=image_batch_size)
    clean_txt_embs = _batched_encode_texts(adapter, captions, batch_size=text_batch_size)

    rows: List[Dict] = []

    if image_corruptors:
        for corruption_name, corruptor in image_corruptors.items():
            for severity in severities:
                rng_master = random.Random(seed * 31 + severity)
                corr_imgs = []
                for img in images:
                    sub_rng = random.Random(rng_master.randint(0, 2**31))
                    corr_imgs.append(corruptor(img, severity, sub_rng))
                corr_img_embs = _batched_encode_images(adapter, corr_imgs, batch_size=image_batch_size)
                agg, per_query = _recall_at_k(corr_img_embs, clean_txt_embs, ks)
                for k, v in agg.items():
                    row = dict(base_row)
                    row["vision_corruption"] = corruption_name
                    row["text_corruption"] = "none"
                    row["image_severity"] = severity
                    row["text_severity"] = 0
                    row["spoke"] = "retrieval"
                    row["metric"] = f"recall_at_{k}_i2t"
                    row["value"] = float(v)
                    row["n_eval"] = n
                    rows.append(row)
                if per_pair_log:
                    for k in ks:
                        for pair_idx, ind in enumerate(per_query[k]):
                            row = dict(base_row)
                            row["pair_id"] = pair_idx
                            row["vision_corruption"] = corruption_name
                            row["text_corruption"] = "none"
                            row["image_severity"] = severity
                            row["text_severity"] = 0
                            row["spoke"] = "retrieval_per_pair"
                            row["metric"] = f"per_pair_recall_at_{k}_i2t"
                            row["value"] = int(ind)
                            row["n_eval"] = n
                            rows.append(row)
                LOG.info("  i2t %-15s sev=%d  R@1=%.3f R@5=%.3f R@10=%.3f",
                         corruption_name, severity, agg[1], agg[5], agg[10])

    if text_corruptors:
        for corruption_name, corruptor in text_corruptors.items():
            for severity in severities:
                rng_master = random.Random(seed * 53 + severity)
                corr_caps = []
                for c in captions:
                    sub_rng = random.Random(rng_master.randint(0, 2**31))
                    corr_caps.append(corruptor(c, severity, sub_rng))
                corr_txt_embs = _batched_encode_texts(adapter, corr_caps, batch_size=text_batch_size)
                agg, per_query = _recall_at_k(corr_txt_embs, clean_img_embs, ks)
                for k, v in agg.items():
                    row = dict(base_row)
                    row["vision_corruption"] = "none"
                    row["text_corruption"] = corruption_name
                    row["image_severity"] = 0
                    row["text_severity"] = severity
                    row["spoke"] = "retrieval"
                    row["metric"] = f"recall_at_{k}_t2i"
                    row["value"] = float(v)
                    row["n_eval"] = n
                    rows.append(row)
                if per_pair_log:
                    for k in ks:
                        for pair_idx, ind in enumerate(per_query[k]):
                            row = dict(base_row)
                            row["pair_id"] = pair_idx
                            row["vision_corruption"] = "none"
                            row["text_corruption"] = corruption_name
                            row["image_severity"] = 0
                            row["text_severity"] = severity
                            row["spoke"] = "retrieval_per_pair"
                            row["metric"] = f"per_pair_recall_at_{k}_t2i"
                            row["value"] = int(ind)
                            row["n_eval"] = n
                            rows.append(row)
                LOG.info("  t2i %-15s sev=%d  R@1=%.3f R@5=%.3f R@10=%.3f",
                         corruption_name, severity, agg[1], agg[5], agg[10])

    return rows
