"""Pipeline orchestrator.

Per (model, seed):
  1. Load N eval pairs from the COCO manifest (deterministic, seeded shuffle).
  2. Sample a K-caption mismatch pool, disjoint from the eval pairs.
  3. Encode all clean items ONCE and cache:
       - per pair: clean image patch_states at 3 depths, clean text pooled_states at 3 depths,
                   clean image pooled embedding, clean text pooled embedding.
       - across pool: pool_text_embs (K, D) and pool_img_embs (K, D).
  4. For each requested vision corruption x severity, for each pair:
       - re-encode the corrupted image,
       - emit image_fidelity rows (3 depths) and match_retention(image_corrupted) rows.
  5. Symmetrically for each text corruption x severity.
  6. (Optional) joint 5x5 grid: for each (img_sev, txt_sev), encode both corrupted sides
     and emit match_retention(joint) rows.

All rows go through schema.empty_row() and into the long-format CSV.
"""
from __future__ import annotations

import csv
import datetime as _dt
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from functools import partial

from .corruptions.image import apply_image_corruption
from .corruptions.text import apply_text_corruption
from .corruptions.severity import VALID_IMAGE_TYPES, VALID_TEXT_TYPES, all_severities
from .data.coco import load_caption_pool, load_pairs
from .models.base import ImageEncoding, ModelAdapter, TextEncoding
from .schema import COLUMNS, empty_row
from .spokes import image_fidelity, match_retention, retrieval, text_fidelity


LOG = logging.getLogger("runner")


# =============================================================================
# Cached clean encodings
# =============================================================================
@dataclass
class PairEncoding:
    """Cached clean activations for one (model, pair_id)."""
    pair_id: int
    manifest_idx: int
    image: object  # PIL.Image
    caption: str
    image_clean: ImageEncoding
    text_clean: TextEncoding


@dataclass
class PoolEncoding:
    pool_text_embs: torch.Tensor   # (K, D)
    pool_img_embs: torch.Tensor    # (K, D)
    pool_indices: list[int] = field(default_factory=list)


# =============================================================================
# Runner
# =============================================================================
class Runner:
    def __init__(
        self,
        adapter: ModelAdapter,
        *,
        image_depth_fractions: Sequence[float],
        text_depth_fractions: Sequence[float],
        k_pool: int,
        seed: int,
        run_tag: str,
    ):
        self.adapter = adapter
        self.seed = seed
        self.run_tag = run_tag
        self.k_pool = k_pool

        self.image_depths: list[int] = adapter.proportional_depths(image_depth_fractions, modality="image")
        self.text_depths: list[int] = adapter.proportional_depths(text_depth_fractions, modality="text")
        if len(self.image_depths) != 3 or len(self.text_depths) != 3:
            raise ValueError("image_depth_fractions and text_depth_fractions must each have length 3")

        self.pairs: list[PairEncoding] = []
        self.pool: Optional[PoolEncoding] = None

    # -------------------------------------------------------------------------
    # Setup: load pairs, encode clean, build pool
    # -------------------------------------------------------------------------
    def precompute(self, n_pairs: int, *, offset: int = 0) -> None:
        LOG.info("precompute: loading %d pairs (seed=%d, offset=%d)", n_pairs, self.seed, offset)
        triples = list(load_pairs(seed=self.seed, n=n_pairs, offset=offset))
        eval_idxs = {idx for _, _, idx in triples}

        for pair_id, (img, cap, midx) in enumerate(triples):
            img_enc = self.adapter.encode_image([img], self.image_depths)
            txt_enc = self.adapter.encode_text([cap], self.text_depths)
            self.pairs.append(
                PairEncoding(
                    pair_id=pair_id,
                    manifest_idx=midx,
                    image=img,
                    caption=cap,
                    image_clean=img_enc,
                    text_clean=txt_enc,
                )
            )
        LOG.info("precompute: encoded %d clean pairs", len(self.pairs))

        # Build mismatch pool, disjoint from eval pairs
        pool_pairs = load_caption_pool(seed=self.seed + 10_000, n=self.k_pool, skip_indices=eval_idxs)
        # The pool needs IMAGE embeddings too for the text_corrupted direction.
        # Re-resolve pool entries to (img, caption) using load_pairs over the pool indices.
        from .data.coco import _load_manifest, manifest_path
        from io import BytesIO
        from PIL import Image
        items = _load_manifest(manifest_path())
        pool_imgs, pool_caps, pool_idxs = [], [], []
        for cap, midx in pool_pairs:
            entry = items[midx]
            pool_imgs.append(Image.open(BytesIO(bytes.fromhex(entry["bytes_hex"]))).convert("RGB"))
            pool_caps.append(entry["caption"])
            pool_idxs.append(midx)

        # Batch the pool encodings — single-item runs were a 20s+ bottleneck on SO400M.
        text_embs, img_embs = [], []
        BATCH = 16
        for i in range(0, len(pool_caps), BATCH):
            batch_caps = pool_caps[i : i + BATCH]
            text_embs.append(
                self.adapter.encode_text(batch_caps, [self.text_depths[-1]]).pooled.detach()
            )
        for i in range(0, len(pool_imgs), BATCH):
            batch_imgs = pool_imgs[i : i + BATCH]
            img_embs.append(
                self.adapter.encode_image(batch_imgs, [self.image_depths[-1]]).pooled.detach()
            )
        self.pool = PoolEncoding(
            pool_text_embs=torch.cat(text_embs, dim=0),
            pool_img_embs=torch.cat(img_embs, dim=0),
            pool_indices=pool_idxs,
        )
        LOG.info("precompute: pool ready (K=%d, disjoint from eval=%s)",
                 self.k_pool, set(pool_idxs).isdisjoint(eval_idxs))

    # -------------------------------------------------------------------------
    # Helpers to build base rows
    # -------------------------------------------------------------------------
    def _base_row(self) -> Dict:
        row = empty_row()
        row["run_tag"] = self.run_tag
        row["timestamp"] = _dt.datetime.now().isoformat(timespec="seconds")
        row["model"] = self.adapter.name
        row["seed"] = self.seed
        return row

    # -------------------------------------------------------------------------
    # Single-modality vision sweep
    # -------------------------------------------------------------------------
    def run_image_corruption(
        self,
        corruption_type: str,
        severities: Sequence[int] = all_severities(),
    ) -> List[Dict]:
        if corruption_type not in VALID_IMAGE_TYPES:
            raise ValueError(f"unknown vision corruption {corruption_type!r}")
        if self.pool is None:
            raise RuntimeError("call precompute() first")

        rows: List[Dict] = []
        for severity in severities:
            for p in self.pairs:
                rng = random.Random((self.seed << 16) ^ (p.pair_id << 8) ^ severity)
                corr_img = apply_image_corruption(p.image, corruption_type, severity, rng=rng)
                corr_enc = self.adapter.encode_image([corr_img], self.image_depths)

                base = self._base_row()
                base["pair_id"] = p.pair_id
                base["vision_corruption"] = corruption_type
                base["image_severity"] = severity
                base["text_corruption"] = "none"
                base["text_severity"] = 0

                # image_fidelity: 3 rows
                rows.extend(image_fidelity.rows_for_pair(
                    base_row=base,
                    clean=p.image_clean,
                    corrupted=corr_enc,
                    depth_layer_indices=self.image_depths,
                ))
                # match_retention image_corrupted: 3 rows
                rows.extend(match_retention.rows_for_pair(
                    base_row=base,
                    direction="image_corrupted",
                    img_corrupt_emb=corr_enc.pooled,
                    txt_corrupt_emb=None,
                    img_clean_emb=p.image_clean.pooled,
                    txt_clean_emb=p.text_clean.pooled,
                    pool_text_embs=self.pool.pool_text_embs,
                    k_pool=self.k_pool,
                ))
        LOG.info("vision %s done: %d rows", corruption_type, len(rows))
        return rows

    # -------------------------------------------------------------------------
    # Single-modality text sweep
    # -------------------------------------------------------------------------
    def run_text_corruption(
        self,
        corruption_type: str,
        severities: Sequence[int] = all_severities(),
    ) -> List[Dict]:
        if corruption_type not in VALID_TEXT_TYPES:
            raise ValueError(f"unknown text corruption {corruption_type!r}")
        if self.pool is None:
            raise RuntimeError("call precompute() first")

        rows: List[Dict] = []
        for severity in severities:
            for p in self.pairs:
                rng = random.Random((self.seed << 16) ^ (p.pair_id << 8) ^ severity ^ 0xCAFE)
                corr_caption = apply_text_corruption(p.caption, corruption_type, severity, rng=rng)
                corr_enc = self.adapter.encode_text([corr_caption], self.text_depths)

                base = self._base_row()
                base["pair_id"] = p.pair_id
                base["vision_corruption"] = "none"
                base["image_severity"] = 0
                base["text_corruption"] = corruption_type
                base["text_severity"] = severity

                rows.extend(text_fidelity.rows_for_pair(
                    base_row=base,
                    clean=p.text_clean,
                    corrupted=corr_enc,
                    depth_layer_indices=self.text_depths,
                ))
                rows.extend(match_retention.rows_for_pair(
                    base_row=base,
                    direction="text_corrupted",
                    img_corrupt_emb=None,
                    txt_corrupt_emb=corr_enc.pooled,
                    img_clean_emb=p.image_clean.pooled,
                    txt_clean_emb=p.text_clean.pooled,
                    pool_img_embs=self.pool.pool_img_embs,
                    k_pool=self.k_pool,
                ))
        LOG.info("text %s done: %d rows", corruption_type, len(rows))
        return rows

    # -------------------------------------------------------------------------
    # Joint 5x5 grid: both modalities corrupted simultaneously
    # -------------------------------------------------------------------------
    def run_joint_grid(
        self,
        vision_corruption: str,
        text_corruption: str,
        severities: Sequence[int] = all_severities(),
    ) -> List[Dict]:
        """Both modalities corrupted simultaneously; emits match_retention rows
        with direction="joint". Only match_retention is computed (fidelity spokes
        are fully covered by the single-modality sweeps).

        Dominance regression (analysis-time, not here):
            Higher severity = WORSE quality, so the "weaker" modality at a joint
            condition is the one with the HIGHER severity number.
              - max(image_severity, text_severity) -> the worse modality (weakest link)
              - min(image_severity, text_severity) -> the better modality
            The dominance claim ("the worse modality dominates the joint score")
            is supported if `retention_margin ~ max(...)` has higher R^2 than
            `retention_margin ~ min(...)` across all five models.
        """
        if vision_corruption not in VALID_IMAGE_TYPES:
            raise ValueError(f"unknown vision corruption {vision_corruption!r}")
        if text_corruption not in VALID_TEXT_TYPES:
            raise ValueError(f"unknown text corruption {text_corruption!r}")
        if self.pool is None:
            raise RuntimeError("call precompute() first")

        rows: List[Dict] = []
        for img_sev in severities:
            for txt_sev in severities:
                for p in self.pairs:
                    seed_seed = (self.seed << 24) ^ (p.pair_id << 12) ^ (img_sev << 6) ^ txt_sev
                    img_rng = random.Random(seed_seed ^ 0xA1)
                    txt_rng = random.Random(seed_seed ^ 0xB2)
                    corr_img = apply_image_corruption(p.image, vision_corruption, img_sev, rng=img_rng)
                    corr_cap = apply_text_corruption(p.caption, text_corruption, txt_sev, rng=txt_rng)
                    img_enc = self.adapter.encode_image([corr_img], [self.image_depths[-1]])
                    txt_enc = self.adapter.encode_text([corr_cap], [self.text_depths[-1]])

                    base = self._base_row()
                    base["pair_id"] = p.pair_id
                    base["vision_corruption"] = vision_corruption
                    base["image_severity"] = img_sev
                    base["text_corruption"] = text_corruption
                    base["text_severity"] = txt_sev

                    # Joint match-retention: both sides corrupted; compare against clean text mismatch pool.
                    rows.extend(self._joint_match_retention_rows(
                        base_row=base,
                        img_corrupt_emb=img_enc.pooled,
                        txt_corrupt_emb=txt_enc.pooled,
                    ))
        LOG.info("joint grid %s x %s done: %d rows", vision_corruption, text_corruption, len(rows))
        return rows

    # -------------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------------
    def run_retrieval_all_corruptions(
        self,
        *,
        n_pairs: int,
        severities: Sequence[int] = all_severities(),
        offset: int = 0,
        image_batch_size: int = 16,
        text_batch_size: int = 32,
        per_pair_log: bool = False,
    ) -> List[Dict]:
        """Retrieval over ALL 3 image corruptions + ALL 3 text corruptions, sharing the clean-encoding pass."""
        triples = list(load_pairs(seed=self.seed + 50_000, n=n_pairs, offset=offset))
        images = [t[0] for t in triples]
        captions = [t[1] for t in triples]

        base = self._base_row()
        base["pair_id"] = ""

        image_corruptors = {ct: partial_image_corruptor(ct) for ct in VALID_IMAGE_TYPES}
        text_corruptors = {ct: partial_text_corruptor(ct) for ct in VALID_TEXT_TYPES}

        rows = retrieval.run_retrieval(
            self.adapter,
            images=images,
            captions=captions,
            base_row=base,
            image_corruptors=image_corruptors,
            text_corruptors=text_corruptors,
            severities=severities,
            seed=self.seed,
            image_batch_size=image_batch_size,
            text_batch_size=text_batch_size,
            per_pair_log=per_pair_log,
        )
        LOG.info("retrieval (all 6 corruptions) done: %d rows", len(rows))
        return rows

    def run_retrieval(
        self,
        *,
        n_pairs: int,
        vision_corruption: str,
        text_corruption: str,
        severities: Sequence[int] = all_severities(),
        offset: int = 0,
        image_batch_size: int = 16,
        text_batch_size: int = 32,
        per_pair_log: bool = False,
    ) -> List[Dict]:
        """Retrieval Recall@k for one (vision, text) corruption pair across severities.

        Pulls a fresh slice of `n_pairs` from the manifest using a different
        offset namespace so retrieval doesn't trample the main eval set.
        """
        if vision_corruption not in VALID_IMAGE_TYPES:
            raise ValueError(f"unknown vision corruption {vision_corruption!r}")
        if text_corruption not in VALID_TEXT_TYPES:
            raise ValueError(f"unknown text corruption {text_corruption!r}")

        triples = list(load_pairs(seed=self.seed + 50_000, n=n_pairs, offset=offset))
        images = [t[0] for t in triples]
        captions = [t[1] for t in triples]

        base = self._base_row()
        base["pair_id"] = ""
        base["vision_corruption"] = vision_corruption
        base["text_corruption"] = text_corruption

        img_corruptor = partial_image_corruptor(vision_corruption)
        txt_corruptor = partial_text_corruptor(text_corruption)

        rows = retrieval.run_retrieval(
            self.adapter,
            images=images,
            captions=captions,
            base_row=base,
            image_corruptor=img_corruptor,
            text_corruptor=txt_corruptor,
            image_corruption_name=vision_corruption,
            text_corruption_name=text_corruption,
            severities=severities,
            seed=self.seed,
            image_batch_size=image_batch_size,
            text_batch_size=text_batch_size,
            per_pair_log=per_pair_log,
        )
        LOG.info("retrieval (%s, %s) done: %d rows", vision_corruption, text_corruption, len(rows))
        return rows

    def _joint_match_retention_rows(
        self,
        *,
        base_row: Dict,
        img_corrupt_emb: torch.Tensor,
        txt_corrupt_emb: torch.Tensor,
    ) -> List[Dict]:
        sim_match = match_retention.cosine(img_corrupt_emb, txt_corrupt_emb)
        sim_mm = match_retention._cosine_to_pool(img_corrupt_emb, self.pool.pool_text_embs)
        margin = sim_match - sim_mm
        out = []
        for metric, val in (
            ("sim_match", sim_match),
            ("sim_mismatch_mean", sim_mm),
            ("retention_margin", margin),
        ):
            row = dict(base_row)
            row["spoke"] = "match_retention"
            row["match_retention_direction"] = "joint"
            row["metric"] = metric
            row["value"] = float(val)
            row["k_pool"] = int(self.k_pool)
            out.append(row)
        return out


# =============================================================================
# CSV writing
# =============================================================================
def partial_image_corruptor(corruption_type: str):
    def f(img, severity, rng):
        return apply_image_corruption(img, corruption_type, severity, rng=rng)
    return f


def partial_text_corruptor(corruption_type: str):
    def f(text, severity, rng):
        return apply_text_corruption(text, corruption_type, severity, rng=rng)
    return f


def append_rows(rows: List[Dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(COLUMNS), extrasaction="ignore")
        if header_needed:
            w.writeheader()
        for r in rows:
            # Round floats for compactness
            r = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}
            w.writerow(r)
