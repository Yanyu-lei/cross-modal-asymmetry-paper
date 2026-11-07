"""Run experiment jobs and write their per-job CSV outputs."""
from __future__ import annotations

import csv
import datetime as _dt
import json
import logging
import os
from pathlib import Path

import torch

from src import file_paths, settings
from src import fidelity_retention_retrieval_metrics as metrics
from src.corruptions import corrupt_caption, corrupt_image, joint_grid_rngs, pair_rngs
from src.load_data import fidelity_retention_pairs, negative_pairs, retrieval_pairs
from src.load_models import encode_captions, encode_images, load_model

LOG = logging.getLogger("job")

# Each row records one measurement; inapplicable fields stay empty.
COLUMNS: tuple[str, ...] = (
    "run_tag",
    "timestamp",
    "model",
    "seed",
    "pair_id",                    # The index refers to this seed's draw; aggregate rows leave it empty.
    "vision_corruption",
    "image_severity",             # Zero denotes a clean input.
    "text_corruption",
    "text_severity",
    "spoke",
    "depth",                      # Fidelity rows use early, mid or late.
    "depth_layer_index",
    "match_retention_direction",
    "metric",
    "value",
    "n_negatives",
    "n_eval",                     # Retrieval rows record the gallery size.
    "notes",
)


def empty_row() -> dict:
    return {c: "" for c in COLUMNS}


def _base(run: str, model: str, seed: int, image_kind="none", image_severity=0, text_kind="none", text_severity=0) -> dict:
    return {**empty_row(), "run_tag": run, "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "model": model, "seed": seed, "vision_corruption": image_kind, "image_severity": image_severity,
            "text_corruption": text_kind, "text_severity": text_severity}


class FidelityRetention:
    """Cache clean encodings and disjoint negatives for one model and seed.

    Clean pairs and damaged inputs are encoded one at a time, and the negatives in batches
    of 16. Changing the batch sizes can change the last digits, so these sizes stay fixed.
    """

    def __init__(self, run: str, adapter, dataset: str, seed: int, n_pairs: int):
        self.run, self.adapter, self.seed = run, adapter, seed
        cfg = settings.MODEL_SETTINGS[adapter.name]
        self.depths = {"image": adapter.proportional_depths(cfg["image_depth_fractions"], "image"),
                       "text": adapter.proportional_depths(cfg["text_depth_fractions"], "text")}
        self.pairs = fidelity_retention_pairs(dataset, seed, n_pairs)
        self.clean = {"image": [adapter.encode_image([p.image], self.depths["image"]) for p in self.pairs],
                      "text": [adapter.encode_text([p.caption], self.depths["text"]) for p in self.pairs]}
        negatives = negative_pairs(dataset, seed, settings.N_NEGATIVES, exclude={p.position for p in self.pairs})
        # The negatives of a damaged image are captions, and those of a damaged caption are images.
        self.negatives = {"image": encode_captions(adapter, [n.caption for n in negatives],
                                                   batch_size=settings.NEGATIVE_CAPTION_BATCH_SIZE),
                          "text": encode_images(adapter, [n.image for n in negatives])}

    def _encode(self, side: str, item, depths):
        return self.adapter.encode_image([item], depths) if side == "image" else self.adapter.encode_text([item], depths)

    def single_rows(self, side: str, kind: str) -> list[dict]:
        """Measure fidelity and retention for one corruption at every severity."""
        other = "text" if side == "image" else "image"
        rows = []
        for severity in settings.SEVERITIES:
            rngs = pair_rngs("fidelity_retention", self.seed, severity, len(self.pairs), side)
            for i, (pair, rng) in enumerate(zip(self.pairs, rngs)):
                if side == "image":
                    damaged = self._encode("image", corrupt_image(pair.image, kind, severity, rng=rng), self.depths["image"])
                    base = _base(self.run, self.adapter.name, self.seed, kind, severity)
                else:
                    damaged = self._encode("text", corrupt_caption(pair.caption, kind, severity, rng=rng), self.depths["text"])
                    base = _base(self.run, self.adapter.name, self.seed, text_kind=kind, text_severity=severity)
                base["pair_id"] = i
                rows += metrics.fidelity_rows(base, side, self.clean[side][i], damaged, self.depths[side])
                margin = metrics.retention_margin(damaged.pooled, self.clean[other][i].pooled, self.negatives[side])
                rows.append(metrics.retention_row(base, f"{side}_corrupted", margin, settings.N_NEGATIVES))
        return rows

    def grid_rows(self) -> list[dict]:
        """Compare each damaged image with its damaged caption and clean caption negatives."""
        image_kind = settings.NOISE_MASK_GRID["image_corruption"]
        text_kind = settings.NOISE_MASK_GRID["text_corruption"]
        rows = []
        for image_severity in settings.SEVERITIES:
            for text_severity in settings.SEVERITIES:
                for i, pair in enumerate(self.pairs):
                    image_rng, text_rng = joint_grid_rngs(self.seed, i, image_severity, text_severity)
                    image = self._encode("image", corrupt_image(pair.image, image_kind, image_severity, rng=image_rng),
                                         [self.depths["image"][-1]])
                    caption = self._encode("text", corrupt_caption(pair.caption, text_kind, text_severity, rng=text_rng),
                                           [self.depths["text"][-1]])
                    base = _base(self.run, self.adapter.name, self.seed, image_kind, image_severity, text_kind, text_severity)
                    base["pair_id"] = i
                    margin = metrics.retention_margin(image.pooled, caption.pooled, self.negatives["image"])
                    rows.append(metrics.retention_row(base, "joint", margin, settings.N_NEGATIVES))
        return rows

    def clean_rows(self) -> list[dict]:
        """Score each undamaged pair in both directions at severity 0 from the cached encodings."""
        rows = []
        for i in range(len(self.pairs)):
            base = _base(self.run, self.adapter.name, self.seed)
            base["pair_id"] = i
            for side, other, direction in (("image", "text", "clean_image_query"), ("text", "image", "clean_caption_query")):
                margin = metrics.retention_margin(self.clean[side][i].pooled, self.clean[other][i].pooled, self.negatives[side])
                rows.append(metrics.retention_row(base, direction, margin, settings.N_NEGATIVES))
        return rows


def fidelity_retention_job(run: str, adapter, dataset: str, seed: int, n_pairs: int) -> tuple[list[dict], list[dict]]:
    """Return the fidelity, retention and pooling-ablation rows for one model and seed."""
    from src.text_pooling_ablation_and_depth_diagnostic import clean_caption_pooling_rows, text_pooling_ablation_rows

    job = FidelityRetention(run, adapter, dataset, seed, n_pairs)
    rows = []
    for kind in settings.IMAGE_CORRUPTIONS:
        rows += job.single_rows("image", kind)
    for kind in settings.TEXT_CORRUPTIONS:
        rows += job.single_rows("text", kind)
    rows += job.grid_rows()
    rows += job.clean_rows()
    return rows, (text_pooling_ablation_rows(run, adapter, dataset, seed, n_pairs)
                  + clean_caption_pooling_rows(run, adapter, dataset, seed, n_pairs))


def object_outlines_by_image(dataset: str) -> dict[int, list]:
    if dataset != "coco":
        raise ValueError("object outlines exist only for COCO")
    with open(file_paths.COCO_OBJECT_OUTLINES) as f:
        return {int(k): v for k, v in json.load(f).items()}


def retrieval_job(run: str, adapter, dataset: str, seed: int, n_pairs: int, image_kinds, text_kinds,
                  gallery_sizes=settings.GALLERY_SIZES, per_pair_gallery_sizes=settings.PER_PAIR_GALLERY_SIZES) -> list[dict]:
    """Score clean and damaged queries against nested galleries of clean candidates."""
    pairs = retrieval_pairs(dataset, seed, n_pairs)
    sizes = [s for s in gallery_sizes if s < n_pairs] + [n_pairs]
    clean = {"image": encode_images(adapter, [p.image for p in pairs]),
             "text": encode_captions(adapter, [p.caption for p in pairs])}
    objects = object_outlines_by_image(dataset) if "object_cutout" in image_kinds else {}
    score = dict(gallery_sizes=sizes, per_pair_gallery_sizes=per_pair_gallery_sizes, ks=settings.RECALL_KS)

    rows = []
    for direction, query, gallery in (("i2t", "image", "text"), ("t2i", "text", "image")):
        rows += metrics.retrieval_rows(_base(run, adapter.name, seed), direction, clean[query], clean[gallery],
                                       clean=True, **score)
    for side, kinds, direction, other in (("image", image_kinds, "i2t", "text"), ("text", text_kinds, "t2i", "image")):
        for kind in kinds:
            for severity in settings.SEVERITIES:
                rngs = pair_rngs("retrieval", seed, severity, n_pairs, side)
                if side == "image":
                    damaged = encode_images(adapter, [corrupt_image(p.image, kind, severity, rng=r, objects=objects.get(p.image_id))
                                                      for p, r in zip(pairs, rngs)])
                    base = _base(run, adapter.name, seed, kind, severity)
                else:
                    damaged = encode_captions(adapter, [corrupt_caption(p.caption, kind, severity, rng=r)
                                                        for p, r in zip(pairs, rngs)])
                    base = _base(run, adapter.name, seed, text_kind=kind, text_severity=severity)
                rows += metrics.shared_space_similarity_rows(base, direction, damaged, clean[side])
                rows += metrics.retrieval_rows(base, direction, damaged, clean[other], clean=False, **score)
                LOG.info("%s %s level %d done", direction, kind, severity)
    return rows


def write_csv(rows: list[dict], path: Path, columns=None, *, round_to: int | None = 6) -> None:
    """Write rows to a temporary CSV and atomically move it into place.

    Model values are rounded to six decimals; damage and depth diagnostics retain full precision.
    Exact damage values are needed for quintile boundaries.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    columns = list(columns or rows[0].keys())
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(v, round_to) if isinstance(v, float) and round_to is not None else v)
                             for k, v in row.items()})
    os.replace(tmp, path)


def job_parts(job: dict) -> list[Path]:
    """Return the paths to a job's part files under results/model_outputs/parts/."""
    folder = file_paths.part_dir(job["run"])
    if job["kind"] == "damage":
        return [folder / f"damage_seed{job['seed']}.csv"]
    if job["kind"] == "text_depth_diagnostic":
        return [folder / f"{job['model']}.csv"]
    name = f"{job['model']}_seed{job['seed']}"
    if job["kind"] == "fidelity_retention":
        return [folder / f"{name}.csv", folder / f"{name}_text_pooling.csv"]
    return [folder / f"{name}.csv"]


def run_job(job: dict) -> None:
    """Execute one job in a fresh process on the device selected by the runner."""
    logging.basicConfig(level=logging.WARNING, format=f"%(asctime)s [{job['run']} {job.get('model', '')} "
                                                   f"seed {job.get('seed', '')}] %(message)s")
    LOG.setLevel(logging.INFO)  # Show job progress without enabling verbose library logs.
    parts = job_parts(job)
    if job["kind"] == "damage":
        from src.input_damage_metrics import damage_rows
        write_csv(damage_rows(job["run_kind"], job["dataset"], job["seed"], job["pairs"],
                              job["image_corruptions"], job["text_corruptions"]), parts[0], round_to=None)
        return
    if job["kind"] == "text_depth_diagnostic":
        from src.text_pooling_ablation_and_depth_diagnostic import text_depth_rows
        write_csv(text_depth_rows(job["model"]), parts[0], round_to=None)
        return

    adapter = load_model(job["model"])
    with torch.no_grad():
        if job["kind"] == "fidelity_retention":
            rows, ablation = fidelity_retention_job(job["run"], adapter, job["dataset"], job["seed"], job["pairs"])
            write_csv(ablation, parts[1])
            write_csv(rows, parts[0], COLUMNS)
        else:
            rows = retrieval_job(job["run"], adapter, job["dataset"], job["seed"], job["pairs"],
                                 job["image_corruptions"], job["text_corruptions"])
            write_csv(rows, parts[0], COLUMNS)
