"""Rebuild the image manifests from the released pair lists.

    python -m scripts.build_manifests --dataset coco
    python -m scripts.build_manifests --dataset flickr30k

The pair lists fix the image IDs, captions and order. Existing manifests and outlines
are kept. Move them aside to rebuild them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import file_paths  # noqa: E402

COCO_IMAGE_URL = "http://images.cocodataset.org/train2017/{:012d}.jpg"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
FLICKR30K_JPEG_QUALITY = 95


def _entry(listed: dict, image_bytes: bytes) -> dict:
    return {"id": listed["id"], "caption": listed["caption"], "bytes_hex": image_bytes.hex(),
            "sha256": hashlib.sha256(image_bytes).hexdigest()}


def _download(url: str, attempts: int = 5) -> bytes:
    """Fetch a URL with exponential backoff after request failures."""
    for attempt in range(attempts):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except requests.RequestException as error:
            if attempt == attempts - 1:
                raise
            print(f"retrying {url} after {error}", flush=True)
            time.sleep(2 ** attempt)


def _write_json(data, path: Path) -> None:
    """Save the JSON file only after the temporary file has been written completely."""
    partial = path.with_name(path.name + ".part")
    with open(partial, "w") as f:
        json.dump(data, f)
    partial.rename(path)


def build_coco(listed: list[dict]) -> list[dict]:
    """Fetch the listed COCO images in order and require their released SHA256 hashes."""
    entries = []
    for i, item in enumerate(listed):
        entries.append(_entry(item, _download(COCO_IMAGE_URL.format(item["id"]))))
        if entries[-1]["sha256"] != item["sha256"]:
            sys.exit(f"COCO picture {item['id']} does not match the released sha256.")
        if (i + 1) % 200 == 0:
            print(f"{i + 1} of {len(listed)} pictures", flush=True)
    return entries


def build_flickr30k(listed: list[dict]) -> list[dict]:
    """Load the listed Flickr30K images in order and encode them as JPEGs at quality 95.

    Report hash mismatches without rejecting the images because Pillow versions can differ.
    """
    from datasets import load_dataset

    wanted = {item["id"] for item in listed}
    found: dict[int, bytes] = {}
    for example in load_dataset("nlphuji/flickr30k", split="test", streaming=True):
        image_id = int(example["img_id"])
        if image_id in wanted and image_id not in found:
            buffer = BytesIO()
            example["image"].convert("RGB").save(buffer, format="JPEG", quality=FLICKR30K_JPEG_QUALITY)
            found[image_id] = buffer.getvalue()
            if len(found) == len(wanted):
                break
    missing = wanted - set(found)
    if missing:
        sys.exit(f"{len(missing)} Flickr30K pictures were not found, for example {sorted(missing)[:5]}.")
    entries = [_entry(item, found[item["id"]]) for item in listed]
    differing = sum(e["sha256"] != item["sha256"] for e, item in zip(entries, listed))
    if differing:
        print(f"Note: {differing} of {len(entries)} re-encoded Flickr30K pictures differ in their bytes from the "
              "released list, probably from a different Pillow version.", flush=True)
    return entries


def build_coco_object_outlines(listed: list[dict]) -> dict[int, list]:
    """Map the listed image IDs to object outlines from the COCO 2017 annotations."""
    archive = file_paths.COCO_ANNOTATIONS / "annotations_trainval2017.zip"
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        print("downloading the COCO 2017 annotations", flush=True)
        # Interrupted downloads must not appear to be complete archives.
        partial = archive.with_suffix(".zip.part")
        with requests.get(COCO_ANNOTATIONS_URL, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(partial, "wb") as f:
                for chunk in response.iter_content(1 << 20):
                    f.write(chunk)
        partial.rename(archive)
    with zipfile.ZipFile(archive) as z, z.open("annotations/instances_train2017.json") as f:
        annotations = json.load(f)
    ids = {item["id"] for item in listed}
    objects: dict[int, list] = {i: [] for i in ids}
    for annotation in annotations["annotations"]:
        if annotation["image_id"] in ids:
            objects[annotation["image_id"]].append(annotation["segmentation"])
    return objects


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/<dataset>_2000_pairs_full.json from the released list.")
    parser.add_argument("--dataset", required=True, choices=file_paths.DATASETS)
    dataset = parser.parse_args().dataset

    with open(file_paths.manifest(dataset)) as f:
        listed = json.load(f)
    if file_paths.full_manifest(dataset).exists():
        print(f"{file_paths.full_manifest(dataset)} already exists, so it is kept.")
    else:
        entries = build_coco(listed) if dataset == "coco" else build_flickr30k(listed)
        _write_json(entries, file_paths.full_manifest(dataset))
        print(f"wrote {len(entries)} pairs to {file_paths.full_manifest(dataset)}")

    if dataset == "coco" and file_paths.COCO_OBJECT_OUTLINES.exists():
        print(f"{file_paths.COCO_OBJECT_OUTLINES} already exists, so it is kept.")
    elif dataset == "coco":
        objects = build_coco_object_outlines(listed)
        _write_json(objects, file_paths.COCO_OBJECT_OUTLINES)
        print(f"wrote the object outlines of {sum(1 for v in objects.values() if v)} of {len(objects)} pictures "
              f"to {file_paths.COCO_OBJECT_OUTLINES}")


if __name__ == "__main__":
    main()
