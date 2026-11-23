"""Acquire a small, pinned UrbanSyn rider and vehicle diagnostic cohort."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import random
import re
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from PIL import Image

from scripts.data.freeze_broad_benchmark import ROOT, record, sha
from scripts.evaluation.compare_human_detection import _box


SEED = 20261123
REVISION = "8599498aceee9072f1e067a4a4efed7f20b11809"
BASE = f"https://huggingface.co/datasets/UrbanSyn/UrbanSyn/resolve/{REVISION}"
BOXES_URL = "https://pegaso.citic.udc.es/urbansyn/zip/bbox2d.zip"
LABELS = {"person": "pedestrian", "rider": "rider", "bicycle": "bicycle", "motorcycle": "motorcycle"}
WIDTH, HEIGHT = 2048, 1024


def target_boxes(rows):
    boxes = []
    for row in rows:
        label = LABELS.get(row["label"])
        if label is None:
            continue
        b = row["bbox"]
        box = _box((b["xMin"] / WIDTH, b["yMin"] / HEIGHT,
                    b["xMax"] / WIDTH, b["yMax"] / HEIGHT))
        boxes.append((label, box))
        if label in {"pedestrian", "rider"}:
            boxes.append(("human", box))
    return boxes


def fetch(path, limit):
    for attempt in range(3):
        try:
            with urlopen(f"{BASE}/{path}?download=true", timeout=30) as response:
                content = response.read(limit + 1)
            break
        except OSError:
            if attempt == 2:
                raise
    if len(content) > limit:
        raise ValueError(f"UrbanSyn resource exceeds size limit: {path}")
    return content


def sample(annotations):
    order = sorted(annotations)
    random.Random(SEED).shuffle(order)
    selected = {}
    for anchor in ("motorcycle", "bicycle", "crowd"):
        count = 0
        for key in order:
            if key in selected:
                continue
            names = [row["label"] for row in annotations[key]]
            qualifies = (names.count("person") >= 4 if anchor == "crowd"
                         else anchor in names and "rider" in names)
            if not qualifies:
                continue
            selected[key] = anchor
            count += 1
            if count == 12:
                break
        if count != 12:
            raise ValueError(f"Insufficient UrbanSyn examples: {anchor}")
    return selected


def freeze(archive_path, output):
    with ZipFile(archive_path) as archive:
        annotations = {}
        rejected = 0
        for path in archive.namelist():
            match = re.fullmatch(r"bbox2d/bbox2d_([0-9]{4})\.json", path)
            if match is None:
                continue
            rows = json.loads(archive.read(path))
            try:
                target_boxes(rows)
            except ValueError:
                rejected += 1
                continue
            annotations[match[1]] = rows
    selected = sample(annotations)
    folder = output / "images"
    folder.mkdir(parents=True, exist_ok=True)

    def download(key):
        rows = annotations[key]
        pinned = fetch(f"bbox2d/bbox2d_{key}.json", 1024 * 1024)
        if json.loads(pinned) != rows:
            raise ValueError(f"UrbanSyn label archive differs from the pinned revision for {key}")
        image = folder / f"rgb_{key}.png"
        raw = image.read_bytes() if image.exists() else fetch(f"rgb/{image.name}", 12 * 1024 * 1024)
        with Image.open(BytesIO(raw)) as decoded:
            if decoded.format != "PNG" or decoded.size != (WIDTH, HEIGHT):
                raise ValueError(f"Unexpected UrbanSyn image format: {key}")
            decoded.verify()
        if not image.exists():
            partial = image.with_suffix(".png.tmp")
            partial.write_bytes(raw)
            partial.replace(image)
        boxes = target_boxes(rows)
        coverage = {label for label, _ in boxes}
        return record(image, "urbansyn_synthetic", key, boxes, coverage, anchor_class=selected[key],
                      source=f"{BASE}/rgb/{image.name}", license="CC-BY-SA-4.0",
                      annotation_sha256=sha(pinned))

    with ThreadPoolExecutor(max_workers=8) as pool:
        frames = list(pool.map(download, selected))
    if len(set(row["sha256"] for row in frames)) != len(frames):
        raise ValueError("UrbanSyn sample contains duplicate image content")
    text = json.dumps({"version": 1, "scope": "public evaluation", "seed": SEED, "frames": frames,
                       "publisher_revision": REVISION, "annotation_archive": BOXES_URL,
                       "source_frames_rejected_for_geometry": rejected},
                      sort_keys=True, indent=2) + "\n"
    (output / "manifest.tmp").write_text(text)
    (output / "manifest.tmp").replace(output / "manifest.json")
    print(json.dumps({"frames": len(frames), "manifest_sha256": sha(text.encode()),
                      "source_frames_rejected_for_geometry": rejected}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "datasets") or (output / "manifest.json").exists():
        raise ValueError("Use a new manifest under the local datasets directory")
    freeze(args.archive, output)


if __name__ == "__main__":
    main()
