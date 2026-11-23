"""Freeze a clip-separated research subset of the BDD100K 2018 detection release."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import os
from pathlib import Path
import random
import re
import time
from urllib.error import HTTPError
from urllib.request import urlopen

from PIL import Image

from scripts.data.freeze_broad_benchmark import ROOT, record, sha
from scripts.evaluation.compare_human_detection import _box


REVISION = "fea86dd9b24ea425d66b70e0fcb6ae0023b5dca5"
BASE = f"https://huggingface.co/datasets/dronefreak/BDD100K/resolve/{REVISION}/data"
SEED = 20261123
WIDTH, HEIGHT = 1280, 720
LABELS = {"person": "pedestrian", "rider": "rider", "bike": "bicycle", "motor": "motorcycle"}
CLASS_IDS = {label: index for index, label in enumerate(LABELS.values())}
METADATA = (("train", "shard_000"), ("train", "shard_004"), ("val", "shard_000"))
COVERAGE = {*CLASS_IDS, "human"}


def target_boxes(rows):
    result = []
    for row in rows:
        label = LABELS.get(row["category"])
        if label is None:
            continue
        b = row["box2d"]
        result.append((label, _box((b["x1"] / WIDTH, b["y1"] / HEIGHT,
                                    b["x2"] / WIDTH, b["y2"] / HEIGHT))))
    return result


def parse_metadata(row):
    objects = row["objects"]
    return target_boxes([
        {"category": category, "box2d": {"x1": x, "y1": y, "x2": x + w, "y2": y + h}}
        for category, (x, y, w, h) in zip(objects["category"], objects["bbox"], strict=True)
    ])


def fetch(relative, limit):
    for attempt in range(4):
        try:
            with urlopen(f"{BASE}/{relative}?download=true", timeout=22) as response:
                content = response.read(limit + 1)
            break
        except OSError as error:
            if attempt == 3 or (isinstance(error, HTTPError) and error.code in {401, 403, 404}):
                raise
            time.sleep(2 ** attempt)
    if len(content) > limit:
        raise ValueError(f"BDD100K resource exceeds size limit: {relative}")
    return content


def read_candidates(root):
    sources = {}
    for split, shard in METADATA:
        path = root / f"{split}_{shard}.jsonl"
        rows = []
        with path.open(encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)
                key = row["file_name"]
                if not re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{8}\.jpg", key):
                    raise ValueError(f"Invalid BDD100K image identifier: {key}")
                try:
                    boxes = parse_metadata(row)
                except ValueError:
                    continue
                counts = Counter(label for label, _ in boxes)
                anchors = set()
                if counts["rider"]:
                    anchors.update(label for label in ("motorcycle", "bicycle") if counts[label])
                if counts["rider"] + counts["pedestrian"] >= 4:
                    anchors.add("crowd")
                if not counts:
                    anchors.add("background")
                if anchors:
                    rows.append((key, split, shard, row["timeofday"], anchors, boxes))
        sources[split, shard] = rows
    return sources


def select(rows, quotas, used_groups):
    order = sorted(rows, key=lambda row: row[0])
    random.Random(SEED).shuffle(order)
    chosen = []
    for anchor, quota in zip(("motorcycle", "bicycle", "crowd", "background"), quotas, strict=True):
        available = 0
        for row in order:
            key = row[0]
            group = key.split("-")[0]
            if group in used_groups or anchor not in row[4]:
                continue
            used_groups.add(group)
            chosen.append((anchor, row))
            available += 1
            if available == quota:
                break
        if available != quota:
            raise ValueError(f"Only {available} of {quota} distinct clip families for {anchor}")
    return chosen


def get_local(path, relative, limit):
    if not path.exists():
        payload = fetch(relative, limit)
        partial = path.with_suffix(path.suffix + ".part")
        partial.write_bytes(payload)
        os.replace(partial, path)
        return payload
    return path.read_bytes()


def prepare(root):
    if (root / "manifest.json").exists():
        raise ValueError("Use a new BDD100K pilot directory")
    candidates = read_candidates(root)
    used = set()
    splits = {"test": select(candidates["val", "shard_000"], (24, 24, 24, 12), used)}
    training = candidates["train", "shard_000"] + candidates["train", "shard_004"]
    splits["val"] = select(training, (16, 16, 16, 8), used)
    splits["train"] = select(training, (80, 80, 80, 32), used)
    jobs = []
    for split, choices in splits.items():
        for folder in ("images", "labels", "source_labels"):
            (root / folder / split).mkdir(parents=True, exist_ok=True)
        jobs.extend((split, *choice) for choice in choices)

    def acquire(job):
        split, anchor, (key, source_split, shard, light, _, metadata_boxes) = job
        stem = key[:-4]
        image = root / "images" / split / key
        image_path = f"images_100k/100k/{source_split}/{shard}/{key}"
        label_path = f"labels/100k/{source_split}/{shard}/{stem}.json"
        source_label = root / "source_labels" / split / (stem + ".json")
        payload = get_local(source_label, label_path, 2_000_000)
        data = json.loads(payload)
        [frame] = data["frames"]
        if data["name"] != stem or frame["timestamp"] != 10000:
            raise ValueError(f"Unexpected BDD100K frame metadata: {key}")
        boxes = target_boxes(frame["objects"])
        if (len(boxes) != len(metadata_boxes) or any(
            label != other_label or any(abs(a - b) > 1e-7 for a, b in zip(box, other_box, strict=True))
            for (label, box), (other_label, other_box) in zip(boxes, metadata_boxes)
        )):
            raise ValueError(f"Mirror index and raw BDD100K boxes differ: {key}")
        with Image.open(BytesIO(get_local(image, image_path, 2_000_000))) as decoded:
            if decoded.format != "JPEG" or decoded.size != (WIDTH, HEIGHT):
                raise ValueError(f"Unexpected BDD100K image format: {key}")
            decoded.load()
        labels = []
        for label, (x1, y1, x2, y2) in boxes:
            values = ((x1 + x2)/2, (y1 + y2)/2, x2 - x1, y2 - y1)
            labels.append(" ".join([str(CLASS_IDS[label]), *(f"{v:.9f}" for v in values)]) + "\n")
        text = "".join(labels)
        (root / "labels" / split / (stem + ".txt")).write_text(text)
        combined = boxes + [("human", box) for label, box in boxes if label in {"pedestrian", "rider"}]
        return record(image, "bdd100k_real_research", stem.split("-")[0], combined, COVERAGE,
                      split=split, release_split=source_split, anchor_class=anchor, light=light,
                      source=f"{BASE}/{image_path}", annotation_sha256=sha(payload),
                      yolo_label_sha256=sha(text.encode()))

    with ThreadPoolExecutor(max_workers=16) as pool:
        frames = list(pool.map(acquire, jobs))
    if len({x["sha256"] for x in frames}) != len(frames):
        raise ValueError("BDD100K cohort repeats image content")
    scope = {"version": 1, "seed": SEED, "scope": "public evaluation", "revision": REVISION}
    (root / "data.yaml").write_text(json.dumps({
        "path": str(root), "train": "images/train", "val": "images/val", "test": "images/test",
        "names": list(CLASS_IDS),
    }, indent=2) + "\n")
    dataset_staging = root / "dataset_manifest.tmp"
    dataset_staging.write_text(json.dumps({**scope, "scope": "research dataset", "frames": frames},
                                         indent=2) + "\n")
    dataset_staging.replace(root / "dataset_manifest.json")
    test_rows = [frame for frame in frames if frame["split"] == "test"]
    (root / "manifest.tmp").write_text(json.dumps({**scope, "frames": test_rows}, indent=2) + "\n")
    (root / "manifest.tmp").replace(root / "manifest.json")
    print(json.dumps({"splits": Counter(x["split"] for x in frames)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "datasets"):
        raise ValueError("Use a BDD100K pilot directory under the local datasets directory")
    prepare(output)


if __name__ == "__main__":
    main()
