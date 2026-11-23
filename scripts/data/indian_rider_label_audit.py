"""Stage the local Indian-road label-convention audit by distinct source clip."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random

from PIL import Image, ImageDraw

from scripts.evaluation.compare_human_detection import load_annotation


ROOT = Path(__file__).resolve().parents[2]
SEED = 20261123
CLASSES = ("pedestrian", "rider", "bicycle", "motorcycle")
PARTITIONS = {
    "training_audit": {"crowd": 5, "rider": 12, "pedestrian": 4, "empty": 3},
    "label_bridge": {"crowd": 4, "rider": 8, "pedestrian": 2, "empty": 2},
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose(frames, quotas, prior_groups):
    stable = sorted(frames, key=lambda row: row["image"])
    random.Random(SEED).shuffle(stable)
    used = set(prior_groups)
    selected = []
    for anchor, quota in quotas.items():
        light = Counter()
        rows = [row for row in stable if row["anchor"] == anchor]
        for _ in range(quota):
            available = [row for row in rows if row["clip_id"] not in used]
            if not available:
                raise ValueError(f"Insufficient distinct clips for {anchor}")
            row = min(available, key=lambda candidate: light[candidate["light"]])
            used.add(row["clip_id"])
            light[row["light"]] += 1
            selected.append(row)
    return selected


def source_rows(partition):
    training = partition == "training_audit"
    if training:
        root = ROOT / "datasets/indian_road_manual280_clean_v1"
        manifest = root / "dataset_manifest.csv"
    else:
        root = ROOT / "datasets/indian_manual_eval_v1"
        manifest = root / "manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if training and row["model_split"] != "train":
                continue
            image = root / row["output_image_path" if training else "relative_image_path"]
            folder = root / "manual_annotations"
            if training:
                folder /= "train"
            annotation = folder / (image.stem + ".json")
            legacy = load_annotation(annotation)
            if len(legacy) >= 4:
                anchor = "crowd"
            elif any(box["class_id"] == 1 for box in legacy):
                anchor = "rider"
            else:
                anchor = "pedestrian" if legacy else "empty"
            yield {"image": image, "annotation": annotation, "clip_id": row["clip_id"],
                   "light": row["timeofday"], "anchor": anchor, "legacy": legacy}


def make_preview(source, target):
    with Image.open(source) as original:
        image = original.convert("RGB").resize((1000, 500))
    canvas = Image.new("RGB", (1040, 530), "#101820")
    canvas.paste(image, (40, 30))
    draw = ImageDraw.Draw(canvas, "RGBA")
    for x in range(0, 1000, 100):
        draw.text((40 + x, 5), str(x), fill="white")
        draw.line((40 + x, 30, 40 + x, 529), fill=(255, 255, 255, 75))
    for y in range(0, 500, 50):
        draw.text((3, 30 + y), str(y), fill="white")
        draw.line((40, 30 + y, 1039, 30 + y), fill=(255, 255, 255, 75))
    canvas.save(target, quality=94)


def prepare(out):
    out.mkdir(parents=True)
    review_dir = out / "reviews"
    review_dir.mkdir()
    previews = out / "previews"
    previews.mkdir()
    groups = set()
    hashes = set()
    frames = []
    for part, quotas in PARTITIONS.items():
        for row in choose(source_rows(part), quotas, groups):
            audit_id = f"{len(frames) + 1:02d}"
            image = row["image"]
            image_hash = digest(image)
            if image_hash in hashes:
                raise ValueError("Audit includes identical image content")
            hashes.add(image_hash)
            groups.add(row["clip_id"])
            make_preview(image, previews / f"{audit_id}.jpg")
            record = {**row, "image": image.relative_to(ROOT).as_posix(),
                      "annotation": row["annotation"].relative_to(ROOT).as_posix(),
                      "image_sha256": image_hash, "annotation_sha256": digest(row["annotation"]),
                      "audit_id": audit_id, "partition": part}
            frames.append(record)
            (review_dir / f"{audit_id}.json").write_text(json.dumps(
                {"image_sha256": image_hash, "reviewed": False, "coverage": [], "boxes": []},
                indent=2) + "\n")
    (out / "selection.json").write_text(json.dumps(
        {"seed": SEED, "scope": "development label audit", "frames": frames}, indent=2) + "\n")
    print(json.dumps({"images": len(frames), "distinct_clips": len(groups),
                      "lighting": Counter(row["light"] for row in frames),
                      "selection_sha256": digest(out / "selection.json")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "datasets"):
        raise ValueError("Place the audit in the local datasets directory")
    prepare(output)


if __name__ == "__main__":
    main()
