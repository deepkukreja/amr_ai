"""Stage real AMR image exports for review and freeze session-separated labels."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from PIL import Image

from scripts.evaluation.compare_human_detection import _box


FIELDS = {"image", "site_id", "session_id", "camera", "capture_utc", "scene", "lighting", "split"}
SPLITS = ("development", "holdout")
LABELS = {
    "pedestrian", "rider", "bicycle", "motorcycle", "car", "van", "truck", "bus",
    "forklift", "pallet_jack", "cart", "pallet", "rack_or_shelf", "bin_or_tote", "box_or_load",
    "dog", "cat", "cattle", "goat", "sheep", "horse", "bird",
}


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def encode(data):
    return json.dumps(data, sort_keys=True, indent=2) + "\n"


def batch_path(batch, relative):
    path = (batch / relative).resolve()
    if not path.is_relative_to(batch.resolve()):
        raise ValueError(f"Input escapes its batch directory: {relative}")
    return path


def check_groups(frames):
    splits = {}
    hashes = set()
    for row in frames:
        session = (row["site_id"], row["session_id"])
        if row["split"] not in SPLITS or splits.setdefault(session, row["split"]) != row["split"]:
            raise ValueError(f"Capture session has an invalid or mixed split: {session}")
        if row["sha256"] in hashes:
            raise ValueError("Capture contains an exact duplicate image")
        hashes.add(row["sha256"])


def stage_batch(index, output, exclude_manifest=None):
    if output.exists():
        raise ValueError("Choose a new field batch directory")
    excluded = set()
    if exclude_manifest is not None:
        excluded = {row["sha256"] for row in json.loads(exclude_manifest.read_text())["frames"]}
    base = index.parent.resolve()
    with index.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if set(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Unexpected capture CSV header")
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
    if not rows:
        raise ValueError("Capture CSV has no image rows")
    files = []
    frames = []
    for index_number, row in enumerate(rows, 1):
        if any(not row[key] for key in FIELDS):
            raise ValueError(f"Capture row {index_number} has an empty field")
        timestamp = datetime.fromisoformat(row["capture_utc"].replace("Z", "+00:00"))
        if timestamp.utcoffset() != timedelta(0):
            raise ValueError("capture_utc requires an explicit UTC timezone")
        row["capture_utc"] = timestamp.isoformat(timespec="auto")
        source = batch_path(base, row["image"])
        if source.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ValueError(f"Unsupported frame format: {source.name}")
        with Image.open(source) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError(f"Unsupported frame encoding: {source.name}")
            width, height = image.size
            image.verify()
        digest = sha(source)
        if digest in excluded:
            raise ValueError(f"Frame is already in the excluded development manifest: {source.name}")
        frame_id = f"frame-{index_number:05d}"
        frames.append({**row, "image": f"images/{frame_id}{source.suffix.lower()}",
                       "annotation": f"annotations/{frame_id}.json", "sha256": digest,
                       "width": width, "height": height})
        files.append(source)
    check_groups(frames)

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".field-stage-", dir=output.parent) as temp:
        batch = Path(temp) / "batch"
        (batch / "images").mkdir(parents=True)
        (batch / "annotations").mkdir()
        for source, row in zip(files, frames):
            destination = batch / row["image"]
            shutil.copyfile(source, destination)
            if sha(destination) != row["sha256"]:
                raise ValueError(f"Image changed while staging: {source.name}")
            annotation = {"image_sha256": row["sha256"], "reviewed": False, "coverage": [], "boxes": []}
            (batch / row["annotation"]).write_text(encode(annotation), encoding="utf-8")
        manifest = {"version": 1, "input_csv_sha256": sha(index), "frames": frames}
        if exclude_manifest is not None:
            manifest["excluded_manifest_sha256"] = sha(exclude_manifest)
        (batch / "capture_manifest.json").write_text(encode(manifest), encoding="utf-8")
        batch.rename(output)
    return output / "capture_manifest.json"


def freeze_batch(batch, split):
    if split not in SPLITS:
        raise ValueError("Unknown field split")
    output = batch / f"{split}.json"
    if output.exists():
        raise ValueError("Use a new version for a corrected field split")
    manifest = batch / "capture_manifest.json"
    data = json.loads(manifest.read_bytes())
    if data["version"] != 1:
        raise ValueError("Unsupported field manifest")
    check_groups(data["frames"])
    frames = []
    for row in data["frames"]:
        if row["split"] != split:
            continue
        image = batch_path(batch, row["image"])
        if sha(image) != row["sha256"]:
            raise ValueError(f"Staged image changed: {image.name}")
        annotation = batch_path(batch, row["annotation"])
        payload = annotation.read_bytes()
        review = json.loads(payload)
        if review["reviewed"] is not True or review["image_sha256"] != row["sha256"]:
            raise ValueError(f"Image requires review or review refers to different pixels: {image.name}")
        classes = review["coverage"]
        if not isinstance(classes, list) or not classes or not set(classes) <= LABELS:
            raise ValueError(f"Declare explicit reviewed class coverage: {image.name}")
        coverage = sorted(set(classes))
        boxes = []
        for item in review["boxes"]:
            label = item["label"]
            if label not in coverage:
                raise ValueError(f"Box label is outside reviewed coverage: {image.name} ({label})")
            boxes.append({"label": label, "box": _box(item["box"])})
        frames.append({**row, "coverage": coverage, "boxes": boxes,
                       "annotation_sha256": hashlib.sha256(payload).hexdigest()})
    if not frames:
        raise ValueError(f"No field frames assigned to {split}")
    report = {"version": 1, "scope": f"field {split}", "capture_manifest_sha256": sha(manifest),
              "sessions": len({(row["site_id"], row["session_id"]) for row in frames}),
              "label_counts": dict(Counter(box["label"] for row in frames for box in row["boxes"])),
              "frames": frames}
    with output.open("x", encoding="utf-8") as file:
        file.write(encode(report))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    stage = actions.add_parser("stage")
    stage.add_argument("--index", type=Path, required=True)
    stage.add_argument("--output", type=Path, required=True)
    stage.add_argument("--exclude-manifest", type=Path)
    freeze = actions.add_parser("freeze")
    freeze.add_argument("--batch", type=Path, required=True)
    freeze.add_argument("--split", choices=SPLITS, required=True)
    args = parser.parse_args()
    if args.action == "stage":
        print(stage_batch(args.index, args.output, args.exclude_manifest))
    else:
        print(freeze_batch(args.batch, args.split))


if __name__ == "__main__":
    main()
