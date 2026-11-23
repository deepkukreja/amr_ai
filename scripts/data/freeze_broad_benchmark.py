"""Freeze a mixed-source AMR development benchmark before running predictions."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
import re
from urllib.request import urlopen

from PIL import Image

from scripts.evaluation.broad_benchmark import select_coverage
from scripts.evaluation.compare_human_detection import _box, load_annotation

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261122
OPEN_LABELS = {
    "Person": "human", "Bicycle": "bicycle", "Car": "car", "Motorcycle": "motorcycle",
    "Bus": "bus", "Truck": "truck", "Van": "van", "Dog": "dog", "Cat": "cat",
    "Cattle": "cattle", "Horse": "horse", "Sheep": "sheep", "Bird": "bird", "Goat": "goat",
}
WAREHOUSE_LABELS = ["human", "forklift_or_truck", "cart_or_pallet_jack", "rack_or_shelf"]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        yield from csv.DictReader(file)


def record(image, cohort, group, boxes, coverage, **metadata):
    return {
        "image": image.relative_to(ROOT).as_posix(), "sha256": sha(image.read_bytes()),
        "cohort": cohort, "group": group, "coverage": sorted(coverage),
        "boxes": [{"label": label, "box": _box(box)} for label, box in boxes], **metadata,
    }


def road_frames():
    root = ROOT / "datasets/indian_manual_eval_v1"
    cache = ROOT / "results/local/model_clean_baseline_20261115/development_comparison.json"
    audited = json.loads(cache.read_text())["models"]["baseline"]["frames"]
    audited = {Path(row["image"]).name: row for row in audited}
    rows = sorted(csv_rows(root / "manifest.csv"), key=lambda row: row["relative_image_path"])
    random.Random(SEED).shuffle(rows)
    unique = {}
    for row in rows:
        unique.setdefault(row["clip_id"], row)
    grouped = defaultdict(list)
    for row in unique.values():
        grouped[row["timeofday"]].append(row)
    chosen = []
    while len(chosen) < min(48, len(unique)):
        for light in sorted(grouped):
            if grouped[light] and len(chosen) < 48:
                chosen.append(grouped[light].pop())
    result = []
    for row in chosen:
        image = root / row["relative_image_path"]
        label = root / "manual_annotations" / (image.stem + ".json")
        prior = audited[image.name]
        if sha(image.read_bytes()) != prior["image_sha256"] or sha(label.read_bytes()) != prior["annotation_sha256"]:
            raise ValueError(f"Reviewed road input changed: {image.name}")
        truth = load_annotation(label)
        boxes = [(("pedestrian", "rider")[obj["class_id"]], obj["box"]) for obj in truth]
        # Human localization and role are distinct views over the same objects.
        boxes += [("human", obj["box"]) for obj in truth]
        result.append(record(image, "road_reviewed", row["clip_id"], boxes, {"human", "pedestrian", "rider"},
                             light=row["timeofday"], weather=row["weather"], label_sha256=prior["annotation_sha256"]))
    return result


def indoor_frames():
    root = ROOT / "datasets/indian_manual280_indoor_replay_20261122"
    manifest = json.loads((root / "manifest.json").read_text())
    result = []
    for row in manifest["frames"]:
        if row["split"] != "probe":
            continue
        if row["session"] != 2:
            raise ValueError("Unexpected THUD capture in the indoor probe")
        frame = record(root / row["image"], "indoor_thud", "store_capture_2",
                       [("human", box) for box in row["boxes"]], {"human"},
                       label_sha256=row["annotation_sha256"])
        if frame["sha256"] != row["image_sha256"]:
            raise ValueError("Indoor image differs from the publisher-audited manifest")
        result.append(frame)
    return result


def warehouse_frames():
    root = ROOT / "datasets/torwic_yolo_clean/val"
    result = []
    for image in sorted((root / "images").glob("*_image_0.png")):
        label = root / "labels" / (image.stem + ".txt")
        boxes = []
        for line in label.read_text().splitlines():
            kind, cx, cy, w, h = line.split()
            kind = int(kind)
            if kind == 4:
                continue
            cx, cy, w, h = map(float, (cx, cy, w, h))
            box = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
            if not all(-1e-6 <= coord <= 1 + 1e-6 for coord in box):
                raise ValueError(f"Invalid warehouse box: {label}")
            boxes.append((WAREHOUSE_LABELS[kind], [min(1, max(0, coord)) for coord in box]))
        coverage = {name for name, _ in boxes}
        if image.name == "000000908_image_0.png" and not boxes:
            coverage = set(WAREHOUSE_LABELS)  # Visually reviewed negative for these four groups.
        result.append(record(image, "warehouse_legacy", image.name.split("_")[0], boxes, coverage,
                             label_sha256=sha(label.read_bytes())))
    return result


def download_image(image_id, folder, split):
    if not re.fullmatch(r"[0-9a-f]{16}", image_id):
        raise ValueError("Invalid Open Images ID")
    image = folder / (image_id + ".jpg")
    if not image.exists():
        url = f"https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
        with urlopen(url, timeout=20) as response:
            data = response.read(20 * 1024 * 1024 + 1)
        if len(data) > 20 * 1024 * 1024:
            raise ValueError(f"Image over the size limit: {image_id}")
        with Image.open(BytesIO(data)) as im:
            im.verify()
        staging = image.with_suffix(".jpg.tmp")
        staging.write_bytes(data)
        staging.replace(image)
    return image


def open_images(folder, root, split, images_per_class):
    with (root / "class-descriptions-boxable.csv").open(newline="") as file:
        names = {key: OPEN_LABELS[name] for key, name in csv.reader(file) if name in OPEN_LABELS}
    meta = {row["ImageID"]: row for row in csv_rows(root / f"{split}-images-with-rotation.csv")
            if row["Subset"] == split and row["Rotation"] in ("", "0", "0.0")
            and row["License"].rstrip("/") in ("https://creativecommons.org/licenses/by/2.0",
                                               "http://creativecommons.org/licenses/by/2.0")
            and row["Author"] and row["OriginalLandingURL"]}
    positives, negatives, incomplete = (defaultdict(set) for _ in range(3))
    for row in csv_rows(root / f"{split}-annotations-human-imagelabels-boxable.csv"):
        if row["LabelName"] in names:
            group = positives if row["Confidence"] == "1" else negatives
            group[row["ImageID"]].add(names[row["LabelName"]])
    boxes = defaultdict(list)
    for row in csv_rows(root / f"{split}-annotations-bbox.csv"):
        image_id = row["ImageID"]
        label = names.get(row["LabelName"])
        if image_id not in meta or label is None:
            continue
        if row["IsGroupOf"] == "1" or row["IsDepiction"] == "1" or row["IsInside"] == "1":
            incomplete[image_id].add(label)
        else:
            box = [float(row[name]) for name in ("XMin", "YMin", "XMax", "YMax")]
            boxes[image_id].append((label, _box(box)))

    cohort = {}
    rng = random.Random(SEED)
    candidates = sorted(boxes)
    rng.shuffle(candidates)
    author_count = Counter()
    for label in OPEN_LABELS.values():
        selected = []
        for image_id in candidates:
            author = meta[image_id]["Author"]
            if image_id in cohort or author_count[author] >= 2 or label in incomplete[image_id]:
                continue
            if label not in positives[image_id] or not any(kind == label for kind, _ in boxes[image_id]):
                continue
            cohort[image_id] = label
            author_count[author] += 1
            selected.append(image_id)
            if len(selected) == images_per_class:
                break
        if len(selected) != images_per_class:
            raise ValueError(f"Missing eligible Open Images positives for {label}: {len(selected)}")

    folder.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(lambda key: download_image(key, folder, split), cohort))
    result = []
    cohort_name = "open_images_v5_test" if split == "test" else "open_images_v5"
    for image_id, path in zip(cohort, paths):
        boxed = {label for label, _ in boxes[image_id]}
        coverage = select_coverage(positives[image_id] & boxed, negatives[image_id] - boxed,
                                   incomplete[image_id] | (positives[image_id] & negatives[image_id]))
        result.append(record(path, cohort_name, image_id,
                             [(label, box) for label, box in boxes[image_id] if label in coverage],
                             coverage, anchor_class=cohort[image_id], author=meta[image_id]["Author"],
                             source=meta[image_id]["OriginalLandingURL"], license=meta[image_id]["License"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--open-images-test", action="store_true",
                        help="Sample only official Open Images V5 test images for public evaluation")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "datasets") or (output / "manifest.json").exists():
        raise ValueError("Use a new manifest under the local datasets directory")
    external = ROOT / "datasets/external"
    if args.open_images_test:
        frames = open_images(output / "images", external / "open_images_test_v5_pilot/metadata",
                             "test", 8)
        scope = "public evaluation"
    else:
        frames = road_frames() + indoor_frames() + warehouse_frames()
        frames += open_images(output / "images", external / "open_images_metadata_v5", "validation", 5)
        if not 120 <= len(frames) <= 180:
            raise ValueError("Development benchmark sample size is out of bounds")
        scope = "development only"
    if len({row["sha256"] for row in frames}) != len(frames):
        raise ValueError("Benchmark contains exact duplicate images")
    manifest = {"version": 1, "seed": SEED, "scope": scope, "frames": frames}
    text = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    (output / "manifest.tmp").write_text(text)
    (output / "manifest.tmp").replace(output / "manifest.json")
    print(json.dumps({"images": len(frames), "cohorts": Counter(row["cohort"] for row in frames),
                      "manifest_sha256": sha(text.encode())}, indent=2))


if __name__ == "__main__":
    main()
