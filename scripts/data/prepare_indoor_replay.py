"""Build a training mix and a separate THUD++ indoor development probe."""
import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
from zipfile import ZipFile

from PIL import Image
import yaml


def publisher_boxes(annotation, size):
    width, height = size
    if size != (annotation["imageWidth"], annotation["imageHeight"]):
        raise ValueError("Publisher annotation dimensions differ from image")
    boxes = []
    for shape in annotation["shapes"]:
        if shape["label"].casefold() != "people":
            continue
        if shape["shape_type"] != "rectangle":
            raise ValueError("Expected People rectangle")
        (x1, y1), (x2, y2) = shape["points"]
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            raise ValueError("Nonfinite publisher box")
        left, right = sorted((x1 / width, x2 / width))
        top, bottom = sorted((y1 / height, y2 / height))
        box = (max(0, left), max(0, top), min(1, right), min(1, bottom))
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("Empty publisher box after clipping")
        boxes.append(box)
    return boxes


def area(box):
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def spaced(rows, gap, limit=48):
    """Pick high-coverage frames with a minimum index gap."""
    selected = []
    for row in sorted(rows, key=lambda item: (-item[1], item[0])):
        if all(abs(row[0] - previous[0]) >= gap for previous in selected):
            selected.append(row)
            if len(selected) == limit:
                break
    return sorted(selected)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def read_yolo(path, classes):
    boxes = []
    for line in path.read_text().splitlines():
        label, cx, cy, w, h = line.split()
        cx, cy, w, h = map(float, (cx, cy, w, h))
        box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        if int(label) not in classes or not all(-1e-6 <= v <= 1 + 1e-6 for v in box) or w <= 0 or h <= 0:
            raise ValueError(f"Invalid YOLO row in {path}")
        boxes.append((int(label), tuple(min(1, max(0, v)) for v in box)))
    return boxes


def session_rows(root, archive, session, local_split):
    converted = root / "datasets/thud_real_people_yolo_v1"
    images = converted / "images" / local_split
    labels = converted / "labels" / local_split
    by_frame = {}
    for image in sorted(images.glob("*.png")):
        match = re.fullmatch(rf"capture_{session}_frame-(\d+)\.color\.png", image.name)
        if not match:
            raise ValueError(f"Unexpected capture member {image.name}")
        label = labels / (image.stem + ".txt")
        boxes = read_yolo(label, {0})
        by_frame[int(match[1])] = (image, max((area(box) for _, box in boxes), default=0))
    eligible = [(frame, largest) for frame, (_, largest) in by_frame.items() if largest >= .02]
    selected = spaced(eligible, gap=8)
    if len(selected) < 12:
        raise ValueError(f"Insufficient close-up coverage in Capture_{session}")

    base = f"RGB-D/Real_Scenes/store/dynamic/Capture_{session}"
    rows = []
    for frame, _ in selected:
        image, _ = by_frame[frame]
        source_name = f"frame-{frame:06d}.color"
        rgb = f"{base}/RGB/{source_name}.png"
        annotation = f"{base}/Label/2D_Object_Detection/{source_name}.json"
        source_image = archive.read(rgb)
        data = archive.read(annotation)
        with Image.open(image) as im, Image.open(BytesIO(source_image)) as original:
            if im.size != original.size or im.convert("RGB").tobytes() != original.convert("RGB").tobytes():
                raise ValueError(f"Local image pixels differ from publisher archive: {image.name}")
            boxes = publisher_boxes(json.loads(data), original.size)
        local_boxes = [box for _, box in read_yolo(labels / (image.stem + ".txt"), {0})]
        if len(local_boxes) != len(boxes) or any(
            abs(x - y) > 2e-6 for actual, expected in zip(sorted(local_boxes), sorted(boxes))
            for x, y in zip(actual, expected)
        ):
            raise ValueError(f"Local annotation differs from publisher: {image.name}")
        rows.append({"image": image, "source_image": source_image, "session": session, "frame": frame, "boxes": boxes,
                     "image_sha256": sha(source_image), "annotation_sha256": sha(data)})
    return rows, len(by_frame), len(eligible)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Output directory already exists")
    source = root / "datasets/indian_road_manual280_clean_v1"
    audit_path = root / "results/local/model_training_20261115/inventory_and_data_audit.json"
    audit = json.loads(audit_path.read_text())
    expected = {Path(row["image"]).resolve(): row for row in audit["fingerprints"]}
    road = {}
    for split in ("train", "val"):
        road[split] = []
        for label in sorted((source / "labels" / split).glob("*.txt")):
            image = source / "images" / split / (label.stem + ".jpg")
            row = expected[image.resolve()]
            if sha(image.read_bytes()) != row["image_sha256"] or sha(label.read_bytes()) != row["label_sha256"]:
                raise ValueError(f"Frozen road pair changed: {image.name}")
            read_yolo(label, {0, 1})
            road[split].append((image, label, row["image_sha256"]))
    if [len(road[split]) for split in ("train", "val")] != [221, 59]:
        raise ValueError("Unexpected road split counts")
    with ZipFile(root / "datasets/external/thud_pp/RGB-D.zip") as archive:
        indoor_train, train_total, train_eligible = session_rows(root, archive, 1, "dev")
        indoor_probe, probe_total, probe_eligible = session_rows(root, archive, 2, "test")
    hashes = [row["image_sha256"] for row in indoor_train + indoor_probe]
    if len(set(hashes)) != len(hashes) or set(hashes) & {h for rows in road.values() for _, _, h in rows}:
        raise ValueError("Exact image duplicate in the selected data")
    for split, entries in road.items():
        for image, label, _ in entries:
            for category, path in (("images", image), ("labels", label)):
                destination = output / category / split / path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(path)
    records = []
    for split, rows in (("train", indoor_train), ("probe", indoor_probe)):
        for row in rows:
            image = row.pop("image")
            destination = output / "images" / split / image.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(row.pop("source_image"))
            if split == "train":
                label = output / "labels/train" / (image.stem + ".txt")
                label.write_text("".join(
                    f"0 {(x1+x2)/2:.9f} {(y1+y2)/2:.9f} {x2-x1:.9f} {y2-y1:.9f}\n"
                    for x1, y1, x2, y2 in row["boxes"]
                ))
                row["label_sha256"] = sha(label.read_bytes())
            records.append({**row, "split": split, "image": str(destination.relative_to(output))})
    config = {"path": str(output), "train": "images/train", "val": "images/val", "names": {0: "person", 1: "rider"}}
    (output / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    metadata = {"source": "https://doi.org/10.5281/zenodo.18459791",
                "road_images": {split: len(rows) for split, rows in road.items()},
                "capture_pool": {"train": [train_total, train_eligible], "probe": [probe_total, probe_eligible]},
                "selected": dict(Counter(row["split"] for row in records)),
                "large_boxes": {split: sum(area(b) >= .1 for row in records if row["split"] == split
                                          for b in row["boxes"]) for split in ("train", "probe")},
                "min_box_area_fraction": .02, "min_frame_gap": 8,
                "validation_includes_indoor_probe": False}
    (output / "manifest.json").write_text(json.dumps({**metadata, "frames": records}, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
