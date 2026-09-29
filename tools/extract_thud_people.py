from pathlib import Path
from zipfile import ZipFile
from collections import defaultdict
import json
import csv
import re

from PIL import Image
import io


ZIP_PATH = Path("datasets/external/thud_pp/RGB-D.zip")
OUT_DIR = Path("datasets/thud_people_yolo_v1")

# Capture-level split to prevent consecutive-frame leakage.
SPLITS = {
    "train": [1, 2, 3, 4],
    "dev": [5],
    "test": [6],
}

BASE = "RGB-D/Synthetic_Scenes/Gym/dynamic/Capture_{capture}/"


def yolo_box(x, y, w, h, img_w, img_h):
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(img_w), x + w)
    y2 = min(float(img_h), y + h)

    if x2 <= x1 or y2 <= y1:
        return None

    xc = ((x1 + x2) / 2.0) / img_w
    yc = ((y1 + y2) / 2.0) / img_h
    nw = (x2 - x1) / img_w
    nh = (y2 - y1) / img_h

    return xc, yc, nw, nh


def process_capture(z, capture_id, split, writer, totals):
    base = BASE.format(capture=capture_id)

    json_files = sorted(
        n for n in z.namelist()
        if n.startswith(base + "Label/captures_")
        and n.endswith(".json")
    )

    image_dir = OUT_DIR / "images" / split
    label_dir = OUT_DIR / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for json_name in json_files:
        with z.open(json_name) as f:
            data = json.load(f)

        for frame in data.get("captures", []):

            # Only the 2D camera annotation stream.
            bbox_ann = next(
                (
                    ann for ann in frame.get("annotations", [])
                    if ann.get("id") == "bounding box"
                ),
                None,
            )

            if bbox_ann is None:
                continue

            filename = frame.get("filename", "")
            if not filename:
                continue

            # JSON contains UUID/rgb_N.png, while ZIP stores RGB/rgb_N.png.
            rgb_name = Path(filename).name

            rgb_zip_path = base + "RGB/" + rgb_name

            try:
                with z.open(rgb_zip_path) as img_file:
                    img_bytes = img_file.read()

                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img_w, img_h = img.size

            except Exception as exc:
                print(f"ERROR image: {rgb_zip_path}: {exc}")
                continue

            boxes = []

            for value in bbox_ann.get("values", []):
                if value.get("label_name") != "People":
                    continue

                box = yolo_box(
                    float(value.get("x", 0)),
                    float(value.get("y", 0)),
                    float(value.get("width", 0)),
                    float(value.get("height", 0)),
                    img_w,
                    img_h,
                )

                if box is not None:
                    boxes.append(box)

            # Make unique names across captures.
            stem = f"gym_dynamic_c{capture_id:02d}_{Path(rgb_name).stem}"

            out_img = image_dir / f"{stem}.png"
            out_lbl = label_dir / f"{stem}.txt"

            img.save(out_img, format="PNG")

            with open(out_lbl, "w") as f:
                for xc, yc, w, h in boxes:
                    f.write(
                        f"0 {xc:.6f} {yc:.6f} "
                        f"{w:.6f} {h:.6f}\n"
                    )

            writer.writerow(
                [
                    split,
                    capture_id,
                    json_name,
                    rgb_name,
                    img_w,
                    img_h,
                    len(boxes),
                ]
            )

            totals[split]["images"] += 1
            totals[split]["positive"] += bool(boxes)
            totals[split]["boxes"] += len(boxes)


def main():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Dataset ZIP not found: {ZIP_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUT_DIR / "metadata.csv"

    totals = defaultdict(
        lambda: {
            "images": 0,
            "positive": 0,
            "boxes": 0,
        }
    )

    with ZipFile(ZIP_PATH) as z, open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "split",
                "capture",
                "annotation_json",
                "rgb_file",
                "width",
                "height",
                "people_boxes",
            ]
        )

        for split, captures in SPLITS.items():
            print(f"\n=== {split.upper()} ===")

            for capture_id in captures:
                print(f"Processing Gym/dynamic/Capture_{capture_id}...")
                process_capture(
                    z,
                    capture_id,
                    split,
                    writer,
                    totals,
                )

    dataset_root = OUT_DIR.resolve()

    yaml_path = OUT_DIR / "thud_people.yaml"

    yaml_path.write_text(
        f"""path: {dataset_root}
train: images/train
val: images/dev
test: images/test

names:
  0: person
"""
    )

    print("\n=== EXTRACTION COMPLETE ===")

    for split in ["train", "dev", "test"]:
        s = totals[split]
        print(
            f"{split:5s}: "
            f"images={s['images']:5d}  "
            f"positive={s['positive']:5d}  "
            f"negative={s['images'] - s['positive']:5d}  "
            f"people_boxes={s['boxes']:6d}"
        )

    print(f"\nDataset: {OUT_DIR}")
    print(f"YAML:    {yaml_path}")


if __name__ == "__main__":
    main()
