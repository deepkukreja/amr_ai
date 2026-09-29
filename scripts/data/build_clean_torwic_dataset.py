from pathlib import Path
from collections import Counter
from PIL import Image
import json
import shutil
import random

SOURCE_ROOT = Path("datasets/torwic_semantic/Labelled_Test_Data")
OUTPUT_ROOT = Path("datasets/torwic_yolo_clean")

CLASSES = {
    "person": 0,
    "fork_truck": 1,
    "cart_pallet_jack": 2,
    "rack_shelf": 3,
    "misc_dynamic_feature": 4,
}

EXCLUDED_PERSON_MASKS = {
    "000000896/image_0.png_raw_labels_2d/person_000000002.png",
    "000000897/image_0.png_raw_labels_2d/person_000000003.png",
    "000000910/image_2.png_raw_labels_2d/person_000000000.png",
    "000000912/image_2.png_raw_labels_2d/person_000000003.png",
    "000000913/image_1.png_raw_labels_2d/person_000000003.png",
    "000000914/image_1.png_raw_labels_2d/person_000000003.png",
    "000001680/image_1.png_raw_labels_2d/person_000000000.png",
    "000001789/image_1.png_raw_labels_2d/person_000000000.png",
    "000001807/image_1.png_raw_labels_2d/person_000000000.png",
    "000001859/image_2.png_raw_labels_2d/person_000000000.png",
}

SEED = 42
TRAIN_GROUPS = 60

random.seed(SEED)

json_files = sorted(
    SOURCE_ROOT.rglob("raw_labels_2d.json")
)

groups = {}

for json_file in json_files:
    rel = json_file.relative_to(SOURCE_ROOT)

    group = rel.parts[0]

    groups.setdefault(group, []).append(json_file)

group_names = sorted(groups)

random.shuffle(group_names)

train_groups = set(group_names[:TRAIN_GROUPS])
val_groups = set(group_names[TRAIN_GROUPS:])

print("=" * 75)
print("BUILDING CLEAN TORWIC YOLO DATASET")
print("=" * 75)

print(f"Total groups:        {len(group_names)}")
print(f"Training groups:     {len(train_groups)}")
print(f"Validation groups:   {len(val_groups)}")
print(f"Training frames:     {sum(len(groups[g]) for g in train_groups)}")
print(f"Validation frames:   {sum(len(groups[g]) for g in val_groups)}")

for split in ("train", "val"):
    (OUTPUT_ROOT / split / "images").mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_ROOT / split / "labels").mkdir(
        parents=True,
        exist_ok=True
    )

counts = {
    "train": Counter(),
    "val": Counter(),
}

frame_counts = Counter()
empty_counts = Counter()

for group in group_names:

    split = "train" if group in train_groups else "val"

    for json_file in sorted(groups[group]):

        folder = json_file.parent
        source_image = folder / "source_image.png"

        if not source_image.exists():
            raise RuntimeError(
                f"Missing source image: {source_image}"
            )

        with open(json_file, "r") as f:
            data = json.load(f)

        mapping = data["labelMapping"]

        rel_folder = folder.relative_to(SOURCE_ROOT)

        frame_name = rel_folder.parts[-1].replace(
            ".png_raw_labels_2d",
            ""
        )

        output_stem = (
            f"{group}_{frame_name}"
        )

        output_image = (
            OUTPUT_ROOT / split / "images" /
            f"{output_stem}.png"
        )

        output_label = (
            OUTPUT_ROOT / split / "labels" /
            f"{output_stem}.txt"
        )

        shutil.copy2(
            source_image,
            output_image
        )

        image = Image.open(source_image)

        width, height = image.size

        yolo_lines = []

        for class_name, class_id in CLASSES.items():

            entry = mapping.get(class_name)

            if entry is None:
                continue

            if isinstance(entry, dict):
                entries = [entry] if entry.get(
                    "numPixels", 0
                ) > 0 else []

            elif isinstance(entry, list):
                entries = [
                    x for x in entry
                    if x.get("numPixels", 0) > 0
                ]

            else:
                entries = []

            if not entries:
                continue

            if class_name == "person":

                png_files = []

                plain = folder / "person.png"

                if plain.exists():
                    png_files.append(plain)

                png_files.extend(
                    sorted(folder.glob("person_*.png"))
                )

            else:

                plain = folder / f"{class_name}.png"

                if plain.exists():
                    png_files = [plain]
                else:
                    png_files = sorted(
                        folder.glob(
                            f"{class_name}_*.png"
                        )
                    )

            if len(entries) != len(png_files):
                raise RuntimeError(
                    f"Count mismatch: {folder} "
                    f"{class_name}: "
                    f"{len(entries)} vs "
                    f"{len(png_files)}"
                )

            for entry, mask_file in zip(
                entries,
                png_files
            ):

                rel_mask = mask_file.relative_to(
                    SOURCE_ROOT
                )

                # Skip the 10 manually validated problematic
                # person masks.
                if (
                    class_name == "person"
                    and str(rel_mask) in EXCLUDED_PERSON_MASKS
                ):
                    continue

                rgba = Image.open(mask_file)

                if rgba.mode != "RGBA":
                    raise RuntimeError(
                        f"Expected RGBA: {mask_file}"
                    )

                alpha = rgba.getchannel("A")

                bbox = alpha.getbbox()

                if bbox is None:
                    continue

                x1, y1, x2, y2 = bbox

                box_w = x2 - x1
                box_h = y2 - y1

                x_center = x1 + box_w / 2
                y_center = y1 + box_h / 2

                xc = x_center / width
                yc = y_center / height
                bw = box_w / width
                bh = box_h / height

                yolo_lines.append(
                    f"{class_id} "
                    f"{xc:.6f} "
                    f"{yc:.6f} "
                    f"{bw:.6f} "
                    f"{bh:.6f}"
                )

                counts[split][class_name] += 1

        with open(output_label, "w") as f:
            if yolo_lines:
                f.write(
                    "\n".join(yolo_lines) + "\n"
                )
            else:
                empty_counts[split] += 1

        frame_counts[split] += 1


# ----------------------------------------------------------------
# YAML
# ----------------------------------------------------------------

yaml_path = OUTPUT_ROOT / "data.yaml"

with open(yaml_path, "w") as f:

    f.write(
        f"path: {OUTPUT_ROOT.resolve()}\n"
    )

    f.write("train: train/images\n")
    f.write("val: val/images\n")

    f.write("\n")

    f.write("names:\n")

    for name, idx in CLASSES.items():
        f.write(
            f"  {idx}: {name}\n"
        )


# ----------------------------------------------------------------
# SPLIT REPORT
# ----------------------------------------------------------------

report = {
    "seed": SEED,
    "total_groups": len(group_names),
    "train_groups": sorted(train_groups),
    "val_groups": sorted(val_groups),
    "excluded_person_masks": sorted(
        EXCLUDED_PERSON_MASKS
    ),
}

with open(
    OUTPUT_ROOT / "split_report.json",
    "w"
) as f:
    json.dump(
        report,
        f,
        indent=2
    )

print("\n" + "=" * 75)
print("DATASET SUMMARY")
print("=" * 75)

for split in ("train", "val"):

    print(f"\n{split.upper()}")

    print(
        f"  frames: "
        f"{frame_counts[split]}"
    )

    print(
        f"  empty labels: "
        f"{empty_counts[split]}"
    )

    for class_name in CLASSES:
        print(
            f"  {class_name:25s}"
            f"{counts[split][class_name]:6d}"
        )

print("\nOutput:")
print(f"  {OUTPUT_ROOT}")
print(f"  {yaml_path}")

print("\nOriginal datasets were not modified.")
