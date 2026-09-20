import json
import shutil
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_ROOT = Path(
    "datasets/torwic_semantic/Labelled_Test_Data"
)

OUTPUT_ROOT = Path(
    "datasets/torwic_detection_raw"
)

CLASSES = {
    "person": 0,
    "fork_truck": 1,
    "cart_pallet_jack": 2,
    "rack_shelf": 3,
    "misc_dynamic_feature": 4,
}


# ============================================================
# HELPERS
# ============================================================

def get_instance_entries(value):
    """
    Convert labelMapping entry into a list of actual instances.

    A dictionary with numPixels == 0 means the class is absent.
    """
    if value is None:
        return []

    if isinstance(value, dict):
        if value.get("numPixels", 0) <= 0:
            return []
        return [value]

    if isinstance(value, list):
        return [
            item
            for item in value
            if item.get("numPixels", 0) > 0
        ]

    return []


def find_instance_pngs(folder, class_name):
    """
    Find the PNG mask files corresponding to a class.
    """
    plain = folder / f"{class_name}.png"

    if plain.exists():
        return [plain]

    return sorted(folder.glob(f"{class_name}_*.png"))


def mask_to_bbox(mask):
    """
    Convert a boolean mask to:
        x1, y1, x2, y2

    Coordinates use image-pixel convention.
    """
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())

    return x1, y1, x2, y2


def bbox_to_yolo(bbox, image_width, image_height):
    """
    Convert pixel bbox into normalized YOLO format:
        x_center y_center width height
    """
    x1, y1, x2, y2 = bbox

    # +1 because x2/y2 are inclusive pixel coordinates.
    width = x2 - x1 + 1
    height = y2 - y1 + 1

    x_center = x1 + width / 2.0
    y_center = y1 + height / 2.0

    return (
        x_center / image_width,
        y_center / image_height,
        width / image_width,
        height / image_height,
    )


# ============================================================
# PREPARE OUTPUT DIRECTORIES
# ============================================================

images_dir = OUTPUT_ROOT / "images"
labels_dir = OUTPUT_ROOT / "labels"

images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# DISCOVER ANNOTATIONS
# ============================================================

json_files = sorted(
    SOURCE_ROOT.rglob("raw_labels_2d.json")
)

print("=" * 75)
print("TORWIC → YOLO DETECTION CONVERTER")
print("=" * 75)

print(f"Source annotation files: {len(json_files)}")
print(f"Output directory:        {OUTPUT_ROOT}")
print()

if not json_files:
    raise RuntimeError("No raw_labels_2d.json files found.")


# ============================================================
# CONVERSION
# ============================================================

image_count = 0
label_count = 0

class_counts = Counter()
empty_images = 0
errors = []

for json_file in json_files:

    folder = json_file.parent

    source_image = folder / "source_image.png"

    if not source_image.exists():
        errors.append(
            f"Missing source image: {source_image}"
        )
        continue

    try:
        with open(json_file, "r") as f:
            data = json.load(f)

        mapping = data.get("labelMapping", {})

        image = Image.open(source_image)

        image_width, image_height = image.size

        # Create unique filename based on sequence/frame path.
        relative_folder = folder.relative_to(SOURCE_ROOT)

        # Example:
        # 000000285/image_0.png_raw_labels_2d
        sequence_id = relative_folder.parts[0]
        annotation_folder_name = relative_folder.parts[-1]

        image_stem = annotation_folder_name.replace(
            ".png_raw_labels_2d", ""
        )

        output_stem = (
            f"{sequence_id}_{image_stem}"
        )

        output_image = images_dir / f"{output_stem}.png"
        output_label = labels_dir / f"{output_stem}.txt"

        # Copy original source image unchanged.
        shutil.copy2(
            source_image,
            output_image
        )

        yolo_lines = []

        for class_name, class_id in CLASSES.items():

            entries = get_instance_entries(
                mapping.get(class_name)
            )

            if not entries:
                continue

            png_files = find_instance_pngs(
                folder,
                class_name
            )

            if len(entries) != len(png_files):
                errors.append(
                    f"{folder}: {class_name} "
                    f"entries={len(entries)} "
                    f"pngs={len(png_files)}"
                )
                continue

            for entry, mask_file in zip(
                entries,
                png_files
            ):

                rgba = np.array(
                    Image.open(mask_file)
                )

                if rgba.ndim != 3 or rgba.shape[2] != 4:
                    errors.append(
                        f"{mask_file}: expected RGBA"
                    )
                    continue

                # Alpha channel defines object pixels.
                mask = rgba[:, :, 3] > 0

                bbox = mask_to_bbox(mask)

                if bbox is None:
                    errors.append(
                        f"{mask_file}: empty mask"
                    )
                    continue

                yolo = bbox_to_yolo(
                    bbox,
                    image_width,
                    image_height
                )

                yolo_lines.append(
                    f"{class_id} "
                    f"{yolo[0]:.6f} "
                    f"{yolo[1]:.6f} "
                    f"{yolo[2]:.6f} "
                    f"{yolo[3]:.6f}"
                )

                class_counts[class_name] += 1
                label_count += 1

        # YOLO allows an empty label file for an image
        # containing no selected detector classes.
        with open(output_label, "w") as f:
            if yolo_lines:
                f.write("\n".join(yolo_lines))
                f.write("\n")
            else:
                empty_images += 1

        image_count += 1

    except Exception as e:
        errors.append(
            f"{folder}: {type(e).__name__}: {e}"
        )


# ============================================================
# DATASET YAML
# ============================================================

yaml_file = OUTPUT_ROOT / "data.yaml"

with open(yaml_file, "w") as f:
    f.write(
        "path: " + str(OUTPUT_ROOT.resolve()) + "\n"
    )
    f.write("train: images\n")
    f.write("val: images\n")
    f.write("test: images\n")
    f.write("\n")
    f.write("names:\n")

    for name, class_id in CLASSES.items():
        f.write(
            f"  {class_id}: {name}\n"
        )


# ============================================================
# SUMMARY
# ============================================================

print("=" * 75)
print("CONVERSION SUMMARY")
print("=" * 75)

print(f"Images converted:       {image_count}")
print(f"Object labels created:  {label_count}")
print(f"Images with no objects: {empty_images}")

print("\nCLASS COUNTS")
print("-" * 75)

for class_name, class_id in CLASSES.items():
    print(
        f"{class_id}: "
        f"{class_name:25s}"
        f"{class_counts[class_name]:5d}"
    )

print("\nERRORS:", len(errors))

if errors:
    print("\nFIRST 20 ERRORS")
    print("-" * 75)

    for error in errors[:20]:
        print(error)

else:
    print("\nSUCCESS")
    print("No conversion errors detected.")

print("\nOutput:")
print(f"  Images: {images_dir}")
print(f"  Labels: {labels_dir}")
print(f"  YAML:   {yaml_file}")
