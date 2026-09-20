import json
from pathlib import Path
from collections import Counter
from PIL import Image
import numpy as np

root = Path("datasets/torwic_semantic/Labelled_Test_Data")

classes = [
    "person",
    "fork_truck",
    "cart_pallet_jack",
    "rack_shelf",
    "misc_dynamic_feature",
]

total = 0
perfect = 0
failed = []

iou_values = []

json_files = sorted(root.rglob("raw_labels_2d.json"))

print("=" * 75)
print("TORWIC GLOBAL OBJECT-MASK VERIFICATION")
print("=" * 75)
print(f"Annotation files: {len(json_files)}")

for json_file in json_files:

    folder = json_file.parent

    indexed_file = folder / "combined_indexedImage.png"

    if not indexed_file.exists():
        failed.append((str(folder), "NO combined_indexedImage.png"))
        continue

    indexed = np.array(Image.open(indexed_file))

    with open(json_file, "r") as f:
        data = json.load(f)

    mapping = data.get("labelMapping", {})

    for cls in classes:

        entry = mapping.get(cls)

        if entry is None:
            continue

        # Build ordered list of (instance_index, expected_pixels)
        if isinstance(entry, dict):
            if entry.get("numPixels", 0) <= 0:
                continue

            instances = [(entry["index"], entry["numPixels"])]

            png_files = [folder / f"{cls}.png"]

        elif isinstance(entry, list):
            instances = [
                (item["index"], item.get("numPixels", 0))
                for item in entry
                if item.get("numPixels", 0) > 0
            ]

            png_files = sorted(folder.glob(f"{cls}_*.png"))

        else:
            continue

        if len(instances) != len(png_files):
            failed.append(
                (
                    str(folder),
                    f"{cls}: instance count {len(instances)} "
                    f"!= PNG count {len(png_files)}"
                )
            )
            continue

        for (instance_id, expected_pixels), png_file in zip(
            instances, png_files
        ):

            if not png_file.exists():
                failed.append(
                    (str(folder), f"missing {png_file.name}")
                )
                continue

            rgba = np.array(Image.open(png_file))

            if rgba.ndim != 3 or rgba.shape[2] != 4:
                failed.append(
                    (str(folder), f"{png_file.name}: not RGBA")
                )
                continue

            png_mask = rgba[:, :, 3] > 0
            indexed_mask = indexed == instance_id

            png_pixels = int(png_mask.sum())
            indexed_pixels = int(indexed_mask.sum())

            intersection = int(
                np.logical_and(png_mask, indexed_mask).sum()
            )

            union = int(
                np.logical_or(png_mask, indexed_mask).sum()
            )

            iou = intersection / union if union else 0.0

            total += 1
            iou_values.append(iou)

            if (
                png_pixels == indexed_pixels
                and abs(iou - 1.0) < 1e-12
            ):
                perfect += 1
            else:
                failed.append(
                    (
                        str(folder),
                        f"{png_file.name}: "
                        f"instance={instance_id}, "
                        f"PNG_pixels={png_pixels}, "
                        f"indexed_pixels={indexed_pixels}, "
                        f"IoU={iou:.6f}"
                    )
                )

print("\nRESULT")
print("-" * 75)
print(f"Object masks checked: {total}")
print(f"Perfect matches:      {perfect}")
print(f"Failures:             {len(failed)}")

if iou_values:
    print(f"Minimum IoU:          {min(iou_values):.6f}")
    print(f"Mean IoU:             {sum(iou_values)/len(iou_values):.6f}")
    print(f"Maximum IoU:          {max(iou_values):.6f}")

if failed:
    print("\nFIRST 20 FAILURES")
    print("-" * 75)

    for folder, reason in failed[:20]:
        print(f"{folder}")
        print(f"  {reason}")

else:
    print("\nSUCCESS")
    print("Every checked object PNG matches its instance ID")
    print("in combined_indexedImage.png exactly.")
