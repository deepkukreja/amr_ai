from pathlib import Path
import json

import numpy as np
from PIL import Image


SRC = Path("torwic_semantic/Labelled_Test_Data")
OUT = Path("torwic_segmentation")

IMAGES_OUT = OUT / "images"
MASKS_OUT = OUT / "masks"

IMAGES_OUT.mkdir(parents=True, exist_ok=True)
MASKS_OUT.mkdir(parents=True, exist_ok=True)

SEMANTIC_IDS = {
    "background": 0,
    "driveable_ground": 1,
    "ceiling": 2,
    "ego_vehicle": 3,
    "wall_fence_pillar": 4,
    "misc_static_feature": 5,
    "rack_shelf": 6,
    "goods_material": 7,
    "fixed_machinery": 8,
    "cart_pallet_jack": 9,
    "pylon_cone": 10,
    "text_region": 11,
    "misc_non_static_feature": 12,
    "person": 13,
    "fork_truck": 14,
    "misc_dynamic_feature": 15,
}

json_files = sorted(SRC.rglob("raw_labels_2d.json"))

converted = 0
skipped = 0

for json_path in json_files:

    base = json_path.parent

    rgb_path = base / "source_image.png"
    indexed_path = base / "combined_indexedImage.png"

    if not rgb_path.exists() or not indexed_path.exists():
        skipped += 1
        continue

    with open(json_path, "r") as f:
        data = json.load(f)

    mapping = {0: 0}

    for class_name, info in data.get("labelMapping", {}).items():

        if class_name not in SEMANTIC_IDS:
            raise RuntimeError(
                f"Unknown class '{class_name}' in {json_path}"
            )

        semantic_id = SEMANTIC_IDS[class_name]

        if isinstance(info, dict) and "index" in info:
            mapping[int(info["index"])] = semantic_id

        elif isinstance(info, list):
            for item in info:
                if isinstance(item, dict) and "index" in item:
                    mapping[int(item["index"])] = semantic_id

    mask = np.array(Image.open(indexed_path))

    unique_ids = np.unique(mask)
    unknown_ids = [
        int(x) for x in unique_ids
        if int(x) not in mapping
    ]

    if unknown_ids:
        raise RuntimeError(
            f"Unmapped IDs {unknown_ids} in {indexed_path}"
        )

    semantic_mask = np.zeros(mask.shape, dtype=np.uint8)

    for instance_id, semantic_id in mapping.items():
        semantic_mask[mask == instance_id] = semantic_id

    sequence = base.parent.name
    image_name = base.name.replace(".png_raw_labels_2d", "")
    stem = f"{sequence}_{image_name}"

    Image.open(rgb_path).convert("RGB").save(
        IMAGES_OUT / f"{stem}.png"
    )

    Image.fromarray(semantic_mask).save(
        MASKS_OUT / f"{stem}.png"
    )

    converted += 1

print()
print("===================================")
print("TorWIC conversion complete")
print("===================================")
print(f"Annotation files found : {len(json_files)}")
print(f"Images converted       : {converted}")
print(f"Images skipped         : {skipped}")
print(f"Output                 : {OUT}")
