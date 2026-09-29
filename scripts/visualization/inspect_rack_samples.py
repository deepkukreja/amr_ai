from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(
    "datasets/torwic_segmentation/split"
)

OUTPUT_DIR = Path(
    "runs/rack_inspection"
)

RACK_CLASS_ID = 6

MAX_SAMPLES = 20

THUMB_WIDTH = 320
THUMB_HEIGHT = 180


# ============================================================
# FIND TRAINING IMAGES CONTAINING RACKS
# ============================================================

image_dir = DATASET_ROOT / "train" / "images"
mask_dir = DATASET_ROOT / "train" / "masks"

images = sorted(
    image_dir.glob("*.png")
)

rack_samples = []


print("Searching training images for rack/shelf pixels...")

for image_path in images:

    mask_path = mask_dir / image_path.name

    mask = np.array(
        Image.open(mask_path).convert("L"),
        dtype=np.int64
    )

    rack_pixels = np.sum(
        mask == RACK_CLASS_ID
    )

    if rack_pixels > 0:

        rack_samples.append(
            (
                image_path,
                rack_pixels
            )
        )


# Sort by number of rack pixels, largest first

rack_samples.sort(
    key=lambda x: x[1],
    reverse=True
)

rack_samples = rack_samples[:MAX_SAMPLES]


print()
print("=" * 72)
print("RACK/SHELF TRAINING SAMPLES")
print("=" * 72)

for i, (image_path, rack_pixels) in enumerate(
    rack_samples
):

    print(
        f"{i + 1:>2}. "
        f"{image_path.name:<35} "
        f"rack pixels: {rack_pixels}"
    )

print("=" * 72)

print(
    f"Found {len(rack_samples)} samples selected "
    f"for visualization."
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CREATE CONTACT SHEET
# ============================================================

columns = 4

rows = (
    len(rack_samples) + columns - 1
) // columns

sheet_width = columns * THUMB_WIDTH
sheet_height = rows * (
    THUMB_HEIGHT + 30
)

sheet = Image.new(
    "RGB",
    (
        sheet_width,
        sheet_height
    ),
    "white"
)

draw = ImageDraw.Draw(sheet)


# ============================================================
# ADD SAMPLES
# ============================================================

for index, (image_path, rack_pixels) in enumerate(
    rack_samples
):

    image = Image.open(
        image_path
    ).convert("RGB")

    image.thumbnail(
        (
            THUMB_WIDTH,
            THUMB_HEIGHT
        )
    )

    x = (
        index % columns
    ) * THUMB_WIDTH

    y = (
        index // columns
    ) * (
        THUMB_HEIGHT + 30
    )

    sheet.paste(
        image,
        (
            x,
            y
        )
    )

    draw.text(
        (
            x + 5,
            y + THUMB_HEIGHT + 5
        ),
        f"{index + 1}: "
        f"{image_path.name} "
        f"({rack_pixels}px)",
        fill="black"
    )


# ============================================================
# SAVE CONTACT SHEET
# ============================================================

output_path = (
    OUTPUT_DIR /
    "rack_training_samples.png"
)

sheet.save(
    output_path
)

print()
print(
    f"Saved contact sheet:\n{output_path}"
)
