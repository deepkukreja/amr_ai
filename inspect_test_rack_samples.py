from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


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


image_dir = DATASET_ROOT / "test" / "images"
mask_dir = DATASET_ROOT / "test" / "masks"

images = sorted(
    image_dir.glob("*.png")
)

rack_samples = []

print("Searching test images for rack/shelf pixels...")

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
            (image_path, rack_pixels)
        )


rack_samples.sort(
    key=lambda x: x[1],
    reverse=True
)

rack_samples = rack_samples[:MAX_SAMPLES]


print()
print("=" * 72)
print("RACK/SHELF TEST SAMPLES")
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

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

columns = 4

rows = (
    len(rack_samples) + columns - 1
) // columns

sheet = Image.new(
    "RGB",
    (
        columns * THUMB_WIDTH,
        rows * (THUMB_HEIGHT + 30)
    ),
    "white"
)

draw = ImageDraw.Draw(sheet)


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
        (x, y)
    )

    draw.text(
        (x + 5, y + THUMB_HEIGHT + 5),
        f"{index + 1}: "
        f"{image_path.name} "
        f"({rack_pixels}px)",
        fill="black"
    )


output_path = (
    OUTPUT_DIR /
    "rack_test_samples.png"
)

sheet.save(output_path)

print()
print(
    f"Saved:\n{output_path}"
)
