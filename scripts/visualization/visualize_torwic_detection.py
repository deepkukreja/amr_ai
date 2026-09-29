from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random

ROOT = Path("datasets/torwic_detection_raw")

IMAGE_DIR = ROOT / "images"
LABEL_DIR = ROOT / "labels"
OUT_DIR = Path("runs/torwic_detection_preview")

OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "person",
    1: "fork_truck",
    2: "cart_pallet_jack",
    3: "rack_shelf",
    4: "misc_dynamic_feature",
}

# Reproducible sample
random.seed(42)

images = sorted(IMAGE_DIR.glob("*.png"))

if not images:
    raise RuntimeError("No converted images found.")

sample = random.sample(
    images,
    min(10, len(images))
)

print("=" * 70)
print("TORWIC YOLO LABEL VISUALIZATION")
print("=" * 70)
print(f"Images available: {len(images)}")
print(f"Preview images:   {len(sample)}")
print(f"Output directory: {OUT_DIR}")

for i, image_path in enumerate(sample, start=1):

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    width, height = image.size

    if label_path.exists():

        lines = label_path.read_text().splitlines()

        for line in lines:

            parts = line.split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            # YOLO normalized → pixel coordinates
            box_w = w * width
            box_h = h * height

            center_x = xc * width
            center_y = yc * height

            x1 = center_x - box_w / 2
            y1 = center_y - box_h / 2
            x2 = center_x + box_w / 2
            y2 = center_y + box_h / 2

            draw.rectangle(
                [x1, y1, x2, y2],
                outline="red",
                width=3,
            )

            name = CLASS_NAMES.get(
                class_id,
                f"class_{class_id}"
            )

            draw.text(
                (x1, max(0, y1 - 15)),
                name,
                fill="red",
            )

    output_path = OUT_DIR / f"preview_{i:02d}.png"
    image.save(output_path)

    print(f"{i:02d}: {image_path.name} -> {output_path}")

print("\nDone.")
