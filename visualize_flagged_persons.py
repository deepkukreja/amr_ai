from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import math

root = Path("datasets/torwic_semantic/Labelled_Test_Data")
out = Path("runs/flagged_person_masks.png")
out.parent.mkdir(parents=True, exist_ok=True)

person_files = sorted(root.rglob("person_*.png"))

flagged = []

for path in person_files:
    rgba = np.array(Image.open(path))
    mask = (rgba[:, :, 3] > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    areas = sorted(
        [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)],
        reverse=True
    )

    if len(areas) >= 2:
        total = sum(areas)
        second_fraction = areas[1] / total

        if second_fraction >= 0.10:
            flagged.append(path)

thumb_w = 480
thumb_h = 270
cols = 2
rows = math.ceil(len(flagged) / cols)

sheet = Image.new(
    "RGB",
    (cols * thumb_w, rows * thumb_h),
    "black"
)

for i, path in enumerate(flagged):

    source_path = path.parent / "source_image.png"

    if not source_path.exists():
        continue

    img = Image.open(source_path).convert("RGB")

    # Resize while preserving the full image.
    img.thumbnail((thumb_w, thumb_h - 35))

    x = (i % cols) * thumb_w
    y = (i // cols) * thumb_h

    sheet.paste(
        img,
        (
            x + (thumb_w - img.width) // 2,
            y + 30
        )
    )

    draw = ImageDraw.Draw(sheet)

    rgba = np.array(Image.open(path))
    mask = (rgba[:, :, 3] > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    areas = sorted(
        [
            int(stats[j, cv2.CC_STAT_AREA])
            for j in range(1, num_labels)
        ],
        reverse=True
    )

    total = sum(areas)
    second_fraction = (
        areas[1] / total
        if len(areas) >= 2
        else 0
    )

    text = (
        f"{path.parent.parent.name}/"
        f"{path.parent.name}\n"
        f"components={len(areas)}  "
        f"largest={areas[0]:,}  "
        f"second={second_fraction:.1%}"
    )

    draw.text(
        (x + 5, y + 5),
        text,
        fill="white"
    )

sheet.save(out)

print("=" * 75)
print("FLAGGED PERSON CONTACT SHEET")
print("=" * 75)
print(f"Flagged masks: {len(flagged)}")
print(f"Saved: {out}")

for i, path in enumerate(flagged, 1):
    print(f"{i:2d}. {path}")
