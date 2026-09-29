from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import cv2
import math

root = Path("datasets/torwic_semantic/Labelled_Test_Data")
out = Path("runs/tiny_person_components.png")
out.parent.mkdir(parents=True, exist_ok=True)

items = []

for mask_path in sorted(root.rglob("person_*.png")):

    rgba = np.array(Image.open(mask_path))
    mask = (rgba[:, :, 3] > 0).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    source_path = mask_path.parent / "source_image.png"
    source = Image.open(source_path).convert("RGB")

    for cid in range(1, n):

        area = int(stats[cid, cv2.CC_STAT_AREA])

        if area >= 50:
            continue

        x = int(stats[cid, cv2.CC_STAT_LEFT])
        y = int(stats[cid, cv2.CC_STAT_TOP])
        w = int(stats[cid, cv2.CC_STAT_WIDTH])
        h = int(stats[cid, cv2.CC_STAT_HEIGHT])

        # Add generous context around the tiny component.
        pad = 40

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(source.width, x + w + pad)
        y2 = min(source.height, y + h + pad)

        crop = source.crop((x1, y1, x2, y2))

        # Draw the component location.
        draw = ImageDraw.Draw(crop)

        draw.rectangle(
            [
                x - x1,
                y - y1,
                x + w - 1 - x1,
                y + h - 1 - y1
            ],
            outline="yellow",
            width=2
        )

        # Enlarge for inspection.
        scale = 4
        crop = crop.resize(
            (
                crop.width * scale,
                crop.height * scale
            ),
            Image.Resampling.NEAREST
        )

        items.append(
            (
                area,
                str(mask_path.relative_to(root)),
                crop
            )
        )

items.sort(key=lambda x: x[0])

print("=" * 75)
print("TINY PERSON COMPONENTS (<50 PIXELS)")
print("=" * 75)
print(f"Components found: {len(items)}")

for i, (area, path, _) in enumerate(items, 1):
    print(f"{i:2d}. {area:2d} px  {path}")

if not items:
    raise RuntimeError("No components below 50 pixels.")

panel_w = 500
panel_h = 350
cols = 2
rows = math.ceil(len(items) / cols)

sheet = Image.new(
    "RGB",
    (panel_w * cols, panel_h * rows),
    "black"
)

draw_sheet = ImageDraw.Draw(sheet)

for i, (area, path, crop) in enumerate(items):

    crop.thumbnail(
        (panel_w - 10, panel_h - 45)
    )

    x0 = (i % cols) * panel_w
    y0 = (i // cols) * panel_h

    sheet.paste(
        crop,
        (
            x0 + (panel_w - crop.width) // 2,
            y0 + 35
        )
    )

    draw_sheet.text(
        (x0 + 5, y0 + 5),
        f"{i+1}: {area}px",
        fill="yellow"
    )

sheet.save(out)

print(f"\nSaved: {out}")
