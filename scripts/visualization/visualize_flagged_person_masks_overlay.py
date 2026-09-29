from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import cv2
import math

root = Path("datasets/torwic_semantic/Labelled_Test_Data")
out = Path("runs/flagged_person_masks_overlay.png")
out.parent.mkdir(parents=True, exist_ok=True)

person_files = sorted(root.rglob("person_*.png"))

flagged = []

for path in person_files:
    rgba = np.array(Image.open(path))
    mask = (rgba[:, :, 3] > 0).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    areas = sorted(
        [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)],
        reverse=True
    )

    if len(areas) >= 2:
        total = sum(areas)
        if areas[1] / total >= 0.10:
            flagged.append(path)

thumb_w = 640
thumb_h = 360
cols = 2
rows = math.ceil(len(flagged) / cols)

sheet = Image.new(
    "RGB",
    (cols * thumb_w, rows * thumb_h),
    "black"
)

for i, mask_path in enumerate(flagged):

    source_path = mask_path.parent / "source_image.png"

    source = np.array(
        Image.open(source_path).convert("RGB")
    )

    rgba = np.array(
        Image.open(mask_path)
    )

    mask = (rgba[:, :, 3] > 0).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    # Sort components by area.
    component_ids = sorted(
        range(1, n),
        key=lambda x: stats[x, cv2.CC_STAT_AREA],
        reverse=True
    )

    # Red overlay for all person-mask pixels.
    overlay = source.copy()
    overlay[mask > 0] = (
        overlay[mask > 0] * 0.4
        + np.array([255, 40, 40]) * 0.6
    ).astype(np.uint8)

    # Draw a yellow box around each connected component.
    for rank, cid in enumerate(component_ids, 1):
        x = stats[cid, cv2.CC_STAT_LEFT]
        y = stats[cid, cv2.CC_STAT_TOP]
        w = stats[cid, cv2.CC_STAT_WIDTH]
        h = stats[cid, cv2.CC_STAT_HEIGHT]

        cv2.rectangle(
            overlay,
            (x, y),
            (x + w - 1, y + h - 1),
            (255, 255, 0),
            4
        )

        cv2.putText(
            overlay,
            f"C{rank}",
            (x + 5, max(20, y + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

    # Resize for contact sheet.
    img = Image.fromarray(overlay)
    img.thumbnail((thumb_w, thumb_h - 55))

    x0 = (i % cols) * thumb_w
    y0 = (i // cols) * thumb_h

    sheet.paste(
        img,
        (
            x0 + (thumb_w - img.width) // 2,
            y0 + 50
        )
    )

    draw = ImageDraw.Draw(sheet)

    # Relative identifier.
    relative = mask_path.relative_to(root)

    draw.text(
        (x0 + 5, y0 + 5),
        str(relative),
        fill="white"
    )

    areas = [
        int(stats[cid, cv2.CC_STAT_AREA])
        for cid in component_ids
    ]

    draw.text(
        (x0 + 5, y0 + 27),
        f"components={len(areas)}  areas={areas[:6]}",
        fill="yellow"
    )

sheet.save(out)

print("=" * 75)
print("FLAGGED PERSON MASK OVERLAY")
print("=" * 75)
print(f"Flagged masks: {len(flagged)}")
print(f"Saved: {out}")
