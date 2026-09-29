from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import cv2
import math

root = Path("datasets/torwic_semantic/Labelled_Test_Data")
out = Path("runs/most_ambiguous_person_masks.png")
out.parent.mkdir(parents=True, exist_ok=True)

targets = [
    "000000912",
    "000000913",
    "000000914",
    "000001789",
    "000001807",
]

files = []

for seq in targets:
    found = sorted(
        root.glob(
            f"{seq}/image_*_raw_labels_2d/person_*.png"
        )
    )
    files.extend(found)

thumb_w = 960
thumb_h = 600
cols = 1
rows = len(files)

sheet = Image.new(
    "RGB",
    (thumb_w, rows * thumb_h),
    "black"
)

for i, mask_path in enumerate(files):

    source_path = mask_path.parent / "source_image.png"

    source = np.array(
        Image.open(source_path).convert("RGB")
    )

    rgba = np.array(
        Image.open(mask_path)
    )

    mask = (rgba[:, :, 3] > 0).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    component_ids = sorted(
        range(1, n),
        key=lambda x: stats[x, cv2.CC_STAT_AREA],
        reverse=True
    )

    overlay = source.copy()

    # Red = all annotated person pixels.
    overlay[mask > 0] = (
        overlay[mask > 0] * 0.4
        + np.array([255, 40, 40]) * 0.6
    ).astype(np.uint8)

    for rank, cid in enumerate(component_ids, 1):

        x = stats[cid, cv2.CC_STAT_LEFT]
        y = stats[cid, cv2.CC_STAT_TOP]
        w = stats[cid, cv2.CC_STAT_WIDTH]
        h = stats[cid, cv2.CC_STAT_HEIGHT]
        area = stats[cid, cv2.CC_STAT_AREA]

        cv2.rectangle(
            overlay,
            (x, y),
            (x + w - 1, y + h - 1),
            (255, 255, 0),
            5
        )

        cv2.putText(
            overlay,
            f"C{rank} ({area:,})",
            (x + 5, max(25, y + 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            3,
            cv2.LINE_AA
        )

    img = Image.fromarray(overlay)
    img.thumbnail((thumb_w, thumb_h - 60))

    x0 = (thumb_w - img.width) // 2
    y0 = i * thumb_h + 50

    sheet.paste(img, (x0, y0))

    draw = ImageDraw.Draw(sheet)

    relative = mask_path.relative_to(root)

    areas = [
        int(stats[cid, cv2.CC_STAT_AREA])
        for cid in component_ids
    ]

    second_fraction = (
        areas[1] / sum(areas)
        if len(areas) >= 2
        else 0
    )

    draw.text(
        (10, i * thumb_h + 5),
        f"{relative} | "
        f"components={len(areas)} | "
        f"areas={areas} | "
        f"second={second_fraction:.1%}",
        fill="white"
    )

sheet.save(out)

print("=" * 75)
print("MOST AMBIGUOUS PERSON MASKS")
print("=" * 75)
print(f"Saved: {out}")

for p in files:
    print(p)
