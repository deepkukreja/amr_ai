from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import cv2

root = Path("datasets/torwic_semantic/Labelled_Test_Data")
out = Path("runs/five_ambiguous_persons.png")
out.parent.mkdir(parents=True, exist_ok=True)

targets = [
    ("000000912", "image_2", "person_000000003.png"),
    ("000000913", "image_1", "person_000000003.png"),
    ("000000914", "image_1", "person_000000003.png"),
    ("000001789", "image_1", "person_000000000.png"),
    ("000001807", "image_1", "person_000000000.png"),
]

panels = []

for seq, image_name, mask_name in targets:

    folder = root / seq / f"{image_name}.png_raw_labels_2d"

    source = np.array(
        Image.open(folder / "source_image.png").convert("RGB")
    )

    rgba = np.array(
        Image.open(folder / mask_name)
    )

    mask = (rgba[:, :, 3] > 0).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    component_ids = sorted(
        range(1, n),
        key=lambda i: stats[i, cv2.CC_STAT_AREA],
        reverse=True
    )

    overlay = source.copy()

    # Show the complete TorWIC person mask in red.
    overlay[mask > 0] = (
        overlay[mask > 0] * 0.4
        + np.array([255, 40, 40]) * 0.6
    ).astype(np.uint8)

    # Draw each connected component.
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
            4
        )

        cv2.putText(
            overlay,
            f"C{rank}: {area:,}",
            (x + 5, max(25, y + 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

    img = Image.fromarray(overlay)

    # Scale each panel to fit 1280x760 while preserving aspect ratio.
    img.thumbnail((1280, 760))

    panel = Image.new("RGB", (1280, 820), "black")
    panel.paste(
        img,
        (
            (1280 - img.width) // 2,
            50
        )
    )

    draw = ImageDraw.Draw(panel)

    areas = [
        int(stats[cid, cv2.CC_STAT_AREA])
        for cid in component_ids
    ]

    draw.text(
        (15, 10),
        f"{seq}/{image_name} | {mask_name} | "
        f"components={len(areas)} | areas={areas}",
        fill="white"
    )

    panels.append(panel)

result = Image.new(
    "RGB",
    (1280, 820 * len(panels)),
    "black"
)

for i, panel in enumerate(panels):
    result.paste(panel, (0, i * 820))

result.save(out)

print("=" * 75)
print("FIVE AMBIGUOUS PERSON MASKS")
print("=" * 75)
print(out)
