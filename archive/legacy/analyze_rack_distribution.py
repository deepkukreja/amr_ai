from pathlib import Path
import numpy as np
from PIL import Image


DATASET_ROOT = Path(
    "datasets/torwic_segmentation/split"
)

RACK_CLASS_ID = 6


def analyze_split(split):

    mask_dir = DATASET_ROOT / split / "masks"

    counts = []

    for mask_path in sorted(mask_dir.glob("*.png")):

        mask = np.array(
            Image.open(mask_path).convert("L"),
            dtype=np.int64
        )

        rack_pixels = int(
            np.sum(mask == RACK_CLASS_ID)
        )

        if rack_pixels > 0:
            counts.append(rack_pixels)

    counts = np.array(
        counts,
        dtype=np.int64
    )

    print()
    print("=" * 72)
    print(f"{split.upper()} RACK DISTRIBUTION")
    print("=" * 72)

    print(
        f"Images containing rack pixels : {len(counts)}"
    )

    if len(counts) == 0:
        print("No rack pixels found.")
        return

    print(
        f"Minimum rack pixels            : {counts.min()}"
    )

    print(
        f"25th percentile                : "
        f"{np.percentile(counts, 25):.0f}"
    )

    print(
        f"Median                         : "
        f"{np.median(counts):.0f}"
    )

    print(
        f"75th percentile                : "
        f"{np.percentile(counts, 75):.0f}"
    )

    print(
        f"Maximum rack pixels            : {counts.max()}"
    )

    for threshold in [
        1000,
        5000,
        10000,
        25000,
        50000,
        100000
    ]:

        number = np.sum(
            counts >= threshold
        )

        percentage = (
            number / len(counts)
        ) * 100

        print(
            f">= {threshold:>6} pixels        : "
            f"{number:>3} "
            f"({percentage:>5.1f}%)"
        )


for split in [
    "train",
    "val",
    "test"
]:
    analyze_split(split)
