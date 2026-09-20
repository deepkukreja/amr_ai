from PIL import Image
from pathlib import Path
import numpy as np

target = Path(
    "datasets/torwic_semantic/Labelled_Test_Data/"
    "000000285/image_0.png_raw_labels_2d"
)

print("=" * 70)
print("TORWIC PNG INSPECTION")
print("=" * 70)

for f in sorted(target.glob("*.png")):
    img = Image.open(f)
    arr = np.array(img)

    unique = np.unique(arr)

    print(f"\nFILE: {f.name}")
    print(f"  mode:        {img.mode}")
    print(f"  size:        {img.size}")
    print(f"  shape:       {arr.shape}")
    print(f"  dtype:       {arr.dtype}")
    print(f"  unique count:{len(unique)}")

    if len(unique) <= 20:
        print(f"  unique vals: {unique.tolist()}")
    else:
        print(f"  first vals:  {unique[:15].tolist()}")

    # Bounding box of non-zero pixels
    if arr.ndim == 2:
        ys, xs = np.where(arr != 0)
    else:
        ys, xs = np.where(np.any(arr != 0, axis=-1))

    if len(xs) > 0:
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        print(
            f"  nonzero bbox: "
            f"x={x1}, y={y1}, "
            f"w={x2-x1+1}, h={y2-y1+1}"
        )
    else:
        print("  nonzero bbox: NONE")
