from pathlib import Path
from PIL import Image, ImageDraw
import random

root = Path("datasets/thud_people_yolo_v1")
img_dir = root / "images/train"
lbl_dir = root / "labels/train"

images = [p for p in img_dir.glob("*.png")]

random.seed(42)
samples = random.sample(images, 12)

for i, img_path in enumerate(samples):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    label_path = lbl_dir / f"{img_path.stem}.txt"

    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue

        cls, xc, yc, w, h = map(float, line.split())

        x1 = (xc - w / 2) * img.width
        y1 = (yc - h / 2) * img.height
        x2 = (xc + w / 2) * img.width
        y2 = (yc + h / 2) * img.height

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    out = Path("/tmp") / f"thud_preview_{i:02d}.jpg"
    img.save(out, quality=90)

    print(out)
