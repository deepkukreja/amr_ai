from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "runs/detect/runs/yolo26s_torwic_960/weights/best.pt"

CLASS_NAMES = {
    0: "person",
    1: "fork_truck",
    2: "cart_pallet_jack",
    3: "rack_shelf",
    4: "misc_dynamic_feature",
}

CLASS_THRESHOLDS = {
    0: 0.30,
    1: 0.25,
    2: 0.35,
    3: 0.30,
    4: 0.35,
}

model = YOLO(MODEL_PATH)

image = "datasets/torwic_yolo_clean/val/images/000001789_image_1.png"

results = model.predict(
    source=image,
    imgsz=960,
    conf=0.001,
    device=0,
    verbose=False,
)

result = results[0]

print("\nAMR OBJECT DETECTION")
print("=" * 70)

if result.boxes is None or len(result.boxes) == 0:
    print("No objects detected.")
else:
    kept = 0

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()

    for box, cls, conf in zip(boxes, classes, confidences):
        threshold = CLASS_THRESHOLDS.get(int(cls), 0.25)

        if float(conf) < threshold:
            continue

        x1, y1, x2, y2 = box

        print(
            f"{CLASS_NAMES[int(cls)]:24s} "
            f"conf={conf:.3f} "
            f"box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
        )

        kept += 1

    print("-" * 70)
    print(f"Objects kept: {kept}")
