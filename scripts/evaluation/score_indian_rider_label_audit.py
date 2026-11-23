"""Compare one fixed detector's predictions under two rider box conventions."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path

from scripts.data.indian_rider_label_audit import CLASSES, ROOT, digest
from scripts.evaluation.compare_human_detection import _box, load_annotation, score_boxes
from scripts.laptop_demo import check_checkpoint, make_profiles, validate_model_names


def paired_matches(legacy, revised, predictions):
    if len(legacy) != len(revised):
        raise ValueError("Each revised box must correspond to one legacy rider")
    old, new = (
        {item["truth_index"] for item in score_boxes(truth, predictions, .5)["matches"]}
        for truth in (legacy, revised)
    )
    return {"legacy_matches": len(old), "revised_matches": len(new), "both": len(old & new),
            "legacy_only": len(old - new), "revised_only": len(new - old),
            "neither": len(legacy) - len(old | new)}


def load_audit(root):
    selection = root / "selection.json"
    data = json.loads(selection.read_text())
    if data["scope"] != "development label audit":
        raise ValueError("Expected the Indian-road development label audit")
    frames = data["frames"]
    if Counter(row["partition"] for row in frames) != {"training_audit": 24, "label_bridge": 16}:
        raise ValueError("Expected 24 training-audit and 16 label-bridge frames")
    if len({row["clip_id"] for row in frames}) != len(frames):
        raise ValueError("Audit frames repeat source clips")

    bridge = {}
    checks = Counter()
    annotations = Counter()
    for frame in frames:
        image = ROOT / frame["image"]
        old_label = ROOT / frame["annotation"]
        legacy = load_annotation(old_label)
        if (digest(image) != frame["image_sha256"] or digest(old_label) != frame["annotation_sha256"]
                or legacy != frame["legacy"]):
            raise ValueError(f"Legacy audit inputs changed: {image.name}")
        review_file = root / "reviews" / (frame["audit_id"] + ".json")
        review = json.loads(review_file.read_text())
        if review["reviewed"] is not True or review["image_sha256"] != frame["image_sha256"]:
            raise ValueError(f"Missing validated review: {image.name}")
        coverage = set(review["coverage"])
        if not coverage <= set(CLASSES):
            raise ValueError(f"Unknown audit coverage: {image.name}")
        for entry in review["boxes"]:
            if entry["label"] not in CLASSES:
                raise ValueError(f"Unknown audit class: {image.name}")
            _box(entry["box"])
            annotations[entry["label"]] += 1
        part = frame["partition"]
        checks[f"{part}_reviewed"] += 1
        checks[f"{part}_complete"] += coverage == set(CLASSES)
        links = review["rider_pairs"]
        if part == "training_audit":
            if links:
                raise ValueError("Rider links belong on the label-bridge frames")
            continue
        seen_legacy = set()
        seen_review = set()
        pairs = []
        for link in links:
            i, j = link["legacy_index"], link["review_index"]
            boxes = review["boxes"]
            if (not 0 <= i < len(legacy) or not 0 <= j < len(boxes)
                    or legacy[i]["class_id"] != 1 or boxes[j]["label"] != "rider"
                    or i in seen_legacy or j in seen_review):
                raise ValueError(f"Invalid one-to-one rider link: {image.name}")
            seen_legacy.add(i)
            seen_review.add(j)
            pairs.append((legacy[i]["box"], boxes[j]["box"]))
        checks["legacy_riders_in_bridge"] += sum(item["class_id"] == 1 for item in legacy)
        bridge[str(image)] = pairs
    return selection, bridge, checks, annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new local score report")
    selection, bridge, checks, annotations = load_audit(args.audit.resolve())
    profile = make_profiles(ROOT)["candidate"]
    check_checkpoint(profile)
    os.environ.pop("ULTRALYTICS_API_KEY", None)
    os.environ.update(YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false", WANDB_MODE="disabled")

    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS
    dict.update(SETTINGS, {key: False for key in ("hub", "sync", "wandb", "mlflow", "comet",
                                                 "clearml", "neptune") if key in SETTINGS})
    dict.update(SETTINGS, {"api_key": ""})
    torch.set_num_threads(2)
    model = YOLO(str(profile.checkpoint))
    validate_model_names(profile, model.names)
    model_device = 0 if torch.cuda.is_available() else "cpu"
    results = model.predict(source=list(bridge), stream=True, imgsz=960, conf=.25, iou=.7, nms=True,
                            max_det=300, device=model_device, classes=[1], batch=1, verbose=False)
    total = Counter()
    processed = set()
    for result in results:
        pairs = bridge[result.path]
        if result.path in processed:
            raise ValueError("A prediction image appeared twice")
        processed.add(result.path)
        if pairs:
            previous, revised = zip(*pairs)
            total.update(paired_matches(previous, revised, result.boxes.xyxyn.cpu().tolist()))
    if processed != bridge.keys():
        raise ValueError("Incomplete paired benchmark inference")
    report = {"scope": "Indian-road development label-convention diagnostic",
              "selection_sha256": digest(selection),
              "checkpoint_sha256": profile.fingerprint,
              "ultralytics": ultralytics.__version__,
              "protocol": {"confidence": .25, "imgsz": 960, "iou": .5, "nms_iou": .7},
              "review_counts": checks, "positive_boxes_seen": annotations,
              "matched_rider_pairs": sum(len(values) for values in bridge.values()),
              "frames_with_rider_pairs": sum(bool(values) for values in bridge.values()),
              "paired_outcomes": total}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
