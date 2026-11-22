"""Compare pinned AMR checkpoints against a frozen mixed-source development set."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

from scripts.evaluation.broad_benchmark import cohort_metrics
from scripts.evaluation.compare_human_detection import _box
from scripts.laptop_demo import check_checkpoint, make_profiles, validate_model_names

ROOT = Path(__file__).resolve().parents[2]
COCO = {
    "person": ("human",),
    "bicycle": ("bicycle",), "car": ("car",), "motorcycle": ("motorcycle",),
    "bus": ("bus",), "truck": ("truck",), "dog": ("dog",), "cat": ("cat",),
    "cow": ("cattle",), "horse": ("horse",), "sheep": ("sheep",), "bird": ("bird",),
}
ROAD = {"person": ("human", "pedestrian"), "rider": ("human", "rider")}
WAREHOUSE = {
    "person": ("human",), "fork_truck": ("forklift_or_truck",),
    "cart_pallet_jack": ("cart_or_pallet_jack",), "rack_shelf": ("rack_or_shelf",),
}
MAPPINGS = {"general": COCO, "reference": ROAD, "candidate": ROAD, "warehouse": WAREHOUSE}


def load_frames(manifest):
    data = json.loads(manifest.read_bytes())
    if data["version"] != 1 or data["scope"] != "development only":
        raise ValueError("Unexpected benchmark format or split")
    frames = {}
    seen = set()
    for frame in data["frames"]:
        source = (ROOT / frame["image"]).resolve()
        if not source.is_relative_to(ROOT / "datasets"):
            raise ValueError("Benchmark image is outside the dataset directory")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != frame["sha256"] or digest in seen:
            raise ValueError(f"Benchmark image changed or repeated: {source.name}")
        seen.add(digest)
        for obj in frame["boxes"]:
            _box(obj["box"])
            if obj["label"] not in frame["coverage"]:
                raise ValueError("Truth includes an unscored label")
        frames[str(source)] = frame
    if not frames:
        raise ValueError("Benchmark has no inputs")
    return frames


def synchronize(torch, device):
    if device != "cpu":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=["0", "cpu"], default="0")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refuse to replace an existing evaluation report")
    frames = load_frames(args.manifest)
    os.environ.pop("ULTRALYTICS_API_KEY", None)
    os.environ.update(YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false", WANDB_MODE="disabled")
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS

    dict.update(SETTINGS, {k: False for k in ("hub", "sync", "wandb", "mlflow", "comet", "clearml", "neptune")
                           if k in SETTINGS})
    dict.update(SETTINGS, {"api_key": ""})
    torch.set_num_threads(2)
    device = 0 if args.device == "0" else "cpu"
    report = {
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scorer_sha256": hashlib.sha256(Path(__file__).with_name("broad_benchmark.py").read_bytes()).hexdigest(),
        "data": dict(Counter(frame["cohort"] for frame in frames.values())),
        "ground_truth_support": {},
        "protocol": {"confidence": .25, "prediction_floor": .01, "iou": .5, "nms_iou": .7, "imgsz": 960,
                     "nms": True, "precision": "FP32", "augment": False, "batch": 1,
                     "ap": "all-point interpolated from confidence-ranked detections above prediction floor"},
        "device": torch.cuda.get_device_name(device) if device != "cpu" else "CPU",
        "torch": torch.__version__, "ultralytics": ultralytics.__version__, "models": {},
    }
    for cohort in report["data"]:
        records = [frame for frame in frames.values() if frame["cohort"] == cohort]
        report["ground_truth_support"][cohort] = dict(Counter(
            obj["label"] for frame in records for obj in frame["boxes"]
        ))
    for key, mapping in MAPPINGS.items():
        profile = make_profiles(ROOT)[key]
        check_checkpoint(profile)
        model = YOLO(str(profile.checkpoint))
        validate_model_names(profile, model.names)
        ids = [cid for cid, label in model.names.items() if label in mapping]
        if {model.names[cid] for cid in ids} != mapping.keys():
            raise ValueError(f"Label mapping differs from {key} checkpoint")
        grouped = defaultdict(list)
        duration = []
        for index, (source, frame) in enumerate(frames.items()):
            synchronize(torch, device)
            started = time.perf_counter()
            result = model.predict(source, imgsz=960, conf=.01, iou=.7, nms=True, max_det=300,
                                   device=device, quantize=None, augment=False, classes=ids, verbose=False)[0]
            boxes = result.boxes.cpu()
            predictions = [
                {"label": label, "box": _box(box), "confidence": score}
                for cid, box, score in zip(boxes.cls.tolist(), boxes.xyxyn.tolist(), boxes.conf.tolist())
                for label in mapping[model.names[int(cid)]]
            ]
            synchronize(torch, device)
            if index:
                duration.append((time.perf_counter() - started) * 1000)
            grouped[frame["cohort"]].append((frame, predictions))
        duration.sort()
        supported = {label for values in mapping.values() for label in values}
        metrics = {}
        for cohort, records in grouped.items():
            labels = set().union(*(frame["coverage"] for frame, _ in records))
            metrics[cohort] = cohort_metrics(records, supported & labels, .25)
        road_by_light = {
            light: cohort_metrics([row for row in grouped["road_reviewed"] if row[0]["light"] == light],
                                  supported & {"human", "pedestrian", "rider"}, .25)
            for light in sorted({frame["light"] for frame, _ in grouped["road_reviewed"]})
        }
        road_by_density = defaultdict(list)
        for row in grouped["road_reviewed"]:
            humans = sum(obj["label"] == "human" for obj in row[0]["boxes"])
            road_by_density["4_or_more" if humans >= 4 else "0_to_3"].append(row)
        report["models"][key] = {
            "checkpoint_sha256": profile.fingerprint, "supported_labels": sorted(supported),
            "metrics": metrics, "road_by_light": road_by_light,
            "road_by_reviewed_human_count": {
                label: cohort_metrics(rows, supported & {"human", "pedestrian", "rider"}, .25)
                for label, rows in road_by_density.items()
            },
            "wall_ms_p50": duration[math.ceil(.5 * len(duration)) - 1],
            "wall_ms_p95": duration[math.ceil(.95 * len(duration)) - 1],
        }
        print(f"{key}: scored {len(frames)} images", flush=True)
        del model
        if device != "cpu":
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        json.dump(report, out, indent=2, sort_keys=True)
        out.write("\n")


if __name__ == "__main__":
    main()
