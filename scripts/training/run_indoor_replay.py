"""Train and evaluate a small replay experiment with two THUD++ captures."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import yaml

from scripts.data.prepare_indoor_replay import area, sha
from scripts.evaluation.compare_human_detection import score_boxes, _metrics
from scripts.laptop_demo import check_checkpoint, make_profiles, validate_model_names


def verify_inputs(root, dataset, manifest):
    config = yaml.safe_load((dataset / "data.yaml").read_text())
    expected_config = {"path": str(dataset), "train": "images/train", "val": "images/val",
                       "names": {0: "person", 1: "rider"}}
    if config != expected_config:
        raise ValueError("Dataset configuration differs from the frozen split and class contract")
    audit = root / "results/local/model_training_20261115/inventory_and_data_audit.json"
    fingerprints = json.loads(audit.read_text())["fingerprints"]
    for entry in fingerprints:
        image = Path(entry["image"])
        split = image.parent.name
        label = dataset / "labels" / split / (image.stem + ".txt")
        if sha((dataset / "images" / split / image.name).read_bytes()) != entry["image_sha256"]:
            raise ValueError(f"Road image changed: {image.name}")
        if sha(label.read_bytes()) != entry["label_sha256"]:
            raise ValueError(f"Road label changed: {label.name}")
    expected = {str(Path(entry["image"]).relative_to(Path(entry["image"]).parents[2]))
                for entry in fingerprints}
    for row in manifest["frames"]:
        if (row["split"], row["session"]) not in {("train", 1), ("probe", 2)}:
            raise ValueError("Indoor capture is assigned to the wrong split")
        expected.add(row["image"])
        image = dataset / row["image"]
        if sha(image.read_bytes()) != row["image_sha256"]:
            raise ValueError(f"Indoor image changed: {image.name}")
        if row["split"] == "train":
            label = dataset / "labels/train" / (image.stem + ".txt")
            if sha(label.read_bytes()) != row["label_sha256"]:
                raise ValueError(f"Indoor label changed: {label.name}")
    actual = {str(path.relative_to(dataset)) for split in ("train", "val", "probe")
              for path in (dataset / "images" / split).iterdir()}
    if actual != expected:
        raise ValueError("Dataset image inventory differs from the audit and manifest")


def evaluate_indoor(model, dataset, frames, confidence):
    matches = []
    misses = []
    false_positives = 0
    for row in frames:
        result = model.predict(str(dataset / row["image"]), imgsz=960, conf=confidence,
                               classes=[0], iou=.7, nms=True, max_det=300, device=0,
                               quantize=None, augment=False, verbose=False)[0]
        boxes = result.boxes.xyxyn.cpu().tolist()
        truth = row["boxes"]
        score = score_boxes(truth, boxes)
        matched = {pair["truth_index"] for pair in score["matches"]}
        matches.extend(box for index, box in enumerate(truth) if index in matched)
        misses.extend(box for index, box in enumerate(truth) if index not in matched)
        false_positives += score["fp"]
    large_tp = sum(area(box) >= .1 for box in matches)
    large_fn = sum(area(box) >= .1 for box in misses)
    return {**_metrics({"tp": len(matches), "fp": false_positives, "fn": len(misses)}),
            "large_tp": large_tp, "large_fn": large_fn, "images": len(frames)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "train", "evaluate"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    manifest_bytes = (dataset / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    verify_inputs(root, dataset, manifest)
    if (output / (args.stage + ".json")).exists() or (output / args.stage).exists():
        raise ValueError("Stage output already exists")
    output.mkdir(parents=True, exist_ok=True)
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
    profiles = make_profiles(root)
    starting = profiles["candidate"]
    check_checkpoint(starting)
    evidence = {
        "manifest_sha256": sha(manifest_bytes),
        "script_sha256": sha(Path(__file__).read_bytes()),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "initialization_sha256": starting.fingerprint,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    if args.stage in ("smoke", "train"):
        model = YOLO(str(starting.checkpoint))
        validate_model_names(starting, model.names)
        smoke = args.stage == "smoke"
        params = dict(data=str(dataset / "data.yaml"), epochs=1 if smoke else 12, patience=12,
                      imgsz=960, batch=2, device=0, workers=2, optimizer="AdamW", lr0=1e-4, lrf=.1,
                      warmup_epochs=1, warmup_bias_lr=1e-4, cos_lr=True, freeze=10, cls_remap=True,
                      seed=20261117, deterministic=True, amp=True, cache=False, close_mosaic=0 if smoke else 3,
                      nms=True, resume=False, pretrained=True, project=str(output), name=args.stage,
                      exist_ok=False, plots=True)
        model.train(**params)
        best = output / args.stage / "weights/best.pt"
        validate_model_names(starting, YOLO(str(best)).names)
        check_checkpoint(starting)
        evidence.update(arguments=params, best_sha256=sha(best.read_bytes()))
    else:
        trained = output / "train/weights/best.pt"
        prior = json.loads((output / "train.json").read_text())
        if prior["manifest_sha256"] != evidence["manifest_sha256"] or prior["best_sha256"] != sha(trained.read_bytes()):
            raise ValueError("Training dataset or checkpoint differs from the recorded run")
        control_path = root / "results/local/zoom_replay_20261117/control_completed.json"
        control = json.loads(control_path.read_text())
        control_weight = Path(control["run"]) / "weights/best.pt"
        if sha(control_weight.read_bytes()) != control["best_sha256"]:
            raise ValueError("Prior control checkpoint changed")
        checkpoints = {
            "pretrained": profiles["general"].checkpoint, "road_reference": profiles["reference"].checkpoint,
            "starting_candidate": starting.checkpoint, "prior_control": control_weight, "indoor_replay": trained,
        }
        check_checkpoint(profiles["general"])
        check_checkpoint(profiles["reference"])
        evidence.update(indoor_confidence=.25, matching_iou=.5, models={})
        frames = [row for row in manifest["frames"] if row["split"] == "probe"]
        for key, checkpoint in checkpoints.items():
            model = YOLO(str(checkpoint))
            validate_model_names(profiles["general"] if key == "pretrained" else starting, model.names)
            row = {"sha256": sha(checkpoint.read_bytes()), "indoor": evaluate_indoor(model, dataset, frames, .25)}
            if key != "pretrained":
                val = model.val(data=str(dataset / "data.yaml"), split="val", imgsz=960, batch=2,
                                device=0, workers=2, conf=.001, iou=.7, nms=True, max_det=300,
                                quantize=None, augment=False, plots=False, project=str(output),
                                name="road_" + key, exist_ok=False)
                row["road_validation"] = {key: float(value) for key, value in val.results_dict.items()}
            evidence["models"][key] = row
            print(key, json.dumps(row), flush=True)
            del model
            torch.cuda.empty_cache()
    (output / (args.stage + ".json")).write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
