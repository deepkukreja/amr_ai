"""Train the research-only BDD100K rider and two-wheeler pilot locally."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path

from scripts.data.freeze_bdd100k_pilot import CLASS_IDS, ROOT, sha
from scripts.laptop_demo import check_checkpoint, make_profiles


def verify_dataset(root):
    config = root / "data.yaml"
    names = list(CLASS_IDS)
    if json.loads(config.read_text()) != {
        "path": str(root), "train": "images/train", "val": "images/val",
        "test": "images/test", "names": names,
    }:
        raise ValueError("Unexpected BDD100K pilot dataset configuration")
    manifest = (root / "dataset_manifest.json").read_bytes()
    data = json.loads(manifest)
    if data["scope"] != "research dataset":
        raise ValueError("Expected the full BDD100K research manifest")
    frames = data["frames"]
    reserved = json.loads((root / "manifest.json").read_bytes())
    if reserved["scope"] != "public evaluation" or reserved["frames"] != [
        row for row in frames if row["split"] == "test"
    ]:
        raise ValueError("The reserved evaluation manifest differs from the BDD100K pilot")
    if Counter(row["split"] for row in frames) != {"train": 272, "val": 56, "test": 84}:
        raise ValueError("Unexpected BDD100K pilot split counts")
    if len({row["group"] for row in frames}) != len(frames):
        raise ValueError("A BDD100K clip family is reused")
    if len({row["sha256"] for row in frames}) != len(frames):
        raise ValueError("BDD100K pilot repeats image content")
    for row in frames:
        split = row["split"]
        image = (ROOT / row["image"]).resolve()
        label = root / "labels" / split / (image.stem + ".txt")
        annotation = root / "source_labels" / split / (image.stem + ".json")
        if image.parent != root / "images" / split or row["release_split"] != ("val" if split == "test" else "train"):
            raise ValueError("Unexpected BDD100K image path or release split")
        for file, key in ((image, "sha256"), (label, "yolo_label_sha256"), (annotation, "annotation_sha256")):
            if sha(file.read_bytes()) != row[key]:
                raise ValueError(f"BDD100K pilot artifact changed: {file.name}")
    return names, sha(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    dataset, output = args.dataset.resolve(), args.output.resolve()
    names, manifest_sha = verify_dataset(dataset)
    profile = make_profiles(ROOT)["general"]
    check_checkpoint(profile)
    output.mkdir(parents=True)
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
    options = dict(data=str(dataset / "data.yaml"), project=str(output), name="train", epochs=10,
                   imgsz=960, batch=4, device=0, workers=2, seed=20261123, amp=False,
                   optimizer="AdamW", lr0=.001, warmup_epochs=1, freeze=10, mosaic=.4,
                   close_mosaic=2, deterministic=True, cls_remap=True, plots=False)
    YOLO(str(profile.checkpoint)).train(**options)
    best = output / "train/weights/best.pt"
    if list(YOLO(str(best)).names.values()) != names:
        raise ValueError("Trained checkpoint does not match the four-class label contract")
    result = {"dataset_manifest_sha256": manifest_sha, "best_sha256": sha(best.read_bytes()),
              "initial_sha256": profile.fingerprint, "device": torch.cuda.get_device_name(0),
              "torch": str(torch.__version__), "ultralytics": ultralytics.__version__,
              "train": options}
    (output / "training.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Best research checkpoint", result["best_sha256"], flush=True)


if __name__ == "__main__":
    main()
