"""Train and evaluate the local LOCO equipment pilot on whole held-out sites."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    dataset, weights, output = (path.resolve() for path in (args.dataset, args.weights, args.output))
    manifest_hash = digest(dataset / "manifest.json")
    initial_hash = digest(weights)
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

    options = dict(imgsz=640, batch=8, device=0, workers=2, project=str(output), exist_ok=False)
    training = dict(data=str(dataset / "data.yaml"), name="train", epochs=6, seed=20261123,
                    deterministic=True, amp=False, optimizer="AdamW", lr0=.001,
                    warmup_epochs=1, freeze=10, mosaic=.5, close_mosaic=2, plots=False)
    torch.set_num_threads(2)
    YOLO(str(weights)).train(**options, **training)
    best = output / "train/weights/best.pt"
    model = YOLO(str(best))
    expected = json.loads((dataset / "data.yaml").read_text())["names"]
    if list(model.names.values()) != expected:
        raise ValueError("Trained checkpoint class order differs from the dataset")

    evidence = {"manifest_sha256": manifest_hash, "initial_sha256": initial_hash, "best_sha256": digest(best),
                "torch": str(torch.__version__), "ultralytics": ultralytics.__version__,
                "gpu": torch.cuda.get_device_name(0), "train": {**options, **training}, "sites": {}}
    for site in ("site1", "site4"):
        scores = model.val(**options, data=str(dataset / (site + ".yaml")), name=site, split="test",
                           conf=.001, iou=.7, nms=True, max_det=300, plots=False, verbose=False)
        support = scores.nt_per_class
        ap = {int(cid): scores.box.class_result(i) for i, cid in enumerate(scores.box.ap_class_index)}
        evidence["sites"][site] = {
            "summary": {k: float(v) for k, v in scores.results_dict.items()},
            "classes": {name: {"ground_truth": int(support[cid]),
                               "ap50": float(ap[cid][2]) if cid in ap else None,
                               "ap50_95": float(ap[cid][3]) if cid in ap else None}
                        for cid, name in enumerate(expected)},
        }
        print(site, evidence["sites"][site], flush=True)
    (output / "evaluation.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
