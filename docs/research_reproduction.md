# Reproducing the November 2026 person/rider research

Tracked [summary](../results/research/20261116/summary.json) records actual measured
counts, checkpoint hashes, split sizes and limitations. Raw images, annotation
JSON, labels, weights and prediction caches stay local. Obtain the authorised
datasets/checkpoints and verify their hashes before running these recipes.
The earlier individual reports link to full local evidence under results/local;
those ignored files will not appear in a fresh clone.

Use the existing training environment used by the experiment: Python 3.12.3,
PyTorch 2.11.0+cu128, Ultralytics 8.4.156, with a working CUDA GPU. Keep ROS
Python separate. Start in the repository root. Inspect dataset provenance,
reviewed labels, classes and clip splits before training; the YAML alone does
not validate them. The 59 validation images have known reference-model ancestry
exposure. The development benchmark has informed selection repeatedly.

## Recorded training recipes

```bash
.venv/bin/yolo detect train cfg=configs/training/manual280_continuation.yaml
.venv/bin/yolo detect train cfg=configs/training/manual280_coco_frozen.yaml
```

The recipes use relative checkpoint/data paths and new output names. They refuse
existing directories; choose a fresh name for each reproduction. Continuation
starts from the prior Manual-280 low-learning-rate best.pt. The COCO recipe starts
from weights/yolo26s.pt and freezes layers 0–9. Actual original runs stopped at
21 and 56 epochs respectively. Default arguments depend on the pinned installed
version. Exact rerun metrics need not match bit-for-bit across GPU/platforms.

Disable authenticated online integrations for a local-only reproduction: remove
ULTRALYTICS_API_KEY from the process environment and disable Ultralytics online
tracking integrations. The recorded training commands did this in-process. Do
not assume that an arbitrary globally configured yolo CLI is offline.

## Validation-only operating thresholds

The runner is `scripts/calibrate_detection_thresholds.py`. It requires the exact
two recorded checkpoints, canonical local reviewed datasets and the preceding
raw development report with image/annotation hashes. The latter defaults to
results/local/model_clean_baseline_20261115/development_comparison.json; override
with AMR_PRIOR_DEVELOPMENT_REPORT when needed. No dataset download occurs.

```bash
export AMR_CALIBRATION_OUTPUT="$PWD/results/local/calibration_reproduction_01"
.venv/bin/python scripts/calibrate_detection_thresholds.py collect_validation
.venv/bin/python scripts/calibrate_detection_thresholds.py select
.venv/bin/python scripts/calibrate_detection_thresholds.py evaluate
```

The runner disables online integrations in-process, freezes the model-bound JSON
before development evaluation and refuses reuse. It re-matches predictions after
filtering and verifies the fixed candidate counts reproduce. Selection uses only
the 59-image validation split, a confidence grid 0.05–0.90, and a reference recall
floor per class. No development threshold search is implemented.

The tracked research descriptor is
configs/inference/person_rider_coco_validation_20261116.json. Its checkpoint path
is normalized relative to this repository; it records the original local frozen
configuration hash separately. These confidence scores are not calibrated
probabilities, and the descriptor is not installed as a ROS runtime profile.

## Checks and data readiness

```bash
python3 -m pytest -q tests
git diff --check
```

93 source tests passed for the recorded calibration increment. The 120-frame
pending annotation batch has zero reviewed files and was not used in training.
It adds spaced frames from existing TRAIN clips, not new scene diversity.
Human review under the current V2 convention and an untouched whole-session
field collection remain the next data steps. Do not manufacture training
negatives from missing labels or mix warehouse and road class ontologies.

## Export parity

The [ONNX reproduction guide](onnx_export_parity_20261116.md) covers the isolated export environment and matching PyTorch/ONNX validation.
