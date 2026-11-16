# Dynamic FP32 ONNX parity — 16 November 2026

Exported the existing 56-epoch COCO-initialized person/rider candidate as dynamic
FP32 ONNX (opset 17), with raw one-to-many outputs and external class-aware NMS.
The original checkpoint and selected reference remain unchanged. This measures
export parity, not improved detection accuracy, Jetson speed or ROS deployment.

The [tracked summary](../results/research/20261116/onnx_parity.json) records hashes,
versions, numerical checks and matched validation results. Generated .pt/.onnx
files, raw validation output and the export environment remain local.

## Measured checks

- ONNX graph validation passed; class metadata is person (0), rider (1).
- Dynamic inputs 1×3×512×960, 2×3×960×960 and 4×3×512×960 passed CPU raw-output
  comparisons. Maximum coordinate error: 0.002503 pixels; maximum score error:
  approximately 1.42e-7. Declared limits were 0.01 pixels and 1e-4 respectively.
- Both formats validated on the same 59 reviewed images, size 960, rectangular
  batches of 4, FP32 CPU, confidence 0.001, NMS IoU 0.70 and max_det 300.
- Aggregate mAP50 gap: 0.0. Aggregate mAP50:95 gap: 2.674e-8; largest class
  mAP50:95 gap: 5.348e-8. Declared limits were 0.001 aggregate / 0.005 per class.
- The training environment retained torch 2.11.0+cu128, NumPy 2.5.3 and
  Ultralytics 8.4.156, with neither ONNX nor ONNX Runtime installed there.

These images are development validation, not a blind field test. The remaining
false-positive/missed-detection tradeoff is unchanged. Export parity does not
resolve the separate ROS NumPy/cv_bridge compatibility issue.

## Reproduce without changing the training environment

Start at the repository root with the recorded training environment available.
Create an ignored exporter environment that can read its packages, then install
only the pinned export additions into that new environment:

```bash
python3 -m venv results/local/onnx_export_env
training_site=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
AMR_TRAINING_SITE="$training_site" results/local/onnx_export_env/bin/python -c 'import os,sysconfig; from pathlib import Path; (Path(sysconfig.get_paths()["purelib"])/"amr_training_readonly.pth").write_text(os.environ["AMR_TRAINING_SITE"]+"\n")'
results/local/onnx_export_env/bin/python -m pip install -r requirements-export.txt
export AMR_ONNX_OUTPUT="$PWD/results/local/onnx_reproduction_01"
results/local/onnx_export_env/bin/python scripts/export_onnx_parity.py export
results/local/onnx_export_env/bin/python scripts/export_onnx_parity.py raw_parity
results/local/onnx_export_env/bin/python scripts/export_onnx_parity.py validate_pt
results/local/onnx_export_env/bin/python scripts/export_onnx_parity.py validate_onnx
results/local/onnx_export_env/bin/python scripts/export_onnx_parity.py summarize
```

The runner verifies the pinned source checkpoint fingerprint, copies it to the
output directory, disables online integrations and automatic installation, and
refuses to replace an existing export/validation record. It needs the authorised
local canonical dataset and original checkpoint, which are not downloaded by
the script. Use a fresh AMR_ONNX_OUTPUT path for each reproduction.

The dynamic model accepts variable batches and rectangular inputs. Outputs are
raw [batch, 6, anchors] values, not final detections: run the same external
class-aware NMS and preserve person/rider names. The calibrated operating profile
was selected for the PyTorch checkpoint; do not bypass its fingerprint binding
by substituting an ONNX path. Any ONNX profile needs the ONNX artifact hash and
matched preprocessing/postprocessing recorded separately.

No inference graph was installed in ROS, no physical motion was requested, and
no target-device benchmark was performed. Keep source/summary updates in Git;
retain binary artifacts and datasets in local evidence or an explicitly chosen
model registry.
