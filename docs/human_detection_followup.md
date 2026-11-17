# Human localization and crop-replay experiment — 17 November 2026

The laptop demo exposed missed foreground and partial-body people in the custom road detector. The existing pretrained COCO model remains the laptop-demo default after the user confirmed webcam detection. This experiment investigates the custom model without changing that working profile.

## Frozen development comparison

All 166 development images and annotations were verified against previously recorded SHA-256 hashes. They contain 346 reviewed boxes: 177 person and 169 rider. This set has repeatedly influenced development; it is not an untouched final test.

The new metric deliberately measures **class-agnostic human localization**, matching both ground-truth classes as humans using descending-IoU, one-to-one matching at IoU 0.50. All models use confidence 0.25, image size 960, FP32, NMS enabled at IoU 0.70, and no test-time augmentation. COCO predictions are restricted to its person class. This metric does not evaluate rider classification or replace the previous two-class results. Rider annotations can include vehicle overlap and therefore differ from COCO person boxes.

| Model | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Pretrained COCO person | 141 | 151 | 205 | 48.3% | 40.8% | 44.2% |
| Retained road reference | 153 | 69 | 193 | 68.9% | 44.2% | 53.9% |
| COCO-initialized road candidate | 170 | 125 | 176 | 57.6% | 49.1% | 53.0% |

These results support retaining distinct general-webcam and road-research profiles. The candidate detects more reviewed road humans with more false positives; the reference retains better aggregate F1 under this protocol. No model is promoted from this comparison.

## Coverage gap

The area groups are custom fractions of image area: small below 1%, medium from 1% to below 10%, large at least 10%. They are not COCO's pixel-area definitions or calibrated physical distances.

| Split | Images | Objects | Small | Medium | Large |
|---|---:|---:|---:|---:|---:|
| Training | 221 | 584 | 513 | 70 | 1 |
| Validation | 59 | 151 | 125 | 23 | 3 |
| Development | 166 | 346 | 318 | 28 | 0 |

The training split has zero large **person** boxes; its one large box is a rider. This is evidence of narrow size coverage, not proof of the sole cause of close-up failure. The development benchmark also cannot establish close-up performance because it contains no large boxes.

## Paired training protocol

Create two new short continuations from the same pinned COCO-initialized road candidate (`2f160e4b51e918d99cd009940bcc6c99e2fc394ff86dc29ebbb60a27359ec7f1`):

- **Control:** the original 221 reviewed training images.
- **Replay:** those originals plus 48 geometric views of 24 person instances from 16 existing training clips. Each selected target has at least 32 pixels of width and 100 pixels of height before cropping. Full-body and upper-body crops retain every positive intersection of every existing person/rider label, including clipped fragments. No detector-generated labels are used.

Both use the unchanged 59-image validation split, 12 epochs, seed 20261117, image size 960, batch 2, AdamW at 0.0001, frozen first 10 layers, cosine decay, mixed precision, and NMS validation. Replay has more training batches because it has additional views; equal nominal epochs do not imply equal image exposures or optimizer updates. The one-epoch smoke run is separate and does not initialize either arm.

The source input pairs and generated crop/label hashes are checked before each run. Validation/development clips are excluded from crop sources. User images and webcam frames are excluded. Crops add scale/framing variation, not independent scene diversity, higher source resolution, or new lighting conditions. Existing annotation errors could also be amplified.

Existing validation fitness selects each arm's `best.pt`. Compare the resulting checkpoints on the frozen development protocol and the user-supplied diagnostic examples, with no threshold search. Those examples are development probes, not untouched tests. Keep the reference and general-demo defaults unless the evidence supports a scoped change.

## Completed experiment and decision

The smoke epoch and both 12-epoch arms completed. Fresh FP32 validation uses batch 2 for all three checkpoints below. Development F1 uses the original fixed person/rider thresholds 0.15/0.25; diagnostic probes use the existing candidate thresholds 0.19/0.25. The starting candidate's earlier development TP/FP/FN counts reproduced exactly.

| Checkpoint | Validation mAP50–95 | Development two-class F1 | Foreground person matches | Upper-body person match |
|---|---:|---:|---:|---:|
| Starting road candidate | 21.90% | 51.02% | 1 / 3 | 0 / 1 |
| Unchanged-data continuation | 20.26% | 50.28% | 1 / 3 | 0 / 1 |
| Crop-replay continuation | 19.87% | 49.79% | 3 / 3 | 0 / 1 |

The probe matches use coarse, selected foreground boxes at IoU 0.50. They are localized diagnostics, not full-image precision estimates. The control also localized one target with the wrong rider class, which is deliberately excluded from its person-match count. All diagnostic image files and target coordinates remain local.

Crop replay recovered the supplied full-body example but reduced road-validation mAP and aggregate development F1. It did not recover the upper-body probe. The human-localization-only F1 also fell slightly (53.04% starting candidate, 53.00% control, 52.85% replay at common confidence 0.25), with increased recall and false positives.

**Decision: retain the current road reference and the pretrained COCO laptop-demo default.** Preserve the experimental weights for research; do not treat the replay arm as a deployment upgrade. The next data work should add separately reviewed near/partial-body examples and scene diversity with session/clip-separated validation. The existing 25 webcam images and 120 pending road frames remain unreviewed and are not training ground truth. Further duplication of the same small road collection does not supply those missing conditions.

The repeated ROS setup failure was outside the model tests: setup-ros constructs its package URL from an unauthenticated latest-release query. CI now preinstalls the verified ROS APT source package 1.3.0, the same version seen in the earlier successful job, and checks its published SHA-256. This removes that lookup while retaining the existing ROS setup action. The [official release](https://github.com/ros-infrastructure/ros-apt-source/releases/tag/1.3.0) and [setup action implementation](https://github.com/ros-tooling/setup-ros/blob/v0.7/src/setup-ros-ubuntu.ts) describe the package and detection path. The updated workflow passed the ROS build and live integration test.

## Reproduction

In the existing inference/training virtual environment, with local data, weights, the prior development comparison cache, and `results/local/model_training_20261115/inventory_and_data_audit.json` present:

```bash
python3 -m pytest -q tests
.venv/bin/python -m scripts.evaluation.compare_human_detection \
  --output results/local/human_detection_20261117 \
  --audit-cache results/local/model_clean_baseline_20261115/development_comparison.json
.venv/bin/python -m scripts.data.prepare_zoom_replay \
  --output datasets/indian_manual280_zoom_replay_20261117
.venv/bin/python -m scripts.training.train_zoom_replay --stage smoke
.venv/bin/python -m scripts.training.train_zoom_replay --stage control
.venv/bin/python -m scripts.training.train_zoom_replay --stage replay
.venv/bin/python -m scripts.evaluation.evaluate_zoom_replay \
  --output results/local/zoom_evaluation_20261117
.venv/bin/python -m scripts.evaluation.compare_human_detection \
  --output results/local/zoom_human_detection_20261117 \
  --audit-cache results/local/model_clean_baseline_20261115/development_comparison.json \
  --extra-checkpoint control=runs/experiments/indian_manual280_zoom_control_20261117/weights/best.pt \
  --extra-checkpoint replay=runs/experiments/indian_manual280_zoom_replay_20261117/weights/best.pt \
  --models control replay
```

Existing output directories are refused. Raw images, derived crops, annotations, weights, training logs and full per-image predictions remain local. Source imagery and reviewed road annotations retain the provenance described in the [research reproduction guide](research_reproduction.md). Public aggregate evidence is in [the dated summary](../results/research/20261117/human_detection.json).

The optional `--probe-dir` for `evaluate_zoom_replay` accepts PNG images and a local `targets.json` mapping each image stem to normalized XYXY person boxes. Probe coordinates and images are not bundled in Git. The measured paired outcome and checkpoint hashes are recorded in [the experiment result](../results/research/20261117/zoom_replay.json).
