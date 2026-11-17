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

## Reproduction

In the existing inference/training virtual environment, with local data and weights present:

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
```

Existing output directories are refused. Raw images, derived crops, annotations, weights, training logs and full per-image predictions remain local. Source imagery and reviewed road annotations retain the provenance described in the [research reproduction guide](research_reproduction.md). Public aggregate evidence is in [the dated summary](../results/research/20261117/human_detection.json).
