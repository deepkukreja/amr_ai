# Fresh COCO-initialized comparison — 15 November 2026

Completed 56 epochs from the existing COCO-pretrained
YOLO26s with layers 0–9 frozen, using the unchanged 221/59 image split.
Best validation CSV epoch: 44.

The new recipe does not pass the aggregate-F1 acceptance rule. Preserve both candidates and keep the previous low-learning-rate model as the reference.

| Model | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Existing reference | 159 | 116 | 187 | 0.5782 | 0.4595 | 0.5121 |
| COCO-initialized candidate | 175 | 165 | 171 | 0.5147 | 0.5058 | 0.5102 |

Both used the established 166-image development benchmark, FP32 at 960,
NMS IoU 0.70, person/rider thresholds 0.15/0.25 and class-aware matching
IoU 0.50. These thresholds were not tuned during this experiment.
The development set has already informed selection and is not an untouched test.

The new initialization metadata records COCO training and no local AMR
fine-tuning. This avoids the old model's known local ancestry exposure to
the 59 validation images. Unknown external pretraining overlap and scene
dependence remain limitations. Multiple training choices change together,
so the comparison does not isolate one causal factor.

The review found small-object weaknesses and prepared visual sheets for 16
unusual-aspect training boxes. Flags are review candidates, not confirmed
label errors. No existing labels were changed. An image-only annotation
batch contains 120 additional frames from 54 existing TRAIN clips: 42 night,
11 dusk and 67 daytime. Validation/development clips and exact duplicate
images are excluded. It adds annotation coverage, not new scene diversity.
All 120 decode, match hashes and load in the existing annotation tool.

Pending workspace: `datasets/indian_manual_train_v3_pending_20261115`.
It contains a manifest, selection audit, current V2 guide and `annotate.py`
launcher. No pseudo-labels, model proposals or training labels were copied,
and no data.yaml was created. None of these frames was used for this run;
human review is required before a new train-only dataset version is built.

Checkpoint: `runs/experiments/indian_manual280_coco_frozen_20261115/weights/best.pt`.
The [local training report](../results/local/model_clean_baseline_20261115/REPORT.md)
and [data-review report](../results/local/model_data_review_20261115/REVIEW.md)
contain all metrics, hashes, plots, review examples and reproduction scripts.
Evidence/data remain local and ignored by Git. No deployed detector,
runtime environment, physical motion, commit or push was changed.

A [portable measured summary](../results/research/20261116/summary.json) and [reproduction instructions](research_reproduction.md) are tracked in Git. Large raw evidence remains local.

Publication note: the experiment records above describe the measured local work. Its source, portable summaries and reproduction notes were subsequently committed and pushed to GitHub on 16 November 2026. Datasets, weights and full caches remain local.
