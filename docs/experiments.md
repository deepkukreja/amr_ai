# AMR Perception Experiment History

This document records the tracked experiment history and reported results
for the AMR perception research project. Raw datasets, model weights, and
generated training outputs are maintained separately from this repository.

## 1. Experiment history

The recorded development sequence is:

| Version | Experiment direction |
|---|---|
| Original | YOLO-based detection baseline trained on TorWIC |
| V1 | TorWIC plus SEW adaptation |
| V2 | Low-light augmentation |
| V3 | Real SEW domain adaptation |
| V4 | Mixed THUD, TorWIC, and SEW training |
| V5b | Targeted scale augmentation |
| V6 | Real-domain adaptation using THUD real-scene data |
| V7 | Realistic low-light adaptation |

The existing project notes identify V6 as the development baseline.
This designation does not imply production readiness or independent
final-test validation.

V7 investigates low-light adaptation that better reflects real camera
conditions. The code for low-light dataset generation is located at
`scripts/data/generate_v7_lowlight.py`.

The presence of this script is not evidence that a V7 model has completed
training or passed evaluation. Record those milestones only after verifying
the generated dataset, training run, checkpoint, and evaluation results.

## 2. Tracked SEW model comparison

Source: `results/sew_model_comparison.csv`

The table below reproduces the values recorded in that file.

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Original | 0.348 | 0.157 | 0.0761 | 0.0150 |
| V1 | 0.674 | 0.399 | 0.4180 | 0.2220 |
| V2 | 0.810 | 0.389 | 0.4390 | 0.2600 |
| V3 | 0.753 | 0.385 | 0.4360 | 0.2380 |

These are reported experiment values, not a guarantee of performance on
new environments. The CSV does not record all evaluation details, such as
the exact checkpoint paths, confidence thresholds, or full split protocol.

When comparing future models, preserve the evaluation protocol and record
the dataset split, checkpoint, thresholds, class-level metrics, and
evaluation script alongside the aggregate metrics.

## 3. SEW condition comparison

Source: `results/sew_condition_comparison.csv`

The following values are the recorded mAP50 results for each condition.

| Condition | V1 mAP50 | V2 mAP50 | V3 mAP50 |
|---|---:|---:|---:|
| Cloudy | 0.8581 | 0.8696 | 0.8865 |
| Indoor | 0.0879 | 0.0663 | 0.0926 |
| Overcast | 0.6717 | 0.8048 | 0.5959 |
| Rain | 0.7993 | 0.8504 | 0.7983 |
| Snow | 0.6764 | 0.7046 | 0.7230 |
| Sunny | 0.8663 | 0.9476 | 0.9079 |

The recorded indoor results are substantially lower than the results for
several other conditions. This supports continued investigation of indoor
and low-light robustness.

The aggregate and per-condition results are stored in separate CSV files.
Their evaluation scopes must be checked before assuming the values are
directly comparable.

## 4. THUD real-scene error analysis

Source: `results/thud_real_capture2_error_analysis_v2_v3.csv`

This file records frame-level observations, including:

- Model and frame filename
- Object-size category
- Border location
- Blur category and blur score
- Detection status

Recorded statuses include `detected`, `low_confidence`, and
`no_useful_overlap`.

This is a frame-level error-analysis record. Its first few rows alone do
not establish an aggregate detection rate or overall model ranking.
Calculate such metrics only after checking the complete file and its
labeling and evaluation conventions.

## 5. Dark-indoor session split

Source: `configs/datasets/amr_dark_indoor_v1_session_split.txt`

The tracked split configuration specifies:

| Partition | Sessions |
|---|---|
| Train | session_01 through session_08 |
| Development | session_09 and session_10 |
| Test | session_11 and session_12 |

The configuration is marked:

`Session-level split — DO NOT CHANGE AFTER DATA COLLECTION`

Preserve the session-level separation. Use development data for model
selection and threshold tuning. Keep test sessions reserved for final
evaluation rather than repeatedly using them to guide development.

Before reporting a result, verify that the actual dataset manifest and
evaluation script follow this configuration.

## 6. Development benchmarks and limitations

The earlier project notes state that a 423-image SEW benchmark was used
repeatedly during development. It should therefore be treated as a
development or transfer benchmark, not as a pristine final test set.

A benchmark's name alone does not make it independent. Independence
depends on how its data were separated and whether the benchmark influenced
model selection, threshold tuning, or other development decisions.

Results from a development benchmark should be labeled accordingly.

## 7. V7 low-light research direction

The V7 direction investigates more realistic low-light adaptation rather
than relying only on exposure reduction.

The original project notes describe a mismatch between synthetic
low-light augmentation and real dark indoor imagery, including a contrast
difference.

The planned research questions are:

1. Does the generated data preserve realistic contrast and image
   characteristics?
2. Are source sessions separated correctly from evaluation sessions?
3. Does training on the generated data improve performance on held-out
   low-light scenes?
4. Are any improvements accompanied by regressions in other conditions?

Do not claim that V7 improves detection performance until the experiment
has been trained and evaluated under a documented protocol.

## 8. Reporting requirements for future experiments

For each new experiment, record:

- Experiment name and research question
- Dataset version and source
- Train, development, and test split definitions
- Training configuration and random seed, when available
- Exact model checkpoint
- Evaluation script and confidence thresholds
- Overall and class-level precision, recall, mAP50, and mAP50-95
- Relevant per-condition results and error analysis
- Limitations, regressions, and known data overlap

Keep dataset contents, model weights, and generated runs out of Git unless
there is a deliberate, documented reason to version a particular artifact.

A reported metric should always be traceable to its dataset split,
checkpoint, and evaluation procedure.
