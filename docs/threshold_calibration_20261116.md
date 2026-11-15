# Validation-only threshold selection — 16 November 2026

Selected person confidence 0.19 / rider 0.25 for the unchanged COCO-initialized
56-epoch checkpoint, using only the fixed 59-image validation split. The rule
maximizes class F1 while retaining at least reference validation recall.
The configuration was frozen before the development comparison.

| Configuration | TP | FP | FN | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Existing reference | 159 | 116 | 187 | 0.4595 | 0.512077 |
| New model, old thresholds | 175 | 165 | 171 | 0.5058 | 0.510204 |
| New model, selected thresholds | 170 | 148 | 176 | 0.4913 | 0.512048 |

The calibrated model has higher recall with more false positives, and
effectively ties reference F1. Keep the existing reference selected and
preserve this profile for research. No superiority/equivalence or deployment
claim is made. The 166-image comparison is repeatedly used development data,
not an untouched test. No development threshold sweep was performed.

Added `scripts/threshold_selection.py` with test-first recall/leakage/count
guards. All 93 source tests passed. Matching is recomputed after filtering;
fixed candidate counts reproduced exactly from the new prediction cache.

See the [full local report](../results/local/threshold_calibration_20261116/REPORT.md)
for cache hashes, the model-bound frozen profile, conditions and limits.
The 120-image training-only annotation batch still has zero reviewed files;
human review and new whole-session field data remain the next data steps.
No model weights, existing annotations, runtime detector settings, cloud
services, dependencies, commits or pushes were changed.

A [portable measured summary](../results/research/20261116/summary.json) and [reproduction instructions](research_reproduction.md) are tracked in Git. Large raw evidence remains local.
