# Broad object detection baseline v1

The initial benchmark is a frozen development comparison. Bootstrapping the current COCO checkpoint gives coverage for common vehicles and animals. The specialist checkpoints supply the existing rider and warehouse labels. The independent field holdout remains future work.

| Cohort | Images | Ground truth and scope |
|---|---:|---|
| Reviewed road | 48 | Previously used person/rider development clips. One image per clip. |
| Indoor THUD++ | 16 | Publisher-audited boxes from Capture 2 in one store. |
| Warehouse legacy | 19 | Converted legacy labels. Grouped categories and possible prior checkpoint exposure. |
| Open Images V5 | 70 | Public validation stills. Positive boxes and explicitly verified negatives. |

Manifest SHA-256 `0435a8380e5846d2f92b0f230d0ca536c393665ed95bb666501902cb59bd4ff3`. No raw frames or file-level annotations are included here.

## Same-cohort comparisons

Cells show matched ground-truth objects at confidence 0.25 and IoU 0.50, followed by false positives. A dash indicates an unsupported class or no applicable ground truth. Scores across cohorts describe different scenes and coverage.

| Label and cohort | Broad COCO | Road reference | Road candidate | Warehouse V6 |
|---|---:|---:|---:|---:|
| Road human | 39/88 (44%), 37 FP | 37/88 (42%), 21 FP | 44/88 (50%), 29 FP | 9/88 (10%), 11 FP |
| Road rider | — | 18/41 (44%), 17 FP | 23/41 (56%), 14 FP | — |
| Indoor human | 21/22 (95%), 1 FP | 4/22 (18%), 10 FP | 17/22 (77%), 5 FP | 18/22 (82%), 3 FP |
| Warehouse legacy human | 9/11 (82%), 1 FP | 8/11 (73%), 1 FP | 9/11 (82%), 2 FP | 6/11 (55%), 0 FP |
| Warehouse legacy forklift/truck | — | — | — | 6/11 (55%), 0 FP |
| Warehouse legacy cart/pallet jack | — | — | — | 10/21 (48%), 15 FP |
| Warehouse legacy rack/shelf | — | — | — | 3/10 (30%), 13 FP |
| Road human, 4+ reviewed humans/frame | 21/52 (40%), 22 FP | 21/52 (40%), 10 FP | 25/52 (48%), 20 FP | 4/52 (8%), 5 FP |
| Road human, 0–3 reviewed humans/frame | 18/36 (50%), 15 FP | 16/36 (44%), 11 FP | 19/36 (53%), 9 FP | 5/36 (14%), 6 FP |

## Broad COCO on public stills

| Class | Scored images | Matched / reviewed boxes | False positives | AP50–95 |
|---|---:|---:|---:|---:|
| bicycle | 6 | 5/9 | 7 | 24% |
| bird | 5 | 5/6 | 1 | 37% |
| bus | 8 | 5/5 | 2 | 43% |
| car | 19 | 13/19 | 8 | 49% |
| cat | 5 | 5/5 | 1 | 47% |
| cattle | 7 | 4/11 | 0 | 31% |
| dog | 9 | 5/6 | 3 | 54% |
| goat | — | — / 8 | — | — |
| horse | 7 | 6/6 | 1 | 90% |
| human | 29 | 50/66 | 29 | 49% |
| motorcycle | 5 | 5/5 | 3 | 58% |
| sheep | 9 | 10/13 | 6 | 45% |
| truck | 10 | 4/6 | 9 | 23% |
| van | — | — / 8 | — | — |

Van and goat have reviewed support in the public cohort. The COCO checkpoint has no dedicated label for either. Warehouse equipment uses the legacy grouped names. Pallets, totes and loads need their own field annotations.

## Runtime and decision

Measured on NVIDIA GeForce RTX 4060 Laptop GPU, FP32, 960 px and batch 1. Wall time includes local file input and mapped detections after the first frame. Jetson and sensor-to-dashboard latency remain unmeasured.

| Profile | p50 | p95 |
|---|---:|---:|
| candidate | 41.5 ms | 57.1 ms |
| general | 57.3 ms | 241.9 ms |
| reference | 40.9 ms | 55.0 ms |
| warehouse | 42.0 ms | 59.5 ms |

The first field annotation pass should concentrate on dense and partially visible people, rider/vehicle separation, and forklift, pallet and pallet-jack distinctions. Capture the next evaluation clips from sessions outside these cohorts and review familiar categories alongside the new ones. Use locally encountered animals to decide which labels merit training.

Use the [label guide and reproduction commands](broad_benchmark_label_guide.md) and the [machine-readable results](../results/research/20261122/broad_detection_dev_v1.json) for the full class metrics, including false negatives, AP50, and lighting and reviewed human-count breakdowns.
