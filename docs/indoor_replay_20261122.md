# Indoor replay pilot, 22 November 2026

The crop experiment recovered the three foreground people in one image, but missed the upper-body probe and reduced road performance. This experiment adds real indoor images with people closer to the camera to the two-class road training mix.

The source is the **store/dynamic** recording in [THUD++ by Li, Li and Zeng](https://doi.org/10.5281/zenodo.18459791). The local copy already contained person-only YOLO labels. The preparation script matches the decoded RGB pixels to the archive and compares each selected box with the publisher's `People` rectangles, clipped to the image boundary. The output uses the original PNG bytes and maps those rectangles to road class 0 (`person`). The road set remains the source of class 1 (`rider`).

| Role | Source | Images |
|---|---|---:|
| Road training | Frozen Manual-280 training split | 221 |
| Indoor training | THUD++ store Capture 1 | 19 |
| Road checkpoint selection | Frozen Manual-280 validation split | 59 |
| Indoor evaluation | THUD++ store Capture 2 | 16 |

Indoor sampling prioritizes frames with a large person box, requires at least one box covering 2% of the image, and leaves at least eight frame indices between selected frames. Eight training boxes and ten evaluation boxes cover at least 10% of their images. Both captures are in the same store and contain related frames. Metrics describe a small, person-positive evaluation cohort and give limited evidence about other stores, empty scenes, road footage, or webcams.

Training starts from the same pinned road candidate used in the November 17 experiment. The continuation matches the earlier control's 12-epoch schedule, seed, optimizer, and frozen layers. Adding frames changes exposure and batch counts. Best checkpoint selection uses road validation only. Indoor metrics use person predictions at confidence 0.25 and one-to-one IoU 0.50 matching. The road mAP comparison runs in FP32 at image size 960, batch 2, and NMS IoU 0.70.

To reproduce from the repository root with the existing GPU environment, dataset archive and earlier local audit:

```bash
.venv/bin/python -m scripts.data.prepare_indoor_replay \
  --output datasets/indian_manual280_indoor_replay_20261122
.venv/bin/python -m scripts.training.run_indoor_replay smoke \
  --dataset datasets/indian_manual280_indoor_replay_20261122 --output results/local/indoor_replay_20261122
.venv/bin/python -m scripts.training.run_indoor_replay train \
  --dataset datasets/indian_manual280_indoor_replay_20261122 --output results/local/indoor_replay_20261122
.venv/bin/python -m scripts.training.run_indoor_replay evaluate \
  --dataset datasets/indian_manual280_indoor_replay_20261122 --output results/local/indoor_replay_20261122
```

The evaluation also reads the recorded checkpoint from the previous unchanged-data control. Images, source annotations, and model weights are local. The experiment uses no user-supplied photographs or webcam footage.

## Results

| Checkpoint | Indoor TP / FP / FN | Indoor F1 | Large boxes matched | Road mAP50–95 |
|---|---:|---:|---:|---:|
| Pretrained COCO person | 21 / 1 / 1 | 95.45% | 9 / 10 | — |
| Road reference | 3 / 8 / 19 | 18.18% | 0 / 10 | 20.36% |
| Starting road candidate | 9 / 0 / 13 | 58.06% | 0 / 10 | 21.90% |
| Prior unchanged-data continuation | 10 / 0 / 12 | 62.50% | 0 / 10 | 20.26% |
| Indoor replay continuation | 16 / 2 / 6 | 80.00% | 5 / 10 | 20.98% |

The replay checkpoint changed indoor true positives by +7, false positives by +2, and road mAP50–95 by -0.92 percentage points relative to its starting checkpoint.
The indoor metrics count the person class only. The road metric evaluates person and rider.
The saved replay checkpoint matched 3 of 3 targets in the earlier foreground photo and 0 of 1 target in the earlier upper-body crop at the same 0.19 person threshold. These local probes contain selected foreground targets.

The road reference and pretrained demo model remain selected. The two captures come from the same store, so the result supports further evaluation across different sites and camera viewpoints before promotion.

[Machine-readable results](../results/research/20261122/indoor_replay.json)
