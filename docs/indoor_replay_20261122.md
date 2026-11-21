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
