# BDD100K rider and two-wheeler research pilot

This is a four-class YOLO26s experiment for pedestrian, rider, bicycle, and motorcycle. The images are real US dashcam frames from BDD100K's original 2018 detection release. Use the pilot for research under the dataset terms. Production use needs the applicable data rights. The existing Indian-road data remains a separate development comparison. [BDD100K paper](https://arxiv.org/abs/1805.04687) · [Original data license](https://github.com/bdd100k/bdd100k/blob/master/doc/source/license.rst)

## Data and checks

The original Berkeley release was available but slow to transfer. The local pilot reads a pinned redistribution with the original BDD100K image names and raw JSON schema. Every selected image decoded as a 1280 × 720 JPEG. Each selected index annotation was checked against the pinned mirror's raw per-image JSON. Keep the original Regents license notice with those local files. [Berkeley archive index](http://128.32.162.150/bdd100k/) · [Pinned redistribution and provenance](https://huggingface.co/datasets/dronefreak/BDD100K/tree/fea86dd9b24ea425d66b70e0fcb6ae0023b5dca5)

| Split | Images | Pedestrian boxes | Rider boxes | Bicycle boxes | Motorcycle boxes |
|---|---:|---:|---:|---:|---:|
| Training | 272 | 1,405 | 206 | 150 | 104 |
| Tuning | 56 | 146 | 37 | 28 | 17 |
| Reserved evaluation | 84 | 376 | 60 | 45 | 34 |

Training and tuning use the publisher's train split. Reserved evaluation uses the publisher's labeled validation split. Selection keeps just one frame from each BDD image-ID prefix group across all three partitions and rejects exact image duplicates. That grouping is a conservative identifier-level check. The source lacks route grouping here. Training includes 53 night frames and 16 dawn/dusk frames. The four sample strata are image-level rider/motorcycle co-occurrence, rider/bicycle co-occurrence, at least four people including riders, and frames with no annotations in the four target categories. Co-occurrence aids sampling. It supplies no individual rider-to-vehicle association.

Training keeps all four object categories in each selected image. Benchmark reports also use a combined human label for comparing the general COCO person model with the specialist checkpoints. The benchmark uses a curated object mix that differs from normal traffic. These vintage labels differ from the revised 2020 detection labels, and results are specific to this sample.

## Reproduce

Run from the WSL checkout. The raw media, annotations, license notice, model weights, and per-image evidence stay in local ignored directories.

```bash
pilot=datasets/external/bdd100k_rider_pilot
base=https://huggingface.co/datasets/dronefreak/BDD100K/resolve/fea86dd9b24ea425d66b70e0fcb6ae0023b5dca5/data/images_100k/100k
mkdir -p "$pilot"
curl -fL --retry 2 -o "$pilot/LICENSE.txt" http://128.32.162.150/bdd100k/LICENSE.txt
curl -fL --retry 2 -o "$pilot/train_shard_000.jsonl" "$base/train/shard_000/metadata.jsonl?download=true"
curl -fL --retry 2 -o "$pilot/train_shard_004.jsonl" "$base/train/shard_004/metadata.jsonl?download=true"
curl -fL --retry 2 -o "$pilot/val_shard_000.jsonl" "$base/val/shard_000/metadata.jsonl?download=true"
.venv/bin/python -m scripts.data.freeze_bdd100k_pilot --output "$pilot"
run=runs/experiments/bdd100k_rider_pilot_v1
out=results/local/bdd100k_rider_pilot_v1
.venv/bin/python -m scripts.training.run_bdd100k_pilot --dataset "$pilot" --output "$run"
.venv/bin/python -m scripts.evaluation.evaluate_broad_benchmark --manifest "$pilot/manifest.json" \
  --rider-vehicle-checkpoint "$run/train/weights/best.pt" --output "$out/evaluation.json"
.venv/bin/python -m scripts.evaluation.evaluate_broad_benchmark --manifest datasets/broad_detection_dev_v1/manifest.json \
  --rider-vehicle-checkpoint "$run/train/weights/best.pt" --output "$out/indian_road_transfer.json"
```

| Local evidence | SHA-256 |
|---|---|
| License notice | `64241b062f669b0b0ef8c4abe37d72f8ca00515f543a92098594ab1f04332533` |
| Train metadata, shard 000 | `4e12f3eb4ea3ddf69f07803144232e7d1f590682930d892b12ae44711e6eb414` |
| Train metadata, shard 004 | `105a0f96d64196ee6bafe3fb0684a0026af43aed9e2c6e9a84ef656cbb5c7ca8` |
| Validation metadata, shard 000 | `0d9319400ead8c5c0212b5e1d2b97072e47a64a4fc548131c2f1d63775f0ef2a` |
| 412-frame dataset manifest | `9601be0a314e8f2892ba57d8a8dffb593c55d436bbf73f6c0ba5e08ca329f058` |
| 84-frame evaluation manifest | `5c00277a8c1a64e1df70471cc9f989c7449d95be55fbe3e8a025c749f7be4a69` |

## Measured result

Fine-tuned the pinned COCO YOLO26s checkpoint for ten epochs at 960 px on the RTX 4060 laptop GPU. Two of four class-head rows transferred by matching class name. The best tuning mAP50–95 was 22.1% at epoch 10. All benchmark figures below use the same 960 px input, 0.25 reporting confidence, 0.50 matching IoU, and 0.70 NMS IoU.

| BDD100K evaluation class | Comparison model | Comparison matches | Pilot matches | Comparison / pilot unmatched predictions |
|---|---|---:|---:|---:|
| Rider | Existing road candidate | 19 / 60 | 30 / 60 | 14 / 23 |
| Pedestrian | Existing road candidate | 166 / 376 | 228 / 376 | 62 / 155 |
| Bicycle | General COCO | 13 / 45 | 17 / 45 | 7 / 17 |
| Motorcycle | General COCO | 10 / 34 | 6 / 34 | 5 / 8 |
| Human including riders | General COCO | 233 / 436 | 271 / 436 | 50 / 165 |

Rider AP50 improved from 31.0% to 45.4% against the existing road candidate on this reserved sample. The stronger recall came with more extra boxes. The general model retained stronger combined-human AP50 (67.1% versus 64.5%). Bicycle recall rose with more extra boxes and motorcycle recall fell. The pilot is a research checkpoint.

Local best checkpoint SHA-256 `50243344680a1193d5a99f80a9fad23256f31ea8b904378a1fc01153f04c9b6b`. Initial weights SHA-256 `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`.

### Transfer to the existing Indian-road benchmark

| Metric at 960 px | Existing road candidate | BDD100K pilot |
|---|---:|---:|
| Rider matches | 23 / 41 | 10 / 41 |
| Unmatched rider predictions | 14 | 28 |
| Human matches | 44 / 88 | 29 / 88 |
| Unmatched human predictions | 29 | 64 |
| Dense-scene rider matches | 9 / 20 | 4 / 20 |

The 48 Indian-road frames are development evidence from a different data source. The audited image hashes share no exact duplicates with the 412 BDD100K frames. The rider annotations also have different conventions. Treat this comparison as performance on the project's existing road labels. Use the established road model for Indian-road work. The next training increment needs real Indian examples with complete pedestrian, rider, bicycle and motorcycle boxes, plus targeted hard negatives for the extra detections observed here.
