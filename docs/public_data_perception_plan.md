# Public data plan for AMR perception

Start development with public camera data and the reviewed clips already on disk. Use public robot recordings for camera viewpoint and occlusion, road clips for riders, and specialist warehouse labels for equipment. Keep the public evaluation sets out of training. A future physical robot and its operating site will have a separate validation phase.

| Source | Use now | Boundary |
|---|---|---|
| TUM LOCO v1 | Train and evaluate detection of forklifts, pallet trucks, pallets, small load carriers, and stillages. | Equipment catalogs and shelving extend beyond the five LOCO classes. People are outside its annotation scope. The publisher lists the dataset under CC0. [Publisher and license](https://github.com/tum-fml/loco) |
| Existing reviewed Indian-road clips | Keep the clip-separated person and rider data as the road development reference and training seed. Prioritize dense traffic, occluded people, passengers, and separate bicycle and motorcycle boxes when expanding this set. | The other traffic classes need review before any joint training. IIIT Hyderabad's IDD Detection and IDD-2WD are the next choices for independent Indian-road data, subject to the publisher's account access and terms. [IDD releases](https://indiadrivingdata.org/) |
| Existing THUD++ sequences | Use the available real robot store sequences for indoor person development and pose/crop investigation. Keep timestamps and capture groups together. | Store captures measure one environment. THUD++ is available for noncommercial research. JRDB is an option for independent crowd and occlusion research if account access is available. [THUD++ paper](https://www.nature.com/articles/s41597-026-07993-8) · [THUD terms](https://jackyzengl.github.io/THUD-Robotic-Dataset.github.io/assets/THUD_Robotic_Dataset_TOS.pdf) · [JRDB license](https://jrdb.erc.monash.edu/dataset/) |
| Open Images V5 official test split | Use a small public regression set for people, bicycles, motorcycles, cars, vans, buses, trucks, dogs, cats, cattle, horses, sheep, birds and goats. | Still images give initial class evidence. The sample has a deliberately balanced anchor count, so it measures neither roadside prevalence nor robot performance. Google licenses its annotations under CC BY 4.0. The manifest records the listed CC BY 2.0 photo license, author and original page for every image. [Open Images V5](https://storage.googleapis.com/openimages/web/factsfigures_v5.html) · [Evaluation protocol](https://storage.googleapis.com/openimages/web/evaluation.html) |

## What to train first

1. **Warehouse equipment.** Use LOCO subsets 2, 3 and 5 for training and tune with whole recording groups withheld from those subsets. Freeze subsets 1 and 4 as site-level evaluation groups before any modeling. Compare per-class recall and false positives per frame with the existing warehouse model wherever its class mapping supports a comparison. Report unsupported classes and site-specific counts. Subset 1 has only five forklift boxes, while subset 4 has no small-load-carrier boxes.
2. **People and two-wheelers.** Keep general pretrained person detection as a control. Run a separate crowded/occluded people and rider experiment with reviewed data. Use temporary sim scenes or controlled crops to test camera height, truncation, distance and lighting. Keep synthetic and real-camera results separate and report when a comparison uses two captures from the same site.
3. **Vehicles and animals.** Measure the broad pretrained checkpoint on the frozen public sample. Keep cars, common two-wheelers and common animals in the regression set while training the specialist gaps. If a missing class proves useful, take examples from a fresh training split or another source with verified boxes and attribution. Use hard negatives along with positive examples.
4. **Unified candidate.** Compare a compact jointly trained detector with the general and specialist checkpoints once the joint labels are reviewed. Preserve a box around the rider separately from the vehicle. Use complete labels for all target classes per image or a verified partial-label training method. A missing source annotation for a person, vehicle or warehouse item is an unknown. Check per-class regressions before changing demo or robot defaults.

## Reproduce the local acquisition

Run these commands from the WSL repository root. Downloaded data, per-image annotations, and model outputs stay in ignored local directories.

```bash
mkdir -p datasets/external/loco_v1
curl -fL --retry 2 -o datasets/external/loco_v1/dataset.zip https://go.mytum.de/239870
curl -fL --retry 2 -o datasets/external/loco_v1/loco-all-v1.json \
  https://raw.githubusercontent.com/tum-fml/loco/b460ab8c37f09162f0613c7546948c496e1b1628/rgb/loco-all-v1.json
unzip -tq datasets/external/loco_v1/dataset.zip
sha256sum datasets/external/loco_v1/{dataset.zip,loco-all-v1.json}

metadata=datasets/external/open_images_test_v5_pilot/metadata
mkdir -p "$metadata"
curl -fL --retry 2 -o "$metadata/class-descriptions-boxable.csv" \
  https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv
curl -fL --retry 2 -o "$metadata/test-annotations-bbox.csv" \
  https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv
curl -fL --retry 2 -o "$metadata/test-annotations-human-imagelabels-boxable.csv" \
  https://storage.googleapis.com/openimages/v5/test-annotations-human-imagelabels-boxable.csv
curl -fL --retry 2 -o "$metadata/test-images-with-rotation.csv" \
  https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv
.venv/bin/python -m scripts.data.freeze_broad_benchmark --open-images-test \
  --output datasets/external/open_images_test_v5_pilot
.venv/bin/python -m scripts.evaluation.evaluate_broad_benchmark \
  --manifest datasets/external/open_images_test_v5_pilot/manifest.json \
  --output results/local/open_images_test_v5_pilot/evaluation.json
```

## LOCO acquisition audit

The 769 MB image archive passed the ZIP checksum test. The pinned publisher COCO file links 5,097 unique image paths to 151,428 boxes. Every image path resolves in the archive and every box has positive dimensions and fits the stated image size. Normalize the leading slash in the COCO `path` field when joining it to archive members. The basename in `file_name` repeats across capture folders. The ZIP contains another 496 JPEGs without an image row in the pinned JSON. Keep them out of automatic training and evaluation until reviewed for intentional negatives or missing labels.

| LOCO group | Linked images | Forklift | Pallet truck | Pallet | Small load carrier | Stillage |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1,075 | 5 | 594 | 13,275 | 8,911 | 1,978 |
| 2 | 1,070 | 138 | 231 | 34,016 | 4,299 | 2,997 |
| 3 | 873 | 188 | 308 | 23,448 | 2,281 | 402 |
| 4 | 1,202 | 119 | 559 | 39,022 | 0 | 29 |
| 5 | 877 | 148 | 1,135 | 10,684 | 6,660 | 1 |

Archive SHA-256 `f3629d989071b824edc63e000f45ccc5826cfbb201a338f16ed87ab3e10bcdcf`.

Pinned COCO annotations SHA-256 `5a794d2e61f308150039e3e1953bc57e8fad208bb3e2b9117245230ec1c22054`.

## Open Images public evaluation

The acquired set contains 112 images, eight positive anchors per category. An image can contain additional classes and instances. The previous 70-image validation sample reproduces exactly, and this test sample shares no image IDs with it and no image SHA-256 hashes with any of the 153 development frames. The new manifest SHA-256 is `469f1e4aec1ee5116696d5dc4682e9525437439fc154b13051ef9ee5a967e6f7`.

The four existing checkpoints ran on the same sample. The general pretrained COCO checkpoint matched **61 of 74 people** at 0.25 confidence and 0.50 IoU. The road candidate matched **36 of 74**, the road reference **21 of 74**, and warehouse V6 **10 of 74**. These road and warehouse checkpoints may perform differently in their intended domains. The sample includes 12 van and 20 goat boxes, two categories outside the COCO checkpoint's vocabulary.

| General COCO class | Matched / annotated | Unmatched predictions on scored images |
|---|---:|---:|
| Bicycle | 10 / 12 | 5 |
| Motorcycle | 11 / 16 | 3 |
| Car | 44 / 58 | 35 |
| Bus | 11 / 12 | 6 |
| Truck | 10 / 14 | 5 |
| Dog | 9 / 9 | 7 |
| Cat | 8 / 9 | 2 |
| Cattle | 9 / 9 | 0 |
| Sheep | 37 / 41 | 11 |
| Horse | 17 / 18 | 0 |
| Bird | 9 / 10 | 2 |
| Human | 61 / 74 | 53 |

An unmatched prediction counts only in the image/class pairs covered by the benchmark. These counts apply to a small, class-targeted public sample. Keep the current general checkpoint as the person/traffic/animal reference. Prioritize LOCO equipment training, then dense people and riders. Review the car and motorcycle misses before picking a separate training source for those classes.

## LOCO equipment training pilot

The first warehouse run fine-tunes YOLO26s for six epochs at 640 px on the laptop RTX 4060. It tests the conversion, grouped splits, training, and evaluation end to end. The pilot has five equipment classes.

| Role | Images | Source |
|---|---:|---|
| Train | 603 | Up to 68 images per capture group, sampled across sorted archive paths in sites 2, 3 and 5 |
| Tune | 219 | All linked images in capture `subset-3/2019-12-17_9` |
| Site 1 evaluation | 1,075 | All linked images for LOCO site 1 |
| Site 4 evaluation | 1,202 | All linked images for LOCO site 4 |

The training split contains 85 forklift boxes across 68 images, 395 pallet-truck boxes and 14,721 pallet boxes. The prepared dataset passed file-hash and exact cross-split duplicate checks for all 3,099 image-label pairs. Its manifest SHA-256 is `528e3aeef0c1157446795c960887d2062a4eaa9c0ec74fc4ba92d192004cf2c9`.

```bash
python3 -m scripts.data.prepare_loco_pilot \
  --source datasets/external/loco_v1 --output datasets/loco_equipment_pilot_v1
.venv/bin/python -m scripts.training.run_loco_pilot \
  --dataset datasets/loco_equipment_pilot_v1 --weights weights/yolo26s.pt \
  --output runs/experiments/loco_equipment_pilot_v1
```

### Held-out warehouse results

| Equipment | Site 1 AP50 (boxes) | Site 4 AP50 (boxes) |
|---|---:|---:|
| Forklift | 0.4% (5) | 7.1% (119) |
| Pallet truck | 20.3% (594) | 18.1% (559) |
| Pallet | 43.3% (13,275) | 35.3% (39,022) |
| Small load carrier | 26.5% (8,911) | No examples (0) |
| Stillage | 26.5% (1,978) | 0.0% (29) |

The two sites scored 9.5% and 5.5% mAP50-95, respectively. Site 4 contributes four classes to its mean. Best tuning mAP50-95 was 5.7%.

The result establishes a reproducible training baseline. Pallets led on both sites, and forklifts and stillages transferred poorly. Keep the candidate for research. For the next candidate, increase diverse forklift and pallet-truck support in the training groups and choose image size and training duration on the tuning split. Treat these two warehouse scores as historical baselines and reserve another site for a later independent check if they start informing model choices.

Local checkpoint SHA-256 `0b1d2236b0cef1006ffd2e65838a865ca455cac551ed629eb216d60f4a41ae7e`. Starting COCO weights SHA-256 `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`.

## LOCO follow-up with more equipment examples

The follow-up keeps every frame from the first training set and adds the remaining publisher-labeled forklift and pallet-truck frames from the same training recordings. It uses the same COCO initialization, seed, six epochs, 640 px input and optimizer settings. There are more optimizer updates per epoch because the training set is larger. This comparison measures the expanded recipe as a whole. The extra frames add examples within known sites.

| Training support | Original | Expanded |
|---|---:|---:|
| Training images | 603 | 1,297 |
| Images with forklifts | 68 | 300 |
| Forklift boxes | 85 | 416 |
| Images with pallet trucks | 194 | 764 |
| Pallet-truck boxes | 395 | 1,569 |

The 219 tuning frames and both site evaluation sets match the first pilot by image hash and label hash. Verified all 3,793 expanded image-label pairs and found no exact image duplicates across splits. Best tuning mAP50-95 rose from 5.7% to 11.3%.

| Metric | Site 1 original | Site 1 expanded | Site 4 original | Site 4 expanded |
|---|---:|---:|---:|---:|
| Forklift AP50 | 0.4% | 0.6% | 7.1% | 16.1% |
| Pallet truck AP50 | 20.3% | 28.0% | 18.1% | 24.9% |
| Pallet AP50 | 43.3% | 51.5% | 35.3% | 43.7% |
| Small load carrier AP50 | 26.5% | 33.7% | No examples | No examples |
| Stillage AP50 | 26.5% | 34.3% | <0.1% | 1.9% |
| Equipment mAP50-95 | 9.5% | 12.7% | 5.5% | 8.4% |

Site 1 has only five forklift boxes. Site 4 has 119. The expanded detector is the stronger warehouse research candidate. Forklift and stillage generalization are still weak, so retain the deployment and demo defaults. A future model should be checked on a warehouse outside LOCO once the repeated site results start guiding model selection.

```bash
python3 -m scripts.data.prepare_loco_pilot --include-equipment-positives \
  --source datasets/external/loco_v1 --output datasets/loco_equipment_expanded_v2
.venv/bin/python -m scripts.training.run_loco_pilot \
  --dataset datasets/loco_equipment_expanded_v2 --weights weights/yolo26s.pt \
  --output runs/experiments/loco_equipment_expanded_v2
```

Manifest SHA-256 `e9dc46edb9f347617c1843ce94030aa5d36ff0a4a1542d97c3bbbab6c3b2dd9f`. Local best checkpoint SHA-256 `6fec5149a5e2d86b0d485a5ce67bd2786fe89f9cd7c2e9a538a6b0ccea2b1994`.

## Crowded-road resolution comparison

The original 960 px results reproduced for all four checkpoints and all four development cohorts. The 1280 px arm uses the same frame labels, checkpoint hashes, 0.25 reporting confidence, 0.50 match IoU, and 0.70 NMS IoU. Both arms use the RTX 4060 laptop GPU in FP32 at batch 1.

| Road candidate metric | 960 px | 1280 px |
|---|---:|---:|
| Human matches in scenes with at least four reviewed people | 25 / 52 | 28 / 52 |
| Unmatched human predictions in those scenes | 20 | 18 |
| Rider matches in those scenes | 9 / 20 | 8 / 20 |
| Human matches across all 48 road images | 44 / 88 | 50 / 88 |
| Unmatched human predictions across all 48 road images | 29 | 37 |
| Rider matches across all 48 road images | 23 / 41 | 23 / 41 |
| Mixed-benchmark file-to-detections p50 / p95 | 34.2 / 46.8 ms | 36.5 / 49.7 ms |

Higher resolution helped people in the dense slice. It did not recover more riders at the fixed threshold, and the whole-road slice added eight unmatched human predictions. Keep the default at 960 px and use 1280 px as a development comparison. The timing includes laptop file input and model inference and applies to this run. The reviewed real-road labels cover people and riders. They cannot score bikes and motorcycles as separate vehicles.

```bash
for size in 960 1280; do
  .venv/bin/python -m scripts.evaluation.evaluate_broad_benchmark --imgsz "$size" \
    --manifest datasets/broad_detection_dev_v1/manifest.json \
    --output "results/local/broad_resolution_${size}_20261123/evaluation.json"
done
```

## UrbanSyn rider and two-wheeler diagnostic

UrbanSyn adds publisher boxes for riders, motorcycles, bicycles and pedestrians as separate objects. Its images are computer-generated. The bounded sample uses 36 scenes. Twelve contain a rider and motorcycle, twelve contain a rider and bicycle, and twelve contain at least four pedestrians. Co-occurrence is a selection rule. The benchmark does not assign individual riders to individual vehicles. The publisher applies size and distance cutoffs. The sampler excludes frames with malformed target boxes, cross-checks each chosen annotation against the pinned publisher revision, and scores the labels with supplied boxes. Unboxed instances can still be present in a positive image. [UrbanSyn documentation and terms](https://urbansyn.org/#terms-of-use) · [Pinned publisher archive](https://huggingface.co/datasets/UrbanSyn/UrbanSyn/tree/8599498aceee9072f1e067a4a4efed7f20b11809)

| Diagnostic | Model | Matches |
|---|---|---:|
| People, including riders | General COCO | 212 / 543 |
| Riders | Road candidate | 10 / 52 |
| Bicycles | General COCO | 13 / 39 |
| Motorcycles | General COCO | 6 / 17 |

Both traffic checkpoints matched 10 of 52 riders. The low coverage supports getting more two-wheeler training data with distinct human and vehicle labels. This diagnostic is a synthetic baseline and stays outside training. Evaluate a future model on real road recordings with all target objects labeled before drawing conclusions about real-road performance.

```bash
sample=datasets/external/urbansyn_rider_pilot
mkdir -p "$sample"
curl -fL --retry 2 -o "$sample/bbox2d.zip" https://pegaso.citic.udc.es/urbansyn/zip/bbox2d.zip
.venv/bin/python -m scripts.data.freeze_urbansyn_pilot --archive "$sample/bbox2d.zip" --output "$sample"
.venv/bin/python -m scripts.evaluation.evaluate_broad_benchmark \
  --manifest "$sample/manifest.json" --output results/local/urbansyn_rider_pilot/evaluation.json
```

The label archive SHA-256 is `1127f4b565a7d7cb4aab1d236b144445c9574d2605f95cc879ba0a0f7b414a0f`. The local sample manifest SHA-256 is `0759866e4976ba1bc7eb968b92d39e89e657a79b520e34d7ba9310d5b3bd1ac8`. The source audit excluded 24 frames with invalid target geometry among 7,539 records.

The real-road follow-up uses the [BDD100K research pilot](bdd100k_rider_pilot.md) for separate pedestrian, rider, bicycle and motorcycle boxes.
