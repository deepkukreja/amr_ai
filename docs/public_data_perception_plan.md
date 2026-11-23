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
