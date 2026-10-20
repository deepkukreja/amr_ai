# Dataset Guide — AMR AI

This document explains the dataset folders in plain language, how the configuration files work, what is actually tracked in Git, and how to prepare and verify local data. It is written for both future collaborators and the project author returning to the work after a break.

## 1. The most important rule

**The Git repository contains code and selected small configuration/manifest files, not the full datasets.** The local dataset collection was estimated at about 39 GB, much of it inside `datasets/external`. Image files, annotation files, source archives, model weights, and training outputs are intentionally excluded from normal commits.

Cloning the repository will therefore not magically download the images referenced in a YAML file or `.txt` manifest. You must obtain the required source dataset through an authorised source and place it in the expected local folder structure.

Read the related documents:

- [`../docs/data_inventory.md`](../docs/data_inventory.md) — inventory of dataset folders present on the machine used for the audit.
- [`../docs/experiments.md`](../docs/experiments.md) — why various dataset variants and training runs were created.
- [`../README.md`](../README.md) — project purpose, current results, environment, and Git workflow.
- [`../docs/learning_guide.md`](../docs/learning_guide.md) — detailed explanation of YAML, manifests, metrics, and verification.

## 2. Understand the three kinds of files

### Dataset images and labels

Images are typically files such as `.jpg` or `.png`. YOLO detection labels are commonly text files containing one object per line, usually a class ID and normalized bounding-box coordinates. Do not assume every `.txt` file is a label: this repository also uses `.txt` files as **image lists**.

### Dataset YAML configuration

A file such as `data.yaml` tells Ultralytics where the images are and which class names/IDs to use. Example:

```yaml
train: images/train
val: images/val
names:
  0: person
  1: rider
```

`train` and `val` are split locations. `names` maps numeric label IDs to class names. If the class IDs in annotation files disagree with `names`, training or evaluation can be wrong even if all paths exist.

### Image-list manifest (`.txt`)

An image-list manifest contains one image path per line. It is different from a YOLO label file. Selected manifests in this repository now use paths starting with `./`, resolved relative to the manifest's own folder by the installed Ultralytics `BaseDataset.get_img_files` logic.

Do not edit a manifest's ordering, drop lines, deduplicate entries, or change relative-path syntax without checking which experiment uses it. The list is part of the experimental input.

## 3. Main dataset groups

Folder names reflect the history of local experiments, not a guarantee of quality or readiness.

### Indian-road / manual annotation data

The following dataset configurations declare the project's two classes, `person` and `rider`:

- `indian_road_combined_v1`
- `indian_road_extension_v1`
- `indian_road_manual280_clean_v1`
- `indian_road_manual80_clean_v1`
- `indian_road_manual80_experiment_v1`
- `indian_road_pilot_v1`

Other nearby folders include:

- `indian_manual_train_v1` and `indian_manual_train_v2` — manually annotated training variants.
- `indian_manual_eval_v1` — manual evaluation examples.
- `indian_manual_independent_test_v1` and `indian_road_independent_test_v1` — folders originally named as independent test collections.
- `indian_road_manual80_experiment_v1` — an experiment that uses an explicit image-list manifest for training.
- `indian_road_manual280_clean_v1`, `indian_road_manual80_clean_v1`, and `indian_road_manual80_experiment_v1` — distinct versions; do not substitute one for another without recording the experiment change.
- `amr_dark_indoor_v1`, `amr_field_validation_v1`, and `indian_road_pilot_v1` — project-specific data or evaluation variants; check their README/configuration and labels before use.

**Important evaluation warning:** the Indian-road development benchmark has 166 images and 346 ground-truth objects (177 `person`, 169 `rider`). It has been used repeatedly for model and threshold selection. The dataset folder named `independent_test` has also been used during development decisions. Do not call it an untouched blind test set unless a separate audit demonstrates that it was never used for selection.

### SEW data and condition-specific evaluations

The `sew/` group contains scene sessions and related configurations. Other named folders include `sew_cloudy_person_only`, `sew_condition_test`, `sew_fullframe_person_test`, `sew_indoor_lowlight_v1`, `sew_person_adaptation`, `sew_rain_adaptation`, `sew_rain_full_adaptation`, `sew_rain_person_only`, `sew_sunny_person`, and `sew_holdout_v1`.

Several SEW configurations use the single class `person`. That is not the same class set as the Indian-road `person`/`rider` detector.

Important lists include:

- `sew/sew_test_images.txt` — 423 images used repeatedly for development/evaluation.
- `sew_eval_images.txt` — a broader image list (1,944 entries in the local audit).
- `sew_test100.txt` — a 100-entry list in the local audit.
- `sew/condition_manifests/{cloudy,indoor,overcast,rain,snow,sunny}.txt` — condition-specific image lists.
- `sew_condition_test/{cloudy,indoor,overcast,rain,snow,sunny}.txt` — smaller condition-specific evaluation lists.
- `sew_dev.yaml` and the associated `torwic_sew_v3_real/dev_images.txt` — the development-image configuration.

The 423-image set is not an untouched final test set because it was used repeatedly during model selection. Weather evaluation configs often set `train` and `val` to the same condition list so a shared Ultralytics YAML can be used for evaluation scripts. Those configs should be treated as **evaluation-only**, not training recipes.

### TORWIC + SEW adaptation experiments

Related folder families include:

- `torwic_detection_raw`
- `torwic_yolo_clean`
- `torwic_sew_adapt_v1`, `torwic_sew_adapt_v2`, `torwic_sew_adapt_v2_lowlight`, `torwic_sew_adapt_v3`
- `torwic_sew_v3_real`, `torwic_sew_adapt_v5_scale`, `torwic_sew_adapt_v6_real`, `torwic_sew_v7_lowlight`
- `torwic_sew_nod_v8`, `torwic_sew_nod_v10_replay`
- `torwic_rain_adapt_v1`, `torwic_rain_full_adapt_v2`
- `torwic_segmentation`, `torwic_semantic`
- `thud_people_yolo_v1`, `thud_real_adapt_v1`, `thud_real_people_yolo_v1`, `thud_torwic_sew_v4`

Several TORWIC/SEW configurations use five classes: `person`, `fork_truck`, `cart_pallet_jack`, `rack_shelf`, and `misc_dynamic_feature`. These are different from the road detector's two-class configuration. Check the configuration and label definitions before comparing outputs or reusing a checkpoint.

Known caveats:

- `torwic_sew_v7_lowlight/torwic_sew_v7_lowlight.yaml` currently points `val` and `test` at the same validation directory. This is not an independent test.
- The NOD V8/V9/V10 low-light training manifests have separate experiment histories and should not be mixed without recording exactly which list was used.
- The V7 generation script creates a candidate low-light dataset; it does not establish that training was performed or that the result improved.

### External sources and other data

`datasets/external/` contains external dataset-related materials and is the largest area of the local data collection. The local inventory also lists an Indian-road dataset source and other data folders. File names and folder structure alone do not prove licence, access terms, annotation quality, provenance, or readiness for training.

The local audit inspected the file names in the nested TIAND/TiHAN camera archive and found 10,919 JPEG images without obvious annotation/label files in that camera archive listing. That is not a full audit of every modality or accompanying document. Treat it as a camera-image source needing annotation/status/terms review, not as a ready-made detection dataset.

Two NOD manifests were converted locally during path work, but their configuration files were left for separate review and those manifests were not included in the recent 43-file commit. They are not part of the portable, tracked manifest set yet. Don't assume that a fresh clone contains them.

## 4. Class compatibility reference

| Dataset/configuration family | Declared classes found in reviewed configs | Typical intended use here | Caution |
|---|---|---|---|
| Indian-road configs | `person`, `rider` | Two-class road-scene detector | Match annotation convention and class IDs exactly. |
| SEW test/condition configs | `person` | Person detection and condition-level evaluation | Not a two-class rider dataset. |
| TORWIC/SEW multi-class configs | `person`, `fork_truck`, `cart_pallet_jack`, `rack_shelf`, `misc_dynamic_feature` | Multi-class dynamic/industrial-scene research | Not directly compatible with a two-class checkpoint without a documented class mapping. |
| THUD person configs | `person` in some configurations | Person-specific adaptation/evaluation | Verify splits and purpose; some configs repeat a directory for train and validation. |
| External NOD configs | `person` in reviewed YAML | External data adaptation or evaluation | Reviewed configs have split-key/path caveats and were not included in the latest portable commit. |
| Raw/segmentation/semantic folders | Varies or no detection names at the checked depth | Source/reference data or other task | Do not assume YOLO detection labels are available. |

## 5. What is currently tracked versus local-only?

The `.gitignore` rule `datasets/*` excludes dataset subfolders by default while allowing `datasets/README.md`. This helps prevent accidental uploads of large or sensitive data. A small set of files was deliberately added using `git add --sparse -f` after review.

The commits reported by the local Git CLI are:

- `960eb03` — `Document local datasets and data management`
- `8e01ee0` — `Add portable Indian road dataset configs`
- `7f93561` — `Make dataset configs and manifests portable`

The latest portability commit added 43 files: 22 YAML configurations and 21 image-list `.txt` manifests. It did **not** add images, YOLO label files, raw data archives, or model weights.

The 21 published image-list manifests were checked locally after path conversion. The 22 selected YAML configurations and their associated manifests were then validated: 34,028 references across configuration uses resolved without path errors. Repeating the check on another machine after retrieving its data is still required. A successful path check does not prove the annotations are correct or the data may be shared.

The local manifest conversion covered 23 lists in total, but the two external NOD manifests were omitted from the 43-file commit while their configurations remain under review. Keep this distinction clear when onboarding a collaborator.

## 6. Getting the data onto a new machine

1. Clone the repository.
2. Read `../docs/data_inventory.md` and identify the one dataset required for the experiment you plan to run.
3. Confirm its source, licence, permission to use/share, class definitions, and split rules.
4. Retrieve the dataset from that authorised source or a separately permissioned storage location.
5. Recreate the expected folder structure. Don't rename folders without updating the associated configuration and verifying all references.
6. Run the path checks described below.
7. Start with a small smoke test before a full training run.

Do not upload tens of gigabytes of raw data to GitHub just to make collaborators' setup easier. Prefer a controlled data-storage location (for example, an access-controlled institutional or team drive) with clear permissions, version/date, archive name, and checksum. Do not put private or unauthorised sharing links in a public README.

## 7. Portable path rules

Ultralytics resolves `train`, `val`, and `test` through the dataset YAML's root `path`. If `path` is omitted, the installed implementation uses the directory containing that YAML. A `.txt` manifest then contains one image path per line. In the selected manifests, entries starting with `./` are resolved relative to the manifest's directory.

Example `data.yaml`:

```yaml
train: images/train
val: images/val
names:
  0: person
  1: rider
```

Example line inside `train.txt`:

```text
./images/train/example.jpg
```

Don't replace manifest entries with bare filenames or paths relative to your terminal's current working directory. Use the expected relative convention and validate the result against the installed Ultralytics version.

## 8. Local validation checklist

Activate the same environment used for training and inspect versions:

```bash
./.venv/bin/python -c "import torch, ultralytics, yaml; print('Torch', torch.__version__); print('CUDA available', torch.cuda.is_available()); print('Ultralytics', ultralytics.__version__); print('PyYAML', yaml.__version__)"
```

For a dataset/configuration, verify:

- YAML can be parsed (calibration files may use OpenCV-specific YAML syntax and are not necessarily Ultralytics dataset YAMLs).
- `names` and numeric class IDs match the annotation files.
- `train`, `val`, `test`, and any text manifests exist.
- every image reference points to a file.
- no machine-specific `/home/...` path is left in a configuration intended for collaborators.
- whether train/validation/test point to the same files/directories is understood and intentional.
- the actual use is training, validation, or evaluation-only.

Git checks:

```bash
git status --short --branch
git diff --cached --name-only
git diff --cached --stat
git diff --cached --check
git check-ignore -v datasets/some_folder/data.yaml
```

`git diff` can appear empty for ignored local data files. Use `git check-ignore` to see why and `git ls-files` to confirm what is actually tracked. If a reviewed, small YAML or image-list manifest needs to be added from a sparse-checkout path, explicitly stage only that file using `git add --sparse -f path/to/file`. Never run `git add -f datasets/` or force-add an entire dataset tree.

### Important distinction

A YOLO **annotation label** `.txt` file is not the same as an **image-list manifest** `.txt` file. Label files are associated with image annotations and should normally remain outside Git unless the project has explicitly decided otherwise. Image-list manifests can be tracked if they are small, contain no restricted information, and are part of a reproducible experiment.

## 9. When is a dataset “ready to use”?

Do not label a dataset ready to train just because it has `images/`, `labels/`, or a YAML file. At minimum, verify:

1. the dataset source and use/sharing terms are documented;
2. the images and annotations exist and are readable;
3. the annotation format and class IDs are correct;
4. the class names in YAML match the labels;
5. the intended train/validation/test split is documented and has no accidental leakage;
6. image-list manifests resolve on the target machine;
7. a small model/data loader smoke test succeeds;
8. the evaluation set's previous use is recorded.

If a check is incomplete, call the data **under review**. That is better than guessing based on the directory name.

## 10. Before you start an experiment

Write down the exact dataset YAML and its Git commit, checkpoint used to initialise training, image size, batch size, seed, number of epochs, optimiser/learning rate, and any augmentation changes. After training, save the results CSV, best checkpoint, class-level metrics, threshold selection method, and error examples outside Git unless they have been intentionally curated for publication.

For this project, manually review false positives and false negatives per class, not only overall mAP. A model may improve rider detection while hurting person detection, or appear better at one confidence threshold while missing many distant people. Record both gains and regressions honestly.
