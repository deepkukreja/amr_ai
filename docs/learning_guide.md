# Learning Guide: How This AMR Perception Repository Works

This guide is meant to be understandable after a break from the project. It explains the terms, the workflow already used, how to verify it, and how to avoid repeating old mistakes.

## 1. The goal in one paragraph

The project explores camera-based perception for an autonomous mobile robot. The current road-scene detection task identifies `person` and `rider`. Experiments use different sources—Indian-road examples, SEW scenes, TORWIC-derived material, and THUD-related data—to study how a detector behaves across different environments and lighting. The project also includes annotation utilities and low-light dataset-preparation tools. Detection is only one part of autonomy; it does not by itself provide safe navigation.

## 2. The basic machine-learning loop

1. **Collect/obtain data:** images must come from a source that can legally and appropriately be used.
2. **Annotate:** boxes and class IDs describe the objects in each image.
3. **Split the data:** training data fits model weights; validation data helps choose settings; test data should be held back for a final check.
4. **Configure the dataset:** a YAML file tells the training/evaluation code where the images are and what the classes mean.
5. **Train:** the model updates weights using training examples.
6. **Validate:** metrics show how it performs on the selected validation set.
7. **Inspect errors:** look at false positives, missed people/riders, small/distant objects, occlusion, motion blur, rain, night scenes, and confusing backgrounds.
8. **Record the experiment:** preserve the code commit, data version, configuration, training arguments, checkpoint, and results.
9. **Test honestly:** do not call a set independent if it influenced any model, threshold, or hyperparameter decisions.

A new script, an image collection, or a directory named `v7` is not proof that training was run or improved.

## 3. Detection vocabulary

### Bounding box

A rectangle around an object. For YOLO detection labels, the common format is:

```text
class_id x_center y_center width height
```

The coordinates are typically normalised between 0 and 1 relative to the image dimensions. Confirm the annotation specification for the source before assuming this format.

### Class ID and class name

A YAML file can define:

```yaml
names:
  0: person
  1: rider
```

This means label ID `0` is `person` and label ID `1` is `rider`. If annotations were created with a different class ordering, the config must be corrected or labels converted using a documented, verified process.

### Precision, recall, F1

- **Precision = TP / (TP + FP)**. Of the detections reported by the model, how many were correct?
- **Recall = TP / (TP + FN)**. Of the ground-truth objects, how many did the model find?
- **F1 = 2 × precision × recall / (precision + recall)**. A balance of precision and recall at a particular operating point.

Here, TP means true positive, FP false positive, and FN false negative, under a specified class, IoU rule, and confidence threshold.

### IoU

Intersection over Union measures how much a predicted box overlaps a ground-truth box. An IoU threshold of 0.50 means a detection usually needs at least 50% overlap to count as a match, subject to the evaluator's matching rules.

### mAP

Mean average precision summarises precision-recall behaviour across confidence thresholds and classes. `mAP@0.50` uses IoU 0.50. `mAP@0.50:0.95` averages over IoU thresholds from 0.50 to 0.95. A larger value is generally better under the same evaluation protocol, but metrics from different datasets, class maps, or split protocols should not be compared as if they were identical experiments.

## 4. YAML configuration versus image-list file

### YAML file

A dataset YAML might say:

```yaml
train: images/train
val: images/val
names:
  0: person
  1: rider
```

`train` and `val` select the images used by each split. `names` defines the class mapping. `path` is optional; in the installed Ultralytics implementation it sets the dataset root. If missing, the YAML file's directory becomes the default root.

### Image-list manifest

A `.txt` manifest used as a split can contain one image path per line. It is not an annotation label. Example:

```text
./images/train/frame_0001.jpg
./images/train/frame_0002.jpg
```

For this project's installed Ultralytics version, `BaseDataset.get_img_files` replaces a leading `./` with the manifest file's parent directory. Other relative lines are not automatically interpreted the same way by that function, which is why the selected lists were converted to `./` entries.

### Why the path conversion was needed

The first audit found 23 image-list manifests and 28,615 image references, most written as absolute paths containing the original machine-specific repository location. These paths work only if a machine has that exact layout. The dry run resolved all referenced images successfully. The conversion then rewrote the entries as manifest-relative `./...` paths, backed up the originals, and validated that every converted entry still pointed to the same existing file. The conversion script reported 23 manifests, 28,615 references, and zero verification failures.

The later configuration validation checked 22 selected YAML files and 21 associated manifests, verifying 34,028 references across configurations (some are counted more than once when a manifest is used by multiple splits/configs) with zero errors.

The two external NOD manifests were converted locally but not included in the latest portability commit because their YAML files were left for separate review. Therefore, a fresh clone will not necessarily contain those local-only manifest edits.

## 5. What the verification commands mean

### Check the repository state

```bash
git status --short --branch
```

- `main...origin/main` with no ahead/behind marker means the local branch and remote tracking branch are aligned.
- `A` in the status output means a file is staged as added.
- `M` means modified.
- `??` means untracked, not committed.

### Check the exact staged files

```bash
git diff --cached --name-only
git diff --cached --stat
git diff --cached --check
```

- `--name-only` lists what will enter the next commit.
- `--stat` summarises changed files/lines.
- `--check` catches certain whitespace problems.

Do this before every commit. A successful `git push` confirms that Git's remote accepted the pushed ref, but verify the final status afterwards.

### Understand `.gitignore`

The project's `.gitignore` includes `datasets/*`, with an exception for `datasets/README.md`. That means normal Git commands hide most content within the dataset directory. If a reviewed YAML or manifest has been intentionally approved for tracking, it may need to be staged explicitly:

```bash
git add --sparse -f path/to/approved_file.yaml
```

Use this only for the exact reviewed path. Never force-add all of `datasets/`; that risks uploading huge images, annotation text, or restricted data. A blank `git diff` does not prove that ignored files were unchanged—use `git check-ignore -v path/to/file` and `git ls-files path/to/file` to understand its state.

### Sparse checkout

A sparse checkout means the working tree includes only selected paths from the repository. A file may exist in the repository tree but Git may warn that it is outside the sparse-checkout definition. The approved `git add --sparse -f ...` workflow was used for the selected configs/manifests. Do not broaden the sparse-checkout settings casually when a targeted staging operation is sufficient.

## 6. What has been committed in the recent cleanup

The local Git history showed these commits as pushed:

| Commit | Why it matters |
|---|---|
| `48d1efe` | Added Indian-road annotation/testing utilities and the V7 low-light dataset generation tool. |
| `26ce880` | Documented the experiment history and workflow. |
| `960eb03` | Added a local dataset inventory and improved dataset data-management guidance. |
| `8e01ee0` | Added five portable two-class Indian-road YAML configurations. |
| `7f93561` | Added 22 YAML configurations and 21 image-list manifests (43 files) after portability checks. |

The dataset folders themselves were not uploaded as part of these configuration commits.

## 7. How to interpret the experiment history

Experiment labels are a record of what was tried, not a ranking of quality:

- **Original baseline:** starting detector. Recorded SEW benchmark metrics were P 0.348, R 0.157, mAP@0.50 0.0761, mAP@0.50:0.95 0.0150.
- **V1 — TorWIC + SEW:** recorded P 0.674, R 0.399, mAP@0.50 0.4180, mAP@0.50:0.95 0.2220.
- **V2 — low-light augmentation:** recorded P 0.810, R 0.389, mAP@0.50 0.4390, mAP@0.50:0.95 0.2600.
- **V3 — real-SEW adaptation:** recorded P 0.753, R 0.385, mAP@0.50 0.4360, mAP@0.50:0.95 0.2380.
- **V4 — THUD + TorWIC + SEW:** a subsequent cross-source experiment; consult the run logs for its actual metrics and arguments.
- **V5b — scale augmentation:** explores the effect of object-scale/resize augmentation; do not infer success without checking its results.
- **V6 — THUD real-domain adaptation:** follows up with real-domain data; consult the recorded run output.
- **V7 — planned low-light direction:** the generation script prepares candidate data. A generated dataset alone does not demonstrate that V7 was trained or improved performance.

The SEW comparison uses a 423-image benchmark that was revisited during model selection. The values are useful for understanding the progression but should not be reported as an independent final benchmark.

For the Indian-road detector, the Manual-80 clean, Manual-280 standard, and Manual-280 low-learning-rate comparisons used the same repeatedly accessed 166-image development set for threshold/model selection. The low-learning-rate candidate had the highest combined F1 in one reported comparison (0.513 at person confidence 0.15, rider confidence 0.25, IoU 0.50), but per-class trade-offs remain. The Manual-280 low-LR candidate's recorded class F1 values in that comparison were 0.508 for person and 0.519 for rider; a separate threshold sweep used person confidence 0.20 and rider confidence 0.25. Always specify threshold settings when quoting metrics.

Read [`experiments.md`](experiments.md) for the fuller project experiment record.

## 8. Do not confuse development with a final test

A test set is only an independent final test if it was not used for choosing models, hyperparameters, annotation rules, confidence thresholds, or other decisions. If the team looks at results on a set and then changes the model because of them, that set has become development feedback for future comparisons.

In this project:

- the 423-image SEW set was used repeatedly during development;
- the 166-image Indian-road development set was used for threshold and candidate selection;
- the folder named `indian_manual_independent_test_v1` has also been used during development decisions;
- the V7 YAML points `val` and `test` to the same directory.

Do not overstate independence. For a stronger final evaluation, define a fresh, appropriately permitted set by scene/session, lock the annotations and thresholds first, and do not use its scores to select the model.

## 9. Reproducing a training run

Before training, record:

- the Git commit (`git rev-parse HEAD`);
- exact dataset YAML, dataset source/version, class map, and split provenance;
- starting checkpoint path and checksum if available;
- image size, batch size, seed, epochs, patience, optimiser, and learning rate;
- augmentations, confidence/IoU thresholds, and evaluation command;
- output directory, `results.csv`, `best.pt`, `last.pt`, and failure examples;
- whether the validation/test data was previously used in model selection.

Use the same configuration and environment when reproducing a result. A checkpoint trained from a Manual-80 checkpoint on Manual-280 data is not directly comparable to a model trained from a different base checkpoint unless the distinction is recorded.

## 10. A small, safe first run

Do not use an evaluation YAML as a training config just because it contains a `train` key. First choose a known training dataset, confirm its class map, and check the train/validation folders. For an experiment that is actually meant to train, the pattern is commonly:

```bash
source .venv/bin/activate
# Example pattern only; replace the model and data paths with the intended, verified ones.
yolo detect train model=/path/to/starting_checkpoint.pt data=/path/to/training_data.yaml imgsz=640 epochs=1 batch=2 device=0
```

An `epochs=1` smoke run tests the data loader and configuration; it is not a meaningful performance experiment. Confirm that its YAML does not point train and validation at the same set by accident. Run this example only after setting its data path to a valid local configuration and confirming that the training and validation splits are appropriate.

## 11. Current machine and software used

Recent work in the project was performed using Ubuntu 24.04 under WSL 2, Python 3.12.3, Ultralytics 8.4.156, PyTorch 2.11.0+cu128, and an NVIDIA RTX 4060 Laptop GPU with 8 GB VRAM. The broader robotics workspace uses Ubuntu 24.04 / ROS 2 Jazzy. These are the recorded working-environment details, not a guarantee of identical behaviour on another machine.

## 12. Recommended next actions

1. Keep this documentation linked from the main README.
2. Prepare a collaborator data-access plan that respects each source's terms and permissions.
3. Audit labels and class compatibility before combining datasets.
4. Add a small automated validator for YAMLs/manifests and run it from a new clone.
5. Treat the NOD config/manifests as still needing a dedicated portability review.
6. Plan a truly fresh final evaluation set if you need deployment-relevant conclusions.
7. Keep reporting transparent: document failures and trade-offs as carefully as improvements.
