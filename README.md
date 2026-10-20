# AMR AI — Perception Research for an Autonomous Mobile Robot

> **Purpose:** develop and evaluate camera-based detection for people and riders in the kinds of scenes an autonomous mobile robot (AMR) may encounter, including Indian roads, mixed traffic, indoor areas, rain, and low light.
>
> **Project status:** research and prototyping. The results documented here are not evidence of a safety-certified or deployment-ready navigation system.

This repository contains code, dataset configuration files, image-list manifests, experiment notes, evaluation outputs, and ROS 2/project tooling. Large image datasets, many annotations, trained weights, and generated training runs stay outside ordinary Git history so that the source repository remains manageable.

## Start here

- **[Dataset guide](datasets/README.md):** what the dataset folders mean, which files are tracked, how to prepare local data, and important class/split cautions.
- **[Experiment history](docs/experiments.md):** the progression of the perception experiments and recorded results.
- **[Dataset inventory](docs/data_inventory.md):** an inventory of local dataset folders. It describes the machine used to create it; it does not guarantee that all listed data is tracked, shareable, labelled, or ready to train.
- **[Learning and verification guide](docs/learning_guide.md):** the beginner-friendly explanation of YAML configs, image lists, paths, metrics, validation, Git checks, and reproducible experiments.

If one of the linked documents is not present in a particular clone, check the current branch and repository history before assuming the file was deleted.

## 1. What is the detector intended to recognise?

The main AMR road-scene task uses two classes:

| Class ID | Class name | Project annotation convention |
|---:|---|---|
| `0` | `person` | A person walking, standing, or otherwise not being treated as a rider under the project convention. Someone standing or walking beside a two-wheeler is labelled `person`. |
| `1` | `rider` | A person operating or riding a motorcycle/scooter, including a driver and a rear passenger. The project convention is to box the rider's body and allow the box to cover the overlapping vehicle portion where that is how the manual labels were defined. |

These conventions are project-specific. They are not automatically the same as the class definitions in public datasets. Before combining data sources, inspect their annotation rules and label IDs. A source that has only `person`, or classes such as forklift/cart/rack, is **not automatically a compatible two-class dataset**.

## 2. Repository layout

The main top-level folders are:

| Folder | What it is for | What to know |
|---|---|---|
| `configs/` | Project and experiment configuration | Check referenced paths and class settings before running a script. |
| `datasets/` | Dataset directories, YAML configs, and image-list manifests | Most raw data is intentionally excluded from Git. Start with `datasets/README.md`. |
| `docs/` | Experiment history, inventory, and technical notes | Read the experiment records before comparing metrics or changing splits. |
| `results/` | Checked-in result summaries, where applicable | `results/local/` is ignored; not every local result file is published. |
| `runs/` | Local Ultralytics training/prediction outputs | Generated runs and model weights are not intended to be checked into Git. |
| `scripts/` | Data preparation, annotation, audit, and research utilities | Inspect each script's arguments and effect before running it. Some scripts may be experimental. |
| `src/` | Project source code | Follow the entry point or instructions for the particular component. |
| `tools/` | Supporting tools | Check whether a tool is general-purpose or tied to one dataset layout. |
| `ros2_ws/` | ROS 2 workspace | Build/install/log outputs are ignored. Follow the ROS package-specific setup. |
| `archive/` | Archived material | Treat as reference material unless a document says otherwise. |

Exact contents can change. Use `git ls-files` to see what the repository actually tracks, and `find` to inspect files present on your local machine.

## 3. Research approach in plain language

The work so far has involved several related but separate questions:

1. **Baseline person detection:** measure the starting detector against the SEW-oriented evaluation data.
2. **Domain adaptation:** fine-tune with data from additional sources so that the detector is less tied to one visual domain.
3. **Low-light adaptation:** test whether realistic low-light examples and augmentation help in dark indoor scenes.
4. **Indian-road adaptation:** develop a two-class `person`/`rider` detector from manually curated Indian-road examples.
5. **Error review:** inspect false positives, missed objects, per-class precision/recall, confidence thresholds, and bounding-box overlap rather than relying on one headline score.
6. **Documentation and reproducibility:** record data sources, splits, configurations, and experiment results so that later work can be checked instead of relying on memory.

Experiment labels V1–V7 represent the history of this research; a later version number does not, by itself, mean it is better. In particular, creating a V7 candidate dataset is not the same as training a model on it or demonstrating improved performance.

## 4. Current research results — read the caveats

### Indian-road two-class fine-tuning

The local training logs record the following validation metrics for the named runs. Precision (P), recall (R), and mean average precision (mAP) are calculated by the training/validation workflow. They are not guarantees of real-world AMR safety.

| Run / checkpoint family | Best recorded epoch for the metric shown | P | R | mAP@0.50 | mAP@0.50:0.95 |
|---|---:|---:|---:|---:|---:|
| Indian Road V2 combined fine-tune | 30 for mAP@0.50:0.95 | 0.7615 | 0.6918 | 0.7576 | 0.5785 |
| Manual-80 clean | 10 | 0.5105 | 0.4965 | 0.4425 | 0.1343 |
| Manual-280 standard | 15 for mAP@0.50:0.95 | 0.6756 | 0.5392 | 0.5742 | 0.2076 |
| Manual-280 low-learning-rate | 48 for mAP@0.50:0.95 | 0.7140 | 0.5707 | 0.6079 | 0.2046 |

These rows are from separate training runs and their validation configurations. Do not compare them as if every image, split, training duration, and initialization were identical. The Manual-280 experiments were initialized from the Manual-80 clean checkpoint and used different learning rates. Check the recorded arguments in the run folders before reproducing them.

### Detection-threshold comparison on the Indian-road development set

A 166-image development benchmark with 346 ground-truth objects (177 `person`, 169 `rider`) has been used repeatedly for model and threshold selection. At person confidence `0.15`, rider confidence `0.25`, and IoU `0.50`, the combined results were:

| Candidate | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Manual-80 clean | 152 | 126 | 194 | 0.547 | 0.439 | 0.487 |
| Manual-280 standard | 137 | 93 | 209 | 0.596 | 0.396 | 0.476 |
| Manual-280 low-learning-rate | 159 | 115 | 187 | 0.580 | 0.460 | 0.513 |

The low-learning-rate Manual-280 model was the strongest overall F1 candidate in this particular comparison, but results differ by class and threshold. This development set has influenced model selection. **It is not an untouched final test set.** Another image collection from the same broad source/domain has also been used during model and threshold selection, so it must not be described as a blind, independent final test.

### SEW-oriented model comparisons

The recorded SEW benchmark comparison includes:

| Model stage | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 |
|---|---:|---:|---:|---:|
| Original baseline | 0.348 | 0.157 | 0.0761 | 0.0150 |
| V1 TorWIC + SEW | 0.674 | 0.399 | 0.4180 | 0.2220 |
| V2 low-light augmentation | 0.810 | 0.389 | 0.4390 | 0.2600 |
| V3 real-SEW adaptation | 0.753 | 0.385 | 0.4360 | 0.2380 |

The 423-image SEW benchmark has been used repeatedly during development and comparisons. It is useful for iterative research, but it is not an untouched independent test set. Read `docs/experiments.md` for additional experiment history and condition-specific measurements.

## 5. Software and machine context

The environment used for the recent work was:

- Ubuntu 24.04 under WSL 2.
- Python 3.12.3 in the project virtual environment.
- Ultralytics 8.4.156.
- PyTorch 2.11.0 with CUDA 12.8 build (`2.11.0+cu128`).
- NVIDIA RTX 4060 Laptop GPU with 8 GB VRAM.
- ROS 2 Jazzy is the current ROS setup in the broader AMR workspace; detection-only tasks may not need ROS.

Your exact local hardware, installed driver, dependencies, and paths may differ. Always run the verification commands on the machine that will train or evaluate the model.

## 6. First-time local setup

### Clone the repository

```bash
git clone https://github.com/deepkukreja/amr_ai.git
cd amr_ai
```

If you already have a working clone, do not clone over it. Check your branch first:

```bash
git status --short --branch
git log -5 --oneline
```

### Create a Python environment

Use the project requirements file as the starting point. If a virtual environment named `.venv` already exists, inspect it before creating another one.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch/CUDA installation can depend on operating system and driver versions. If the requirements file does not install a working GPU build on your system, use the official PyTorch installation selector for your system rather than copying another machine's CUDA command blindly.

### Verify the environment

```bash
./.venv/bin/python -c "import sys, torch, ultralytics, yaml; print('Python:', sys.version.split()[0]); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Ultralytics:', ultralytics.__version__); print('PyYAML:', yaml.__version__)"
```

A successful import confirms that packages are available. It does not prove that every dataset path or GPU operation is valid.

## 7. Dataset setup is separate from cloning code

The repository does **not** contain all images and annotations. The local `datasets/` collection has been estimated at roughly 39 GB, with much of it under `datasets/external`. Many sources have their own licences, terms, or access restrictions. A collaborator must obtain each source through an authorised route and place it into the expected local layout.

Start with [the dataset guide](datasets/README.md) and [the inventory](docs/data_inventory.md). Do not assume that a folder name proves that a dataset is annotated or safe to redistribute. The inventory is a local snapshot and can be stale on another machine.

Once the data is available, validate the YAML configuration and all referenced files before training. Do not download or extract every dataset just to begin learning; start with one small, authorised, labelled dataset and its documented configuration.

## 8. Important path rules

There are two separate path-resolution steps:

1. In a YOLO dataset YAML, `path` sets the dataset root. If it is omitted, the installed Ultralytics version uses the YAML file's directory as the default root. `train`, `val`, and `test` are resolved from that root.
2. In an image-list `.txt` file, each line identifies an image. The project converted selected manifest entries to paths beginning with `./`, so Ultralytics resolves each such entry relative to the manifest file's directory.

Examples:

```yaml
train: images/train
val: images/val
names:
  0: person
  1: rider
```

And an image-list line:

```text
./images/train/example.jpg
```

Do not remove `./` from these manifest entries casually. The behavior was checked against the installed `BaseDataset.get_img_files` implementation. The YAML files and manifests need to remain consistent with each other.

## 9. Evaluation and split discipline

- **Training split:** used to update model weights.
- **Validation split:** used during development for monitoring, model selection, and threshold/parameter decisions.
- **Test split:** ideally held back until choices are finished and used for a final estimate.
- **Development benchmark:** an iterative feedback set. Once used to pick models or thresholds, it is not an untouched final test.

Some existing evaluation YAML files set `train` and `val` to the same image list because Ultralytics expects those fields even when a particular script uses the YAML for evaluation. Treat those configurations as evaluation-only unless the experiment notes explicitly say otherwise. Never launch training just because a YAML file has a `train:` key.

Known cautions include:

- `datasets/sew/sew_test.yaml` and `datasets/sew_test.yaml` point their train/validation fields at the same 423-image list.
- Weather-specific SEW configs reuse each condition list for train/validation to support evaluation workflows; these are not independent splits.
- `datasets/torwic_sew_v7_lowlight/torwic_sew_v7_lowlight.yaml` has `val` and `test` pointing to the same validation directory. It must not be described as an independent test.
- The 166-image Indian-road development benchmark has been repeatedly used to compare models and confidence thresholds.
- The Indian-road `independent_test` folder has also been used in development decisions; do not call it a blind holdout without a new audit proving it was never used for selection.

Preserve established split assignments until their provenance and intended use are understood. A change to a split is a change to the experiment, not merely a formatting fix.

## 10. How to verify before committing

Review tracked files and Git status:

```bash
git status --short --branch
git log -5 --oneline
git diff --check
git diff --cached --check
```

See whether Git ignores a dataset file:

```bash
git check-ignore -v datasets/some_folder/data.yaml
```

The project currently ignores dataset contents by default. The approved YAML and manifest files were deliberately force-staged with `git add --sparse -f` because the local clone also uses sparse-checkout. That does **not** mean all dataset files should be force-added.

Before a data-related commit, stage only the reviewed small configuration or manifest files. Then inspect the list:

```bash
git diff --cached --name-only
git diff --cached --stat
git diff --cached --check
```

For this project, use particular care not to stage images, YOLO label `.txt` files, archives, weights, recordings, or raw datasets accidentally. Image-list `.txt` files are not the same as annotation label files, even though both use the `.txt` extension.

## 11. Recent repository work

The following commits were reported by the local Git CLI as pushed to `origin/main`:

| Commit | Message | Summary |
|---|---|---|
| `48d1efe` | Add Indian road annotation and low-light tools | Added manual annotation/testing tools and a V7 low-light dataset-generation utility. |
| `26ce880` | Document AMR perception experiments and workflow | Updated README and experiment notes. |
| `960eb03` | Document local datasets and data management | Added dataset inventory and updated dataset guidance. |
| `8e01ee0` | Add portable Indian road dataset configs | Added five two-class Indian-road YAML configs. |
| `7f93561` | Make dataset configs and manifests portable | Added 22 YAML configs and 21 image-list manifests (43 files total), preserving the recorded split assignments. |

The recent configuration verification checked 22 selected configurations, 21 associated manifests, and 34,028 references across configuration uses with zero path validation errors. These are local validation results; a different machine must repeat the checks after obtaining the actual data.

At the end of this work, the reported status was `main...origin/main`, with one local untracked backup script: `scripts/manual_annotator_independent_test_before_fix.py`. Keep it only if you still need the backup. Do not add it accidentally to commits.

## 12. Recommended way to continue

1. Read `docs/experiments.md` before interpreting old results.
2. Choose one task and one dataset configuration; don't combine sources simply because the class names look similar.
3. Confirm dataset permissions, annotations, class IDs, split design, and path resolution.
4. Run a small smoke test before a long training job.
5. Record the exact checkpoint, dataset YAML, commit, image size, seed, batch size, epochs, optimiser/learning rate, thresholds, IoU, and metric definitions.
6. Audit false positives and missed detections by class and by scene condition.
7. Keep development data distinct from an untouched final test where possible.
8. Document what was actually run. A script, proposed dataset, or planned experiment is not proof that a model was trained or improved.

## 13. Safety and scope

This is a research project, not a certified autonomous-driving or robot-safety system. A high detection score does not itself make navigation safe. A mobile robot also needs robust tracking, sensor fusion, localization, planning, motion control, emergency stops, fault handling, and testing under representative environmental and operational conditions. Do not rely on this detector alone to make safety-critical movement decisions.

For practical questions about a dataset or experiment, the most useful sequence is: check this README, read `datasets/README.md`, inspect `docs/data_inventory.md`, then read `docs/experiments.md` and the configuration used for the run.
