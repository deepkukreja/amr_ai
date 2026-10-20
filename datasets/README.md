# Dataset Management

This directory is used for datasets, dataset configurations, annotations,
and locally generated dataset variants for AMR perception research.

**The actual datasets are not included in this Git repository.**

For the list of datasets present in the development workspace, see
[Local Dataset Inventory](../docs/data_inventory.md).

## Storage Policy

Keep raw datasets, large archives, MCAP recordings, and generated
training datasets outside Git.

The repository tracks this README, but the dataset contents are excluded
by the root `.gitignore`. Do not force-add large or restricted dataset
files to Git.

Use the original authorized source or approved shared storage to obtain
datasets. Confirm the applicable access agreement and redistribution
permissions before sharing data with another researcher.

## Dataset Groups

The local workspace contains several families of datasets:

| Group | Purpose |
|---|---|
| TorWIC | Source data and derived variants used in perception research |
| SEW | Environmental-condition, weather, and low-light research |
| THUD++ | Real-scene pedestrian perception and domain adaptation |
| Indian-road datasets | Pilot data, manually annotated images, training sets, and evaluation sets |
| AMR validation data | Specialized field and indoor validation collections |
| TiHAN / TIAND | External camera and multimodal sensor data requiring further inspection |

These descriptions indicate intended research roles. They do not establish
that every local folder is annotated, complete, or ready for training.
Check the inventory and the contents of each dataset before using it.

## Preparing a Local Workspace

1. Obtain the required dataset through its authorized source.
2. Check the dataset's access terms and any redistribution restrictions.
3. Place the data in the expected local directory.
4. Inspect the dataset configuration, usually a `data.yaml` file, where one exists.
5. Verify image paths, label formats, class mappings, and train/validation splits.
6. Confirm that validation and test data do not overlap with training data.
7. Record the dataset version, preparation procedure, and relevant scripts.

Dataset preparation utilities are located under `scripts/data/`.

Do not assume that similarly named dataset variants contain the same images,
labels, or train/test splits.

## Person and Rider Detection

Some Indian-road datasets use two detection classes:

- `person`
- `rider`

Before training or evaluation, verify that the selected dataset uses the
expected class mapping and annotation conventions. Other source datasets
may use different labels or formats and may require conversion.

Keep independently reserved test data separate from development data used
for model selection and confidence-threshold tuning.

## TiHAN / TIAND Status

The local `OneDrive_2026-10-16.zip` archive was inspected. Its camera archive
contains 10,919 JPEG images, and no obvious annotation files were found in
the camera archive listing.

Treat this as an unverified external camera-data source, not a confirmed
ready-to-use pedestrian/rider detection dataset. Inspect its metadata,
annotation availability, and applicable usage terms before preparing it
for training or external evaluation.

## Reproducibility

When preparing or using a dataset, document:

- Original source and version
- Local folder name
- Annotation format and class mapping
- Preparation script and configuration
- Train, development, and test split definitions
- Known limitations and data overlap

See [Experiment History](../docs/experiments.md) for the project's
recorded experiment history and evaluation guidance.
