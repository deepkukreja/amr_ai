# Datasets

The datasets used by this project are not stored in Git because of their size
and licensing/distribution constraints.

The local workspace may contain:

- TorWIC
- SEW
- THUD++
- Derived YOLO-format datasets
- Experimental train/validation splits

Expected structure:

datasets/
├── torwic/
├── sew/
├── external/
│   └── thud_pp/
└── <derived datasets>/

Dataset preparation scripts are available under:

scripts/data/

Do not commit raw datasets, MCAP recordings, ZIP archives, or generated
training datasets to the repository.
