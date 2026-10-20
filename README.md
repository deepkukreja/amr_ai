# AMR AI

Autonomous Mobile Robot (AMR) perception research and development.

This repository contains the perception pipeline, dataset preparation tools,
evaluation utilities, ROS 2 integration, and experiment documentation used
during development of an autonomous mobile robot platform.

## Current focus

The current work focuses on robust visual perception for mobile robots,
particularly person detection across changing environments, illumination,
weather, and sensor/domain conditions.

The primary detector is based on Ultralytics YOLO26s.

## Repository structure

```text
amr_ai/
├── configs/          Configuration and dataset split files
├── datasets/         Dataset documentation; raw data stays local
├── docs/             Experiment and research documentation
├── scripts/
│   ├── data/         Dataset preparation
│   ├── evaluation/   Evaluation utilities
│   ├── training/    Training utilities
│   └── visualization/
├── src/
│   └── perception/   Reusable perception modules
├── tools/            Dataset/utility tools
├── ros2_ws/          ROS 2 integration
└── archive/
    └── legacy/       Historical experiments and debugging code

```

## Research Status

V6 is the development baseline recorded in the experiment history.
V7 explores more realistic low-light adaptation. Its generation script
is available, but model training and evaluation must be verified separately.

## Documentation

- [Experiment history](docs/experiments.md)
- [Dataset guidance](datasets/README.md)
- [SEW model comparison](results/sew_model_comparison.csv)
- [SEW condition comparison](results/sew_condition_comparison.csv)
- [THUD error analysis](results/thud_real_capture2_error_analysis_v2_v3.csv)
- [Dark-indoor session split](configs/datasets/amr_dark_indoor_v1_session_split.txt)

## Data and Model Artifacts

Raw datasets, large recording archives, trained model weights, and generated
experiment outputs are managed separately from the source repository.
Check dataset licensing and preserve the defined train, development, and
test separation when preparing or evaluating models.
