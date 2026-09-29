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
