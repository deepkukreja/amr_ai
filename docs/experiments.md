# Experiment History

## Perception experiments

The project began with a YOLO-based person/object detection baseline trained
on the TorWIC dataset.

The major adaptation stages were:

- Original TorWIC baseline
- V1: TorWIC + SEW adaptation
- V2: low-light augmentation
- V3: real SEW domain adaptation
- V4: THUD + TorWIC + SEW mixed training
- V5b: targeted scale augmentation
- V6: real-domain adaptation using THUD real-scene data
- V7: realistic low-light adaptation (planned)

## Current milestone

V6 is the current development baseline.

V6 demonstrated strong performance on the real THUD development scene and
generalized strongly to the separate SEW development sessions.

A separate 423-image SEW benchmark has also been used repeatedly during
development. It should therefore be treated as a development/transfer
benchmark rather than a pristine final test set.

## V7 direction

The next experiment targets realistic low-light domain adaptation.

The motivation is a measured difference between:

- normal-light SEW training data
- the dark SEW benchmark indoor session
- the previous V2 synthetic low-light augmentation

The V2 augmentation matched average brightness but produced substantially
lower image contrast than the real dark indoor images.

V7 will therefore investigate low-light transformations that preserve
contrast and introduce realistic camera degradation rather than simply
reducing exposure.
