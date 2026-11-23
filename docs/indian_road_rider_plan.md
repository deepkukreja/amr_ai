# Indian-road rider and two-wheeler plan

The next experiment should test Indian-road data with the rider and their vehicle boxed separately. On the reserved US BDD100K sample, the new model improved rider matches from 19/60 to 30/60 and increased unmatched rider predictions from 14 to 23. On the existing Indian-road development frames, matches fell from 23/41 to 10/41. Motorcycle matches on BDD100K fell from 10/34 with general COCO to 6/34 with the pilot. Changes in road scenes and rider box conventions can both affect these comparisons. [Pilot and transfer results](bdd100k_rider_pilot.md)

## 1. Establish the evaluation before training

The Third Eye Labs Indian Road Detection Benchmark is the first evaluation choice. The provider reports 450 labeled Delhi NCR frames with separate person, rider, bicycle and motorcycle boxes. Their two annotation passes yield agreed boxes and ignore regions. The provider publishes a grouped COCO benchmark that combines pedestrian and rider. AMR needs to keep the four classes separate for its main comparison and retain the ignore regions when scoring. The release is CC BY 4.0. The repository card and revision metadata were readable, while direct access to `annotations/dev.json` returned HTTP 401. The Hugging Face repository uses its access gate. Pin revision `d367b3b9b69d411db386e38f408696411189a2d0` for this audit once access is available. [Dataset and scoring notes](https://huggingface.co/datasets/thirdeyelabs/indian-road-benchmark) · [Publisher leaderboard](https://thirdeyelabs.ai/leaderboard)

Before using it as a holdout, check clip and drive IDs against the 221 prior training frames, the 59 tuning frames, the 166 development frames, the 120 pending frames and the archived runs' source manifests. Compare image hashes and inspect near duplicates because the published benchmark applies blurring. Reserve whole drives or routes when traceable. If the overlap audit leaves too few fresh clips, use a new licensed source. IIIT Hyderabad's IDD Detection lists the four classes separately in its official tooling. Check a small sample for geometry, crowd conventions and use terms after publisher access is available. IDD-2WD is also a targeted source to inspect for riding scenes and label semantics. [IDD access](https://indiadrivingdata.org/) · [IDD class and format documentation](https://github.com/AutoNUE/public-code#detection)

## 2. Start label work with data already on disk

The current local manual set has 221 training frames from 89 clips and 59 tuning frames from 22 different clips. Those reviews cover the prior human and rider classes. The 120 pending frames are unreviewed and come from 54 of the existing training clips. They can add difficult training conditions. Fresh evaluation clips need a different source group. Third Eye Labs' larger road collection supplies images and machine-generated proposals. Its documentation groups motorcyclists with the mounted vehicle in the rider category and lists unridden motorcycles separately. A full four-class review is needed. [Original source and label definitions](https://huggingface.co/datasets/thirdeyelabs/indian-road-dataset)

1. Review 40 frames before scaling up. Use 24 from training clips to test the new workflow and 16 from the legacy development set to compare old and new annotation conventions. Keep the latter in development. Review a portion twice before using model suggestions broadly.
2. Use the [project rules](broad_benchmark_label_guide.md). Box each visible mounted human, including passengers, separately from the bicycle or powered two-wheeler. Box a person beside a parked bike as a pedestrian. Treat scooters as powered two-wheelers. Keep uncertain and heavily occluded cases in review or use the benchmark's ignore protocol. Train ordinary YOLO only on images checked for all four classes.
3. Score fixed existing predictions against both versions on the 16 development frames to estimate how much the box convention changes matches. Use the first audit to settle ambiguous conventions before expanding the training and tuning pool toward 200–300 reviewed frames. Include crowded junctions, night and dusk, small or truncated riders, passenger cases, and hard negatives such as pedestrians next to parked bikes. Prefer varied clips over consecutive frames.

## 3. Run a controlled comparison

Use the same reviewed Indian training and tuning images, resolution, augmentations, seed and training budget for each experimental arm.

| Arm | Initialization | Purpose |
|---|---|---|
| A | General COCO YOLO26s | Establish performance from the existing broad pretrained model with Indian four-class labels. |
| B | BDD100K four-class pilot | Measure whether prior rider and vehicle training helps once both arms see identical Indian labels. Keep this arm in research under the BDD100K data terms. |
| References | Existing Indian-road candidate and general COCO | Compare human/rider classes against the existing road baseline and vehicle classes against general COCO. |

Select checkpoints and confidence thresholds on tuning clips. Use rider recall at a comparable level of unmatched rider predictions per frame as the first measure. Track pedestrians, bicycles and motorcycles separately. Include precision, AP50, AP50–95, counts by lighting and density, and uncertainty estimated across clips. Score same-frame human and vehicle coverage as a diagnostic when both boxes exist. A per-rider link needs explicitly reviewed rider-to-vehicle assignments.

For a research screening decision, aim for at least a 10 percentage point increase in rider recall at the existing model's unmatched prediction rate, and check pedestrian recall and vehicle precision for regressions. Agree a meaningful tolerance on tuning data before opening the holdout. Aim for at least 100 rider instances across varied held-out clips. Report weaker class and lighting slices with their actual support and distinguish exploratory gains from measured readiness. Any change to the runtime profile would require its own integrated validation.

The immediate implementation task is the 40-frame four-class label audit and the benchmark access/overlap check. Use that evidence to choose the first training batch and protect a reserved Indian cohort before model selection.

The [40-frame audit report](indian_rider_label_audit.md) records the implemented sample, paired results, class coverage, and independent benchmark access issue.
