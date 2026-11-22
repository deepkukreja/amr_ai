# AMR object labels and first development benchmark

Use this guide with [the object detection roadmap](broad_object_detection_roadmap.md). The protocol compares existing checkpoints before any change to training data, runtime defaults, or thresholds.

## Label rules

| Label | Rule |
|---|---|
| `pedestrian` | Person walking, standing, seated beside a vehicle, or otherwise off a two-wheeler. Include visible parts of a partly hidden person when the instance is identifiable. |
| `rider` | Human riding a bicycle or powered two-wheeler, including passengers. Box each visible human separately. A person standing beside a parked bike is a pedestrian. A person actively mounted on a stopped two-wheeler counts as a rider. |
| `human` | Evaluation and operator grouping that covers both pedestrian and rider. A generic COCO person prediction supports this view. It supplies no rider-role label. |
| `bicycle`, `motorcycle` | Box the vehicle separately from its occupants. Use motorcycle for a powered motorcycle or scooter under the agreed local mapping. Flag uncertain vehicle type for review. |
| `car`, `van`, `truck`, `bus` | Box the road vehicle. Use van when body shape is clear and the source distinguishes it. Keep warehouse forklifts separate from road trucks in new annotations. |
| `forklift`, `pallet_jack`, `cart`, `pallet`, `rack_or_shelf`, `bin_or_tote`, `box_or_load` | Box the named physical equipment or load instead of a combined obstruction. A load, its pallet, and the handling vehicle may each have their own box when visible and relevant. |
| `dog`, `cat`, `cattle`, `goat`, `sheep`, `horse`, `bird` | Box the visible animal. Use the species grouping only when identifiable. Keep uncertain examples in review and record them separately for follow-up. |

Draw a tight box around visible extent and clip it to the image. For heavy occlusion, record the occlusion and cover the identifiable visible parts. Avoid inventing boxes behind walls or other opaque objects. Mark reflections, printed people, posters, statues, and realistic toys as negatives for the corresponding real-object category when the frame has been reviewed for it. If a crowd is too dense for reliable individual boxes, mark that image/class unscored for instance evaluation and retain it for qualitative review. Record close or cut-off people, distant small people, crowded scenes, blur, lighting, and empty scenes for separate analysis. This benchmark reproduces legacy rider boxes as-is, including any overlap with the vehicle. Apply the new convention to newly reviewed footage.

## Frozen input policy

The manifest records exact file hashes, grouping, label coverage, and normalized visible boxes per image. The sampler fixes its seed before predictions and rejects duplicate image bytes. The benchmark uses these cohorts.

| Cohort | Selection and interpretation |
|---|---|
| Reviewed road | Up to 48 distinct clips from the existing 166-image development set, stratified by time of day. Recheck original image and reviewed annotation hashes. Score human localization and the existing pedestrian/rider roles. Vehicle predictions are outside the label coverage of these images. |
| Indoor THUD++ | 16 publisher-audited person images from the separate Capture 2 already used in the indoor replay study. Scores describe one store and a person-positive sample. |
| Warehouse legacy | One image from each of 19 local TorWIC validation IDs. A contact sheet was checked for class meaning. Use grouped forklift/truck, cart/pallet jack, rack/shelf and person reference boxes, with a visually checked empty frame. Scores are diagnostics against converted legacy labels. Their coverage, scene independence and past checkpoint exposure require a separate audit before any release claim. |
| Open Images V5 | Five positive anchors each for 14 categories, with a fixed selection, listed author/license metadata, and at most two images per listed author. Score positively verified classes with boxes and explicitly verified negatives. Skip the image/class pair when group boxes, depictions, inside boxes, or verification conflicts make coverage unsuitable. Download only selected validation images. [Open Images V5](https://storage.googleapis.com/openimages/web/factsfigures_v5.html) · [Partial-label evaluation](https://storage.googleapis.com/openimages/web/evaluation.html) |

Vehicle and animal annotations here come from public still images. A new site/session holdout with fully reviewed vehicles, equipment and animals is required to measure deployment-domain performance. Legacy warehouse classes remain grouped for scoring. The new pallet, bin, and box classes need separate reviewed annotations. Keep raw data, names of private sites and camera captures in the local workspace.

## Compare the checkpoints

Score the pretrained COCO checkpoint without its person-only demo filter, the road reference and candidate, and warehouse V6. Model class IDs are mapped by checked checkpoint names. A model's unsupported class stays unsupported in the report even when ground truth exists.

The score uses one-to-one confidence-ordered matching at IoU 0.50 and confidence 0.25. AP50 and AP50–95 use all-point interpolation over predictions from confidence 0.01 upward and the same image/class masks. This is a project metric with coverage restrictions, separate from official Open Images challenge scoring. Report ground-truth count, scored images, true positives, false positives, false negatives, and precision/recall per cohort and class. Report road results by lighting and reviewed human density. Report model inference wall time from file input to mapped detections at batch 1 after the first image. Jetson and sensor-to-operator timing need separate measurements.

After installing the staged scripts in `amr_ai`, run these commands from its root:

```bash
python3 -m pytest -q tests
.venv/bin/python -m scripts.data.freeze_broad_benchmark \
  --output datasets/broad_detection_dev_v1
.venv/bin/python -m scripts.evaluation.evaluate_broad_benchmark \
  --manifest datasets/broad_detection_dev_v1/manifest.json \
  --output results/local/broad_detection_dev_v1/evaluation.json
```

Review results before making any training decision. Freeze the benchmark before scoring, retain the manifest locally, and publish only aggregate evidence and reproducible tooling. Keep the existing webcam and robot profiles in place during this evaluation.
