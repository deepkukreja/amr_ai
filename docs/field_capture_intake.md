# Field capture and annotation intake

Use the [broad benchmark label guide](broad_benchmark_label_guide.md) for boxes and class definitions. The first recording priority is small and partially visible people in busy scenes, people and vehicles around two-wheelers, and warehouse scenes containing forklifts, pallet jacks, carts, empty pallets and loads. Include quiet scenes and lookalikes such as parked bikes, posters, trolleys and empty racks.

| Recording | What to capture |
|---|---|
| Indoor operating area | Aisle crossings, close or cropped people, moving and parked equipment, racks, bins and loads. Include glare, uneven light and empty aisles. |
| Outdoor yard or site access road | People crossing or beside parked vehicles, bicycles, motorcycles or scooters, riders and passengers, cars and larger vehicles. Include near and distant examples. |
| Different session or site | Reserve this recording for holdout before reviewing scores. Capture natural conditions and any animals that already occur at the site. |

Use the intended camera and mount wherever available. Record the original capture time, camera identity, physical site and session. Keep only spaced and informative frames from each recording. Export images with their camera acquisition timestamps. If a bag or video is the only source, preserve it locally and record its identity and time basis when exporting frames.

Keep development and holdout recordings in the **same intake batch** so the tool can check that a session stays together. Use a different site for holdout when practical. Inspect alternate clips for repeated passes through the same scene and near duplicates. The automated check detects exact file duplication within the batch and, when supplied, against the broad benchmark manifest. Confirm the selected sessions are absent from earlier training and tuning before calling the holdout independent. Save a separate batch version when changing reviewed data.

## Prepare the image index

Store JPEG or PNG frames in a local capture folder. Create `captures.csv` there using the header below. The `image` path is relative to the CSV. Keep the full table, images and labels out of Git.

```csv
image,site_id,session_id,camera,capture_utc,scene,lighting,split
```

Use a short stable site identifier and session identifier, a UTC time ending in `Z`, and a consistent scene name such as `warehouse` or `loading_yard`. Set `split` to `development` or `holdout` before tuning. The CSV may describe frames already exported from a ROS bag. The intake works with files and does not access any camera or ROS topic.

From the AMR source root:

```bash
python3 -m scripts.data.field_intake stage \
  --index /path/to/capture/captures.csv \
  --output datasets/amr_field_objects_v1 \
  --exclude-manifest datasets/broad_detection_dev_v1/manifest.json
```

The command copies the original image bytes and creates a pending JSON file for every frame. Set `reviewed` to `true` after checking the image. Set `coverage` to the labels actually reviewed, including supported labels confirmed absent. Enter each box as `{"label":"rider","box":[0.31,0.22,0.44,0.88]}` in normalized XYXY coordinates. A reviewed negative has nonempty coverage and an empty `boxes` list. Keep the image hash supplied by the tool.

```bash
python3 -m scripts.data.field_intake freeze \
  --batch datasets/amr_field_objects_v1 --split development
.venv/bin/python -m scripts.evaluation.evaluate_broad_benchmark \
  --manifest datasets/amr_field_objects_v1/development.json \
  --output results/local/amr_field_objects_v1/development_scores.json
```

The evaluator scores the selected model's supported classes under each frame's reviewed mask. It can also compare the existing merged warehouse labels when both constituent classes were reviewed. The same rule applies to generic human localization across pedestrians and riders. Check class support and missed examples before training.

After the final candidate and thresholds are fixed, freeze the `holdout` split and evaluate its manifest with `--allow-field-holdout`. Keep intermediate model decisions on development data and publish only aggregate results.
