# Indian-road rider label audit

This is the first implementation step of the [Indian-road rider plan](indian_road_rider_plan.md). The audit uses 40 real Indian dashcam images from 40 distinct local clips. It includes 18 daytime, 15 night and seven dusk frames. The image bytes and previous annotations were hashed and checked again before scoring. The 24 frames drawn from training clips and the 16 legacy development frames retain those roles.

## Review outcome

The assistant visually checked each frame for pedestrians, riders, bicycles and motorcycles. Where clear, it drew or accepted a visually checked visible-extent box. Pinned detector proposals and previous boxes served as aids. Four legacy development frames were viewed without overlays before checking proposed labels and enlarged crops. This was one review process. Independent human adjudication is future work.

| Source role | Frames checked | Frames with complete four-class coverage |
|---|---:|---:|
| Prior training clips | 24 | 4 |
| Legacy development | 16 | 3 |
| Total | 40 | 7 |

Across complete and partial frames, the audit recorded 72 pedestrian, 45 rider, 38 motorcycle and eight bicycle boxes. These are visible positive observations. Glare, dark areas, occlusion and overlapping vehicles leave uncertain classes on many frames. Class coverage is explicit in each local review. Use the incomplete frames for review and diagnostics. Ordinary YOLO training needs complete labels for the target categories. Only the four complete training-clip frames are applicable to that pool at this stage. The wider batch should prioritize clearer bicycles, hard negatives such as pedestrians beside parked bikes, and motorcycle boundaries where riders overlap.

## Compare the same riders under both box conventions

Of 24 legacy rider labels on 16 development frames, 16 riders across nine distinct clips could be linked to the same visible person with a separate revised box. Other legacy labels were left outside the paired metric because the corresponding person or visible extent was ambiguous. The existing Indian-road candidate ran once over the 16 images at 960 px. Both sets were scored from the identical predictions, at confidence 0.25, NMS IoU 0.70 and one-to-one matching IoU 0.50.

| Outcome | Riders |
|---|---:|
| Matched under both conventions | 5 |
| Matched under legacy only | 3 |
| Matched under revised only | 2 |
| Matched under neither | 6 |
| Total paired riders | 16 |

The model matched 8/16 riders with legacy boxes and 7/16 with visible-extent human boxes. Five individual matches changed. Some old labels covered only the torso and others included more of the vehicle. This shows that label geometry changes individual match outcomes in this subset. The small subset and its incomplete class coverage limit conclusions about the larger transfer gap. The paired result reports recall on linked riders. Use fully reviewed frames for precision and false-positive rates.

## External evaluation and reproduction

The configured Hugging Face session received HTTP 403 for the pinned Third Eye Labs annotation file. Activation of dataset access is needed to check the benchmark IDs for overlap and reserve the independent evaluation. The local development audit can be repeated separately. [Third Eye Labs benchmark](https://huggingface.co/datasets/thirdeyelabs/indian-road-benchmark)

With the project's local Indian-road source data and pinned candidate installed, run from the WSL checkout:

```bash
.venv/bin/python -m scripts.data.indian_rider_label_audit \
  --output datasets/indian_rider_label_audit_v1
.venv/bin/python -m scripts.evaluation.score_indian_rider_label_audit \
  --audit datasets/indian_rider_label_audit_v1 \
  --output results/local/indian_rider_label_audit_v1/evaluation.json
```

The first command stages pending review files and grid previews. The second reads completed local reviews. Set `reviewed` after visually inspecting the frame, keep its supplied image hash, declare complete per-class `coverage`, and record visible positives in `boxes` with normalized XYXY coordinates. The `rider_pairs` list links zero-based `legacy_index` and `review_index` values for the same visible rider. Images, individual boxes and the full review records stay local.

Selection SHA-256 `b27b1978e0e7e42b2e11072adaeb3e04a9cbed1f44aa4a0cbf92f58eaa694166`. Pinned scorer checkpoint SHA-256 `2f160e4b51e918d99cd009940bcc6c99e2fc394ff86dc29ebbb60a27359ec7f1`.
