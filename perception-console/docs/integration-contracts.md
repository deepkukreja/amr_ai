# Integration contracts

## Version 1 operator world snapshot

Source adapters supply acquisition time and provenance independently of receipt time. Coordinates are right-handed, X forward, Y left, Z up in `base_link`. Positions are metres, speed is m/s and angles and angular rate are radians and rad/s. Each snapshot has `version`, `mode`, `session`, `sequence`, `stampMs`, `site`, `frame`, `clock` and `scenario`. The fixtures use `sim_warehouse` or `sim_yard` and elapsed milliseconds in one scenario cycle. Each cycle has a distinct session. The Gazebo importer uses `map`, distinguishes `sourceStampMs` from playback elapsed time, and retains `GAZEBO` provenance. A prospective live adapter uses source ROS time and supplies receipt time separately. Consumers reset histories at a session change. Unknown values use null or explicitly absent channels. Visualization converts ROS coordinates to Three.js as `(x,y,z) → (x,z,-y)`.

| Channel | Payload and semantics |
| --- | --- |
| Robot and localization | Pose X/Y/yaw, speed, steering, battery and localization state/variance. `robot.modelId` selects a configured profile, with a heading marker when model geometry is unavailable. Localization uncertainty uses m². Production adapters keep original ROS headers and full covariance. |
| Environment and coverage | Fixture structure is known static geometry. The local occupancy grid separately carries per-cell unknown/free/occupied evidence with resolution and minimum corner. Ray traversal marks observed free cells, endpoints mark occupied cells and occlusion stays unknown. Accompany clouds with acquisition timestamp and validity. |
| Semantic observations and tracks | Source ID, stamp, frame, class, score, 3D extent. Display tracks have session-scoped IDs, position, extent, planar velocity, confidence, observation age and history. When semantic association is ambiguous or absent, geometry can remain unknown-class. |
| Navigation | Mission state and destination, planned route, local trajectory, steering prediction and traveled history. All paths share the frame or are explicitly unavailable. Prediction is a kinematic view aid. |
| Health and events | Sensor ID, frame, nominal rate, last source stamp, age and status. Warnings have stable codes and simulation timeline markers. A disconnected browser ages the transport independently. |
| Replay | A versioned ordered frame collection with source provenance, sample interval, duration and timeline events. The browser changes presentation mode to REPLAY while retaining SIMULATION or GAZEBO as the recorded source. |

Single Nav2 planning captures also use replay presentation. They retain the ROS map/pose/path frame and source stamp, have no timeline, and omit motion or sensor telemetry. See the [Nav2 capture contract and fixture](nav2-plan-view.md).

The optional ROS `depth` channel carries `frame`, `stampMs`, `width`, `height` and `mm16`. The frame is the depth image's optical frame. The base64 buffer stores the complete row-major image as little-endian unsigned 16-bit optical depth in millimetres, with zero for unknown. Each dimension is at most 160 pixels. Depth has an independent one-second freshness limit and clears when the browser loses the stream. This preview is sampled in camera image space. RGB association and map registration require their own calibration and time checks.

## Upstream topic and model boundaries

Topic profiles are launch-specific and remappable. The targets below are starting budgets in steady operation. Verify each rate, frame, QoS and topic publisher from the actual graph before adopting a profile.

| Producer | Boundary | Initial adapter policy |
| --- | --- | --- |
| SLAM Toolbox or map server | `/map` `nav_msgs/OccupancyGrid`; static PGM + YAML in `rbot/maps` | Match transient-local delivery where offered. Use map identity, frame, origin, resolution and width/height. Retain unknown cells. Send on revision or throttled for live mapping. |
| EKF, AMCL and TF | `/odometry/filtered` `nav_msgs/Odometry`, `/amcl_pose` `geometry_msgs/PoseWithCovarianceStamped`, `/tf`, `/tf_static` | Verify the existing `odom → base_footprint → base_link` transform chain and `map → odom` authority. Budget up to 20 Hz world pose. Gate map registration on source age, valid covariance and TF at the measurement stamp. Keep covariance and pose provenance. |
| 2D and 3D LiDAR | `/scan` `sensor_msgs/LaserScan`, `/lidar_3d/points_raw` `sensor_msgs/PointCloud2` and the current `rlai_lidar_processing` output | Subscribe with confirmed sensor QoS. Use stamp, range validity and per-point time when present. Start at 5–10 Hz for data consumed by the browser; bound points and carry filter/deskew status. |
| RGB-D and stereo | `/depth_camera/image_raw`, `/depth_camera/depth`, `/depth_camera/camera_info` plus stereo topics; `sensor_msgs/Image` and `CameraInfo` | Match frame, calibration version and encoding. Use depth validity and alignment before projection. Start with a single calibrated RGB-D source; track age, skew, latency and independent stream state per device. |
| Model adapter | Existing PyTorch profile or pinned FP32 ONNX input/output. ROS detector and depth fusion use `vision_msgs/Detection2DArray` and `Detection3DArray`. | Bind artifact SHA-256, artifact format, input color/resize/letterbox, dtype, shape, class map, thresholds and NMS to one profile. Preserve original image coordinates, source stamp and model status. Start at 5 Hz semantics and measure budget. A 2D detection has no inferred global position without depth/TF. Use a separate inference environment where ROS NumPy/cv_bridge compatibility requires it. |
| Existing Nav2 and local supervisor | `/plan` `nav_msgs/Path`, confirmed controller trajectory output, existing read-only control/ownership reports | Keep path type and validity explicit. Preserve existing platform V1 status age independently of pose. Never infer mission readiness from odometry. Pause/resume and request expiry belong to a later validated, authenticated API. |
| Joint and auxiliary feedback | `/joint_states` `sensor_msgs/JointState`, configured IMU, battery/drive and other sensor topics | Consume steering feedback only with a verified joint name and sign. Add diagnostics and calibration profiles per input. Unavailable battery, E-stop, GNSS and bumpers remain unknown. |

Keep planned routes read-only in the console. An eventual remote gateway needs authenticated operator and robot identities, exclusive locally granted control, bounded speed/steering, command expiry, heartbeat, teardown on link loss, and append-only operator audit events through the existing local supervision boundary. Independent physical E-stop and protective functions are separate hardware responsibilities.
