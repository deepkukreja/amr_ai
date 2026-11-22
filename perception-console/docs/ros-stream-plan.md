# Perception integration and hardware bring-up

1. **Continuous read-only view.** Receive the existing map and Nav2 path, resolve the robot pose, and register LiDAR at the sensor timestamp. Add one opt-in RGB camera with matching CameraInfo. Show a path overlay only when the calibration, frames and timestamps agree. Keep the operator shell, saved replays and message provenance intact.
2. **Validate the data before using it.** Exercise the saved warehouse map and Nav2 route with an isolated ROS sensor fixture. Trace one LiDAR return and the calibrated camera overlay to the browser. Drop each feed and verify the other stays available. Stop the producer and check that an old snapshot cannot be presented as a new one. Use the existing Gazebo publishers in a separate session before adopting their configuration.
3. **Stationary bench integration.** Inventory the actual sensor models, firmware, timestamps and ROS topics. Verify the D3 URDF, measured sensor mounts, camera calibration, ground clearance, steering sign and vehicle footprint. Compare labeled camera targets and measured LiDAR landmarks indoors and outdoors. Exercise unplugging, occlusion, localization resets and network loss. Record development and held-out field runs separately.
4. **Physical motion qualification.** Confirm the D3 controller and local supervisor, exclusive control, heartbeat loss, hardware stop and braking under measured payload and speed conditions. Begin in a controlled test area and expand the operating envelope only against measured evidence. A later remote-operations release needs authentication and independent local stop enforcement.

The continuous bridge is file-based across WSL and Windows. A ROS subscriber writes a bounded, atomically replaced snapshot to an operator-selected local path. The loopback server emits new snapshots only after it sees the publisher sequence advance. Freshness uses ROS source time and monotonic receipt time on the producer, and monotonic time since the last new snapshot in the browser. A stopped publisher cannot make an existing file new. Camera transport is opt-in. The browser has no ROS publisher or command endpoint.

## Where Gaussian splatting fits

Cartken describes depth and color cameras, sensor fusion, human-annotated HD maps with route restrictions, and escalation to remote operators. Its camera supplier describes a visual SLAM system combining monocular and stereo features with semantic recognition. Neither public technical description identifies Gaussian splatting as Cartken's map or visualization representation. [Cartken technology](https://www.cartken.com/) · [RealSense engineering case study](https://www.realsenseai.com/case-studies/cartken/)

| Layer | Use in this console |
| --- | --- |
| SLAM occupancy and Nav2 | The operational map and planned path, including unknown cells. |
| Live LiDAR and calibrated RGB | Current obstacles, the live view, and independent sensor freshness. |
| Coloured point cloud or simple mesh | The first optional 3D site layer when we have a registered capture. |
| Gaussian splatting | An optional surveyed site view for operators, replay, training and demos. |

3D Gaussian splatting can render a surveyed scene photorealistically from new angles. Experimental visual SLAM systems also estimate poses using splats. A rendered splat's opacity is a visual parameter. Metric clearance, current occupancy, unknown space, and collision decisions would still use the existing navigation and perception stack. [Original 3DGS research](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) · [Gaussian Splatting SLAM](https://arxiv.org/abs/2312.06741)

Run the first splat trial offline on one warehouse entrance that spans indoors and outdoors. Capture calibrated overlapping RGB views with metric poses, align the resulting asset against measured landmarks in the SLAM frame, and label its survey date and map version. Compare alignment, coverage, load time and GPU cost with a coloured point cloud. The site layer should be optional and should not hide current LiDAR, unknown space or warnings. Select a renderer and license only after that trial is justified.

## Running the observer

Use the ROS 2 Jazzy system Python with the ROS sensor messages, `sensor_msgs_py`, NumPy and OpenCV. Run the observer after starting the existing ROS stack. Select **simulation** for synthetic or Gazebo sources and **live** only for the physical setup. Add `--ros-args -p use_sim_time:=true` when the graph uses a simulation clock. Camera topic flags are optional and must be supplied together.

```sh
source /opt/ros/jazzy/setup.bash
python3 -B perception-console/ros/stream_ros.py /path/shared/with/console/world.json \
  --site "Small warehouse" --mode simulation --robot-model d3-prototype \
  --camera-topic /depth_camera/image_raw --camera-info /depth_camera/camera_info
```

Set `RBW_ROS_WORLD_FILE` to the same local file from the console host and start the built console. Select **ROS sensor stream**, or open the console URL with `?source=ros-world`. The default ROS topics are `/map`, `/plan` and `/lidar_3d/points`, with `base_footprint` as the robot frame. Change these with documented command-line flags when the deployment uses other names. The observer retries brief file sharing violations and skips a snapshot if the file remains locked.

LiDAR points are transformed to the map frame at acquisition time. The observer bounds the display to 800 points within 30 m, 2,000 route poses, and a JPEG with a longest edge of 480 pixels. The map retains free, unknown and occupied evidence after downsampling. Camera overlays require a stamped optical transform, matching CameraInfo and a path from at or before the image timestamp. Supported raw camera encodings are RGB8, BGR8 and MONO8, with pinhole, plumb-bob and rational-polynomial calibration as applicable. Unmatched calibration leaves the image visible without the overlay. Camera and LiDAR clear independently after one second without usable data, pose after 1.2 seconds, and path after three seconds. Use the published source rate to assess whether those development limits fit the robot.

## Reproduce the isolated sensor test

```sh
source /opt/ros/jazzy/setup.bash
ROS_DOMAIN_ID=123 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
  python3 -B perception-console/ros/verify_stream.py \
  /path/to/the/existing/nav2-world.json /tmp/rbw-ros-fixture
```

Use an unused domain and a new output directory. The fixture republishes the saved warehouse map and captured Nav2 coordinates with simulation-fixture timestamps. It supplies a synthetic LiDAR landmark, RGB frames and known camera intrinsics. The test observes independent LiDAR, CameraInfo and RGB dropout and recovery. The optional `--serve` flag keeps the healthy fixture available after verification for browser review.

## Verified example

The Jazzy/Fast DDS fixture delivered all 67 route poses and 497 LiDAR points, including the independent landmark at (2.25, −0.85, 0.84) m in the map frame. The camera overlay contained 49 projected path points and its far endpoint matched a separate pinhole calculation. A LiDAR dropout cleared the cloud while the camera overlay remained visible. CameraInfo dropout cleared only the overlay. RGB dropout cleared the image and left LiDAR available. Feeds recovered and the isolated test exposed zero velocity topics.

The production build and sixteen console tests passed. Desktop and phone-width browser checks exercised the map, camera inset, expanded image, point-cloud view and diagnostics. A separate browser check showed that an unchanged file cannot announce a live stream, an advancing snapshot restores data, and stopped updates clear the camera and planned path. Physical hardware, camera calibration on the real robot, and production WAN transport remain unverified.

## Stationary Gazebo integration

The existing rbot warehouse launch ran on ROS 2 Jazzy with Gazebo Sim 8.14 and a private, localhost-only simulation session. SLAM Toolbox published `/map` and the map transform. The Gazebo depth camera supplied `/depth_camera/image_raw` and matching `/depth_camera/camera_info`. The existing LiDAR filter supplied the observer's default `/lidar_3d/points` topic. The observer and ROS publishers shared `/clock`. The browser identified the stream as **Simulation** and used the generic ROS robot marker for the simulated chassis.

The browser received three successive snapshots with advancing simulation timestamps. The sampled frame contained an occupancy grid of 57 × 57 display cells at 0.30 m, including 3,150 unknown cells, 783 timestamp-aligned LiDAR returns, and a 480 × 360 RGB preview. A planner-only Nav2 server used the live SLAM map with the static and inflation layers. A `ComputePathToPose` request for (0.95, 0) m returned 14 poses. They matched the observer's path pose for pose, and seven projected path points were visible in the camera overlay. The robot stayed stationary throughout the test.

The browser cleared the overlay when the plan expired and kept showing the camera. When the observer stopped, the same browser showed **Stale** and removed the old camera frame. No browser exceptions occurred during the verified run. This covers the Gazebo sensor, SLAM, planner and console boundary for one stationary pose in the warehouse.

To repeat the sensor pass, source Jazzy and the existing rbot overlay, choose an unused localhost ROS domain and Gazebo partition, and launch the installed small warehouse with 3D LiDAR, depth camera and SLAM enabled. Set `extra_gz_resource_path` to the installed resource directory containing the `rlai_meshes` package so Gazebo can resolve `model://rlai_meshes/...` URIs. Use the observer command above with simulation time enabled and open the bridge's ROS sensor stream. To repeat the path check, activate only the Nav2 planner with the same clock and live `/map`, and issue a planning request within SLAM-observed free space. Physical mount calibration, the real camera and LiDAR, outdoor localization and field sensor outages still require bench or site evidence.
