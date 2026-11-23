# Camera perception roadmap

## What Tesla has described

Tesla announced the start of the **FSD (Supervised) v15** rollout on November 21, 2026, including **Tesla Reality View**. Tesla describes the display as a live generated view. Elon Musk described a conditional mean-flow video model that uses features from the same vision encoder as the driving network. This is a company description of the display. The full production architecture remains proprietary. [Tesla rollout announcement](https://x.com/Tesla/status/2124222447855743019) · [Tesla v15 release notes](https://x.com/TeslaSI/status/2124229673056113100) · [Musk's technical description](https://x.com/elonmusk/status/2124458557525397780)

Tesla's public engineering overview describes per-camera detection, segmentation and monocular depth, plus a multiview network for top-down geometry. Tesla AI also discussed generative Gaussian splatting in its 2025 ICCV presentation. These sources establish the company's work on scene reconstruction and simulation. The public v15 description leaves the use of an explicit splat representation in the shipping display unconfirmed. [Tesla engineering overview](https://www.tesla.com/SI) · [ICCV presentation linked by Tesla](https://www.tesla.com/fsd-evidence-dashboard)

## Built and verified

The existing ROS observer can now subscribe to a depth image alongside RGB, LiDAR, TF and the SLAM map. Select **ROS sensor stream**, then **Front depth**. The panel shows distance along the depth camera's optical axis. The colour scale uses metres, and unknown pixels have a separate swatch. Each source can go stale independently. The console keeps the current SLAM and Nav2 view available for operators.

Use the [observer command](ros-stream-plan.md#running-the-observer) with `--depth-topic /depth_camera/depth` in the existing Gazebo warehouse. The observer accepts `32FC1` metres and `16UC1` millimetres, checks row layout and byte order, and samples up to 160 pixels on the longest side without interpolating unknown depth. The view is a bounded preview in the depth optical frame. [ROS depth image specification](https://github.com/ros-infrastructure/rep/blob/master/rep-0118.rst)

Gazebo supplied a 640 × 480 `32FC1` depth image. One browser preview contained 15,170 valid and 4,030 unknown samples at 160 × 120. The sampled valid values ranged from about 0.25 to 5.67 m. With only depth paused, the view cleared and RGB and LiDAR continued. Depth recovered when resumed. The desktop and 390-pixel-wide browser checks passed. Evidence applies to the Gazebo warehouse and its simulated sensor.

## Ranged objects in the console

The observer accepts the existing ROS `Detection3DArray` topic with `--objects-topic /amr/detections_3d`. In the console, **Objects** lists the class, horizontal range from the robot and producer class score. The map shows a marker at the registered location. Each item is a current observation. The console uses stamped TF to get the position in the map frame and the range in the robot base frame. Missing or expired observations clear independently of RGB, depth and LiDAR.

Run the upstream depth fusion with an RGB image and depth already registered into the same optical geometry. The existing producer scales pixel coordinates between resolutions. Separate optical frames and camera models need real registration before fusion, even when both images have the same dimensions. Use valid neighbouring depth pixels and retain the original timestamps. [ROS depth_image_proc](https://github.com/ros-perception/image_pipeline/tree/jazzy/depth_image_proc)

In an isolated Jazzy test, the existing fusion node received a prescribed `person` detection at pixel (283, 135), synthetic depth patches at 2 m and 4 m, and matching optical calibration. The camera was 0.4 m ahead of the robot origin, and the map pose included a 90° yaw and a translation. The console showed horizontal ranges of 2.43 m and 4.47 m, within 1 mm of the fixture calculation. The map positions matched within 2 mm. Invalid depth removed the observation. Detection dropout cleared its marker and range while RGB, depth and LiDAR continued. The verified path covers fusion, timestamped TF, the read-only bridge and desktop/mobile rendering. Detector accuracy still needs a test with model-produced boxes.

## Next increments

1. **Detector verification.** Feed real detector results from a labeled simulation or synchronized replay with registered depth. Compare recognition and range against ground truth across occlusion, distance and lighting. Record the checkpoint and separate classification errors from range errors.
2. **Camera coverage and operator layout.** Use the same controls across robot profiles, with each feed showing its own availability. Compare a focused layout with a grid on the same recorded tasks and test link degradation. [Teleoperator camera layout and evaluation](teleoperator-camera-layout.md)
3. **Camera-only benchmark and reconstruction.** Evaluate a small monocular model offline against synchronized depth in indoor and outdoor data. Label relative depth or metric estimates with their actual provenance. Compare stability, missing-surface errors and latency before using estimates in an operator view. Keep Gaussian splats in the [site survey](site-survey-pilot.md) for optional appearance and replay. [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
