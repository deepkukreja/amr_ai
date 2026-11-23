# Camera perception roadmap

## What Tesla has described

Tesla announced the start of the **FSD (Supervised) v15** rollout on November 21, 2026, including **Tesla Reality View**. Tesla describes the display as a live generated view. Elon Musk described a conditional mean-flow video model that uses features from the same vision encoder as the driving network. This is a company description of the display. The full production architecture remains proprietary. [Tesla rollout announcement](https://x.com/Tesla/status/2124222447855743019) · [Tesla v15 release notes](https://x.com/TeslaSI/status/2124229673056113100) · [Musk's technical description](https://x.com/elonmusk/status/2124458557525397780)

Tesla's public engineering overview describes per-camera detection, segmentation and monocular depth, plus a multiview network for top-down geometry. Tesla AI also discussed generative Gaussian splatting in its 2025 ICCV presentation. These sources establish the company's work on scene reconstruction and simulation. The public v15 description leaves the use of an explicit splat representation in the shipping display unconfirmed. [Tesla engineering overview](https://www.tesla.com/SI) · [ICCV presentation linked by Tesla](https://www.tesla.com/fsd-evidence-dashboard)

## Built and verified

The existing ROS observer can now subscribe to a depth image alongside RGB, LiDAR, TF and the SLAM map. Select **ROS sensor stream**, then **Front depth**. The panel shows distance along the depth camera's optical axis. The colour scale uses metres, and unknown pixels have a separate swatch. Each source can go stale independently. The console keeps the current SLAM and Nav2 view available for operators.

Use the [observer command](ros-stream-plan.md#running-the-observer) with `--depth-topic /depth_camera/depth` in the existing Gazebo warehouse. The observer accepts `32FC1` metres and `16UC1` millimetres, checks row layout and byte order, and samples up to 160 pixels on the longest side without interpolating unknown depth. The view is a bounded preview in the depth optical frame. [ROS depth image specification](https://github.com/ros-infrastructure/rep/blob/master/rep-0118.rst)

Gazebo supplied a 640 × 480 `32FC1` depth image. One browser preview contained 15,170 valid and 4,030 unknown samples at 160 × 120. The sampled valid values ranged from about 0.25 to 5.67 m. With only depth paused, the view cleared and RGB and LiDAR continued. Depth recovered when resumed. The desktop and 390-pixel-wide browser checks passed. Evidence applies to the Gazebo warehouse and its simulated sensor.

## Next increments

1. **Range for detected objects.** Reuse the current detector interface. Start with a class supported by the deployed checkpoint, such as person. Register depth into the RGB optical frame, match timestamps and intrinsics, and estimate object range from supported pixels inside the detection. Keep class confidence, range validity and original stamps. Verify against known objects at multiple simulated distances, including occlusion and an input dropout. ROS already provides components for registration and point-cloud projection. [ROS depth_image_proc](https://github.com/ros-perception/image_pipeline/tree/jazzy/depth_image_proc)
2. **A common top-down view.** Register verified observations to the SLAM frame and place them beside current LiDAR, occupancy and paths. Use the same operator controls across robot profiles, with source health showing which inputs are available. Preserve unknown space and keep the display read-only.
3. **Camera-only benchmark and reconstruction.** Evaluate a small monocular model offline against synchronized depth in indoor and outdoor data. Label relative depth or metric estimates with their actual provenance. Compare stability, missing-surface errors and latency before using estimates in an operator view. Keep Gaussian splats in the [site survey](site-survey-pilot.md) for optional appearance and replay. [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
