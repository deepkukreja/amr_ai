# Gaussian splat site-view pilot

Open **Site survey** from the console or go to `/survey`. Approach, Threshold and Yard apply the same camera pose to both renderers. Use the renderer controls to compare or inspect one view by itself.

## Results

Six simulated RGB-D viewpoints produced 62,797 ray hits and 27,838 unique samples on a 6.5 cm voxel grid. The Three.js point cloud and Spark Gaussian renderer use the same sample positions and source colors. The three synthetic doorway landmarks matched the known fixture within numerical precision. This measures consistency in the simulated frame.

| Threshold view | RGB points | Gaussian splats |
| --- | ---: | ---: |
| Visible canvas coverage | 25.64% | 58.93% |
| Median browser time to renderer ready | 752 ms | 1,108 ms |

Coverage counts pixels whose most different RGB channel differs from the background by over 18 levels. The two 684 × 456 canvases used the same camera. Coverage varies with the chosen point size, Gaussian footprint and opacity. Larger footprints filled gaps in the lane markings, doorway and crates and softened some edges. Coverage describes rendered appearance.

Startup timings are five fresh-context loads of the comparison page in headless Edge with SwiftShader software rendering. These timings include page, fixture and renderer startup. Spark added 2.80 MB on the uncompressed local bridge. The main console loaded without the survey or Spark chunks. Browser checks covered synchronized viewpoints, individual renderers, stacked mobile panels, and point rendering during a failed Spark load. These are desktop browser measurements with no onboard GPU or physical sensor capture.

## Product decision and field trial

Keep the shared operator view centered on the robot's SLAM map, pose, planned path and live LiDAR or camera inputs as available. Different robot profiles and sites can supply the layers they have. The point cloud remains the lighter geometry view. An optional registered Gaussian asset is worth a field trial for operator orientation and replay at visually confusing entrances and outdoor transitions. This pilot uses Gaussians initialized analytically from synthetic depth. A real survey will test a reconstruction trained from capture.

1. Capture one approved indoor–outdoor threshold from several angles, including an angle withheld for evaluation. Record calibrated camera intrinsics and metric poses, and build a colored point-cloud baseline from synchronized depth or LiDAR where available.
2. Register both representations to the same site map ID and revision. Use independent surveyed landmarks to measure alignment and inspect occlusion, fine edges and lighting changes.
3. Compare whether operators can recognize landmarks and changes in layout. Measure first render, camera-movement latency, asset size and GPU memory on the target operator workstation. Enable the optional layer where the visual improvement justifies the cost. Date each capture so operators can distinguish survey appearance from live observations.

### What to hand over from the first site

- One local ROS bag or equivalent synchronized export with original RGB, matching camera intrinsics and distortion, and timestamped camera poses or the transforms needed to resolve the optical frame in the site map. Include aligned depth or LiDAR and its calibration for the colored point-cloud baseline. Record the acquisition timestamps and clock domain. The browser's downsampled previews are unsuitable as source data for this comparison.
- The exact SLAM map used for the recording, including frame, origin, resolution and map revision. Include which camera and sensor mounts were used. Keep at least one overlapping view out of reconstruction and include independently measured doorway landmarks for the alignment check.
- Capture date, site ID, and the names of the topics or files in the recording. Keep raw captures local and publish aggregate alignment and rendering results.

The [stationary Gazebo test](ros-stream-plan.md#stationary-gazebo-integration) proved that the running simulator's map, filtered LiDAR and calibrated camera reach the console. Site registration and learned splat reconstruction need the source recording above.

## References

- Cartken describes camera-based perception, mapping and remote assistance. Its RealSense case study describes depth-based visual SLAM with monocular and stereo features and semantics. The published sources leave the display representation unspecified. [Cartken](https://www.cartken.com/) · [RealSense case study](https://www.realsenseai.com/case-studies/cartken/)
- The original Gaussian splatting work describes optimizing anisotropic Gaussians from calibrated images. Spark supplies the browser renderer and on-demand rendering used in this pilot. [3D Gaussian Splatting paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) · [Spark on-demand rendering](https://sparkjs.dev/docs/on-demand-rendering/)
