# Recorded Nav2 plan view

The console can display a real Nav2 path on a saved occupancy map with the robot pose resolved from TF at the same ROS timestamp. The orange-and-white D3 visualization uses the checked prototype wheel locations, cargo deck and sensor mast. In the planner-only fixture it is a design profile placed at the fixture start pose. The fixture retains the upstream planner motion model and footprint.

## Data path

1. An existing map server or SLAM node publishes `nav_msgs/OccupancyGrid`. The reader preserves the origin, map yaw, resolution, dimensions and cell values.
2. The existing planner publishes `nav_msgs/Path`. The reader requires matching map and path frames and the robot transform at the path timestamp. It writes a new local capture file once the three sources line up. A missing transform or frame mismatch prevents capture.
3. The loopback bridge validates the capture and downscales occupancy for WebGL. An aggregate cell remains unknown unless every covered source cell is free or at least one source cell is occupied. The renderer draws the original path coordinates as a ribbon with a centreline. Ribbon width is for visibility.
4. The console identifies the result as replay. Vehicle speed, steering, battery, localization confidence, objects and warnings remain unavailable in a plan-only capture. No ROS write interface is exposed to the browser.

The product shell uses the shared website theme tokens and website wordmark. The map view retains a dark background for path and obstacle contrast.

## Capture a plan from an existing ROS graph

Run the read-only observer in the existing ROS 2 Jazzy environment and use a fresh output filename. Its defaults are `/map`, `/plan` and `base_footprint`. Adjust topics and base frame to match the running graph. When the graph uses a simulation clock, append `--ros-args -p use_sim_time:=true`.

```sh
source /opt/ros/jazzy/setup.bash
python3 perception-console/ros/read_nav2.py /absolute/path/nav2-world.json \
  --site "Small warehouse"
```

Set `RBW_NAV2_CAPTURE` to the capture path before starting the console. The Windows workspace also discovers the original local fixture artifact under `outputs/nav2_console_20261122/nav2-world.json`. Open the console and select **Recorded Nav2 plan** from Source. Capture inputs are local evidence.

## Reproduce the planner-only fixture

Use the existing validated warehouse map YAML/PGM pair and its Nav2 parameter snapshot. Pick an unused localhost ROS domain and a fresh output directory. The runner starts its own map and planner servers, supplies a stationary transform, sends a planning action, and stops its owned processes. It removes the live obstacle layer for this static-map fixture, declines unknown traversal and checks that no velocity topics appear.

```sh
source /opt/ros/jazzy/setup.bash
ROS_DOMAIN_ID=119 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
  python3 -B perception-console/ros/verify_planner.py \
  --map "$evidence/reference_configuration/map/nav2_map.yaml" \
  --parameters "$evidence/reference_configuration/nav2_params.yaml" \
  --output /tmp/rbw-nav2-plan-check
```

The fixture checks that the published path matches the planning action result, its endpoints match the selected start and goal, and a missing timestamped transform prevents an overlay. A real robot feed still needs continuous state and freshness, LiDAR/camera integration, and validation of its D3 controller profile.

## Verified example

The localhost-only Jazzy/Fast DDS fixture loaded the saved warehouse map at its native 399 × 400 cells and 0.05 m resolution. Smac Hybrid-A* returned 67 poses over 5.071 m from the stationary TF start to goal (4, −3), and the path subscription matched the planning result pose for pose. The missing-transform check rejected capture and the fixture exposed zero velocity topics. An earlier route to (4, −5) returned no path because the selected goal was occupied. No planner tolerance was relaxed to obtain the passing run.

Thirteen console tests, the production build, and browser checks passed. The browser checks covered the recorded path in perspective and occupancy views, the website wordmark and shared tokens, switching to existing sources, and layout at phone and desktop widths. The browser screenshot and the ROS capture are local evidence.
