# One operator view for indoor and outdoor AMRs

## Decision

Use one operating view with a robot selector supplied by the fleet platform, one workspace viewport and a small inspector with Overview, Objects and Diagnostics. Show the robot, its current reference frame, the active route and the most important operational issue first. Keep perspective and map controls in the viewport and expose time navigation whenever the source is a recording.

The viewport takes its robot representation from a versioned robot profile. Prefer the released URDF and stamped joint transforms once available. A footprint profile can render valid geometry without joint feedback. When neither is verified, use a heading marker. The prototype exercises the dimensioned D3 body in simulation and the pose-only fallback for an older differential-drive Gazebo capture. The app shell works independently of the robot geometry.

| Existing system | Pattern to reuse | Robot Buildworks choice |
| --- | --- | --- |
| Foxglove | TF-aware 3D and URDF rendering, separate geo-map channel and layouts independent of where messages came from. | Keep synchronized source time and frame IDs in the operator view. Expose detailed TF, images and channels through standard ROS or Foxglove schemas for engineering. |
| Open-RMF Web | Fleet, task and map concerns sit above individual robot adapters. | Keep identity and mission ownership in the existing fleet backend. Make the operator view reusable at one robot route, with fleet comparison as a separate view. |
| Rerun | Spatial data, timelines and diagnostics use a data stream distinct from its viewer blueprint. | Share data across viewport, replay and diagnostics. Use Rerun/Foxglove for richer analysis when the standard tools suffice. |
| Robot Web Tools | ROS messages can be rendered in a browser independently of Gazebo or RViz. | Avoid granting the browser direct general-purpose ROS access. Give it a curated read-only world contract. |

Primary references include [Foxglove 3D and URDF](https://docs.foxglove.dev/docs/visualization/panels/3d), [Foxglove map view](https://docs.foxglove.dev/docs/visualization/panels/map), [Open-RMF Web](https://github.com/open-rmf/rmf-web), [Rerun](https://github.com/rerun-io/rerun), and [Robot Web Tools](https://github.com/RobotWebTools/ros3djs).

## Reference frames and profiles

| Situation | Display contract |
| --- | --- |
| Indoor mapped operation | Use the configured facility, level and map version. Display map occupancy in the map frame. Use odometry for local tracks and register them into the map only when the source transform is valid. |
| Outdoor site with valid global localization | Keep the perception viewport in metric local coordinates, typically configured ENU. Accept a separate georeferencing layer with explicit datum, altitude reference and transform provenance. Use GNSS fix/covariance age and configured localization quality before showing the global overlay. |
| Outdoor without a valid geo-fix, unmapped operation, or indoor/outdoor transition | Keep the robot-relative or odom view and label its frame. Hide the geographic overlay or discontinuous map path until valid registration returns. Retain unknown terrain, and avoid bridging pose history across source, map or localization resets. |
| Multiple robots | Identify each snapshot by robot, session, site and frame. Only overlay robots with verified transforms into the same site map revision or datum. Keep robot-local frames separate otherwise. |

The outdoor loading yard runs locally using the same robot profile, observations and world contract as the indoor scenario. It has its own metric frame, layout and route. It requires neither GPS nor map tiles. For a later documented geographic layer, use the upstream localization transform and consider MapLibre when an approved GeoJSON or tile source and offline policy exist. Geographic basemaps are context. Occupancy and free-space evidence still come from perception or the approved site map. [Nav2 localization in Jazzy](https://docs.nav2.org/jazzy/configuration_and_development/first_time_robot_setup_guide/odom/setup_robot_localization/) · [Nav2 GPS navigation concepts](https://docs.nav2.org/jazzy/getting_started/navigation_concepts/navigation_servers/) · [MapLibre GL JS](https://github.com/maplibre/maplibre-gl-js)

## Acceptance for the next live adapter

- Use the same read-only UI and world fields across two real robot profiles. Include stamps and freshness for each missing or available channel.
- Replay a warehouse capture and an outdoor capture with datum loss, coordinate corrections and site/level changes. Confirm no cross-frame track or path overlay.
- Keep optional live RGB/depth streams and model diagnostics under their capability profile. Geographic tiles, GNSS and video must fail independently of the local world stream.
- Measure latency and coverage in outdoor glare, precipitation and changing terrain before using any perception result for motion decisions.
