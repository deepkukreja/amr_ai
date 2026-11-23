# Robot Buildworks perception console

A read-only browser view for AMR perception and operations. Warehouse and outdoor loading-yard scenarios use the same D3 robot profile, world stream and operator layout. The viewport also accepts an older Gazebo route replay as recorded, including its map, truth pose, wheel speed and localization covariance.

## Run locally

With Node.js 22.12 or newer and a WebGL 2 browser, run these commands from this directory.

```sh
npm ci
npm test
npm run build
npm start
```

Open [the console](http://127.0.0.1:4180/). Choose an indoor scenario, the outdoor loading yard or an available recording using **Source**. **View** selects Follow, Bird’s-eye, Point cloud, Semantic, Occupancy or Debug. The inspector switches between robot state, object tracks and sensor diagnostics. **Freeze view** pauses the browser view. Playback adds seek, speed and event markers.

The server listens on IPv4 loopback. Set `RBW_PORT` to use another port. There are no command endpoints.

## Existing Gazebo evidence

The route replay entry appears only when the local route evidence is present. The bridge reads the existing Windows delivery in `../outputs/route_validation` when available. To use it from a different checkout, set `RBW_ROUTE_EVIDENCE` to the local `route_validation` directory before starting. It expects `evidence/nominal_91` and `reference_configuration/map` under that directory. The recorded source remains Gazebo simulation. Its reference map is downsampled conservatively for viewing.

## Engineering notes

- [System architecture and project inventory](docs/system-architecture.md)
- [Integration contracts and ROS topics](docs/integration-contracts.md)
- [Indoor/outdoor operator-view research](docs/operator-view-research.md)
- [Development status and acceptance gaps](docs/development-status.md)
- [Recorded Nav2 plan view and ROS capture](docs/nav2-plan-view.md)
- [Continuous ROS stream, Gaussian splatting research and integration plan](docs/ros-stream-plan.md)
- [Camera perception roadmap and verified front depth view](docs/camera-perception-roadmap.md)
- [Four-camera replay trial and teleoperator evaluation](docs/teleoperator-camera-layout.md#replay-trial)

Simulation provides generated sensor observations and scripted scenario events. The console also supports replay and an opt-in read-only ROS stream for maps, navigation paths, LiDAR, RGB and metric depth previews. Physical sensor and controller validation are tracked in the integration plan.
