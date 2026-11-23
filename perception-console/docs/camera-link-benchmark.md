# Camera priority on a constrained link

The [single-browser benchmark](camera-transport-benchmark.md) measured three camera arrangements on loopback. This diagnostic runs the same doorway clip between two independent headless Chromium browser processes on one machine. The sender publishes four authentic WebRTC video tracks over UDP. The receiver matches the video markers to the submitted frames and measures presentation in a shared host clock. The script brackets both browser clocks against its own and fails if their combined check exceeds 15 ms.

| Condition | Uplink (kb/s) | Encoder cap sum (kb/s) | Delay added each way | Configured random loss |
| --- | ---: | ---: | ---: | ---: |
| Unshaped | Host link | 720 | None | None |
| Constrained | 420 | 720 | 70 ms | 1% |
| Headroom | 420 | 330 | 70 ms | 1% |

The shaped runs apply Chromium's packet emulator to the sending page. It delays packets in both directions, caps sender upload and bounds the queue at 20 packets. The observed peer RTT must reflect configured delay. The encoder gives half of its budget to the selected view and splits the rest equally. Before raising the new view's cap, the sender lowers the previous cap. Each profile measures a 12-second clip with front prioritized, then repeats the clip with rear prioritized after a four-second settling interval on the same peer connection. [Chromium packet emulator implementation](https://chromium.googlesource.com/chromium/src/%2B/main/services/network/throttling/throttling_p2p_network_interceptor.cc) · [WebRTC encoding parameters](https://www.w3.org/TR/webrtc/#dom-rtcrtpencodingparameters-maxbitrate)

The script reports transmitted and received RTP video payload, matched and submitted frame counts, and P50/P95 time from source submission to expected composition. A frame must reach the receiver inside the same measurement window to count as matched. Compare latency with coverage because delayed frames can miss the window. Received payload excludes transport overhead. The first new rear frame after the switch measures continuity. The steady window measures where the bitrate goes. [Frame presentation API](https://wicg.github.io/video-rvfc/)

In the headroom condition, the script then sets loss to 100%. After allowing 1.2 seconds for buffered video, it verifies the receiver presents zero new frames for 1.5 seconds while the source keeps submitting. It restores the same constrained profile and records each track's first fresh frame after a conservative source sequence cutoff. This verifies that packet loss actually reaches the media path.

## Results

Two complete runs used Edge 158 with AMD Radeon graphics through ANGLE/D3D11. The browser clock checks bounded their combined uncertainty to 3.1 ms. All eight measured windows on the constrained link used 512 × 288 VP8 video. The table spans both priority directions and both runs.

| Condition | Selected received payload (kb/s) | Each auxiliary (kb/s) | Selected P95 presentation (ms) | Matched per feed |
| --- | ---: | ---: | ---: | ---: |
| Unshaped, 720 encoder cap | 357–363 | 117–124 | 52–70 | 99.3–100% |
| 420 uplink, 720 encoder cap | 77–90 | 78–85 | 266–388 | 92.4–97.9% |
| 420 uplink, 330 encoder cap | 151–163 | 50–59 | 256–478 | 91.7–97.9% |

With the larger cap, congestion left each camera with roughly the same payload rate. With headroom, the selected camera received 2.8–3.0 times the mean auxiliary rate. The tail latency ranges overlap, and the headroom setting had the higher observed maximum for the selected view. Treat the encoder cap as a budget and assess latency and image usability separately.

All switches kept the same tracks and ICE pair. Cap updates took 2.1–3.9 ms. The first new rear frame appeared 219–374 ms after application in the shaped cases. Shaped UDP routes showed 151–189 ms RTT, compared with 1 ms on the unshaped routes. The two full-loss drills each received zero new frames during the quiet window while the source submitted 18–19 per track. All four tracks showed a fresh frame 1.02–1.21 seconds after restoring the link.

For the next prototype, move priority by keeping the active stream and reducing auxiliary frame rate or resolution as available uplink falls. Measure whether the active camera stays readable and whether its frame delay and recovery meet the maneuver's needs. The budgets above are test points for a single scene and device.

## Reproduction

Run `node benchmarks/camera-link.mjs` from the perception console directory after installing the console dependencies. The diagnostic uses a locally available Playwright installation and Microsoft Edge. An absolute package path can be supplied in `RBW_PLAYWRIGHT_MODULE`, or a local installation can be used with `npm install --no-save --package-lock=false playwright`. Set `RBW_CHROMIUM_CHANNEL=chrome` to use Google Chrome. The temporary Vite server binds loopback, and the runner closes the server and browser processes. Standard output contains JSON measurements and standard error shows progress.

The simulation uses a short clip and two browser processes on one clock. Tests with separate machines, actual camera image detail, uplink variability and operator performance are needed before setting production defaults.
