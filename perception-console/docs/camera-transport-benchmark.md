# Camera transport benchmark

Open **Transport benchmark** from **Camera trial** or visit `/cameras/transport`. Select **Run benchmark** and leave the tab visible. The diagnostic uses browser-generated video of the same simulated doorway crossing. It creates a fresh pair of WebRTC peers for each profile. Signaling stays in the page and the peers use local ICE candidates. The diagnostic needs no camera permission or external media service. [WebRTC canvas capture example](https://webrtc.github.io/samples/src/content/capture/canvas-pc/)

| Profile | Front | Rear | Left | Right |
| --- | ---: | ---: | ---: | ---: |
| Front and rear | 360 | 360 | — | — |
| Four equal feeds | 180 | 180 | 180 | 180 |
| Front prioritized | 360 | 120 | 120 | 120 |

Caps are in kb/s and sum to 720 for each profile. The source requests 512 × 288 video at up to 12 frames/s for each active camera. After warm-up, each profile uses a 12-second replay window. The caps configure the senders. Actual video payload rate comes from the change in outbound RTP bytes over the corresponding stats interval and excludes network overhead. The active codec, decoded frame size and mean receiver jitter-buffer residence time are reported per track. [WebRTC sender parameters](https://www.w3.org/TR/webrtc/#dom-rtcrtpencodingparameters-maxbitrate) · [WebRTC statistics](https://www.w3.org/TR/webrtc-stats/)

A luminance marker identifies the source frame through encoding and decoding. The timestamp is taken immediately before the source requests capture. A receiver reads the marker at `requestVideoFrameCallback` and subtracts the source timestamp from the expected display time, using a shared browser clock. P50 and P95 describe matched frames whose timestamps fall in the measurement window. Compare them with the matched and submitted frame counts. Low coverage can hide missing or late frames. The browser estimate covers transmission through expected composition and excludes scene rendering before capture and physical display scanout. [Video frame callback timing](https://wicg.github.io/video-rvfc/)

After the prioritized measurement window, the source stops drawing and requesting the left track for three seconds. A feed becomes interrupted after 800 ms without a newly matched presentation. A feed that still advances but whose latest frame is at least 500 ms old shows **Delayed video**. The drill records time from pause to interruption, time from resume to the next newly generated frame reaching expected display, and the number of new frames across the other three feeds during the pause. Recovery times out after five seconds. **Download JSON** exports the per-feed measurements, drill outcomes and browser identification locally.

## Local measurements

Three consecutive runs used headless Edge 158 at 1440 × 900 with AMD Radeon graphics through ANGLE/D3D11. All tracks decoded at 512 × 288 with VP8. At least 99% of submitted frames per feed matched inside each measurement window.

| Profile | Total payload (kb/s) | Front payload (kb/s) | P95 across track results (ms) |
| --- | ---: | ---: | ---: |
| Front and rear | 665–675 | 338–342 | 40–63 |
| Four equal feeds | 694–700 | 173–174 | 41–66 |
| Front prioritized | 690–693 | 340–341 | 42–90 |

Each cell gives the observed range across repeated runs, with latency spanning the per-track P95 values. Prioritization roughly doubled front payload versus four equal feeds at comparable total payload. The left feed flagged after 824–882 ms and presented a new frame 59–89 ms after resume. Each of the other three feeds presented 32–36 new frames during the interruption.

In a separate browser check, the test harness withheld left-frame capture after resume. The benchmark reported the five-second recovery timeout and retained all three profiles in the exported results.

Earlier diagnostic runs forced software rendering and observed frames more than 700 ms old while video still advanced. The table uses hardware-rendered runs without that diagnostic instrumentation. Rendering and decoding capacity on the operator device should be part of the next test.

## What to test next

The [two-process camera link benchmark](camera-link-benchmark.md) exercises a front-to-rear priority switch and UDP impairment on one host. Carry the same experiment to separate robot-side and operator-side devices with actual camera streams. Compare stable throughput, latency tails, image detail during motion, and failed freezes or reconnects while moving the priority between directions. Measure WAN latency with a shared visible timing reference or calibrated clock synchronization. A join or reconnect also needs a first-frame time measurement.

Use participant comparison results and camera coverage around a loaded chassis to choose which views are needed for each maneuver. This local diagnostic measures video transport behavior for a shared process and a synthetic scene.
