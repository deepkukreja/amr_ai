# Teleoperator camera layout

Support front, rear, left and right as configurable camera slots across robot profiles. Start by proving coverage with front and rear, adding side views where payload overhang, tight docking or traffic at aisle exits creates a blind area. Camera field of view, mounting height, near-ground coverage and the loaded robot's outline determine which positions earn a place. Four streams alone leave coverage unverified.

The default desktop layout should show **one large operator-selected feed**, direction-labelled thumbnails for the other available views, and a small map with the robot's heading. Front is the default. Rear is one action away during reversing, and the operator can pin a side view for docking or inspecting clearance. Provide an on-demand **2×2 grid** to inspect all available directions during a stop or a tight maneuver. Keep feed orientation and position consistent. A small robot reference in the image can help operators judge clearance. [Teleoperation camera-placement study](https://www.cs.uml.edu/~holly/papers/YancoMaxwellSSRR2006.pdf)

For the media prototype, use independently timestamped video tracks with a higher-quality focused feed and lower-bandwidth auxiliary views. Report age and degraded or missing status per feed. Under constrained bandwidth, preserve the selected feed first and reduce the quality or rate of the auxiliary streams. Evaluate WebRTC video separately from the existing sensor snapshot stream. Keep local motion supervision independent of browser video. [WebRTC specification](https://www.w3.org/TR/webrtc/) · [Cartken remote assistance](https://www.cartken.com/)

## Decision test

1. Mark visible floor and cargo-edge coverage for each camera pose with both an empty and loaded footprint. Run the same exercise indoors and against outdoor glare or shadow. Record which areas are visible only from the side cameras.
2. Compare front/rear, focused-plus-thumbnails and four equal tiles on the same replay of a reverse dock, a doorway with cross-traffic and a close pass beside the payload. Measure time to identify the hazard, missed hazards, time to switch views, and operator errors.
3. Repeat with reduced bandwidth and with one frozen feed. Measure capture-to-display delay, frame age and recovery per feed. Qualify the design only for the tasks, visibility and link conditions measured.

The current implementation covers the single front camera and separate map, depth and ranged observations. The four-view proposal is a design for a read-only simulation or replay trial.
