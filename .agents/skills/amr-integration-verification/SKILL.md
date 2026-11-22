---
name: amr-integration-verification
description: Verify Robot Buildworks AMR changes that cross a component boundary, such as model to ROS, ROS to fleet dashboard, or recorded evidence to the website. Use when a change affects a producer and consumer or a claim about their integrated behavior.
---

# Verify AMR integration

1. Identify the changed boundary and its consumers. Read the existing domain instructions and the shared AMR handoff before editing. Check the actual launch, producer, schema, and consumer code relevant to that change.
2. State the expected contract from source to consumer. Check only the fields that matter to this change. Typical fields include units, class names, frame and map identity, source clock and age, provenance, and unavailable or fault states. For command interfaces, trace ownership and enforcement to the local supervisor.
3. Exercise one representative input through the affected producer and consumer. Exercise the most relevant failure path, such as stale telemetry, a missing transform, or a rejected command. Reuse existing tests, fixtures, or recorded runs and use the smallest setup that actually crosses the boundary. For a final operator interface, check the result in the normal UI.
4. Record the result in the chat handoff and link the relevant commit, recording, or test. State which boundary was exercised, the environment (fixture, replay, simulation, bench, or hardware), and what remains unverified. A result reused on the website or dashboard must keep its original environment and scope.

For a final AMR interface, read the project theme guide and use the website's design tokens and approved brand assets. Keep functional verification proportional to the task. Skip brand work for test fixtures, diagnostic views, training scripts, and temporary demos.
