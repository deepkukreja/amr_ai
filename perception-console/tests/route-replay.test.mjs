import assert from 'node:assert/strict';
import test from 'node:test';
import { routeRecording, decodeReferenceMap } from '../server/route-replay.mjs';

test('recorded Gazebo truth keeps ROS timestamps and missing hardware telemetry', () => {
  const truth = [0, 0.2, 0.4, 0.6].map((offset, i) => ({
    stamp: 10 + offset, at: 120 + offset, x: i / 10, y: 0, yaw: 0, stage: 'route',
  }));
  const recording = routeRecording({
    truth, wheel: [{ stamp: 10.2, speed: 0.42 }],
    localization: [{ stamp: 10.1, max_xy_variance: 0.02 }],
    goals: [{ goal: [0.3, 0, 0], accepted: true, score: { passed: true }, completion_ros_time: 10.4 }],
    occupancy: { width: 1, height: 1, origin: [0, 0], resolution: 0.4, cells: [-1] },
  });
  assert.equal(recording.sourceMode, 'GAZEBO');
  assert.equal(recording.frames[1].sourceStampMs, 10200);
  assert.equal(recording.frames[1].frame, 'map');
  assert.equal(recording.frames[1].robot.speed, 0.42);
  assert.equal(recording.frames[1].robot.battery, null);
  assert.equal(recording.frames[1].robot.steering, null);
  assert.equal(recording.frames[1].robot.localization.varianceM2, 0.02);
  assert.equal(recording.frames[2].robot.speed, null);
  assert.deepEqual(recording.frames[1].nav.planned, []);
  assert.equal(recording.events[0].stampMs, 400);
});

test('PGM map rows are flipped to positive Y and downsampling preserves occupied and unknown space', () => {
  const pgm = Buffer.concat([Buffer.from('P5\n# note\n4 4\n255\n'), Buffer.from([
    254, 254, 205, 254,
    254, 254, 254, 254,
    254,   0, 254, 254,
    254, 254, 254, 254,
  ])]);
  const yaml = 'mode: trinary\nresolution: 0.05\norigin: [-1.5, -2, 0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196';
  const map = decodeReferenceMap(pgm, yaml, 2);
  assert.deepEqual(map, { width: 2, height: 2, resolution: 0.1, origin: [-1.5, -2], cells: [100, 0, 0, -1] });
});
