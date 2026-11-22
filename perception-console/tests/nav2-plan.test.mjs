import assert from 'node:assert/strict';
import test from 'node:test';
import { planRecording } from '../server/nav2-plan.mjs';

const capture = {
  version: 1, source: 'NAV2_PLANNING_FIXTURE', site: 'Small warehouse', frame: 'map', clock: 'ROS_SYSTEM',
  map: { frame: 'map', stampMs: 10, width: 3, height: 2, resolution: 0.05, origin: [-1, -1], yaw: 0.2, cells: [0, 0, 100, -1, 0, 35] },
  pose: { frame: 'map', stampMs: 20, x: -0.6, y: -0.4, yaw: 0.5 },
  plan: { frame: 'map', stampMs: 20, points: [[-0.6, -0.4, 0], [0.4, 0.2, 0], [1.0, 0.3, 0]] },
};

test('recorded Nav2 plan preserves coordinates, pose stamps, map rotation and absent motion data', () => {
  const recording = planRecording(capture);
  assert.equal(recording.sourceMode, 'NAV2_PLANNING_FIXTURE');
  const [frame] = recording.frames;
  assert.equal(frame.robot.modelId, 'd3-prototype');
  assert.equal(frame.robot.speed, null);
  assert.equal(frame.robot.steering, null);
  assert.equal(frame.frame, 'map');
  assert.equal(frame.sourceStampMs, capture.plan.stampMs);
  assert.equal(frame.occupancy.yaw, 0.2);
  assert.deepEqual(frame.occupancy.cells, [0, 0, 100, -1, 0, -1]);
  assert.deepEqual(frame.nav.planned, capture.plan.points);
  assert.deepEqual(frame.nav.prediction, []);
  assert.deepEqual(frame.nav.history, []);
});

test('rejects inconsistent ROS frames, absent pose, invalid cells and non-finite plan data', () => {
  assert.throws(() => planRecording({ ...capture, plan: { ...capture.plan, frame: 'odom' } }));
  assert.throws(() => planRecording({ ...capture, pose: null }));
  assert.throws(() => planRecording({ ...capture, map: { ...capture.map, cells: [0] } }));
  assert.throws(() => planRecording({ ...capture, plan: { ...capture.plan, points: [[NaN, 0, 0], [0, 0, 0]] } }));
});
