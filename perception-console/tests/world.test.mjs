import assert from 'node:assert/strict';
import test from 'node:test';
import { Occupancy, transformPoint, acceptSample, associateObservations, Tracker } from '../server/perception.mjs';
import { scenarios, createSimulation, buildRecording } from '../server/simulation.mjs';

test('timestamp and frame checks reject invalid, future, foreign and stale samples', () => {
  const sample = { frame: 'sim_warehouse', stampMs: 900 };
  const window = { frame: 'sim_warehouse', nowMs: 1000, maxAgeMs: 200 };
  assert.equal(acceptSample(sample, window), true);
  for (const stampMs of [NaN, -1, 1001, 799]) {
    assert.equal(acceptSample({ ...sample, stampMs }, window), false);
  }
  assert.equal(acceptSample({ ...sample, frame: 'other' }, window), false);
  assert.equal(acceptSample(null, window), false);
});

test('transform applies a metre-space translation and ROS-positive yaw', () => {
  const [x, y, z] = transformPoint([2, 0, 3], { x: 1, y: 4, z: 0.5, yaw: Math.PI / 2 });
  assert.ok(Math.abs(x - 1) < 1e-9);
  assert.equal(y, 6);
  assert.equal(z, 3.5);
});

test('rays leave unobserved space unknown and occupied endpoints win over free evidence', () => {
  const grid = new Occupancy([0, 0], 8, 3, 1);
  grid.observe([0.5, 1.5], [3.5, 1.5], true);
  grid.observe([0.5, 1.5], [6.5, 1.5], false);
  assert.equal(grid.cells[8], 0);
  assert.equal(grid.cells[11], 100);
  assert.equal(grid.cells[15], -1);
  assert.equal(grid.cells[20], -1);
});

const geometry = { source: 'lidar', frame: 'sim_warehouse', stampMs: 900, position: [1, 3, 0], size: [1, 1, 1] };
const semantic = { ...geometry, source: 'rgbd', label: 'Person', confidence: 0.8 };

test('semantic fusion preserves geometry when class or time alignment is uncertain', () => {
  const context = { frame: 'sim_warehouse', nowMs: 1000, maxAgeMs: 200 };
  assert.equal(associateObservations([geometry], [semantic], context)[0].label, 'Person');
  assert.equal(associateObservations([geometry], [{ ...semantic, stampMs: 1000 }], context)[0].label, 'Unknown');
  assert.equal(associateObservations([geometry], [semantic, { ...semantic, label: 'Cart' }], context)[0].label, 'Unknown');
  const located = associateObservations([], [semantic], context);
  assert.equal(located[0].label, 'Person');
  assert.deepEqual(located[0].sources, ['rgbd']);
});

test('a track survives a brief occlusion, estimates velocity and expires', () => {
  const tracker = new Tracker();
  const first = tracker.update([{ ...geometry, stampMs: 0, label: 'Unknown', sources: ['lidar'] }], 0)[0];
  const second = tracker.update([{ ...semantic, stampMs: 100, position: [1.2, 3, 0], sources: ['rgbd'] }], 100)[0];
  assert.equal(second.id, first.id);
  assert.ok(second.velocity[0] > 0);
  assert.equal(tracker.update([], 400)[0].status, 'coasting');
  assert.equal(tracker.update([], 1000).length, 0);
});

test('all eleven scenarios produce deterministic world frames with independent faults', () => {
  assert.equal(scenarios.length, 11);
  for (const { id } of scenarios) {
    const a = createSimulation(id);
    const b = createSimulation(id);
    for (let i = 0; i < 82; i++) assert.deepEqual(a.next(), b.next());
  }
  const frameAt = (id, index) => {
    const simulation = createSimulation(id);
    let frame;
    for (let i = 0; i <= index; i++) frame = simulation.next();
    return frame;
  };
  const clear = frameAt('clear', 60);
  assert.equal(clear.mode, 'SIMULATION');
  assert.ok(clear.robot.pose.x > frameAt('clear', 0).robot.pose.x + 2);
  assert.ok(clear.nav.history.length > 2);
  assert.ok(clear.nav.prediction.length > 2);
  const initialPallet = frameAt('clear', 0).tracks.find(track => track.label === 'Pallet');
  assert.ok(initialPallet);
  assert.equal(clear.tracks.find(track => track.id === initialPallet.id).label, 'Unknown');
  assert.ok(frameAt('pedestrian', 60).tracks.some(track => track.label === 'Person' && Math.abs(track.velocity[1]) > 0.1));
  assert.ok(frameAt('forklift', 60).tracks.some(track => track.label === 'Forklift'));
  assert.ok(frameAt('blocked', 70).tracks.some(track => track.label === 'Unknown'));
  assert.equal(frameAt('blocked', 70).robot.speed, 0);
  assert.equal(frameAt('camera-loss', 80).sensors.find(sensor => sensor.id === 'rgbd').status, 'stale');
  const noLidar = frameAt('lidar-loss', 80);
  assert.equal(noLidar.cloud, null);
  assert.ok(noLidar.occupancy.cells.every(cell => cell === -1));
  assert.equal(frameAt('localization', 80).robot.localization.status, 'uncertain');
  assert.equal(frameAt('assistance', 80).nav.state, 'Assistance requested');
  assert.equal(frameAt('paused', 80).nav.state, 'Paused');
  assert.ok(frameAt('direction', 85).tracks.some(track => track.label === 'Cart' && track.velocity[1] < 0));
});

test('outdoor and indoor fixtures share the vehicle profile while preserving distinct site frames and routes', () => {
  const indoor = createSimulation('clear').next();
  const outdoor = createSimulation('yard').next();
  assert.equal(indoor.version, outdoor.version);
  assert.equal(indoor.robot.modelId, outdoor.robot.modelId);
  assert.ok(outdoor.robot.modelId);
  assert.notEqual(indoor.site, outdoor.site);
  assert.equal(outdoor.frame, 'sim_yard');
  assert.equal(outdoor.cloud.frame, outdoor.frame);
  assert.notDeepEqual(indoor.nav.planned, outdoor.nav.planned);
  assert.equal(outdoor.nav.destination, 'Gate 02');
});

test('recording frames preserve their acquisition stamps and event positions through replay', () => {
  const recording = buildRecording('lidar-loss');
  assert.equal(recording.sourceMode, 'SIMULATION');
  assert.equal(recording.frames[0].stampMs, 0);
  assert.equal(recording.frames.at(-1).stampMs, recording.durationMs);
  assert.ok(recording.frames.every((frame, i) => frame.stampMs === i * recording.intervalMs));
  assert.ok(recording.events.some(event => event.code === 'LIDAR_STALE' && event.stampMs > 0));
});
