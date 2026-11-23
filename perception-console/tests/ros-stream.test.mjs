import assert from 'node:assert/strict';
import { mkdtemp, rmdir, unlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { setTimeout as delay } from 'node:timers/promises';
import { createBridge } from '../server/bridge.mjs';
import { acceptWorld, sequenceGate } from '../server/ros-stream.mjs';

const snapshot = {
  version: 1, sequence: 1, session: 'fixture-1', mode: 'SIMULATION', source: 'ROS_SENSOR_FIXTURE', clock: 'ROS_SYSTEM',
  site: 'Small warehouse', frame: 'map', stampMs: 1000,
  occupancy: { origin: [0, 0], yaw: 0, resolution: 1, width: 1, height: 1, cells: [0] },
  pose: { x: 0, y: 0, yaw: 0, stampMs: 980 },
  plan: { frame: 'map', stampMs: 900, points: [[0, 0, 0], [2, 0, 0]] },
  cloud: { frame: 'map', stampMs: 970, points: [[1, 0, 0.2]] },
  camera: { frame: 'optical', stampMs: 990, width: 320, height: 180, jpeg: '/9j/2Q==',
    overlay: 'registered', planStampMs: 900, segments: [[[0.1, 0.5], [0.8, 0.6]]] },
  sensors: [{ id: 'lidar', name: 'LiDAR', status: 'fresh', ageMs: 30, stampMs: 970 }],
};

const depth = { frame: 'depth_optical', stampMs: 995, width: 2, height: 1, mm16: '6AMAAA==' };
const objects = { frame: 'map', stampMs: 985, items: [
  { classId: 'person', score: 0.95, rangeM: 2.433, position: [1.4, 0.9, 0.58] },
] };

test('ranged detections require a fresh map frame and valid evidence, without changing other layers', () => {
  assert.equal(acceptWorld(snapshot).objects, null);
  const valid = acceptWorld({ ...snapshot, objects });
  assert.deepEqual(valid.objects, objects);
  assert.deepEqual(valid.tracks, []);
  assert.deepEqual(acceptWorld({ ...snapshot, objects: { ...objects, items: [] } }).objects.items, []);
  for (const invalid of [
    { ...objects, frame: 'odom' },
    { ...objects, stampMs: 1001 },
    { ...objects, items: [{ ...objects.items[0], rangeM: -1 }] },
    { ...objects, items: [{ ...objects.items[0], score: 1.2 }] },
    { ...objects, items: [{ ...objects.items[0], position: [1, 0, Infinity] }] },
  ]) {
    const result = acceptWorld({ ...snapshot, objects: invalid,
      sensors: [{ id: 'objects', name: 'Ranged objects', status: 'fresh' }] });
    assert.equal(result.objects, null);
    assert.equal(result.sensors[0].status, 'unregistered');
    assert.deepEqual(result.camera, valid.camera);
    assert.deepEqual(result.cloud, valid.cloud);
  }
  assert.equal(acceptWorld({ ...snapshot, stampMs: 2200, objects }).objects, null);
});

test('depth is optional, keeps its own optical frame and ages independently of RGB', () => {
  assert.equal(acceptWorld(snapshot).depth, null);
  const valid = acceptWorld({ ...snapshot, depth });
  assert.deepEqual(valid.depth, depth);
  assert.ok(valid.camera);
  assert.ok(valid.cloud);
  for (const candidate of [
    { ...depth, stampMs: 1001 },
    { ...depth, frame: '' },
    { ...depth, width: 161 },
    { ...depth, height: 2 },
    { ...depth, mm16: '6AMAAA!?' },
    { ...depth, width: 3, mm16: 'AAAAAAA\n' },
  ]) {
    const invalid = acceptWorld({ ...snapshot, depth: candidate,
      sensors: [{ id: 'depth', name: 'Depth camera', status: 'fresh' }] });
    assert.equal(invalid.depth, null);
    assert.equal(invalid.sensors[0].status, 'unregistered');
    assert.deepEqual(invalid.camera, valid.camera);
    assert.deepEqual(invalid.cloud, valid.cloud);
  }
  assert.equal(acceptWorld({ ...snapshot, stampMs: 2200, depth }).depth, null);
});

test('an on-disk snapshot alone cannot establish a running source', () => {
  const seen = sequenceGate();
  assert.equal(seen(snapshot), false);
  assert.equal(seen(snapshot), false);
  assert.equal(seen({ ...snapshot, sequence: 2 }), true);
  assert.equal(seen({ ...snapshot, sequence: 1 }), false);
  assert.equal(seen({ ...snapshot, session: 'fixture-2' }), false);
  assert.equal(seen({ ...snapshot, session: 'fixture-2', sequence: 2 }), true);
});

test('the ROS world keeps registered layers and removes only the layer with bad evidence', () => {
  const valid = acceptWorld(snapshot);
  assert.deepEqual(valid.cloud.points, [[1, 0, 0.2]]);
  assert.deepEqual(valid.camera.segments, snapshot.camera.segments);
  assert.equal(valid.robot.speed, null);
  const mismatched = acceptWorld({ ...snapshot, cloud: { ...snapshot.cloud, frame: 'odom' } });
  assert.equal(mismatched.cloud, null);
  assert.equal(mismatched.sensors[0].status, 'unregistered');
  assert.ok(mismatched.camera);
  assert.equal(acceptWorld({ ...snapshot, mode: 'LIVE' }), null);
  assert.equal(acceptWorld({ ...snapshot, camera: { ...snapshot.camera, stampMs: 1001 } }).camera, null);
});

test('the HTTP stream skips an old file and delivers the next advancing snapshot', async t => {
  const directory = await mkdtemp(join(tmpdir(), 'rbw-stream-test-'));
  const file = join(directory, 'snapshot.json');
  await writeFile(file, JSON.stringify(snapshot));
  const server = createBridge({ rosFile: file });
  t.after(async () => {
    server.closeAllConnections();
    server.close();
    await unlink(file);
    await rmdir(directory);
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  const response = await fetch(`${base}/api/v1/world?scenario=ros-world`, { signal: AbortSignal.timeout(4500) });
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  await delay(500);
  await writeFile(file, JSON.stringify({ ...snapshot, sequence: 2, depth, objects }));
  let raw = '';
  while (!raw.includes('\n\n')) raw += (await reader.read()).value;
  const frame = JSON.parse(raw.split('\ndata: ')[1].trim());
  await reader.cancel();
  assert.equal(frame.sequence, 2);
  assert.deepEqual(frame.cloud.points, snapshot.cloud.points);
  assert.deepEqual(frame.depth, depth);
  assert.deepEqual(frame.objects, objects);
});
