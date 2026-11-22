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
  await writeFile(file, JSON.stringify({ ...snapshot, sequence: 2 }));
  let raw = '';
  while (!raw.includes('\n\n')) raw += (await reader.read()).value;
  const frame = JSON.parse(raw.split('\ndata: ')[1].trim());
  await reader.cancel();
  assert.equal(frame.sequence, 2);
  assert.deepEqual(frame.cloud.points, snapshot.cloud.points);
});
