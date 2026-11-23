import assert from 'node:assert/strict';
import test from 'node:test';
import { cameraCaps, changePriority, presentationStats } from '../benchmarks/camera-link-metrics.mjs';

test('priority moves within the encoder budget and releases the old cap first', async () => {
  const caps = cameraCaps('front', 720_000);
  assert.deepEqual(caps, { front: 360_000, rear: 120_000, left: 120_000, right: 120_000 });
  const changes = [];
  await changePriority('front', 'rear', 720_000, async (id, cap) => {
    caps[id] = cap;
    changes.push([id, cap]);
    assert.ok(Object.values(caps).reduce((sum, value) => sum + value, 0) <= 720_000);
  });
  assert.deepEqual(changes, [['front', 120_000], ['rear', 360_000]]);
  assert.deepEqual(caps, cameraCaps('rear', 720_000));
  assert.deepEqual(cameraCaps('rear', 330_000), { front: 55_000, rear: 165_000, left: 55_000, right: 55_000 });
  await assert.rejects(changePriority('front', 'rear', 720_000, async (id) => {
    assert.equal(id, 'front');
    throw new Error('Sender rejected the lower cap');
  }), /rejected/);
});

test('presentation statistics reject old, duplicated and late frames', () => {
  const sent = [[1, 80], [2, 100], [3, 120], [4, 150], [5, 200]];
  const received = [[1, 130], [2, 150], [2, 160], [3, 140], [4, 200], [6, 170]];
  assert.deepEqual(presentationStats(sent, received, 100, 200), {
    submitted: 3, matched: 2, p50Ms: 20, p95Ms: 50,
  });
});
