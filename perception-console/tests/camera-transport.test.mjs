import assert from 'node:assert/strict';
import test from 'node:test';
import { bufferDelay, decodeMarker, encodeMarker, feedHealth, frameLatency, mediaRate, transportCases } from '../src/camera-transport-metrics.js';

test('the profiles share one encoder budget', () => {
  assert.deepEqual(transportCases.map(item => Object.keys(item.caps).length), [2, 4, 4]);
  for (const bench of transportCases) {
    assert.equal(Object.values(bench.caps).reduce((sum, cap) => sum + cap, 0), 720_000);
  }
  assert.equal(transportCases[2].caps.front, 360_000);
  assert.equal(transportCases[2].caps.left, 120_000);
});

test('a marker survives luma shifts and rejects mismatched or missing data', () => {
  const samples = value => encodeMarker(value).map(bit => bit ? 222 : 28);
  for (const value of [1, 18, 193, 255, 256, 400]) {
    assert.equal(decodeMarker(samples(value)), value);
  }
  const corrupt = samples(18);
  corrupt[0] = corrupt[0] === 28 ? 222 : 28;
  assert.equal(decodeMarker(corrupt), null);
  assert.equal(decodeMarker(Array(20).fill(20)), null);
  assert.equal(decodeMarker(samples(1).slice(1)), null);
  assert.throws(() => encodeMarker(0), RangeError);
  assert.throws(() => encodeMarker(1024), RangeError);
});

test('presentation latency only uses matched frames in the measurement window', () => {
  const submitted = new Map([[6, 4099], [7, 4105], [8, 4110]]);
  assert.equal(frameLatency(submitted, 7, 4152, 4100, 4200), 47);
  assert.equal(frameLatency(submitted, 6, 4152, 4100, 4200), null);
  assert.equal(frameLatency(submitted, 8, 4105, 4100, 4200), null);
  assert.equal(frameLatency(submitted, 8, 4210, 4100, 4200), null);
  assert.equal(frameLatency(submitted, 9, 4152, 4100, 4200), null);
});

test('payload rate uses deltas and a valid stats time interval', () => {
  assert.equal(mediaRate({ timestamp: 100, bytesSent: 4000 }, { timestamp: 2100, bytesSent: 19000 }), 60);
  assert.equal(mediaRate({ timestamp: 100, bytesSent: 4000 }, { timestamp: 100, bytesSent: 9000 }), null);
  assert.equal(mediaRate({ timestamp: 100, bytesSent: 4000 }, { timestamp: 200, bytesSent: 3999 }), null);
  assert.equal(mediaRate(null, { timestamp: 200, bytesSent: 9000 }), null);
});

test('old but advancing video is delayed, while no new presentation becomes stale', () => {
  assert.equal(feedHealth(null, 2_000), 'waiting');
  assert.equal(feedHealth({ sentAt: 1750, displayedAt: 1900 }, 2_000), 'receiving');
  assert.equal(feedHealth({ sentAt: 1000, displayedAt: 1900 }, 2_000), 'delayed');
  assert.equal(feedHealth({ sentAt: 1000, displayedAt: 1200 }, 2_000), 'stale');
});

test('average jitter buffer delay uses emitted frames in the current measurement interval', () => {
  const before = { jitterBufferDelay: 8, jitterBufferEmittedCount: 20 };
  assert.equal(bufferDelay(before, { jitterBufferDelay: 12, jitterBufferEmittedCount: 30 }), 400);
  assert.equal(bufferDelay(before, before), null);
  assert.equal(bufferDelay(before, { jitterBufferDelay: 1, jitterBufferEmittedCount: 30 }), null);
});
