import assert from 'node:assert/strict';
import test from 'node:test';
import { cameraRig, cameraPose, clips, sampleClip, feedProfile, nextFrame, feedState } from '../src/camera-trial.js';

test('four distinct camera mounts rotate with the robot in the metric frame', () => {
  const rig = Object.keys(cameraRig);
  assert.deepEqual(rig, ['front', 'rear', 'left', 'right']);
  const pose = { x: 2, y: 3, yaw: Math.PI / 2 };
  const frames = rig.map(id => cameraPose(id, pose));
  assert.equal(new Set(frames.map(frame => frame.yaw)).size, 4);
  assert.ok(frames[0].y > 3 && frames[1].y < 3 && frames[2].x < 2 && frames[3].x > 2);
  assert.ok(frames.every(frame => frame.z > 0));
});

test('layouts replay identical motion and only schedule their visible feeds', () => {
  assert.equal(sampleClip('crossing', 0).robot.x, -5.8);
  assert.ok(sampleClip('reverse', 12000).robot.x < sampleClip('reverse', 0).robot.x);
  assert.ok(sampleClip('crossing', 8000).person.y < sampleClip('crossing', 3500).person.y);
  assert.equal(sampleClip('passing', 4500).loaded, true);
  assert.equal(clips.length, 3);
  assert.equal(feedProfile('left', 'pair', 'front', 'normal'), null);
  const active = feedProfile('rear', 'grid', 'rear', 'constrained');
  const preview = feedProfile('left', 'grid', 'rear', 'constrained');
  assert.ok(active.fps > preview.fps && active.width > preview.width && active.delayMs < preview.delayMs);
});

test('feed expiry and recovery depend on its own simulated acquisition clock', () => {
  const profile = feedProfile('front', 'focus', 'front', 'normal');
  const first = nextFrame(null, 500, profile, false);
  assert.equal(feedState(first, 500), 'ready');
  assert.equal(nextFrame(first, 501, profile, false), first);
  assert.equal(nextFrame(first, 1900, profile, true), first);
  assert.equal(feedState(first, 1900), 'stale');
  assert.equal(feedState(nextFrame(null, 1900, profile, true), 1900), 'unavailable');
  const recovered = nextFrame(first, 1920, profile, false);
  assert.equal(feedState(recovered, 1920), 'ready');
  assert.ok(recovered.stampMs > first.stampMs);
});
