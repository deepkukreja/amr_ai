import assert from 'node:assert/strict';
import test from 'node:test';
import { captureEntrance, project, surveyViews } from '../src/survey-data.js';

test('one calibrated metric survey feeds both renderers with deterministic in-bounds surface samples', () => {
  const first = captureEntrance();
  const second = captureEntrance();
  assert.equal(first.frame, 'survey_entrance_sim');
  assert.ok(first.samples.length >= 1000);
  assert.deepEqual(second.samples, first.samples);
  assert.ok(first.samples.every(sample => Number.isFinite(sample.x) && Number.isFinite(sample.y)
    && sample.z >= -0.12 - 1e-9 && sample.z < 4 && sample.axis >= 0 && sample.axis <= 2));
  assert.equal(new Set(first.samples.map(sample => sample.voxel)).size, first.samples.length);
  assert.equal(first.registeredLandmarks.length, 3);
  assert.ok(first.registeredLandmarks.every(item => item.errorM < 1e-6));
  assert.ok(first.measurements.rayHits > first.samples.length);
});

test('calibrated viewpoints project targets consistently and reject points behind the camera', () => {
  for (const view of surveyViews) {
    const center = project(view, view.target);
    assert.ok(Math.abs(center.u - 0.5) < 1e-10);
    assert.ok(Math.abs(center.v - 0.5) < 1e-10);
    assert.equal(project(view, view.eye.map((coordinate, axis) => coordinate * 2 - view.target[axis])), null);
  }
});
