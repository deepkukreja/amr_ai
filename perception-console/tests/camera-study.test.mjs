import assert from 'node:assert/strict';
import test from 'node:test';
import { makeSchedule, scoreRun, summarizeStudy, studyCsv, studyLength } from '../src/camera-study.js';

test('the six sequences balance layout order and cover every maneuver-layout pair', () => {
  const schedules = Array.from({ length: 6 }, (_, index) => makeSchedule(index + 1));
  assert.equal(studyLength, 9);
  for (const schedule of schedules) {
    assert.equal(schedule.length, studyLength);
    assert.equal(new Set(schedule.map(run => `${run.clip}/${run.layout}`)).size, studyLength);
    assert.deepEqual(schedule.map(run => run.clip), schedules[0].map(run => run.clip));
  }
  for (const clip of ['reverse', 'crossing', 'passing']) {
    const orders = schedules.map(schedule => schedule.filter(run => run.clip === clip).map(run => run.layout).join(','));
    assert.equal(new Set(orders).size, 6);
  }
  assert.throws(() => makeSchedule(0), RangeError);
});

test('only correct identifications contribute to median recognition time', () => {
  const first = { clip: 'reverse', layout: 'focus' };
  const second = { clip: 'crossing', layout: 'focus' };
  const third = { clip: 'passing', layout: 'focus' };
  const good = scoreRun(first, 'barrier', 952.4, 2);
  const wrong = scoreRun(second, 'pallet', 755, 0);
  const miss = scoreRun(third, null, 12000, 1);
  assert.deepEqual(good, { ...first, answer: 'barrier', outcome: 'correct', elapsedMs: 952, viewChanges: 2 });
  assert.equal(wrong.outcome, 'incorrect');
  assert.equal(miss.elapsedMs, null);
  assert.equal(scoreRun(first, 'barrier', 12000, 0).outcome, 'missed');
  assert.deepEqual(summarizeStudy([good, wrong, miss])[0],
    { layout: 'focus', runs: 3, correct: 1, incorrect: 1, missed: 1, medianMs: 952, viewChanges: 3 });
  assert.equal(summarizeStudy([])[0].medianMs, null);
});

test('download records retain sequence, simulation profile and per-run outcomes', () => {
  const entry = scoreRun(makeSchedule(2)[0], 'barrier', 1200, 1);
  const csv = studyCsv({ id: 'S1', sequence: 2, profile: 'constrained', records: [entry] });
  const lines = csv.trimEnd().split('\n');
  assert.match(lines[0], /delivery_profile/);
  assert.match(lines[0], /elapsed_ms,view_changes$/);
  assert.match(lines[1], /^1,S1,2,simulation,constrained,1,reverse,focus,barrier,correct,1200,1$/);
});
