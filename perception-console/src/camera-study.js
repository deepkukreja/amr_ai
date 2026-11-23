import { clipLengthMs, clips } from './camera-trial.js';

export const layoutNames = { focus: 'Focused', pair: 'Front and rear', grid: 'Four cameras' };
export const studyCases = {
  reverse: { prompt: 'Identify the obstacle directly behind the robot as it reverses.', answer: 'barrier' },
  crossing: { prompt: 'Identify what moves across the robot’s path.', answer: 'person' },
  passing: { prompt: 'Identify what the loaded robot passes close to on its right.', answer: 'pallet' },
};
export const answers = { barrier: 'Dock barrier', person: 'Person', pallet: 'Pallet load' };
const orders = [
  ['focus', 'pair', 'grid'], ['focus', 'grid', 'pair'],
  ['pair', 'focus', 'grid'], ['pair', 'grid', 'focus'],
  ['grid', 'focus', 'pair'], ['grid', 'pair', 'focus'],
];
export const studyLength = clips.length * Object.keys(layoutNames).length;

export function makeSchedule(sequence) {
  const order = orders[sequence - 1];
  if (!Number.isInteger(sequence) || !order) throw new RangeError('Choose a sequence from 1 to 6.');
  return order.flatMap((_, round) =>
    clips.map(({ id }, index) => ({ clip: id, layout: order[(index + round) % order.length] })));
}

export function scoreRun(run, answer, elapsedMs, viewChanges) {
  const missed = answer === null || elapsedMs >= clipLengthMs;
  return {
    ...run, answer: missed ? null : answer,
    outcome: missed ? 'missed' : answer === studyCases[run.clip].answer ? 'correct' : 'incorrect',
    elapsedMs: missed ? null : Math.round(elapsedMs), viewChanges,
  };
}

export function summarizeStudy(records) {
  return Object.keys(layoutNames).map(layout => {
    const runs = records.filter(run => run.layout === layout);
    const correct = runs.filter(run => run.outcome === 'correct').map(run => run.elapsedMs).sort((a, b) => a - b);
    const middle = Math.floor(correct.length / 2);
    const medianMs = correct.length === 0 ? null
      : correct.length % 2 ? correct[middle] : (correct[middle - 1] + correct[middle]) / 2;
    return {
      layout, runs: runs.length, correct: correct.length,
      incorrect: runs.filter(run => run.outcome === 'incorrect').length,
      missed: runs.filter(run => run.outcome === 'missed').length,
      medianMs, viewChanges: runs.reduce((total, run) => total + run.viewChanges, 0),
    };
  });
}

export function studyCsv(session) {
  const header = 'schema_version,session_id,sequence,source,delivery_profile,run,clip,layout,answer,outcome,elapsed_ms,view_changes';
  const rows = session.records.map((record, index) => [
    1, session.id, session.sequence, 'simulation', session.profile, index + 1, record.clip,
    record.layout, record.answer, record.outcome, record.elapsedMs, record.viewChanges,
  ].join(','));
  return [header, ...rows, ''].join('\n');
}
