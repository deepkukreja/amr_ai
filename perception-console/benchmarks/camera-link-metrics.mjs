import { cameraRig } from '../src/camera-trial.js';
import { frameLatency, percentile } from '../src/camera-transport-metrics.js';

export const directions = Object.keys(cameraRig);

export function cameraCaps(priority, budget) {
  const main = Math.floor(budget / 2);
  const auxiliary = Math.floor((budget - main) / (directions.length - 1));
  return Object.fromEntries(directions.map(id => [id, id === priority ? main : auxiliary]));
}

export async function changePriority(previous, selected, budget, applyCap) {
  if (previous === selected) return;
  const caps = cameraCaps(selected, budget);
  await applyCap(previous, caps[previous]);
  await applyCap(selected, caps[selected]);
}

export function presentationStats(sent, received, from, until) {
  const source = new Map(sent.filter(([, stamp]) => stamp >= from && stamp < until));
  const matched = new Map();
  for (const [sequence, displayedAt] of received) {
    const age = frameLatency(source, sequence, displayedAt, from, until);
    if (age !== null && !matched.has(sequence)) matched.set(sequence, age);
  }
  const ages = [...matched.values()];
  return { submitted: source.size, matched: ages.length, p50Ms: percentile(ages, 0.50), p95Ms: percentile(ages, 0.95) };
}
