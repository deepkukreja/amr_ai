import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

export const recordedRoute = { id: 'recorded-route', label: 'Recorded Gazebo route', replayOnly: true };
const defaultDirectory = fileURLToPath(new URL('../../outputs/route_validation/', import.meta.url));

export function decodeReferenceMap(pgm, yaml, stride = 8) {
  let offset = 0;
  const whitespace = byte => [9, 10, 13, 32].includes(byte);
  function token() {
    while (offset < pgm.length) {
      if (whitespace(pgm[offset])) offset++;
      else if (pgm[offset] === 35) {
        while (offset < pgm.length && pgm[offset] !== 10) offset++;
      } else break;
    }
    const start = offset;
    while (offset < pgm.length && !whitespace(pgm[offset])) offset++;
    return pgm.toString('ascii', start, offset);
  }
  const magic = token();
  const columns = Number(token());
  const rows = Number(token());
  const peak = Number(token());
  if (pgm[offset] === 13 && pgm[offset + 1] === 10) offset++;
  offset++;
  const property = name => new RegExp(`^${name}:\\s*([^\\r\\n]+)`, 'm').exec(yaml)?.[1].trim();
  const resolution = Number(property('resolution'));
  const [x, y, yaw] = JSON.parse(property('origin') ?? '[]');
  const occupied = Number(property('occupied_thresh'));
  const free = Number(property('free_thresh'));
  if (magic !== 'P5' || peak !== 255 || !Number.isInteger(columns) || columns <= 0 || !Number.isInteger(rows) || rows <= 0
    || pgm.length - offset !== columns * rows || property('mode') !== 'trinary' || property('negate') !== '0'
    || yaw !== 0 || ![x, y, resolution, occupied, free].every(Number.isFinite)
    || resolution <= 0 || free < 0 || occupied > 1 || free >= occupied) {
    throw new Error('Unsupported recorded map encoding');
  }
  const width = Math.ceil(columns / stride);
  const height = Math.ceil(rows / stride);
  const cells = Array.from({ length: width * height }, (_, index) => {
    const row = Math.floor(index / width) * stride;
    const column = (index % width) * stride;
    let unknown = row + stride > rows || column + stride > columns;
    for (let dy = row; dy < Math.min(row + stride, rows); dy++) {
      for (let dx = column; dx < Math.min(column + stride, columns); dx++) {
        const probability = 1 - pgm[offset + (rows - 1 - dy) * columns + dx] / 255;
        if (probability > occupied) return 100;
        if (probability >= free) unknown = true;
      }
    }
    return unknown ? -1 : 0;
  });
  return { width, height, origin: [x, y], resolution: resolution * stride, cells };
}

const stampMs = sample => Math.round(sample.stamp * 1000);

function sampler(samples) {
  const ordered = samples.filter(sample => Number.isFinite(sample.stamp)).sort((a, b) => a.stamp - b.stamp);
  let next = 0;
  return function sampleAt(time, maxAge) {
    while (next < ordered.length && stampMs(ordered[next]) <= time) next++;
    const sample = ordered[next - 1];
    return sample && time - stampMs(sample) <= maxAge ? sample : null;
  };
}

export function routeRecording({ truth, wheel, localization, goals, occupancy }) {
  const poses = truth.filter(sample => sample.stage === 'route'
    && [sample.stamp, sample.x, sample.y, sample.yaw].every(Number.isFinite)).sort((a, b) => a.stamp - b.stamp);
  if (poses.length < 2) throw new Error('No usable route captures');
  const start = stampMs(poses[0]);
  const intervalMs = 200;
  const durationMs = Math.floor((stampMs(poses.at(-1)) - start) / intervalMs) * intervalMs;
  const atPose = sampler(poses);
  const atWheel = sampler(wheel);
  const atLocalization = sampler(localization);
  const completed = goals.filter(goal => goal.accepted && goal.score?.passed && Number.isFinite(goal.completion_ros_time));
  const events = completed.map((goal, index) => ({
    code: `WAYPOINT_${index + 1}`, label: `Waypoint ${index + 1} completed`,
    stampMs: Math.round(goal.completion_ros_time * 1000) - start,
  })).filter(event => event.stampMs >= 0 && event.stampMs <= durationMs);
  let history = [];
  const frames = Array.from({ length: durationMs / intervalMs + 1 }, (_, sequence) => {
    const elapsed = sequence * intervalMs;
    const time = elapsed + start;
    const pose = atPose(time, 100);
    const motion = atWheel(time, 100);
    const estimate = atLocalization(time, 5000);
    const varianceM2 = Number.isFinite(estimate?.max_xy_variance) ? estimate.max_xy_variance : null;
    if (pose) history = [...history, [pose.x, pose.y, 0]].slice(-120);
    const next = events.findIndex(event => event.stampMs > elapsed);
    return {
      version: 1, mode: 'SIMULATION', provenance: 'GAZEBO', clock: 'ROS_SIM', session: 'recorded-gazebo-nominal-91',
      sequence, stampMs: elapsed, sourceStampMs: pose ? stampMs(pose) : null, durationMs, frame: 'map',
      scenario: recordedRoute.id, site: 'small-warehouse',
      robot: {
        name: 'rbot / Gazebo', modelId: null, pose: pose ? { x: pose.x, y: pose.y, yaw: pose.yaw } : null, speed: motion?.speed ?? null,
        steering: null, battery: null, localization: { status: varianceM2 === null ? 'unavailable' : 'recorded', varianceM2 },
      },
      sensors: [], tracks: [], cloud: null, occupancy, events: [],
      nav: {
        state: next === -1 ? 'Route completed' : 'Recorded route',
        destination: next === -1 ? null : `Waypoint ${next + 1} / ${events.length}`,
        planned: [], local: [], prediction: [], history,
      },
    };
  });
  return { version: 1, scenario: recordedRoute.id, sourceMode: 'GAZEBO', durationMs, intervalMs, frames, events };
}

export async function loadRouteRecording(directory = process.env.RBW_ROUTE_EVIDENCE ?? defaultDirectory) {
  const samples = join(directory, 'evidence', 'nominal_91');
  const map = join(directory, 'reference_configuration', 'map');
  const readJson = async basename => JSON.parse(await readFile(join(samples, basename), 'utf8'));
  const [truth, wheel, localization, report, pgm, yaml] = await Promise.all([
    readJson('truth_samples.json'), readJson('wheel_samples.json'), readJson('localization_errors_samples.json'), readJson('report.json'),
    readFile(join(map, 'nav2_map.pgm')), readFile(join(map, 'nav2_map.yaml'), 'utf8'),
  ]);
  return routeRecording({ truth, wheel, localization, goals: report.goals, occupancy: decodeReferenceMap(pgm, yaml) });
}
