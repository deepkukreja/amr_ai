import { Box3, Ray, Vector3 } from 'three';

const vector = values => new Vector3(...values);
const up = new Vector3(0, 0, 1);
const resolution = { width: 144, height: 96 };
const voxelSizeM = 0.065;
export const surveyFov = 58;
const verticalScale = Math.tan(surveyFov * Math.PI / 360);
const horizontalScale = verticalScale * resolution.width / resolution.height;

export const surveyViews = [
  { name: 'Approach', eye: [-5.9, -2.4, 2.05], target: [1.7, 0.1, 1.22] },
  { name: 'Threshold', eye: [-2.9, 0.44, 1.55], target: [3.9, -0.35, 1.17] },
  { name: 'Yard', eye: [-6.1, 4.8, 2.8], target: [0.30, -0.3, 1.15] },
];

const captureViews = [
  ...surveyViews,
  { eye: [5.2, 3.2, 2.05], target: [-0.9, -0.85, 1.16] },
  { eye: [4.3, -3.7, 1.9], target: [-1.6, 0.5, 1.24] },
  { eye: [1.4, -0.85, 1.65], target: [-3.4, 0.3, 1.3] },
];

const surfaces = [
  [[-10, -8, -0.12], [0, 8, 0], 0x817f76, 'asphalt'],
  [[0, -4.65, -0.12], [10, 4.65, 0], 0xa5a195, 'concrete'],
  [[0, -4.65, 0], [0.26, -1.64, 3.65], 0xbabfc3, 'facade'],
  [[0, 1.64, 0], [0.26, 4.65, 3.65], 0xbabfc3, 'facade'],
  [[0, -1.64, 2.77], [0.26, 1.64, 3.65], 0xbabfc3, 'facade'],
  [[-0.08, -1.76, 0], [0.28, -1.60, 2.82], 0x3b515c],
  [[-0.08, 1.60, 0], [0.28, 1.76, 2.82], 0x3b515c],
  [[-0.08, -1.76, 2.70], [0.28, 1.76, 2.86], 0x3b515c],
  [[0.26, -4.65, 0], [9.8, -4.45, 3.65], 0x8b979e, 'facade'],
  [[0.26, 4.45, 0], [9.8, 4.65, 3.65], 0x8b979e, 'facade'],
  [[9.6, -4.45, 0], [9.8, 4.45, 3.65], 0x737f87, 'facade'],
  [[-1.32, -2.40, 0], [-1.06, -2.14, 1.04], 0xe4af33],
  [[-1.32, 2.14, 0], [-1.06, 2.40, 1.04], 0xe4af33],
  [[-3.7, 2.50, 0], [-2.20, 3.65, 0.87], 0x9d6843, 'crate'],
  [[-3.45, 2.56, 0.87], [-2.45, 3.48, 1.48], 0xb98859, 'crate'],
  [[2.1, -3.75, 0], [4.0, -2.0, 1.22], 0x977557, 'crate'],
  [[2.3, -3.65, 1.22], [3.52, -2.21, 2.01], 0xb8976b, 'crate'],
  [[2.4, 2.0, 0], [4.85, 3.7, 0.17], 0x4c6270],
  [[2.4, 2.0, 0.95], [4.85, 3.7, 1.07], 0x4c6270],
  [[2.4, 2.0, 1.94], [4.85, 3.7, 2.06], 0x4c6270],
  [[2.4, 2.0, 0], [2.53, 2.14, 2.34], 0x476b7b],
  [[4.72, 2.0, 0], [4.85, 2.14, 2.34], 0x476b7b],
  [[2.4, 3.57, 0], [2.53, 3.7, 2.34], 0x476b7b],
  [[4.72, 3.57, 0], [4.85, 3.7, 2.34], 0x476b7b],
  [[2.59, 2.27, 0.17], [3.49, 3.46, 0.73], 0x567b75, 'crate'],
  [[3.48, 2.25, 1.07], [4.5, 3.35, 1.69], 0xb1845d, 'crate'],
].map(([min, max, color, finish]) => ({ bounds: new Box3(vector(min), vector(max)), color, finish }));

function basis(view) {
  const eye = vector(view.eye);
  const forward = vector(view.target).sub(eye).normalize();
  const right = forward.clone().cross(up).normalize();
  return { eye, forward, right, vertical: right.clone().cross(forward).normalize() };
}

function direction({ forward, right, vertical }, u, v, result) {
  return result.copy(forward).addScaledVector(right, (2 * u - 1) * horizontalScale)
    .addScaledVector(vertical, (1 - 2 * v) * verticalScale).normalize();
}

export function project(view, point) {
  const { eye, forward, right, vertical } = basis(view);
  const offset = vector(point).sub(eye);
  const depth = offset.dot(forward);
  if (depth <= 0) return null;
  return {
    u: (1 + offset.dot(right) / (depth * horizontalScale)) * 0.5,
    v: (1 - offset.dot(vertical) / (depth * verticalScale)) * 0.5,
    rangeM: offset.length(),
  };
}

function nearestSurface(ray, intersection, point) {
  let range = 16;
  let surface = null;
  for (const candidate of surfaces) {
    if (!ray.intersectBox(candidate.bounds, intersection)) continue;
    const distance = intersection.distanceTo(ray.origin);
    if (distance >= range) continue;
    range = distance;
    surface = candidate;
    point.copy(intersection);
  }
  return surface;
}

function surfaceAxis(point, { min, max }) {
  const tolerance = 1e-7;
  if (Math.min(Math.abs(point.x - min.x), Math.abs(point.x - max.x)) < tolerance) return 0;
  if (Math.min(Math.abs(point.y - min.y), Math.abs(point.y - max.y)) < tolerance) return 1;
  return 2;
}

function surfaceColor({ x, y, z }, surface, axis) {
  const laneEdge = Math.abs(Math.abs(y) - 1.23) < 0.065;
  if (surface.finish === 'asphalt' && x > -7 && x < -0.4 && laneEdge) return 0xefca63;
  if (surface.finish === 'concrete' && x < 7.7 && laneEdge) return 0xefe6b9;
  const seam = surface.finish === 'facade' && z % 0.41 < 0.029;
  const slat = surface.finish === 'crate' && axis !== 2 && z % 0.24 < 0.034;
  const grain = Math.sin(x * 39.17 + y * 59.3 + z * 49.97) * 0.047;
  const factor = [0.92, 0.83, 1.03][axis] * (seam || slat ? 0.71 : 1) + grain;
  const channel = shift => Math.round(Math.min(255, ((surface.color >> shift) & 255) * factor));
  return (channel(16) << 16) + (channel(8) << 8) + channel(0);
}

export function captureEntrance() {
  const samples = new Map();
  const hit = new Vector3();
  const point = new Vector3();
  const ray = new Ray();
  let rayHits = 0;
  for (const view of captureViews) {
    const camera = basis(view);
    ray.origin.copy(camera.eye);
    for (let row = 0; row < resolution.height; row++) {
      for (let column = 0; column < resolution.width; column++) {
        direction(camera, (column + 0.5) / resolution.width, (row + 0.5) / resolution.height, ray.direction);
        const surface = nearestSurface(ray, hit, point);
        if (!surface) continue;
        rayHits++;
        const { x, y, z } = point;
        const voxel = [x, y, z].map(value => Math.round(value / voxelSizeM)).join(',');
        if (samples.has(voxel)) continue;
        const axis = surfaceAxis(point, surface.bounds);
        samples.set(voxel, { x, y, z, axis, voxel, color: surfaceColor(point, surface, axis) });
      }
    }
  }
  const controlPoints = [
    ['Left jamb', [-0.08, -1.69, 1.62], surveyViews[0]],
    ['Right jamb', [-0.08, 1.67, 1.21], surveyViews[0]],
    ['Lintel', [-0.08, 0.32, 2.77], surveyViews[1]],
  ];
  const registeredLandmarks = controlPoints.map(([name, landmark, view]) => {
    const camera = basis(view);
    const pixel = project(view, landmark);
    ray.origin.copy(camera.eye);
    direction(camera, pixel.u, pixel.v, ray.direction);
    const surface = nearestSurface(ray, hit, point);
    return { name, errorM: surface ? point.distanceTo(vector(landmark)) : Infinity };
  });
  return {
    frame: 'survey_entrance_sim',
    samples: [...samples.values()],
    registeredLandmarks,
    measurements: { rayHits, captures: captureViews.length, voxelSizeM },
  };
}
