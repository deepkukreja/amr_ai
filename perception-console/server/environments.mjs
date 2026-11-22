// Visualization dimensions come from output/Robot_Buildworks_AMR/Geometry_checks.json.
export const delta = Object.freeze({ wheelbase: 0.72, rearTrack: 0.8, tireRadius: 0.125, tireWidth: 0.085 });
export const robotProfiles = { 'd3-prototype': { kind: 'delta-steer', geometry: delta } };

const warehouse = {
  id: 'warehouse-b03-fixture-v1',
  name: 'Warehouse A',
  zone: 'Zone B-03',
  setting: 'Indoor',
  frame: 'sim_warehouse',
  bounds: [-9, -8, 22, 8],
  route: { start: -3.8, goal: 14, centerY: 0, amplitude: 0.68, wavelength: 4.1, destination: 'Dock B-04' },
  structures: [
    ...[-1, 1].flatMap(side => [-5, -0.3, 4.4, 9.1, 13.8].map((x, index) => ({
      id: `rack-${side}-${index}`, kind: 'rack', position: [x, side * 3.8, 0], size: [4.15, 1.45, 3.1],
    }))),
    { id: 'wall-back', kind: 'wall', position: [19.5, 0, 0], size: [0.3, 16, 4.1] },
    { id: 'wall-left', kind: 'wall', position: [6, 7.8, 0], size: [27, 0.25, 4.1] },
    { id: 'wall-right', kind: 'wall', position: [6, -7.8, 0], size: [27, 0.25, 4.1] },
  ],
};

const yard = {
  id: 'loading-yard-fixture-v1',
  name: 'Loading yard',
  zone: 'Gate 02',
  setting: 'Outdoor',
  frame: 'sim_yard',
  bounds: [-10, -10, 23, 10],
  route: { start: -4.2, goal: 17, centerY: -0.55, amplitude: 1.4, wavelength: 5.8, destination: 'Gate 02' },
  structures: [
    { id: 'trailer-1', kind: 'container', position: [4, 5.5, 0], size: [6, 2.5, 2.6] },
    { id: 'trailer-2', kind: 'container', position: [10.8, 5.5, 0], size: [6, 2.5, 2.6] },
    ...[-3.8, 0.5, 4.8, 9.1, 13.4].map((x, index) => ({
      id: `barrier-${index}`, kind: 'barrier', position: [x, -4.9, 0], size: [3.1, 0.62, 0.75],
    })),
    ...[-2.1, 2.7].map((y, index) => ({
      id: `gate-${index}`, kind: 'bollard', position: [17, y, 0], size: [0.36, 0.36, 1.6],
    })),
  ],
};

export const environments = { [warehouse.id]: warehouse, [yard.id]: yard };
export const indoorSite = warehouse.id;
export const outdoorSite = yard.id;
export const round = value => Math.round(value * 1000) / 1000;
export const routeY = (route, x) => route.centerY + route.amplitude * Math.sin((x - route.start) / route.wavelength);
export const routePoint = (route, x) => [round(x), round(routeY(route, x)), 0];

export function actorsAt(scenario, time) {
  if (scenario === 'yard') return [
    { label: 'Cart', position: [1.8 + Math.min(time, 10) * 0.38, -2.3, 0], size: [1.0, 0.61, 0.79], confidence: 0.89 },
    { label: 'Unknown', position: [7.8, 2.9, 0], size: [0.92, 0.83, 0.65], confidence: null },
  ];
  const actors = [
    { label: 'Pallet', position: [2.8, -2.35, 0], size: [1.13, 0.84, 0.73], confidence: 0.92 },
    { label: 'Unknown', position: [6.1, 2.7, 0], size: [0.8, 0.78, 0.9], confidence: null },
  ];
  if (scenario === 'pedestrian') actors.push({
    label: 'Person', position: [2.1 + time * 0.07, Math.max(-2.7, 2.85 - time * 0.55), 0],
    size: [0.52, 0.54, 1.72], confidence: 0.9,
  });
  if (scenario === 'forklift' || scenario === 'pedestrian') actors.push({
    label: 'Forklift', position: [Math.max(4.6, (scenario === 'forklift' ? 10.1 : 8.7) - time * (scenario === 'forklift' ? 0.54 : 0.18)), -1.65, 0],
    size: [1.55, 0.95, 1.86], confidence: 0.87,
  });
  if (scenario === 'blocked' && time >= 4) actors.push({
    label: 'Unknown', position: [1.75, routeY(warehouse.route, 1.75), 0], size: [0.9, 1.25, 1.02], confidence: null,
  });
  if (scenario === 'direction') actors.push({
    label: 'Cart', position: [3.8, 0.3 + (time < 6 ? time : 12 - time) * 0.32, 0],
    size: [0.92, 0.65, 1.01], confidence: 0.84,
  });
  return actors;
}

export function cameraSees(environment, pose, actor, actors) {
  const origin = [pose.x, pose.y];
  const offset = actor.position.slice(0, 2).map((value, axis) => value - origin[axis]);
  const distance = Math.hypot(...offset);
  const direction = offset.map(value => value / distance);
  const forward = Math.cos(pose.yaw) * direction[0] + Math.sin(pose.yaw) * direction[1];
  return distance >= 0.25 && distance <= 13 && forward >= 0.63
    && !environment.structures.concat(actors).some(shape => shape !== actor && hitDistance(origin, direction, shape) < hitDistance(origin, direction, actor));
}

function hitDistance(origin, direction, shape) {
  let near = 0;
  let far = 13.5;
  for (const axis of [0, 1]) {
    const low = shape.position[axis] - shape.size[axis] / 2 - origin[axis];
    const high = low + shape.size[axis];
    if (Math.abs(direction[axis]) < 1e-8) {
      if (low > 0 || high < 0) return Infinity;
      continue;
    }
    const enter = low / direction[axis];
    const exit = high / direction[axis];
    near = Math.max(near, Math.min(enter, exit));
    far = Math.min(far, Math.max(enter, exit));
  }
  return near <= far ? near : Infinity;
}

export function scanEnvironment(environment, pose, actors, grid) {
  const shapes = environment.structures.concat(actors);
  const visible = new Set();
  const points = [];
  const origin = [pose.x, pose.y];
  for (let ray = 0; ray < 160; ray++) {
    const angle = pose.yaw + ray * Math.PI / 80;
    const direction = [Math.cos(angle), Math.sin(angle)];
    let distance = 13.5;
    let hit;
    for (const shape of shapes) {
      const near = hitDistance(origin, direction, shape);
      if (near < distance) { distance = near; hit = shape; }
    }
    const end = origin.map((value, axis) => value + direction[axis] * distance);
    grid.observe(origin, end, Boolean(hit));
    if (!hit) continue;
    visible.add(hit);
    for (let layer = 0; layer < 6; layer++) {
      const height = hit.size[2] * (layer + 0.35 + 0.35 * Math.sin(ray * 5.3 + layer * 2.1)) / 6;
      points.push([round(end[0]), round(end[1]), round(height)]);
    }
  }
  return { points, visible };
}
