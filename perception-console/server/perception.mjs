export function acceptSample(sample, { frame, nowMs, maxAgeMs }) {
  return sample?.frame === frame && Number.isFinite(sample.stampMs)
    && sample.stampMs >= 0 && sample.stampMs <= nowMs && nowMs - sample.stampMs <= maxAgeMs;
}

export function transformPoint([x, y, z], pose) {
  const { yaw, x: tx, y: ty, z: tz } = pose;
  return [tx + x * Math.cos(yaw) - y * Math.sin(yaw), ty + x * Math.sin(yaw) + y * Math.cos(yaw), tz + z];
}

export class Occupancy {
  constructor(origin, width, height, resolution) {
    this.origin = origin;
    this.width = width;
    this.height = height;
    this.resolution = resolution;
    this.cells = new Int8Array(width * height).fill(-1);
  }

  observe(start, end, occupied) {
    const cell = (coordinate, axis) => Math.floor((coordinate - this.origin[axis]) / this.resolution);
    let [x, y] = start.map(cell);
    const [ex, ey] = end.map(cell);
    const dx = Math.abs(ex - x);
    const dy = -Math.abs(ey - y);
    const sx = Math.sign(ex - x);
    const sy = Math.sign(ey - y);
    let error = dx + dy;
    while (x !== ex || y !== ey) {
      this.write(x, y, 0);
      const doubled = 2 * error;
      if (doubled >= dy) { error += dy; x += sx; }
      if (doubled <= dx) { error += dx; y += sy; }
    }
    if (occupied) this.write(ex, ey, 100);
  }

  write(x, y, evidence) {
    if (x < 0 || x >= this.width || y < 0 || y >= this.height) return;
    const index = y * this.width + x;
    this.cells[index] = Math.max(this.cells[index], evidence);
  }

  snapshot() {
    return { origin: this.origin, width: this.width, height: this.height, resolution: this.resolution, cells: Array.from(this.cells) };
  }
}

const planarDistance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

export function associateObservations(geometry, semantics, window) {
  const shapes = geometry.filter(sample => acceptSample(sample, window));
  const classes = semantics.filter(sample => acceptSample(sample, window) && sample.confidence >= 0.55);
  const candidates = shapes.map(shape => classes.filter(observation =>
    Math.abs(shape.stampMs - observation.stampMs) <= 50 && planarDistance(shape.position, observation.position) <= 0.6));
  const merged = shapes.map((shape, index) => {
    const [observation] = candidates[index];
    const unique = candidates[index].length === 1 && candidates.filter(matches => matches.includes(observation)).length === 1;
    return {
      ...shape, label: unique ? observation.label : 'Unknown', confidence: unique ? observation.confidence : null,
      sources: unique ? [shape.source, observation.source] : [shape.source],
    };
  });
  const locatedOnly = classes.filter(observation => !candidates.some(matches => matches.includes(observation))
    && !shapes.some(shape => planarDistance(shape.position, observation.position) <= 0.6));
  return merged.concat(locatedOnly.map(observation => ({ ...observation, sources: [observation.source] })));
}

export class Tracker {
  constructor() {
    this.tracks = new Map();
    this.counter = 0;
  }

  update(observations, nowMs) {
    const unused = new Set(this.tracks.keys());
    for (const [id, track] of this.tracks) {
      if (nowMs - track.stampMs > 600) {
        this.tracks.delete(id);
        unused.delete(id);
      }
    }

    const candidates = [];
    for (const observation of observations) {
      for (const track of this.tracks.values()) {
        const elapsed = (observation.stampMs - track.stampMs) / 1000;
        if (elapsed <= 0 || (track.label !== 'Unknown' && observation.label !== 'Unknown' && track.label !== observation.label)) continue;
        const predicted = track.position.map((coordinate, axis) => coordinate + (track.velocity[axis] ?? 0) * elapsed);
        const distance = planarDistance(predicted, observation.position);
        if (distance < 0.45 + elapsed * 0.6) candidates.push({ observation, track, distance });
      }
    }

    const matches = new Map();
    candidates.sort((a, b) => a.distance - b.distance);
    for (const { observation, track } of candidates) {
      if (unused.has(track.id) && !matches.has(observation)) {
        matches.set(observation, track);
        unused.delete(track.id);
      }
    }
    for (const observation of observations) {
      const old = matches.get(observation);
      const elapsed = old ? (observation.stampMs - old.stampMs) / 1000 : 0;
      const velocity = old ? [0, 1].map(axis =>
        0.6 * (observation.position[axis] - old.position[axis]) / elapsed + 0.4 * old.velocity[axis]) : [0, 0];
      const id = old?.id ?? `T-${String(++this.counter).padStart(3, '0')}`;
      const history = [...(old?.history ?? []), observation.position].slice(-16);
      this.tracks.set(id, { ...observation, id, velocity, history });
    }
    return Array.from(this.tracks.values(), track => ({
      ...track, ageMs: nowMs - track.stampMs, status: nowMs === track.stampMs ? 'observed' : 'coasting',
    }));
  }
}
