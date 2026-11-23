import { open } from 'node:fs/promises';

export const rosWorld = { id: 'ros-world', label: 'ROS sensor stream', streamOnly: true };
const maxBytes = 768 * 1024;
const stamp = value => Number.isSafeInteger(value) && value >= 0;
const tuple = (value, size) => Array.isArray(value) && value.length === size && value.every(Number.isFinite);
const points = (value, max) => Array.isArray(value) && value.length <= max && value.every(point => tuple(point, 3));
const fresh = (value, now, allowedAge) => stamp(value) && value <= now && now - value <= allowedAge;

function acceptDepth(depth, now) {
  if (!depth || typeof depth.frame !== 'string' || !depth.frame.trim() || !fresh(depth.stampMs, now, 1000)
    || ![depth.width, depth.height].every(size => Number.isInteger(size) && size > 0 && size <= 160)) return null;
  const bytes = 2 * depth.width * depth.height;
  if (typeof depth.mm16 !== 'string' || depth.mm16.length !== 4 * Math.ceil(bytes / 3)
    || !/^[A-Za-z0-9+/]+={0,2}$/.test(depth.mm16) || Buffer.byteLength(depth.mm16, 'base64') !== bytes) return null;
  return depth;
}

function acceptObjects(layer, frame, now) {
  if (!layer || layer.frame !== frame || !fresh(layer.stampMs, now, 1000)
    || !Array.isArray(layer.items) || layer.items.length > 32) return null;
  return layer.items.every(item => typeof item?.classId === 'string' && item.classId.trim() && item.classId.length <= 64
    && Number.isFinite(item.score) && item.score >= 0 && item.score <= 1
    && Number.isFinite(item.rangeM) && item.rangeM >= 0 && item.rangeM <= 30
    && tuple(item.position, 3)) ? layer : null;
}

export function sequenceGate() {
  let marker;
  return frame => {
    if (!marker || frame.session !== marker.session) {
      marker = frame;
      return false;
    }
    if (frame.sequence <= marker.sequence || frame.stampMs < marker.stampMs) return false;
    marker = frame;
    return true;
  };
}

export function acceptWorld(input) {
  if (input?.version !== 1 || !stamp(input.sequence) || !stamp(input.stampMs) || typeof input.session !== 'string' || !input.session
    || !['ROS_MONITOR', 'ROS_SENSOR_FIXTURE'].includes(input.source)
    || !['LIVE', 'SIMULATION'].includes(input.mode) || (input.source === 'ROS_SENSOR_FIXTURE' && input.mode !== 'SIMULATION')
    || !['ROS_SYSTEM', 'ROS_SIM'].includes(input.clock) || typeof input.site !== 'string' || !input.site
    || typeof input.frame !== 'string' || !input.frame) return null;

  const map = input.occupancy;
  if (!map || ![map.width, map.height].every(size => Number.isInteger(size) && size > 0)
    || map.width * map.height > 4096 || !tuple(map.origin, 2) || !Number.isFinite(map.yaw)
    || !Number.isFinite(map.resolution) || map.resolution <= 0
    || !Array.isArray(map.cells) || map.cells.length !== map.width * map.height
    || !map.cells.every(cell => [-1, 0, 100].includes(cell))) return null;
  const now = input.stampMs;
  const pose = input.pose && [input.pose.x, input.pose.y, input.pose.yaw].every(Number.isFinite)
    && fresh(input.pose.stampMs, now, 1200) ? input.pose : null;
  const registered = (layer, age, max) => layer?.frame === input.frame
    && fresh(layer.stampMs, now, age) && points(layer.points, max);
  const plan = registered(input.plan, 3000, 2000) ? input.plan : null;
  const cloud = registered(input.cloud, 1000, 1000) ? input.cloud : null;
  const depth = acceptDepth(input.depth, now);
  const objects = acceptObjects(input.objects, input.frame, now);
  const image = input.camera;
  let camera = null;
  if (image && typeof image.frame === 'string' && fresh(image.stampMs, now, 1000)
    && [image.width, image.height].every(size => Number.isInteger(size) && size > 0 && size <= 640)
    && typeof image.jpeg === 'string' && image.jpeg.length < 400000 && /^\/9j\/[A-Za-z0-9+/]*={0,2}$/.test(image.jpeg)) {
    const drawable = image.overlay === 'registered' && fresh(image.planStampMs, image.stampMs, 3000)
      && Array.isArray(image.segments) && image.segments.length <= 200
      && image.segments.every(segment => Array.isArray(segment) && segment.length >= 2 && segment.length <= 2000
        && segment.every(point => tuple(point, 2) && point.every(value => value >= 0 && value <= 1)));
    camera = { ...image, overlay: drawable ? 'registered' : 'unavailable', segments: drawable ? image.segments : [] };
  }

  const rejected = new Set();
  if (input.cloud && !cloud) rejected.add('lidar');
  if (input.camera && !camera) rejected.add('rgbd');
  if (input.depth && !depth) rejected.add('depth');
  if (input.objects && !objects) rejected.add('objects');
  if (input.pose && !pose) rejected.add('localization');
  if (input.plan && !plan) rejected.add('plan');
  const sensors = (Array.isArray(input.sensors) ? input.sensors : []).slice(0, 8).filter(sensor => typeof sensor?.id === 'string'
    && typeof sensor.name === 'string' && ['fresh', 'stale', 'unregistered', 'unavailable'].includes(sensor.status))
    .map(sensor => ({ ...sensor, ...(rejected.has(sensor.id) ? { status: 'unregistered', ageMs: null } : {}) }));
  const d3 = input.robotModel === 'd3-prototype' || input.source === 'ROS_SENSOR_FIXTURE';
  return {
    ...input, pose, plan, cloud, camera, depth, objects, sensors, version: 1, provenance: 'ROS_STREAM', scenario: rosWorld.id,
    robot: {
      name: d3 ? 'D3 · Design profile' : 'ROS robot', modelId: d3 ? 'd3-prototype' : null, pose,
      speed: null, steering: null, battery: null, localization: { status: pose ? 'transform available' : 'unavailable', varianceM2: null },
    },
    tracks: [], events: [],
    nav: { state: plan ? 'Plan available' : 'Plan unavailable', destination: null, planned: plan?.points ?? [], history: [], prediction: [], local: [] },
  };
}

export function rosSnapshotReader(path) {
  const isNew = sequenceGate();
  return async () => {
    let file;
    try {
      file = await open(path, 'r');
      const stat = await file.stat();
      if (!stat.isFile() || stat.size > maxBytes) return null;
      const frame = acceptWorld(JSON.parse(await file.readFile('utf8')));
      return frame && isNew(frame) ? frame : null;
    } catch {
      return null;
    } finally {
      await file?.close();
    }
  };
}
