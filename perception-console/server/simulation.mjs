import { Occupancy, Tracker, associateObservations, transformPoint } from './perception.mjs';
import { actorsAt, cameraSees, environments, indoorSite, outdoorSite, robotProfiles, round, routePoint, routeY, scanEnvironment } from './environments.mjs';

export const intervalMs = 100;
const durationMs = 20000;
const modelId = 'd3-prototype';
const { geometry: delta } = robotProfiles[modelId];

export const scenarios = [
  { id: 'pedestrian', label: 'Pedestrian crossing', start: 4.4, end: 8.5, code: 'CROSSING', event: 'Pedestrian crossing the planned route' },
  { id: 'clear', label: 'Clear corridor' },
  { id: 'forklift', label: 'Approaching forklift', start: 3.4, end: 9.7, code: 'FORKLIFT_NEAR', event: 'Approaching forklift in the adjacent lane' },
  { id: 'blocked', label: 'Blocked aisle', start: 4, code: 'PATH_BLOCKED', event: 'Unclassified obstacle blocks the aisle' },
  { id: 'direction', label: 'Cart changes direction', start: 6, end: 10.5, code: 'TRACK_TURN', event: 'Cart changes its direction of travel' },
  { id: 'camera-loss', label: 'Camera stream loss', start: 6.5, code: 'CAMERA_STALE', event: 'No recent RGB-D observations' },
  { id: 'lidar-loss', label: 'LiDAR data loss', start: 6.5, code: 'LIDAR_STALE', event: 'No recent LiDAR observations' },
  { id: 'localization', label: 'Uncertain localization', start: 5.5, code: 'LOCALIZATION_UNCERTAIN', event: 'Simulated pose variance exceeds the fixture threshold' },
  { id: 'assistance', label: 'Assistance requested', start: 5, code: 'ASSISTANCE', event: 'Robot requests operator assistance' },
  { id: 'paused', label: 'Navigation paused', start: 4.5, code: 'NAV_PAUSED', event: 'Scenario navigation pause' },
  { id: 'yard', label: 'Loading yard', site: outdoorSite },
].map(scenario => ({ site: indoorSite, ...scenario }));
const sensorProfiles = [
  { id: 'lidar', name: '3D LiDAR', frame: 'sim_lidar', hz: 10, maxAgeMs: 400, offset: [0.44, 0, 0.54] },
  { id: 'rgbd', name: 'Front RGB-D', frame: 'sim_camera_optical', hz: 10, maxAgeMs: 500, offset: [0.63, 0, 0.43] },
  { id: 'odom', name: 'Odometry', frame: 'sim_warehouse', hz: 10, maxAgeMs: 400 },
  { id: 'steer', name: 'Steering encoder', frame: 'sim_front_steer', hz: 10, maxAgeMs: 400 },
];
const stopStates = {
  pedestrian: 'Yielding', blocked: 'Blocked', localization: 'Localization hold',
  assistance: 'Assistance requested', paused: 'Paused', 'lidar-loss': 'Perception hold',
};

function sensorState(stamps, stampMs) {
  return sensorProfiles.map(profile => {
    const last = stamps.get(profile.id) ?? null;
    const ageMs = last === null ? null : stampMs - last;
    const status = ageMs === null ? 'unavailable' : ageMs > profile.maxAgeMs ? 'stale' : 'fresh';
    return { ...profile, stampMs: last, ageMs, status };
  });
}

function predictedArc(pose, steering) {
  const curvature = Math.tan(steering) / delta.wheelbase;
  return Array.from({ length: 16 }, (_, index) => {
    const distance = index * 0.15;
    if (Math.abs(curvature) < 1e-6) return [pose.x + distance * Math.cos(pose.yaw), pose.y + distance * Math.sin(pose.yaw), 0];
    const heading = pose.yaw + curvature * distance;
    return [pose.x + (Math.sin(heading) - Math.sin(pose.yaw)) / curvature,
      pose.y + (Math.cos(pose.yaw) - Math.cos(heading)) / curvature, 0];
  });
}

export function createSimulation(id, sessionPrefix = 'fixture') {
  const scenario = scenarios.find(item => item.id === id);
  if (!scenario) throw new RangeError('Unknown scenario');
  const environment = environments[scenario.site];
  const { frame, route } = environment;
  const [minX, minY, maxX, maxY] = environment.bounds;
  let sequence = 0;
  let cycle = 0;
  let x = route.start;
  let tracker = new Tracker();
  let history = [];
  const stamps = new Map();

  return {
    next() {
      if (sequence * intervalMs > durationMs) {
        sequence = 0;
        cycle++;
        x = route.start;
        tracker = new Tracker();
        history = [];
        stamps.clear();
      }
      const stampMs = sequence * intervalMs;
      const time = stampMs / 1000;
      const activeEvent = time >= scenario.start && time < (scenario.end ?? Infinity);
      const lidarAvailable = !(id === 'lidar-loss' && activeEvent);
      const cameraAvailable = !(id === 'camera-loss' && activeEvent);
      if (lidarAvailable) stamps.set('lidar', stampMs);
      if (cameraAvailable) stamps.set('rgbd', stampMs);
      stamps.set('odom', stampMs);
      stamps.set('steer', stampMs);
      const sensors = sensorState(stamps, stampMs).map(sensor => sensor.id === 'odom' ? { ...sensor, frame } : sensor);
      const warning = activeEvent && (id !== 'lidar-loss' && id !== 'camera-loss'
        || sensors.some(sensor => sensor.status === 'stale'));
      const navState = warning ? stopStates[id] ?? 'En route' : 'En route';
      const dx = navState === 'En route' ? 0.64 : 0;
      if (sequence > 0) x += dx * intervalMs / 1000;
      const phase = (x - route.start) / route.wavelength;
      const slope = (route.amplitude / route.wavelength) * Math.cos(phase);
      const curvature = -route.amplitude * Math.sin(phase) / (route.wavelength ** 2 * (1 + slope ** 2) ** 1.5);
      const pose = { x: round(x), y: round(routeY(route, x)), yaw: round(Math.atan(slope)) };
      const steering = Math.atan(delta.wheelbase * curvature);
      const grid = new Occupancy([minX, minY], (maxX - minX) * 2, (maxY - minY) * 2, 0.5);
      const actors = actorsAt(id, time);
      const sensorPose = offset => {
        const [x, y, z] = transformPoint(offset, { ...pose, z: 0 });
        return { x, y, z, yaw: pose.yaw };
      };
      const scan = lidarAvailable ? scanEnvironment(environment, sensorPose(sensorProfiles[0].offset), actors, grid) : null;
      const cameraPose = sensorPose(sensorProfiles[1].offset);
      const sample = actor => ({ position: actor.position.map(round), size: actor.size, frame, stampMs });
      const geometry = actors.filter(actor => scan?.visible.has(actor)).map(actor => ({ ...sample(actor), source: 'lidar' }));
      const semantics = cameraAvailable ? actors.filter(actor => actor.confidence && cameraSees(environment, cameraPose, actor, actors)).map(actor => ({
        ...sample(actor), source: 'rgbd', label: actor.label, confidence: actor.confidence,
      })) : [];
      const observations = associateObservations(geometry, semantics, { frame, nowMs: stampMs, maxAgeMs: 200 });
      if (sequence % 3 === 0) history = [...history, [pose.x, pose.y, 0]].slice(-48);
      const planned = Array.from({ length: 37 }, (_, index) => routePoint(route, x + (route.goal - x) * index / 36));
      return {
        version: 1, mode: 'SIMULATION', clock: 'SCENARIO_ELAPSED', session: `${sessionPrefix}-${id}-${cycle}`,
        sequence: sequence++, stampMs, durationMs, frame, scenario: id, site: environment.id,
        robot: {
          name: 'RBW–D3', modelId, pose, speed: round(dx * Math.hypot(1, slope)), steering: round(steering),
          battery: round(82 - time / 110), localization: {
            status: id === 'localization' && warning ? 'uncertain' : 'valid',
            varianceM2: id === 'localization' && warning ? 0.46 : 0.012,
          },
        },
        sensors, tracks: tracker.update(observations, stampMs), occupancy: grid.snapshot(),
        cloud: scan ? { points: scan.points, stampMs, frame } : null,
        nav: {
          state: navState, destination: route.destination, planned, history,
          local: dx ? planned.filter(point => point[0] <= x + 3.3) : [],
          prediction: dx ? predictedArc(pose, steering) : [],
        },
        events: warning ? [{ code: scenario.code, label: scenario.event }] : [],
      };
    },
  };
}

export function buildRecording(id) {
  const simulation = createSimulation(id, 'recorded');
  const frames = [];
  const events = [];
  let previous = '';
  for (let step = 0; step * intervalMs <= durationMs; step++) {
    const snapshot = simulation.next();
    frames.push(snapshot);
    const [event] = snapshot.events;
    if (event && event.code !== previous) events.push({ ...event, stampMs: snapshot.stampMs });
    previous = event?.code;
  }
  return { version: 1, scenario: id, sourceMode: 'SIMULATION', intervalMs, durationMs, frames, events };
}
