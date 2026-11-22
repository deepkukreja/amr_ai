import { open } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

export const nav2Plan = { id: 'nav2-plan', label: 'Recorded Nav2 plan', replayOnly: true, snapshotOnly: true };
const defaultFile = fileURLToPath(new URL('../../outputs/nav2_console_20261122/nav2-world.json', import.meta.url));
const maxBytes = 12 * 1024 * 1024;

export function planRecording(capture) {
  const { frame, source, clock, site, map, pose, plan } = capture;
  if (capture.version !== 1 || !['NAV2_PLANNING_FIXTURE', 'ROS_CAPTURE'].includes(source)
    || !['ROS_SYSTEM', 'ROS_SIM'].includes(clock) || typeof site !== 'string' || !site || site.length > 100
    || typeof frame !== 'string' || !frame || ![map, pose, plan].every(layer => layer?.frame === frame
      && Number.isSafeInteger(layer.stampMs) && layer.stampMs >= 0)
    || ![pose.x, pose.y, pose.yaw].every(Number.isFinite) || pose.stampMs !== plan.stampMs
    || !Array.isArray(plan.points) || plan.points.length < 2 || plan.points.length > 20000
    || !plan.points.every(point => Array.isArray(point) && point.length === 3 && point.every(Number.isFinite))) {
    throw new Error('Invalid or unregistered Nav2 capture');
  }
  const { width, height, resolution, origin, yaw, cells } = map;
  if (![width, height].every(size => Number.isInteger(size) && size > 0)
    || width * height > 2000000 || !Number.isFinite(resolution) || resolution <= 0
    || !Number.isFinite(yaw) || !Array.isArray(origin) || origin.length !== 2 || !origin.every(Number.isFinite)
    || !Array.isArray(cells) || cells.length !== width * height
    || !cells.every(cell => Number.isInteger(cell) && cell >= -1 && cell <= 100)) {
    throw new Error('Invalid occupancy grid');
  }
  let stride = Math.max(1, Math.ceil(Math.sqrt(cells.length / 4000)));
  while (Math.ceil(width / stride) * Math.ceil(height / stride) > 4096) stride++;
  const columns = Math.ceil(width / stride);
  const rows = Math.ceil(height / stride);
  const displayCells = Array.from({ length: columns * rows }, (_, index) => {
    const column = (index % columns) * stride;
    const row = Math.floor(index / columns) * stride;
    let unknown = column + stride > width || row + stride > height;
    for (let y = row; y < Math.min(row + stride, height); y++) {
      for (let x = column; x < Math.min(column + stride, width); x++) {
        const cell = cells[y * width + x];
        if (cell === 100) return 100;
        if (cell !== 0) unknown = true;
      }
    }
    return unknown ? -1 : 0;
  });
  const visualModel = capture.robotModel === 'd3-prototype' || source === 'NAV2_PLANNING_FIXTURE';
  return {
    version: 1, scenario: nav2Plan.id, sourceMode: source, durationMs: 0, intervalMs: 1, events: [],
    frames: [{
      version: 1, mode: 'REPLAY', provenance: 'NAV2_PLAN', source, clock, frame, site,
      session: `nav2-plan-${plan.stampMs}`, sequence: 0, stampMs: 0, sourceStampMs: plan.stampMs, durationMs: 0, scenario: nav2Plan.id,
      robot: {
        name: visualModel ? 'D3 · Design profile' : 'ROS robot',
        modelId: visualModel ? 'd3-prototype' : null, pose: { x: pose.x, y: pose.y, yaw: pose.yaw },
        speed: null, steering: null, battery: null, localization: { status: 'unavailable', varianceM2: null },
      },
      occupancy: { origin, yaw, width: columns, height: rows, resolution: resolution * stride, cells: displayCells },
      sensors: [], tracks: [], cloud: null, events: [],
      nav: { state: 'Plan captured', destination: null, planned: plan.points, local: [], history: [], prediction: [] },
    }],
  };
}

export async function loadNav2Plan(path = process.env.RBW_NAV2_CAPTURE ?? defaultFile) {
  const file = await open(path, 'r');
  try {
    const { size, mode } = await file.stat();
    if ((mode & 0o170000) !== 0o100000 || size > maxBytes) throw new Error('Unsupported Nav2 capture file');
    return planRecording(JSON.parse(await file.readFile('utf8')));
  } finally {
    await file.close();
  }
}
