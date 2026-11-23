export const clipLengthMs = 12_000;

export const cameraRig = {
  front: { label: 'Front', offset: [0.75, 0, 0.98], yaw: 0 },
  rear: { label: 'Rear', offset: [-0.24, 0, 0.81], yaw: Math.PI },
  left: { label: 'Left', offset: [0.15, 0.34, 0.78], yaw: Math.PI / 2 },
  right: { label: 'Right', offset: [0.15, -0.34, 0.78], yaw: -Math.PI / 2 },
};

export const clips = [
  { id: 'reverse', label: 'Reverse dock', site: 'Loading yard' },
  { id: 'crossing', label: 'Doorway crossing', site: 'Warehouse entrance' },
  { id: 'passing', label: 'Close pass', site: 'Warehouse aisle' },
];

const clampUnit = value => Math.min(1, Math.max(0, value));

export function sampleClip(id, stampMs) {
  const t = clampUnit(stampMs / clipLengthMs);
  if (id === 'crossing') {
    const approach = 3.6 * clampUnit(t / 0.32);
    const depart = 3.4 * clampUnit((t - 0.73) / 0.27);
    return {
      robot: { x: -5.8 + approach + depart, y: 0, yaw: 0 },
      person: { x: 0.74, y: 2.9 - 5.8 * clampUnit((t - 0.25) / 0.46) },
      loaded: false,
    };
  }
  const reversing = id === 'reverse';
  return {
    robot: { x: reversing ? -1.75 - 4.35 * t : 1.7 + 7 * t, y: reversing ? 0 : -0.51, yaw: 0 },
    person: null,
    loaded: true,
  };
}

export function cameraPose(id, robot) {
  const { offset: [x, y, z], yaw } = cameraRig[id];
  const cosine = Math.cos(robot.yaw);
  const sine = Math.sin(robot.yaw);
  return { x: robot.x + x * cosine - y * sine, y: robot.y + x * sine + y * cosine, z, yaw: robot.yaw + yaw };
}

const standardFocus = { fps: 12, width: 768, height: 432, delayMs: 65 };
const standardPreview = { fps: 6, width: 352, height: 198, delayMs: 65 };
const standardGrid = { fps: 8, width: 512, height: 288, delayMs: 65 };
const limitedFocus = { fps: 10, width: 640, height: 360, delayMs: 135 };
const limitedGrid = { fps: 8, width: 448, height: 252, delayMs: 160 };
const limitedPreview = { fps: 3, width: 256, height: 144, delayMs: 310 };

export function feedProfile(id, layout, focused, link) {
  if (layout === 'pair' && id !== 'front' && id !== 'rear') return null;
  if (link === 'constrained') {
    if (id !== focused) return limitedPreview;
    return layout === 'focus' ? limitedFocus : limitedGrid;
  }
  if (layout !== 'focus') return standardGrid;
  return id === focused ? standardFocus : standardPreview;
}

export function nextFrame(previous, nowMs, profile, frozen) {
  if (frozen || (previous?.profile === profile && nowMs - previous.deliveredAt < 1000 / profile.fps)) return previous;
  return { stampMs: Math.max(previous?.stampMs ?? 0, nowMs - profile.delayMs), deliveredAt: nowMs, profile };
}

export function feedState(frame, nowMs) {
  if (!frame) return 'unavailable';
  return nowMs - frame.deliveredAt >= 800 ? 'stale' : 'ready';
}
