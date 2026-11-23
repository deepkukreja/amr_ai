import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const world = ([x, y, z]) => new THREE.Vector3(x, z, -y);
const standard = (color, roughness = 0.73, metalness = 0.14) => new THREE.MeshStandardMaterial({ color, roughness, metalness });
const boxGeometry = new THREE.BoxGeometry(1, 1, 1);
const wheelGeometry = new THREE.CylinderGeometry(1, 1, 1, 24);
const colors = { rack: '#466385', beam: '#b16a3d', steel: '#6b8090', pallet: '#8e7356', cyan: '#41dcf7', amber: '#ffad56' };
const hasMap = frame => ['GAZEBO', 'NAV2_PLAN', 'ROS_STREAM'].includes(frame?.provenance);
const hasNav2 = frame => ['NAV2_PLAN', 'ROS_STREAM'].includes(frame?.provenance);

function box(parent, x, y, z, width, height, depth, material) {
  const mesh = new THREE.Mesh(boxGeometry, material);
  mesh.position.set(x, y, z);
  mesh.scale.set(width, height, depth);
  parent.add(mesh);
  return mesh;
}

function batch(scene, specifications, material) {
  if (!specifications.length) return;
  const meshes = new THREE.InstancedMesh(boxGeometry, material, specifications.length);
  const matrix = new THREE.Matrix4();
  specifications.forEach(([x, y, z, width, height, depth], index) => {
    matrix.makeScale(width, height, depth).setPosition(x, y, z);
    meshes.setMatrixAt(index, matrix);
  });
  scene.add(meshes);
}

function makeEnvironment(scene, environment) {
  const racks = new THREE.Group();
  const walls = new THREE.Group();
  scene.add(racks, walls);
  if (!environment) return { racks, walls };
  const [minX, minY, maxX, maxY] = environment.bounds;
  const outdoor = environment.setting === 'Outdoor';
  box(scene, (minX + maxX) / 2, -0.055, -(minY + maxY) / 2, maxX - minX, 0.1, maxY - minY, standard(outdoor ? '#344453' : '#223348'));
  if (environment.structures.some(shape => shape.kind === 'rack')) {
    const width = maxY - minY;
    box(walls, (minX + maxX) / 2, 4.12, 0, maxX - minX, 0.14, width, standard('#26384e'));
    const fixture = standard('#546979');
    const emitter = new THREE.MeshBasicMaterial({ color: '#b1e8ff', toneMapped: false });
    for (let x = minX + 3; x < maxX - 3; x += 4) {
      box(walls, x, 3.97, 0, 0.13, 0.16, width, fixture);
      box(walls, x, 3.83, 0, 0.17, 0.036, 1.61, emitter);
    }
  }
  const grid = new THREE.BufferGeometry();
  const lines = [];
  for (let x = minX; x <= maxX; x += 0.25) lines.push(x, 0.003, -minY, x, 0.003, -maxY);
  for (let y = minY; y <= maxY; y += 0.25) lines.push(minX, 0.003, -y, maxX, 0.003, -y);
  grid.setAttribute('position', new THREE.Float32BufferAttribute(lines, 3));
  scene.add(new THREE.LineSegments(grid, new THREE.LineBasicMaterial({ color: '#427e99', transparent: true, opacity: 0.34 })));
  const bins = { posts: [], beams: [], crates: [], gray: [], pallets: [], guards: [] };
  const fixtures = {
    container: standard('#4d7185'), barrier: standard('#697780'), bollard: standard('#ce9147'),
  };
  const seams = standard('#8197a0');
  for (const shape of environment.structures) {
    const [x, y] = shape.position;
    const [length, depth, height] = shape.size;
    const z = -y;
    if (shape.kind === 'wall') {
      box(walls, x, height / 2, z, length, height, depth, standard('#263b50'));
      if (length < 1) {
        box(walls, x - 0.18, 1.4, 0, 0.07, 2.75, 2.65, standard('#173044'));
        for (let h = 0.2; h < 2.75; h += 0.15) box(walls, x - 0.231, h, 0, 0.025, 0.035, 2.58, standard('#3a5064'));
      }
      continue;
    }
    if (shape.kind !== 'rack') {
      box(racks, x, height / 2, z, length, height, depth, fixtures[shape.kind] ?? seams);
      if (shape.kind === 'container') {
        for (let offset = -length / 2 + 0.16; offset < length / 2; offset += 0.25) {
          box(racks, x + offset, height / 2, z + depth / 2, 0.021, height * 0.88, 0.023, seams);
        }
        box(racks, x, height + 0.008, z, length, 0.025, depth, seams);
      }
      if (shape.kind === 'bollard') box(racks, x, height * 0.78, z, length + 0.015, 0.14, depth + 0.015, seams);
      continue;
    }
    for (const end of [-1, 1]) {
      for (const side of [-1, 1]) {
        const pierX = x + end * (length / 2 - 0.055);
        const pierZ = z + side * (depth / 2 - 0.045);
        bins.posts.push([pierX, height / 2, pierZ, 0.11, height, 0.09]);
        if (side === -Math.sign(z)) bins.guards.push([pierX, 0.28, pierZ + side * 0.11, 0.15, 0.56, 0.14]);
      }
    }
    for (const level of [0.18, 1.11, 2.04, 2.96]) {
      for (const side of [-1, 1]) bins.beams.push([x, level, z + side * (depth / 2 - 0.04), length, 0.11, 0.09]);
      bins.posts.push([x, level - 0.035, z, length, 0.06, depth - 0.17]);
      if (level > 2.9) continue;
      for (let item = 0; item < 5; item++) {
        const local = (item - 2) * 0.73;
        const cargoHeight = 0.34 + 0.22 * (1 + Math.sin(x * 3.7 + y + item * 3.15 + level));
        bins.pallets.push([x + local, level + 0.09, z, 0.66, 0.085, 1.03]);
        const collection = (item + Math.round(level)) % 3 ? bins.crates : bins.gray;
        collection.push([x + local, level + 0.133 + cargoHeight / 2, z, 0.59, cargoHeight, 0.95]);
        collection.push([x + local, level + 0.14 + cargoHeight, z, 0.63, 0.04, 0.99]);
      }
    }
  }
  batch(racks, bins.posts, standard(colors.rack));
  batch(racks, bins.beams, standard(colors.beam));
  batch(racks, bins.pallets, standard(colors.pallet));
  batch(racks, bins.crates, standard('#4b657a'));
  batch(racks, bins.gray, standard('#697f91'));
  batch(racks, bins.guards, standard('#bd8645'));
  const stripe = new THREE.MeshBasicMaterial({ color: outdoor ? '#d6b96f' : '#3d879e', transparent: true, opacity: outdoor ? 0.7 : 0.4 });
  for (const y of outdoor ? [-3.6, 3.8] : [-2.6, 2.6]) {
    for (let x = minX + 1; x < maxX - 3; x += 1.9) box(scene, x, 0.01, -y, 0.96, 0.008, outdoor ? 0.065 : 0.031, stripe);
  }
  return { racks, walls };
}

export function makeRobot(geometry) {
  const group = new THREE.Group();
  const { wheelbase, rearTrack, tireRadius, tireWidth } = geometry;
  const rubber = standard('#171d26', 0.94, 0);
  const rims = standard('#738797', 0.39, 0.62);
  const orange = standard('#f08035', 0.47, 0.22);
  const ivory = standard('#e0e5e4', 0.49, 0.17);
  const deck = standard('#273d50', 0.55, 0.36);
  const black = standard('#142332');
  const light = new THREE.MeshBasicMaterial({ color: '#ffd0a0' });
  const rimLight = new THREE.MeshBasicMaterial({ color: '#6fe8fa' });
  const outline = new THREE.Shape();
  outline.moveTo(-0.23, -0.34);
  outline.lineTo(0.34, -0.34);
  outline.lineTo(0.72, -0.23);
  outline.lineTo(0.91, -0.14);
  outline.lineTo(0.91, 0.14);
  outline.lineTo(0.72, 0.23);
  outline.lineTo(0.34, 0.34);
  outline.lineTo(-0.23, 0.34);
  outline.closePath();
  const body = new THREE.Mesh(new THREE.ExtrudeGeometry(outline, {
    depth: 0.17, bevelEnabled: true, bevelSize: 0.024, bevelThickness: 0.021, bevelSegments: 2, steps: 1,
  }), ivory);
  body.rotation.x = -Math.PI / 2;
  body.position.y = 0.195;
  group.add(body);
  box(group, 0.13, 0.373, 0, 0.71, 0.035, 0.60, deck);
  box(group, -0.236, 0.392, 0, 0.063, 0.125, 0.65, orange);
  box(group, -0.266, 0.247, 0, 0.026, 0.064, 0.46, black);
  box(group, 0.69, 0.48, 0, 0.29, 0.33, 0.48, ivory);
  box(group, 0.70, 0.749, 0, 0.21, 0.22, 0.36, orange);
  box(group, 0.707, 0.867, 0, 0.268, 0.033, 0.435, ivory);
  box(group, 0.812, 0.777, 0, 0.017, 0.057, 0.344, black);
  box(group, 0.895, 0.277, 0, 0.037, 0.053, 0.49, black);
  for (const side of [-1, 1]) {
    box(group, 0.112, 0.392, side * 0.324, 0.717, 0.128, 0.04, orange);
    box(group, 0.112, 0.456, side * 0.324, 0.717, 0.022, 0.043, black);
    box(group, -0.27, 0.29, side * 0.27, 0.02, 0.029, 0.074, light);
    box(group, 0.827, 0.777, side * 0.115, 0.016, 0.042, 0.075, rimLight);
  }
  const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.062, 0.074, 0.096, 24), black);
  mast.position.set(0.68, 0.929, 0);
  group.add(mast);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.063, 0.005, 6, 26), rimLight);
  ring.rotation.x = Math.PI / 2;
  ring.position.set(0.68, 0.94, 0);
  group.add(ring);
  const front = new THREE.Group();
  front.position.set(wheelbase, tireRadius, 0);
  group.add(front);
  const wheel = (parent, x, y, z, radius, width, material) => {
    const mesh = new THREE.Mesh(wheelGeometry, material);
    mesh.scale.set(radius, width, radius);
    mesh.rotation.x = Math.PI / 2;
    mesh.position.set(x, y, z);
    parent.add(mesh);
  };
  for (const side of [-1, 1]) {
    wheel(group, 0, tireRadius, side * rearTrack / 2, tireRadius, tireWidth, rubber);
    wheel(group, 0, tireRadius, side * (rearTrack + tireWidth) / 2, tireRadius * 0.64, 0.014, rims);
  }
  wheel(front, 0, 0, 0, tireRadius, tireWidth, rubber);
  wheel(front, 0, 0, 0, tireRadius * 0.66, tireWidth + 0.012, rims);
  group.userData.steering = front;
  return group;
}

const materials = {
  person: standard('#b6d994'), personDark: standard('#306d94'), face: standard('#cdb299'),
  lift: standard('#d3934a'), frame: standard('#324857'), cargo: standard('#819cae'), pallet: standard(colors.pallet),
};

function makeTrack(track) {
  const group = new THREE.Group();
  const [length, width, height] = track.size;
  const color = track.label === 'Forklift' ? colors.amber : track.label === 'Unknown' ? '#a4b3c5' : colors.cyan;
  const geometry = new THREE.EdgesGeometry(boxGeometry);
  const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.82 });
  const border = new THREE.LineSegments(geometry, material);
  border.scale.set(length + 0.13, height + 0.12, width + 0.13);
  border.position.y = height / 2;
  group.add(border);
  const model = new THREE.Group();
  group.add(model);
  if (track.label === 'Person') {
    const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.17, 0.34, 4, 9), materials.person);
    body.position.y = 1.13;
    model.add(body);
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.13, 12, 8), materials.face);
    head.position.y = 1.58;
    model.add(head);
    for (const side of [-1, 1]) {
      box(model, side * 0.055, 0.44, side * 0.105, 0.14, 0.8, 0.13, materials.personDark).rotation.z = side * 0.13;
      box(model, 0, 1.08, side * 0.19, 0.11, 0.42, 0.09, materials.person).rotation.x = side * 0.13;
    }
  } else if (track.label === 'Forklift') {
    box(model, 0.18, 0.34, 0, 0.95, 0.53, 0.78, materials.lift);
    box(model, 0.14, 0.67, 0, 0.66, 0.2, 0.72, materials.lift);
    box(model, 0.17, 1.50, 0, 0.69, 0.08, 0.73, materials.frame);
    for (const side of [-1, 1]) {
      box(model, -0.10, 1.07, side * 0.31, 0.059, 0.84, 0.07, materials.frame);
      box(model, 0.44, 1.07, side * 0.31, 0.059, 0.84, 0.07, materials.frame);
      box(model, -0.52, 0.88, side * 0.22, 0.09, 1.56, 0.09, materials.frame);
      box(model, -0.65, 0.12, side * 0.22, 0.29, 0.045, 0.08, materials.frame);
      const tire = new THREE.Mesh(wheelGeometry, materials.frame);
      tire.rotation.x = Math.PI / 2;
      tire.position.set(0.43, 0.21, side * 0.37);
      tire.scale.set(0.21, 0.14, 0.21);
      model.add(tire);
    }
  } else {
    box(model, 0, height * 0.51, 0, length * 0.95, height * 0.83, width * 0.94, materials.cargo);
    box(model, 0, height * 0.94, 0, length, 0.048, width, materials.cargo);
    for (const side of [-1, 1]) {
      box(model, 0, 0.095, width * 0.34 * side, length * 0.99, 0.13, width * 0.18, materials.pallet);
      box(model, side * length * 0.37, height * 0.53, 0, 0.06, height * 0.82, width * 0.97, materials.frame);
    }
  }
  group.userData = { label: track.label, geometry, material, border, model };
  return group;
}

function route(color, dashed) {
  const material = dashed
    ? new THREE.LineDashedMaterial({ color, dashSize: 0.18, gapSize: 0.13, transparent: true, opacity: 0.86 })
    : new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95 });
  return new THREE.Line(new THREE.BufferGeometry(), material);
}

function setPoints(object, points) {
  const old = object.geometry;
  object.geometry = new THREE.BufferGeometry().setFromPoints(points);
  old.dispose();
}

function updateRoute(object, coordinates, elevation) {
  setPoints(object, coordinates.map(point => world([point[0], point[1], elevation])));
  if (object.isLine) object.computeLineDistances();
  object.visible = coordinates.length > 1;
}

function updateRibbon(object, points) {
  const vertices = [];
  const indices = [];
  points.forEach(([x, y], index) => {
    const a = points[Math.max(0, index - 1)];
    const b = points[Math.min(index + 1, points.length - 1)];
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const length = Math.hypot(dx, dy) || 1;
    const halfWidth = (index === 0 ? 0.12 : 0.28) / length;
    vertices.push(x - dy * halfWidth, 0.048, -y - dx * halfWidth,
      x + dy * halfWidth, 0.048, -y + dx * halfWidth);
    if (index) {
      const i = index * 2;
      indices.push(i - 2, i - 1, i, i - 1, i + 1, i);
    }
  });
  object.geometry.dispose();
  object.geometry = new THREE.BufferGeometry();
  object.geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  object.geometry.setIndex(indices);
  object.visible = points.length > 1;
}

function dotTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 32;
  const context = canvas.getContext('2d');
  context.fillStyle = '#fff';
  context.beginPath();
  context.arc(16, 16, 11, 0, Math.PI * 2);
  context.fill();
  return new THREE.CanvasTexture(canvas);
}

export function createScene(host, labels, environment, profiles) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.6));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.38;
  renderer.domElement.setAttribute('aria-label', 'Interactive 3D environment');
  host.append(renderer.domElement);
  const scene = new THREE.Scene();
  const background = environment?.setting === 'Outdoor' ? '#344f68' : '#0c1523';
  scene.background = new THREE.Color(background);
  scene.fog = new THREE.Fog(background, 17, 42);
  scene.add(new THREE.HemisphereLight('#c9e8ff', '#253750', 2.15));
  const key = new THREE.DirectionalLight('#e1efff', 2.0);
  key.position.set(-3, 10, 5);
  scene.add(key);
  const backdrop = new THREE.Group();
  scene.add(backdrop);
  const { racks, walls } = makeEnvironment(backdrop, environment);
  const chassis = new Map();
  const poseMarker = new THREE.Group();
  const marker = new THREE.Mesh(new THREE.ConeGeometry(0.27, 0.55, 3), new THREE.MeshBasicMaterial({ color: '#ffaa5d' }));
  marker.rotation.z = -Math.PI / 2;
  poseMarker.add(marker);
  scene.add(poseMarker);
  poseMarker.visible = false;
  for (const [id, profile] of Object.entries(profiles)) {
    if (profile.kind !== 'delta-steer') continue;
    const robot = makeRobot(profile.geometry);
    const { tireWidth, wheelbase, rearTrack } = profile.geometry;
    const halfWidth = (rearTrack + tireWidth) / 2 + 0.035;
    const nose = wheelbase + 0.22;
    const footprint = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(
      [[-0.29, -halfWidth], [nose, -halfWidth], [nose, halfWidth], [-0.29, halfWidth]]
        .map(([x, z]) => new THREE.Vector3(x, 0.017, z))),
    new THREE.LineBasicMaterial({ color: '#65dfea', transparent: true, opacity: 0.55 }));
    robot.add(footprint);
    robot.visible = false;
    chassis.set(id, robot);
    scene.add(robot);
  }

  const dot = dotTexture();
  const planned = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({
    color: '#49dfff', map: dot, alphaTest: 0.4, size: 0.17,
  }));
  const routeRibbon = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({
    color: '#32ddcf', side: THREE.DoubleSide, transparent: true, opacity: 0.52, depthWrite: false, toneMapped: false,
  }));
  routeRibbon.visible = false;
  routeRibbon.renderOrder = 2;
  const planCenter = route('#d0fcff', false);
  planCenter.renderOrder = 3;
  const local = route('#6ce8fd', false);
  const history = route('#5aa5bd', false);
  const prediction = route('#e7f0fa', true);
  const heading = route('#e7f0fa', false);
  const cloud = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({
    color: '#5be8ff', map: dot, alphaTest: 0.4, size: 0.065, transparent: true, opacity: 0.65,
  }));
  const ranged = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({
    color: '#d6a4ff', map: dot, alphaTest: 0.4, size: 11, sizeAttenuation: false, fog: false,
  }));
  const grid = new THREE.InstancedMesh(boxGeometry, new THREE.MeshBasicMaterial({ vertexColors: false, transparent: true, opacity: 0.86 }), 4096);
  grid.frustumCulled = false;
  grid.count = 0;
  scene.add(planned, routeRibbon, planCenter, local, history, prediction, heading, cloud, ranged, grid);
  const camera = new THREE.PerspectiveCamera(53, 1, 0.055, 100);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.maxDistance = 47;
  controls.minDistance = 2.0;
  controls.maxPolarAngle = Math.PI * 0.495;
  let mode = 'follow';
  let frame;
  let selected = '';
  let currentSession;
  let lastPose;
  let renderedOccupancy;
  const tracks = new Map();
  const occupancyColors = { '-1': new THREE.Color('#202d45'), 0: new THREE.Color('#286b78'), 100: new THREE.Color('#c47842') };
  const planMapColors = { '-1': new THREE.Color('#202933'), 0: new THREE.Color('#465563'), 100: new THREE.Color('#8b7c6d') };
  const matrix = new THREE.Matrix4();

  function reset() {
    const center = lastPose ?? new THREE.Vector3(-3.8, 0, 0);
    if (mode === 'occupancy' || mode === 'bird') {
      const map = frame?.occupancy;
      const recorded = hasMap(frame);
      const bounds = environment?.bounds ?? [-10, -10, 10, 10];
      const yaw = map?.yaw ?? 0;
      const halfWidth = (map?.width ?? 0) * (map?.resolution ?? 0) / 2;
      const halfHeight = (map?.height ?? 0) * (map?.resolution ?? 0) / 2;
      const x = recorded ? map.origin[0] + halfWidth * Math.cos(yaw) - halfHeight * Math.sin(yaw) : (bounds[0] + bounds[2]) / 2;
      const z = recorded ? -map.origin[1] - halfWidth * Math.sin(yaw) - halfHeight * Math.cos(yaw) : -(bounds[1] + bounds[3]) / 2;
      controls.target.set(x, 0, z);
      const overhead = recorded ? Math.max(9, 1.45 * Math.max(halfWidth, halfHeight) * 2) : 25;
      camera.position.set(x + (mode === 'occupancy' ? 0 : -10.2), overhead, z + (mode === 'occupancy' ? 0.02 : 11.8));
    } else if (mode === 'cloud' || mode === 'debug') {
      controls.target.copy(center).add(new THREE.Vector3(4.2, 0.42, -0.1));
      camera.position.copy(center).add(new THREE.Vector3(-6.4, 7.0, 5.4));
    } else if (hasNav2(frame) && frame.nav.planned.length) {
      const last = world(frame.nav.planned.at(-1));
      const bearing = last.clone().sub(center).setY(0).normalize();
      const side = new THREE.Vector3(-bearing.z, 0, bearing.x);
      controls.target.copy(center).lerp(last, 0.36).setY(0.18);
      camera.position.copy(center).addScaledVector(bearing, -3.0).addScaledVector(side, 1.85).setY(3.6);
    } else {
      controls.target.copy(center).add(new THREE.Vector3(3.4, 0.52, -0.28));
      camera.position.copy(center).add(new THREE.Vector3(-3.65, 2.52, 1.06));
    }
    controls.update();
  }

  function updateOccupancy(occupancy) {
    if (renderedOccupancy === occupancy) return;
    renderedOccupancy = occupancy;
    const { cells, width, resolution, origin, yaw = 0 } = occupancy;
    const cosine = Math.cos(yaw);
    const sine = Math.sin(yaw);
    const scale = new THREE.Vector3(resolution * 0.97, 0.008, resolution * 0.97);
    const palette = hasNav2(frame) ? planMapColors : occupancyColors;
    grid.count = Math.min(cells.length, 4096);
    for (let i = 0; i < grid.count; i++) {
      const x = (i % width + 0.5) * resolution;
      const y = (Math.floor(i / width) + 0.5) * resolution;
      matrix.makeRotationY(yaw).scale(scale).setPosition(origin[0] + x * cosine - y * sine, 0.021, -origin[1] - x * sine - y * cosine);
      grid.setMatrixAt(i, matrix);
      grid.setColorAt(i, palette[cells[i]]);
    }
    grid.instanceMatrix.needsUpdate = true;
    if (grid.instanceColor) grid.instanceColor.needsUpdate = true;
  }

  function updateMode() {
    const recorded = hasMap(frame);
    backdrop.visible = !recorded;
    racks.visible = mode !== 'occupancy' && mode !== 'cloud' && mode !== 'semantic';
    walls.visible = mode === 'follow';
    grid.visible = mode === 'occupancy' || recorded;
    cloud.visible = Boolean(frame?.cloud) && ['follow', 'cloud', 'debug'].includes(mode);
    ranged.visible = Boolean(frame?.objects?.items.length) && mode !== 'cloud';
    for (const { mesh } of tracks.values()) {
      mesh.visible = mode !== 'cloud';
      mesh.userData.model.visible = !['occupancy', 'debug', 'semantic'].includes(mode);
    }
    if (grid.visible && frame) updateOccupancy(frame.occupancy);
  }

  function removeTrack(id) {
    const { mesh, tag } = tracks.get(id);
    scene.remove(mesh);
    mesh.userData.geometry.dispose();
    mesh.userData.material.dispose();
    mesh.traverse(child => {
      if (child.geometry && child.geometry !== boxGeometry && child.geometry !== wheelGeometry && child !== mesh.userData.border) child.geometry.dispose();
    });
    tag.remove();
    tracks.delete(id);
  }

  function update(snapshot) {
    frame = snapshot;
    const { pose, steering } = frame.robot;
    const recorded = hasMap(frame);
    const nav2 = hasNav2(frame);
    const position = pose && world([pose.x, pose.y, 0]);
    if (currentSession !== frame.session || (position && !lastPose)) {
      currentSession = frame.session;
      for (const id of tracks.keys()) removeTrack(id);
      lastPose = position;
      reset();
    } else if (position && lastPose && (mode === 'follow' || mode === 'semantic')) {
      const delta = position.clone().sub(lastPose);
      controls.target.add(delta);
      camera.position.add(delta);
    }
    const robot = chassis.get(frame.robot.modelId);
    for (const model of chassis.values()) model.visible = Boolean(position) && model === robot;
    poseMarker.visible = Boolean(position) && !robot;
    if (position) {
      lastPose = position;
      poseMarker.position.copy(position).setY(0.12);
      poseMarker.rotation.y = pose.yaw;
      if (robot) {
        robot.position.copy(position);
        robot.rotation.y = pose.yaw;
        robot.userData.steering.rotation.y = steering ?? 0;
      }
    }
    updateRoute(planned, nav2 ? [] : frame.nav.planned, 0.042);
    if (nav2) {
      updateRibbon(routeRibbon, frame.nav.planned);
      updateRoute(planCenter, frame.nav.planned, 0.059);
    } else {
      routeRibbon.visible = false;
      planCenter.visible = false;
    }
    updateRoute(local, frame.nav.local, 0.025);
    updateRoute(history, frame.nav.history, 0.028);
    updateRoute(prediction, frame.nav.prediction, 0.061);
    updateRoute(heading, pose && !recorded ? [
      [pose.x, pose.y, 0], [pose.x + Math.cos(pose.yaw) * 0.98, pose.y + Math.sin(pose.yaw) * 0.98, 0],
    ] : [], 0.045);
    setPoints(cloud, (frame.cloud?.points ?? []).map(world));
    if (frame.objects?.items.length) {
      setPoints(ranged, frame.objects.items.map(({ position: [x, y] }) => world([x, y, 0.08])));
    }
    const seen = new Set();
    for (const track of frame.tracks) {
      seen.add(track.id);
      if (tracks.has(track.id) && tracks.get(track.id).mesh.userData.label !== track.label) removeTrack(track.id);
      if (!tracks.has(track.id)) {
        const mesh = makeTrack(track);
        const tag = document.createElement('span');
        tag.className = 'scene-label';
        labels.append(tag);
        scene.add(mesh);
        tracks.set(track.id, { mesh, tag });
      }
      const item = tracks.get(track.id);
      item.mesh.position.copy(world(track.position));
      item.mesh.userData.material.opacity = track.status === 'observed' ? 0.82 : 0.29;
      item.track = track;
      item.tag.textContent = `${track.label} · ${track.id}`;
      item.tag.dataset.kind = track.label.toLowerCase();
      item.tag.dataset.status = track.status;
    }
    for (const id of tracks.keys()) if (!seen.has(id)) removeTrack(id);
    updateMode();
  }

  function resize() {
    const width = host.clientWidth;
    const height = host.clientHeight;
    if (!width || !height) return;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
  const observer = new ResizeObserver(resize);
  observer.observe(host);
  reset();
  resize();
  let lastRender = 0;
  let animation;
  function render(timestamp) {
    animation = requestAnimationFrame(render);
    if (timestamp - lastRender < 28) return;
    lastRender = timestamp;
    controls.update();
    renderer.render(scene, camera);
    for (const [id, { tag, mesh, track }] of tracks) {
      const position = mesh.position.clone();
      position.y += track.size[2] + 0.22;
      position.project(camera);
      tag.hidden = !mesh.visible || position.z > 1 || position.z < -1 || Math.abs(position.x) > 0.93 || Math.abs(position.y) > 0.91
        || (position.y < -0.70 && id !== selected);
      tag.style.left = `${(position.x + 1) * 50}%`;
      tag.style.top = `${(1 - position.y) * 50}%`;
      tag.dataset.selected = String(id === selected);
    }
  }
  animation = requestAnimationFrame(render);

  return {
    update,
    setMode(next) { if (next !== mode) { mode = next; updateMode(); reset(); } },
    select(id) { selected = id; },
    reset,
    zoom(scale) {
      const direction = camera.position.clone().sub(controls.target);
      direction.setLength(THREE.MathUtils.clamp(direction.length() * scale, controls.minDistance, controls.maxDistance));
      camera.position.copy(controls.target).add(direction);
      controls.update();
    },
    dispose() {
      cancelAnimationFrame(animation);
      observer.disconnect();
      controls.dispose();
      for (const id of tracks.keys()) removeTrack(id);
      scene.traverse(object => {
        if (object.geometry && object.geometry !== boxGeometry && object.geometry !== wheelGeometry) object.geometry.dispose();
        if (object.material && !Object.values(materials).includes(object.material)) object.material.dispose();
        if (object.isInstancedMesh) object.dispose();
      });
      dot.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    },
  };
}
