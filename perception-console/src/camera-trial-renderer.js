import * as THREE from 'three';
import { delta } from '../server/environments.mjs';
import { makeRobot } from './scene.js';
import { cameraPose, sampleClip } from './camera-trial.js';

const width = 768;
const height = 432;
const aspect = width / height;

export function createTrialRenderer(onLost) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#b4c7d3');
  scene.fog = new THREE.Fog('#c8d4d9', 18, 43);
  scene.add(new THREE.HemisphereLight('#f0f4ff', '#919998', 2.25));
  const sun = new THREE.DirectionalLight('#fff5e1', 1.45);
  sun.position.set(-7, 12, -5);
  scene.add(sun);
  const geometry = new THREE.BoxGeometry();
  const materials = new Map();
  function surface(color) {
    if (!materials.has(color)) materials.set(color, new THREE.MeshStandardMaterial({ color, roughness: 0.8 }));
    return materials.get(color);
  }
  function box(x, y, z, length, breadth, tall, color, parent = scene) {
    const mesh = new THREE.Mesh(geometry, surface(color));
    mesh.scale.set(length, tall, breadth);
    mesh.position.set(x, z + tall / 2, -y);
    parent.add(mesh);
    return mesh;
  }
  const yard = '#78828a', concrete = '#a3a5a0', panel = '#a9b5bc', steel = '#405568', rail = '#b87950';
  const yellow = '#edb738', stripe = '#d7d8d3', carton = '#b08b66';
  box(-11, 0, -0.135, 22, 22, 0.122, yard);
  box(8.1, 0, -0.135, 16.2, 15, 0.122, concrete);
  const roof = box(8, 0, 4.05, 16.3, 15, 0.12, '#c3c6c5');
  for (const side of [-1, 1]) {
    box(0, side * 4.65, 0, 0.32, 5.7, 4.13, panel);
    box(0.04, side * 1.85, 0, 0.4, 0.17, 3.11, steel);
    box(7.95, side * 7.45, 0, 16, 0.2, 4.04, panel);
    box(-0.85, side * 2.1, 0, 0.20, 0.20, 0.91, yellow);
    box(4.9, side * 1.25, 0, 8.8, 0.055, 0.008, yellow);
    box(-4.25, side * 1.28, 0, 6.6, 0.055, 0.009, stripe);
    for (const x of [3.3, 7.7, 12.1]) {
      for (const column of [-1, 1]) {
        box(x + column * 1.9, side * 4.15, 0, 0.13, 1.38, 3.03, steel);
        box(x + column * 1.9, side * 3.42, 0, 0.2, 0.16, 0.43, yellow);
      }
      for (const [tier, z] of [0.19, 1.11, 2.04].entries()) {
        box(x, side * 4.15, z, 3.85, 1.38, 0.10, rail);
        for (const slot of [-1, 0, 1]) {
          box(x + slot * 1.1, side * 4.15, z + 0.11, 0.95, 1.10, 0.50 + 0.09 * (slot + tier), carton);
        }
      }
    }
    box(-7.25, side * 2.65, 0, 1.5, 0.90, 0.58, '#65818f');
    box(-7.25, side * 2.65, 0.58, 1.12, 0.85, 0.52, '#65818f');
    box(-8.0, side * 1.50, 0, 0.2, 0.20, 0.64, yellow);
    box(-12.5, side * 10.2, 0, 14, 1.2, 0.31, '#627667');
  }
  box(0, 0, 3.0, 0.32, 3.7, 1.18, panel);
  box(-0.03, 0, 2.95, 0.39, 3.86, 0.17, steel);
  box(15.7, 0, 0, 0.22, 15, 4.12, panel);
  for (const x of [2, 5.7, 9.4, 13.1]) box(x, 0, 3.88, 0.16, 11.5, 0.08, '#e5e9e4');
  box(-8.65, 0, 0, 0.30, 3.5, 0.42, steel);
  for (let stripeIndex = -4; stripeIndex <= 4; stripeIndex++) {
    box(-8.49, stripeIndex * 0.38, 0.13, 0.03, 0.24, 0.24, yellow);
    box(-7.5, stripeIndex * 0.36, 0, 0.05, 0.23, 0.01, yellow);
  }
  box(5.08, -1.88, 0, 1.63, 0.81, 0.94, carton);
  box(5.08, -1.88, 0.92, 1.63, 0.81, 0.035, '#c9a87d');
  box(-4.58, 5.55, 0, 4.3, 2.0, 2.12, '#537487');
  for (let rib = -4; rib <= 4; rib++) box(-4.58 + rib * 0.43, 4.53, 0.10, 0.045, 0.045, 1.9, '#6b8795');
  for (let line = -5; line <= 15; line += 2.5) box(line, 0, 0, 0.01, 13, 0.002, '#828887');

  const robot = makeRobot(delta);
  scene.add(robot);
  const cargo = box(0.12, 0, 0.39, 0.69, 0.70, 0.31, '#6b818b', robot);
  const person = new THREE.Group();
  const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.18, 0.36, 4, 10), surface('#dd9740'));
  body.position.y = 1.16;
  person.add(body);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.14, 14, 10), surface('#a57155'));
  head.position.y = 1.62;
  person.add(head);
  const legs = [-1, 1].map(side => box(side * 0.10, 0, 0.08, 0.13, 0.13, 0.82, '#364555', person));
  scene.add(person);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height, false);
  renderer.setScissorTest(true);
  renderer.domElement.addEventListener('webglcontextlost', onLost);
  const camera = new THREE.PerspectiveCamera(57, aspect, 0.07, 45);
  const mapCamera = new THREE.OrthographicCamera(-12, 12, 6.75, -6.75, 0.1, 45);
  mapCamera.up.set(0, 0, -1);
  mapCamera.position.set(1.8, 15, 0);
  mapCamera.lookAt(1.8, 0, 0);
  const mapHeading = new THREE.Mesh(new THREE.ConeGeometry(0.30, 0.88, 4).rotateZ(-Math.PI / 2),
    new THREE.MeshBasicMaterial({ color: '#ffd283', fog: false }));
  mapHeading.visible = false;
  scene.add(mapHeading);
  const locations = new WeakMap();

  function setScene(id, stampMs) {
    const state = sampleClip(id, stampMs);
    robot.position.set(state.robot.x, 0, -state.robot.y);
    robot.rotation.y = state.robot.yaw;
    cargo.visible = state.loaded;
    person.visible = Boolean(state.person);
    if (state.person) {
      person.position.set(state.person.x, 0, -state.person.y);
      person.rotation.y = Math.PI / 2;
      legs.forEach((leg, i) => { leg.rotation.x = Math.sin(stampMs / 150) * (i ? -0.27 : 0.27); });
    }
    return state.robot;
  }

  function paint(target, viewpoint, resolution) {
    const { width: w, height: h } = resolution;
    renderer.setViewport(0, height - h, w, h);
    renderer.setScissor(0, height - h, w, h);
    renderer.render(scene, viewpoint);
    if (target.width !== w || target.height !== h) {
      target.width = w;
      target.height = h;
    }
    if (!locations.has(target)) locations.set(target, target.getContext('2d', { alpha: false }));
    locations.get(target).drawImage(renderer.domElement, 0, 0, w, h, 0, 0, w, h);
  }

  return {
    paintFeed(target, id, clip, frame) {
      const pose = cameraPose(id, setScene(clip, frame.stampMs));
      camera.position.set(pose.x, pose.z, -pose.y);
      camera.lookAt(pose.x + Math.cos(pose.yaw), pose.z - (id === 'front' ? 0.10 : 0.28), -pose.y - Math.sin(pose.yaw));
      paint(target, camera, frame.profile);
    },
    paintMap(target, clip, nowMs) {
      const pose = setScene(clip, nowMs);
      mapHeading.position.set(pose.x, 1.18, -pose.y);
      mapHeading.rotation.y = pose.yaw;
      mapHeading.visible = true;
      roof.visible = false;
      paint(target, mapCamera, { width: 384, height: 216 });
      roof.visible = true;
      mapHeading.visible = false;
    },
    dispose() {
      renderer.domElement.removeEventListener('webglcontextlost', onLost);
      const geometries = new Set();
      const usedMaterials = new Set();
      scene.traverse(mesh => {
        if (mesh.geometry) geometries.add(mesh.geometry);
        if (mesh.material) usedMaterials.add(mesh.material);
      });
      geometries.forEach(item => item.dispose());
      usedMaterials.forEach(item => item.dispose());
      renderer.dispose();
      renderer.forceContextLoss();
    },
  };
}
