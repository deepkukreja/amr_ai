import {
  BufferGeometry, Color, Float32BufferAttribute, PerspectiveCamera, Points, PointsMaterial,
  Quaternion, Scene, Vector3, WebGLRenderer,
} from 'three';
import { surveyFov } from './survey-data.js';

const background = 0x192229;
const normals = [new Vector3(1, 0, 0), new Vector3(0, 0, -1), new Vector3(0, 1, 0)];
const sceneVector = ([x, y, z]) => new Vector3(x, z, -y);

function addPoints(scene, samples) {
  const positions = [];
  const colors = [];
  const color = new Color();
  for (const { x, y, z, color: rgb } of samples) {
    positions.push(x, z, -y);
    color.setHex(rgb);
    colors.push(color.r, color.g, color.b);
  }
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new Float32BufferAttribute(colors, 3));
  const material = new PointsMaterial({ size: 0.048, vertexColors: true });
  scene.add(new Points(geometry, material));
  return () => {
    geometry.dispose();
    material.dispose();
  };
}

function addSplats(scene, samples, renderer, onDirty, { SparkRenderer, SplatMesh }) {
  const spark = new SparkRenderer({ renderer, onDirty, enableLod: false });
  scene.add(spark);
  const center = new Vector3();
  const scales = new Vector3(0.042, 0.042, 0.008);
  const localNormal = new Vector3(0, 0, 1);
  const orientations = normals.map(normal => new Quaternion().setFromUnitVectors(localNormal, normal));
  const color = new Color();
  const splats = new SplatMesh({
    maxSplats: samples.length,
    raycastable: false,
    constructSplats(packed) {
      for (const { x, y, z, axis, color: rgb } of samples) {
        packed.pushSplat(center.set(x, z, -y), scales, orientations[axis], 0.96, color.setHex(rgb));
      }
    },
  });
  scene.add(splats);
  return {
    isReady: () => spark.activeSplats > 0,
    initialized: splats.initialized,
    dispose() {
      spark.dispose();
      splats.dispose();
    },
  };
}

export function createSurveyViewport(host, samples, sparkModule, onReady, onError) {
  const renderer = new WebGLRenderer({ antialias: false });
  const scene = new Scene();
  scene.background = new Color(background);
  const camera = new PerspectiveCamera(surveyFov, 1, 0.08, 24);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  renderer.domElement.setAttribute('aria-hidden', 'true');
  host.appendChild(renderer.domElement);

  let frame = null;
  let ready = false;
  let disposed = false;
  function requestRender() {
    if (disposed || frame !== null) return;
    frame = requestAnimationFrame(() => {
      frame = null;
      renderer.render(scene, camera);
      if (!ready && (!splats || splats.isReady())) {
        ready = true;
        onReady();
      }
    });
  }
  const splats = sparkModule ? addSplats(scene, samples, renderer, requestRender, sparkModule) : null;
  const disposeGeometry = splats ? splats.dispose : addPoints(scene, samples);
  splats?.initialized.catch(() => { if (!disposed) onError(); });

  function resize() {
    const { clientWidth: width, clientHeight: height } = host;
    if (!width || !height) return;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
    requestRender();
  }
  const observer = new ResizeObserver(resize);
  observer.observe(host);

  function loseContext() {
    onError();
  }
  renderer.domElement.addEventListener('webglcontextlost', loseContext);
  return {
    setView(view) {
      camera.position.copy(sceneVector(view.eye));
      camera.lookAt(sceneVector(view.target));
      requestRender();
    },
    dispose() {
      disposed = true;
      cancelAnimationFrame(frame);
      observer.disconnect();
      disposeGeometry();
      renderer.domElement.removeEventListener('webglcontextlost', loseContext);
      renderer.dispose();
      renderer.forceContextLoss();
      renderer.domElement.remove();
    },
  };
}
