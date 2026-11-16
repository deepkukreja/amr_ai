'use strict';
const $ = id => document.getElementById(id);
const model = $('model'), video = $('video'), image = $('resultImage');
let profiles = [], busy = false, cameraRunning = false, startingCamera = false;
let stream = null, timer = null, controller = null, serial = 0, frameStarted = 0;

function error(message) { $('error').textContent = message || ''; $('error').hidden = !message; }
function status(message) { $('status').textContent = message; }
function setBusy(value) {
  busy = value;
  for (const id of ['file','imageUrl','urlButton','sample','model']) $(id).disabled = value || cameraRunning;
  $('camera').disabled = startingCamera || (value && !cameraRunning);
}
function selected() { return profiles.find(p => p.id === model.value); }
function describeModel() {
  const p = selected();
  $('modelDescription').textContent = p ? p.description : 'No model is available.';
  $('classTags').replaceChildren(); $('thresholdValues').replaceChildren();
  if (!p) return;
  for (const label of p.classes) {
    const tag = document.createElement('span'); tag.className = 'tag';
    tag.textContent = label.replaceAll('_',' '); $('classTags').append(tag);
    const row = document.createElement('div'); row.textContent = `${label.replaceAll('_',' ')}: ${p.thresholds[label].toFixed(2)}`;
    $('thresholdValues').append(row);
  }
}
function showResult(data) {
  image.src = 'data:image/jpeg;base64,' + data.image;
  image.hidden = false; video.hidden = true; $('emptyState').hidden = true;
  $('count').textContent = data.detections.length;
  $('resultModel').textContent = data.profile_label;
  $('latency').textContent = `${Math.round(data.elapsed_ms)} ms`;
  $('imageSize').textContent = `${data.width} × ${data.height}`;
  $('detections').replaceChildren();
  if (!data.detections.length) {
    const p = document.createElement('p'); p.className = 'no-detections';
    p.textContent = 'No detections above the profile confidence levels.'; $('detections').append(p);
  }
  for (const item of data.detections) {
    const row = document.createElement('div'); row.className = 'detection';
    const name = document.createElement('span'); name.className = 'name';
    const dot = document.createElement('i'); name.append(dot, document.createTextNode(item.label.replaceAll('_',' ')));
    const score = document.createElement('strong'); score.textContent = (item.confidence * 100).toFixed(1) + '%';
    row.append(name,score); $('detections').append(row);
  }
}
async function analyze(endpoint, body, contentType, label) {
  if (busy) return;
  const ticket = ++serial; controller = new AbortController();
  error(''); setBusy(true); status('Analyzing… First use of a detector may take a moment.');
  $('sourceLabel').textContent = label;
  try {
    const response = await fetch(`${endpoint}?model=${encodeURIComponent(model.value)}`, {
      method:'POST', body, headers:contentType ? {'Content-Type':contentType} : {}, signal:controller.signal
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'The request failed.');
    if (ticket !== serial) return;
    showResult(data);
    status(cameraRunning ? `Camera live · ${(1000 / Math.max(1,performance.now()-frameStarted)).toFixed(1)} processed frames/s` : 'Analysis complete.');
  } catch (e) {
    if (e.name !== 'AbortError' && ticket === serial) {
      error(e.message); status('Ready to try another image.');
      if (cameraRunning) stopCamera();
    }
  } finally { if (ticket === serial) setBusy(false); }
}
function upload(file) {
  if (!file || busy || cameraRunning) return;
  if (file.size > 12 * 1024 * 1024) { error('Use an image smaller than 12 MB.'); return; }
  analyze('/api/predict',file,'application/octet-stream',file.name || 'Uploaded image');
}
$('file').addEventListener('change',e => { upload(e.target.files[0]); e.target.value = ''; });
$('sample').addEventListener('click',() => analyze('/api/sample',null,null,'Sample image'));
$('urlForm').addEventListener('submit',e => {
  e.preventDefault(); if (!$('imageUrl').value.trim() || busy || cameraRunning) return;
  analyze('/api/url',JSON.stringify({url:$('imageUrl').value.trim()}),'application/json','Internet image');
});
for (const event of ['dragover','dragleave','drop']) $('dropzone').addEventListener(event,e => {
  e.preventDefault(); $('dropzone').classList.toggle('over',event === 'dragover');
  if (event === 'drop') upload(e.dataTransfer.files[0]);
});
model.addEventListener('change',() => { describeModel(); error(''); status('Detector selected. Choose an input to run it.'); });

async function capture() {
  if (!cameraRunning) return;
  const canvas = $('captureCanvas'); canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  if (!canvas.width) { timer = setTimeout(capture,150); return; }
  canvas.getContext('2d').drawImage(video,0,0);
  const blob = await new Promise(resolve => canvas.toBlob(resolve,'image/jpeg',.86));
  if (!cameraRunning || !blob) return;
  frameStarted = performance.now();
  await analyze('/api/predict',blob,'application/octet-stream','Laptop webcam');
  if (cameraRunning) timer = setTimeout(capture,Math.max(0,250-(performance.now()-frameStarted)));
}
function stopCamera() {
  cameraRunning = false; startingCamera = false; ++serial;
  clearTimeout(timer); if (controller) controller.abort();
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null; video.srcObject = null; video.hidden = true;
  $('liveBadge').hidden = true; $('cameraText').textContent = 'Start webcam';
  $('camera').classList.remove('stop'); $('cameraIcon').textContent = '◉';
  if (image.hidden) $('emptyState').hidden = false;
  setBusy(false); status('Camera stopped.');
}
async function startCamera() {
  error('');
  if (!navigator.mediaDevices?.getUserMedia) { error('Open this localhost page in Edge or Chrome to use the webcam.'); return; }
  startingCamera = true; setBusy(false); status('Waiting for camera permission…');
  try {
    stream = await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720}},audio:false});
    video.srcObject = stream; await video.play();
    cameraRunning = true; startingCamera = false;
    $('emptyState').hidden = true; image.hidden = true; video.hidden = false;
    $('liveBadge').hidden = false; $('cameraText').textContent = 'Stop webcam';
    $('camera').classList.add('stop'); $('cameraIcon').textContent = '■';
    setBusy(false); capture();
  } catch (e) {
    stopCamera();
    error(e.name === 'NotAllowedError' ? 'Camera permission was denied. Allow camera access in your browser and try again.' : 'The camera is unavailable. Close other camera apps and try again.');
  }
}
$('camera').addEventListener('click',() => cameraRunning ? stopCamera() : startCamera());
window.addEventListener('beforeunload',stopCamera);
document.addEventListener('visibilitychange',() => { if (document.hidden && cameraRunning) stopCamera(); });

async function initialize() {
  try {
    const response = await fetch('/api/models');
    if (!response.ok) throw new Error('Could not connect to the local detector.');
    profiles = (await response.json()).models;
    model.replaceChildren();
    for (const p of profiles) {
      const option = document.createElement('option'); option.value = p.id;
      option.textContent = p.label + (p.available ? '' : ' · weights missing'); option.disabled = !p.available;
      model.append(option);
    }
    const ready = profiles.find(p => p.available);
    if (ready) model.value = ready.id;
    describeModel();
    if (!ready) { setBusy(true); error('The trained model weights are not installed in this checkout.'); }
  } catch (e) { setBusy(true); error(e.message); status('Local detector unavailable.'); }
}
initialize();
