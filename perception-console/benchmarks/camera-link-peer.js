import { frameSize, createMarkerReader, writeMarker } from '../src/camera-frame-marker.js';
import { clipLengthMs, cameraRig } from '../src/camera-trial.js';
import { createTrialRenderer } from '../src/camera-trial-renderer.js';
import { cameraCaps, changePriority, directions } from './camera-link-metrics.mjs';

const clock = () => performance.timeOrigin + performance.now();
const makePeer = () => new RTCPeerConnection({ iceServers: [], bundlePolicy: 'max-bundle' });
const fps = 12;

async function localDescription(peer, sdp) {
  await peer.setLocalDescription(sdp);
  const deadline = performance.now() + 5_000;
  while (peer.iceGatheringState !== 'complete') {
    if (performance.now() > deadline) throw new Error('ICE gathering timed out.');
    await new Promise(resolve => setTimeout(resolve, 25));
  }
  return peer.localDescription.toJSON();
}

function selectedRoute(peerStats) {
  const transport = [...peerStats.values()].find(item => item.type === 'transport' && item.selectedCandidatePairId);
  const candidatePair = peerStats.get(transport?.selectedCandidatePairId);
  if (!candidatePair) return null;
  return {
    id: candidatePair.id,
    protocol: peerStats.get(candidatePair.localCandidateId)?.protocol,
    rttMs: Number.isFinite(candidatePair.currentRoundTripTime) ? candidatePair.currentRoundTripTime * 1000 : null,
  };
}

async function snapshot(peer, feeds, sending) {
  const stats = await peer.getStats();
  const type = sending ? 'outbound-rtp' : 'inbound-rtp';
  const rows = await Promise.all([...feeds.values()].map(async ({ id, transceiver }) => {
    const endpoint = sending ? transceiver.sender : transceiver.receiver;
    const trackStats = await endpoint.getStats();
    const record = [...trackStats.values()].find(item => item.type === type && item.kind === 'video');
    return [id, record ? {
      timestamp: record.timestamp,
      bytes: sending ? record.bytesSent : record.bytesReceived,
      packetsLost: record.packetsLost ?? null,
      width: record.frameWidth ?? null,
      height: record.frameHeight ?? null,
      codec: trackStats.get(record.codecId)?.mimeType ?? null,
    } : null];
  }));
  return { route: selectedRoute(stats), rows: Object.fromEntries(rows) };
}

const events = feeds => Object.fromEntries([...feeds].map(([id, feed]) => [id, feed.events]));

export function startSender(budget) {
  document.querySelector('#role').textContent = 'Sender · Simulated doorway crossing';
  const peer = makePeer();
  const renderer = createTrialRenderer(() => { throw new Error('Sender graphics context was lost.'); });
  const feeds = new Map();
  let phaseStart;
  let priority = 'front';
  let sequence = 0;
  let timer;
  for (const id of directions) {
    const canvas = Object.assign(document.createElement('canvas'), frameSize);
    const context = canvas.getContext('2d', { alpha: false });
    const stream = canvas.captureStream(0);
    const [track] = stream.getVideoTracks();
    const transceiver = peer.addTransceiver(track, { streams: [stream], direction: 'sendonly' });
    feeds.set(id, { id, canvas, context, stream, track, transceiver, events: [] });
  }

  const applyCap = async (id, cap) => {
    const sender = feeds.get(id).transceiver.sender;
    const parameters = sender.getParameters();
    parameters.encodings[0].maxBitrate = cap;
    await sender.setParameters(parameters);
    if (sender.getParameters().encodings[0]?.maxBitrate !== cap) throw new Error(`Rejected encoder cap for ${id}.`);
  };

  return {
    streams: Object.fromEntries([...feeds].map(([id, feed]) => [feed.stream.id, id])),
    async offer() { return localDescription(peer, await peer.createOffer()); },
    async connect(answer) {
      await peer.setRemoteDescription(answer);
      await Promise.all(Object.entries(cameraCaps(priority, budget)).map(([id, cap]) => applyCap(id, cap)));
      timer = setInterval(() => {
        try {
          const stampMs = phaseStart === undefined ? 0 : Math.min(clipLengthMs, clock() - phaseStart);
          sequence++;
          for (const feed of feeds.values()) {
            renderer.paintFeed(feed.canvas, feed.id, 'crossing', { stampMs, profile: frameSize });
            writeMarker(feed.context, sequence);
            feed.events.push([sequence, clock()]);
            feed.track.requestFrame();
          }
        } catch (error) {
          clearInterval(timer);
          throw error;
        }
      }, 1000 / fps);
    },
    replay() { phaseStart = clock(); return phaseStart; },
    async prioritize(id) {
      const startedAt = clock();
      await changePriority(priority, id, budget, applyCap);
      priority = id;
      return { startedAt, appliedAt: clock(), sequence, caps: cameraCaps(priority, budget) };
    },
    sequence: () => sequence,
    events: () => events(feeds),
    stats: () => snapshot(peer, feeds, true),
    close() {
      clearInterval(timer);
      for (const feed of feeds.values()) feed.track.stop();
      peer.close();
      renderer.dispose();
    },
  };
}

export function startReceiver(streams) {
  document.querySelector('#role').textContent = 'Receiver · Simulated doorway crossing';
  const host = document.querySelector('#feeds');
  const peer = makePeer();
  const readMarker = createMarkerReader();
  const feeds = new Map(directions.map(id => {
    const figure = document.createElement('figure');
    const label = document.createElement('figcaption');
    label.textContent = cameraRig[id].label;
    const video = Object.assign(document.createElement('video'), { autoplay: true, muted: true, playsInline: true });
    video.setAttribute('aria-label', cameraRig[id].label);
    figure.append(label, video);
    host.append(figure);
    return [id, { id, video, events: [] }];
  }));

  peer.ontrack = event => {
    const id = streams[event.streams[0]?.id];
    if (!feeds.has(id)) throw new Error('Received an unidentified video track.');
    const feed = feeds.get(id);
    const { video } = feed;
    feed.transceiver = event.transceiver;
    const frame = (_, metadata) => {
      const sequence = readMarker(video);
      if (sequence > (feed.events.at(-1)?.[0] ?? 0)) {
        feed.events.push([sequence, performance.timeOrigin + metadata.expectedDisplayTime]);
      }
      feed.callback = video.requestVideoFrameCallback(frame);
    };
    feed.callback = video.requestVideoFrameCallback(frame);
    video.srcObject = event.streams[0];
  };

  return {
    async answer(offer) {
      await peer.setRemoteDescription(offer);
      return localDescription(peer, await peer.createAnswer());
    },
    sequences: () => Object.fromEntries([...feeds].map(([id, feed]) => [id, feed.events.at(-1)?.[0] ?? 0])),
    events: () => events(feeds),
    stats: () => snapshot(peer, feeds, false),
    trackIds: () => [...feeds.values()].map(feed => feed.transceiver?.receiver.track.id),
    close() {
      for (const feed of feeds.values()) {
        if (feed.callback !== undefined) feed.video.cancelVideoFrameCallback(feed.callback);
        feed.video.srcObject = null;
      }
      peer.close();
    },
  };
}
