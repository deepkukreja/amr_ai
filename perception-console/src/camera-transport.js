import { clipLengthMs } from './camera-trial.js';
import { createTrialRenderer } from './camera-trial-renderer.js';
import { bufferDelay, decodeMarker, encodeMarker, feedHealth, frameLatency, markerCells, mediaRate, percentile } from './camera-transport-metrics.js';

const frameSize = { width: 512, height: 288 };
const cellWidth = 8;
const markerWidth = markerCells * cellWidth;
const markerHeight = 16;
const fps = 12;
const freezeMs = 3_000;
const recoveryWindowMs = 5_000;

function writeMarker(context, sequence) {
  encodeMarker(sequence).forEach((bit, index) => {
    context.fillStyle = bit ? '#fff' : '#000';
    context.fillRect(index * cellWidth, 0, cellWidth, markerHeight);
  });
}

function readMarker(context, video) {
  const scaleX = video.videoWidth / frameSize.width;
  const scaleY = video.videoHeight / frameSize.height;
  context.drawImage(video, 0, (markerHeight / 2 - 2) * scaleY, markerWidth * scaleX, 4 * scaleY,
    0, 0, markerWidth, 1);
  const { data } = context.getImageData(0, 0, markerWidth, 1);
  return decodeMarker(Array.from({ length: markerCells }, (_, i) => data[(i * cellWidth + cellWidth / 2) * 4]));
}

async function videoStats(endpoint, type) {
  const stats = await endpoint.getStats();
  for (const stat of stats.values()) {
    if (stat.type === type && stat.kind === 'video') {
      return { ...stat, codec: stats.get(stat.codecId)?.mimeType ?? null };
    }
  }
  return null;
}

function trackStats(feeds) {
  return Promise.all(feeds.map(async feed => {
    const [outbound, inbound] = await Promise.all([
      videoStats(feed.sender, 'outbound-rtp'), videoStats(feed.receiver, 'inbound-rtp'),
    ]);
    return { outbound, inbound };
  }));
}

export async function measureTransport(profile, videos, { signal, onStatus, onFeeds }) {
  if (!globalThis.RTCPeerConnection || !HTMLCanvasElement.prototype.captureStream
    || !globalThis.CanvasCaptureMediaStreamTrack?.prototype.requestFrame
    || !HTMLVideoElement.prototype.requestVideoFrameCallback) {
    throw new Error('This browser cannot capture and time the video frames required by the benchmark.');
  }

  const peers = [];
  const feeds = [];
  let renderer, timer, failure, startedAt, endedAt, frozenAt, resumedAt;
  let freeze = false;
  let finished = false;
  let publishedHealth = {};

  const waitUntil = async (condition, timeoutMs, message) => {
    const deadline = performance.now() + timeoutMs;
    while (true) {
      signal.throwIfAborted();
      if (failure) throw failure;
      if (condition()) return;
      if (performance.now() >= deadline) throw new Error(message);
      await new Promise(resolve => setTimeout(resolve, 25));
    }
  };

  const render = () => {
    try {
      const now = performance.now();
      const stampMs = startedAt === undefined ? 0 : (now - startedAt) % clipLengthMs;
      for (const feed of feeds) {
        if (freeze && feed.id === 'left') continue;
        renderer.paintFeed(feed.canvas, feed.id, 'crossing', { profile: frameSize, stampMs });
        const sequence = ++feed.sequence;
        writeMarker(feed.context, sequence);
        feed.sent.set(sequence, performance.now());
        feed.track.requestFrame();
      }
    } catch (error) {
      failure = error;
    }
  };

  const update = () => {
    const now = performance.now();
    const health = Object.fromEntries(feeds.map(feed => [feed.id, feedHealth(feed.last, now)]));
    if (feeds.every(feed => health[feed.id] === publishedHealth[feed.id])) return;
    publishedHealth = health;
    onFeeds(health);
  };

  try {
    signal.throwIfAborted();
    renderer = createTrialRenderer(() => { failure = new Error('The graphics context was lost.'); });
    const options = { iceServers: [], bundlePolicy: 'max-bundle' };
    const senderPeer = new RTCPeerConnection(options);
    peers.push(senderPeer);
    const receiverPeer = new RTCPeerConnection(options);
    peers.push(receiverPeer);
    for (const [id, cap] of Object.entries(profile.caps)) {
      const canvas = document.createElement('canvas');
      Object.assign(canvas, frameSize);
      const context = canvas.getContext('2d', { alpha: false });
      const stream = canvas.captureStream(0);
      const [track] = stream.getVideoTracks();
      const video = videos[id];
      const feed = { id, cap, canvas, context, track, video, sent: new Map(), sequence: 0, display: [], samples: [], invalid: 0 };
      feeds.push(feed);
      const transceiver = senderPeer.addTransceiver(track, { direction: 'sendonly', streams: [stream] });
      feed.sender = transceiver.sender;
      feed.streamId = stream.id;
    }
    const markerCanvas = document.createElement('canvas');
    markerCanvas.width = markerWidth;
    markerCanvas.height = 1;
    const markerContext = markerCanvas.getContext('2d', { willReadFrequently: true });

    receiverPeer.ontrack = event => {
      const feed = feeds.find(item => item.streamId === event.streams[0]?.id);
      if (!feed || finished) return;
      feed.receiver = event.receiver;
      const { video } = feed;
      const presented = (_, metadata) => {
        feed.callback = video.requestVideoFrameCallback(presented);
        try {
          const sequence = readMarker(markerContext, video);
          const sentAt = feed.sent.get(sequence);
          const displayedAt = metadata.expectedDisplayTime;
          if (sentAt === undefined) {
            if (displayedAt >= startedAt && displayedAt < endedAt) feed.invalid++;
            return;
          }
          if (sequence <= (feed.last?.sequence ?? 0) || displayedAt < sentAt) return;
          feed.last = { sequence, sentAt, displayedAt };
          feed.display.push(feed.last);
          const age = frameLatency(feed.sent, sequence, displayedAt, startedAt, endedAt);
          if (age !== null) feed.samples.push({ age, width: metadata.width, height: metadata.height });
        } catch (error) {
          failure = error;
        }
      };
      feed.callback = video.requestVideoFrameCallback(presented);
      video.srcObject = event.streams[0];
      video.play().catch(error => { if (!finished) failure = error; });
    };

    onStatus('Connecting local peers');
    await senderPeer.setLocalDescription(await senderPeer.createOffer());
    await waitUntil(() => senderPeer.iceGatheringState === 'complete', 5_000, 'Sender ICE gathering timed out.');
    await receiverPeer.setRemoteDescription(senderPeer.localDescription);
    await receiverPeer.setLocalDescription(await receiverPeer.createAnswer());
    await waitUntil(() => receiverPeer.iceGatheringState === 'complete', 5_000, 'Receiver ICE gathering timed out.');
    await senderPeer.setRemoteDescription(receiverPeer.localDescription);
    await Promise.all(feeds.map(async feed => {
      const parameters = feed.sender.getParameters();
      parameters.encodings[0].maxBitrate = feed.cap;
      await feed.sender.setParameters(parameters);
      if (feed.sender.getParameters().encodings[0]?.maxBitrate !== feed.cap) {
        throw new Error(`The browser did not retain the ${feed.id} encoder cap.`);
      }
    }));

    onStatus('Warming up the video feeds');
    timer = setInterval(() => { render(); update(); }, 1000 / fps);
    await waitUntil(() => feeds.every(feed => feed.last), 10_000, 'A video feed did not deliver a readable frame.');
    const warmUntil = performance.now() + 1_500;
    await waitUntil(() => performance.now() >= warmUntil, 2_500, 'Video warm-up timed out.');
    const before = await trackStats(feeds);
    startedAt = performance.now();
    endedAt = startedAt + clipLengthMs;
    onStatus('Measuring the doorway replay');
    await waitUntil(() => performance.now() >= endedAt, clipLengthMs + 1_000, 'The replay did not finish.');
    const after = await trackStats(feeds);

    let interruption = null;
    if (profile.id === 'priority') {
      const side = feeds.find(feed => feed.id === 'left');
      frozenAt = performance.now();
      const alreadyStale = feedHealth(side.last, frozenAt) === 'stale';
      freeze = true;
      onStatus('Pausing the left source');
      let staleAt;
      await waitUntil(() => {
        const now = performance.now();
        if (!alreadyStale && feedHealth(side.last, now) === 'stale') staleAt ??= now;
        return now >= frozenAt + freezeMs;
      }, freezeMs + 1_000, 'The interruption window did not finish.');
      onStatus('Restoring the left source');
      resumedAt = performance.now();
      freeze = false;
      const recoveryDeadline = resumedAt + recoveryWindowMs;
      await waitUntil(() => side.last?.sentAt >= resumedAt || performance.now() >= recoveryDeadline,
        recoveryWindowMs + 1_000, 'The recovery window did not finish.');
      const restored = side.display.find(frame => frame.sentAt >= resumedAt && frame.displayedAt <= recoveryDeadline);
      interruption = {
        alreadyStale,
        staleAfterMs: staleAt === undefined ? null : staleAt - frozenAt,
        resumeToDisplayMs: restored ? restored.displayedAt - resumedAt : null,
        freezeMs,
        recoveryWindowMs,
        otherFrames: Object.fromEntries(feeds.filter(feed => feed !== side).map(feed => [feed.id,
          feed.display.filter(frame => frame.sentAt >= frozenAt && frame.displayedAt <= resumedAt).length])),
      };
    }

    const rows = feeds.map((feed, index) => {
      const ages = feed.samples.map(sample => sample.age);
      const last = feed.samples.at(-1);
      const kbps = mediaRate(before[index].outbound, after[index].outbound);
      if (!last || kbps === null) throw new Error(`Missing presentation or RTP data for the ${feed.id} feed.`);
      return {
        id: feed.id, capKbps: feed.cap / 1000, kbps,
        sent: [...feed.sent.values()].filter(at => at >= startedAt && at < endedAt).length,
        matched: ages.length, invalid: feed.invalid, p50Ms: percentile(ages, 0.50), p95Ms: percentile(ages, 0.95),
        width: last.width, height: last.height, codec: after[index].outbound.codec,
        averageJitterBufferMs: bufferDelay(before[index].inbound, after[index].inbound),
      };
    });
    return { id: profile.id, durationMs: clipLengthMs, rows, interruption };
  } finally {
    finished = true;
    clearInterval(timer);
    for (const feed of feeds) {
      if (feed.callback !== undefined) feed.video.cancelVideoFrameCallback(feed.callback);
      feed.video.srcObject = null;
      feed.track.stop();
    }
    for (const peer of peers) peer.close();
    renderer?.dispose();
  }
}
