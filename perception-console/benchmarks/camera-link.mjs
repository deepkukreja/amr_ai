import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { setTimeout as delay } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import { clipLengthMs } from '../src/camera-trial.js';
import { cameraCaps, directions, presentationStats } from './camera-link-metrics.mjs';
import { mediaRate } from '../src/camera-transport-metrics.js';

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.RBW_PLAYWRIGHT_MODULE || 'playwright');

const constrained = { urlPattern: '', uploadThroughput: 52_500, downloadThroughput: -1,
  latency: 70, packetLoss: 1, packetQueueLength: 20, packetReordering: false };
const scenarios = [
  { id: 'unshaped', budget: 720_000, link: null },
  { id: 'constrained', budget: 720_000, link: constrained },
  { id: 'headroom', budget: 330_000, link: constrained, checkOutage: true },
];
const clock = () => performance.timeOrigin + performance.now();
const server = await createServer({
  root: fileURLToPath(new URL('../', import.meta.url)), configFile: false, logLevel: 'error',
  server: { host: '127.0.0.1', port: 0, strictPort: true },
});
const browsers = [];

function shape(cdp, link) {
  return cdp.send('Network.emulateNetworkConditionsByRule', {
    offline: false, matchedNetworkConditions: link ? [link] : [],
  });
}

async function bracketClock(page) {
  const samples = [];
  for (let i = 0; i < 5; i++) {
    const from = clock();
    const remote = await page.evaluate(() => performance.timeOrigin + performance.now());
    const until = clock();
    samples.push({ duration: until - from, bound: Math.max(Math.abs(remote - from), Math.abs(remote - until)) });
  }
  return samples.sort((a, b) => a.duration - b.duration)[0].bound;
}

const stats = page => page.evaluate(() => window.cameraLink.stats());
const readEvents = page => page.evaluate(() => window.cameraLink.events());
const latest = page => page.evaluate(() => window.cameraLink.sequence());
const received = page => page.evaluate(() => window.cameraLink.sequences());

function rate(before, after) {
  return mediaRate(before && { timestamp: before.timestamp, bytesSent: before.bytes },
    after && { timestamp: after.timestamp, bytesSent: after.bytes });
}

async function measure(sender, receiver, priority, budget) {
  const [sendBefore, receiveBefore] = await Promise.all([stats(sender), stats(receiver)]);
  const from = await sender.evaluate(() => window.cameraLink.replay());
  const until = from + clipLengthMs;
  await delay(Math.max(0, until - clock()));
  const [sendAfter, receiveAfter, source, seen] = await Promise.all([
    stats(sender), stats(receiver), readEvents(sender), readEvents(receiver),
  ]);
  const caps = cameraCaps(priority, budget);
  const rows = directions.map(id => {
    const presentation = presentationStats(source[id], seen[id], from, until);
    const sent = sendAfter.rows[id];
    const receivedTrack = receiveAfter.rows[id];
    const sendKbps = rate(sendBefore.rows[id], sent);
    const receiveKbps = rate(receiveBefore.rows[id], receivedTrack);
    assert.ok(presentation.submitted > 30 && presentation.matched > 0, `No measured video on ${id}.`);
    assert.ok(sendKbps !== null && receiveKbps !== null, `Missing RTP counters for ${id}.`);
    return { id, capKbps: caps[id] / 1000, sendKbps, receiveKbps, ...presentation,
      width: receivedTrack.width, height: receivedTrack.height, codec: receivedTrack.codec };
  });
  assert.equal(sendAfter.route?.protocol, 'udp');
  assert.equal(receiveAfter.route?.protocol, 'udp');
  return { priority, rows, rttMs: sendAfter.route.rttMs, routeId: sendAfter.route.id };
}

async function outage(sender, receiver, cdp, link) {
  await shape(cdp, { ...link, packetLoss: 100 });
  await delay(1_200);
  const [paused, sourceAtPause] = await Promise.all([received(receiver), latest(sender)]);
  await delay(1_500);
  const [pausedEnd, sourceAtEnd] = await Promise.all([received(receiver), latest(sender)]);
  assert.deepEqual(pausedEnd, paused, 'All receiver tracks must stop advancing under full packet loss.');
  assert.ok(sourceAtEnd - sourceAtPause > 8, 'Source must keep submitting frames during the outage.');
  await shape(cdp, link);
  const restoredAt = clock();
  const cutoff = await latest(sender);
  await receiver.waitForFunction(cutoff => Object.values(window.cameraLink.sequences()).every(sequence => sequence > cutoff),
    cutoff, { timeout: 15_000 });
  const seen = await readEvents(receiver);
  return {
    quietWindowMs: 1500, submittedDuringQuietWindow: sourceAtEnd - sourceAtPause,
    arrivingDuringQuietWindow: 0,
    recoveryMs: Object.fromEntries(directions.map(id => [id, seen[id].find(([seq]) => seq > cutoff)[1] - restoredAt])),
  };
}

async function runScenario(config, url) {
  const [sender, receiver] = await Promise.all(browsers.map(browser => browser.newPage({ viewport: { width: 1280, height: 900 } })));
  const pageErrors = [];
  for (const page of [sender, receiver]) page.on('pageerror', error => pageErrors.push(error.message));

  try {
    await Promise.all([sender, receiver].map(page => page.goto(url)));
    const clockBounds = await Promise.all([bracketClock(sender), bracketClock(receiver)]);
    assert.ok(clockBounds.reduce((sum, bound) => sum + bound, 0) < 15, 'Capture and presentation clocks disagree.');
    const cdp = await sender.context().newCDPSession(sender);
    await shape(cdp, config.link);
    const { offer, streams } = await sender.evaluate(async budget => {
      const { startSender } = await import('/benchmarks/camera-link-peer.js');
      window.cameraLink = startSender(budget);
      return { offer: await window.cameraLink.offer(), streams: window.cameraLink.streams };
    }, config.budget);
    const answer = await receiver.evaluate(async ({ offer, streams }) => {
      const { startReceiver } = await import('/benchmarks/camera-link-peer.js');
      window.cameraLink = startReceiver(streams);
      return window.cameraLink.answer(offer);
    }, { offer, streams });
    await sender.evaluate(answer => window.cameraLink.connect(answer), answer);
    await receiver.waitForFunction(() => Object.values(window.cameraLink.sequences()).every(value => value > 0),
      {}, { timeout: 14_000 });
    const trackIds = await receiver.evaluate(() => window.cameraLink.trackIds());
    assert.equal(new Set(trackIds).size, directions.length);
    await delay(4_000);
    console.error(`${config.id}: front prioritized`);
    const front = await measure(sender, receiver, 'front', config.budget);
    const change = await sender.evaluate(() => window.cameraLink.prioritize('rear'));
    await delay(4_000);
    console.error(`${config.id}: rear prioritized`);
    const rear = await measure(sender, receiver, 'rear', config.budget);
    assert.deepEqual(await receiver.evaluate(() => window.cameraLink.trackIds()), trackIds);
    assert.equal(front.routeId, rear.routeId, 'Priority changes must keep the original ICE pair.');
    if (config.link) assert.ok(rear.rttMs >= 100, 'The UDP route did not show the configured delay.');
    const firstRear = (await readEvents(receiver)).rear.find(([sequence]) => sequence > change.sequence);
    assert.ok(firstRear, 'No rear frame arrived after changing priority.');
    const interruption = config.checkOutage ? await outage(sender, receiver, cdp, config.link) : null;
    const renderer = await sender.evaluate(() => {
      const gl = document.createElement('canvas').getContext('webgl2');
      const debug = gl?.getExtension('WEBGL_debug_renderer_info');
      return debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : null;
    });
    assert.deepEqual(pageErrors, []);
    return {
      id: config.id, encoderBudgetKbps: config.budget / 1000,
      link: config.link && { uplinkKbps: config.link.uploadThroughput / 125, latencyPerDirectionMs: config.link.latency,
        packetLossPercent: config.link.packetLoss, queuePackets: config.link.packetQueueLength },
      clockBoundMs: clockBounds.reduce((sum, bound) => sum + bound, 0), renderer, protocol: 'udp',
      phases: [front, rear].map(({ routeId, ...phase }) => phase),
      switch: { sameTracksAndIcePair: true, capApplyMs: change.appliedAt - change.startedAt,
        firstPostSwitchRearFrameMs: firstRear[1] - change.appliedAt },
      interruption,
    };
  } finally {
    await Promise.allSettled([sender, receiver].map(async page => {
      try { await page.evaluate(() => window.cameraLink?.close()); }
      finally { await page.close(); }
    }));
  }
}

try {
  await server.listen();
  const url = `http://127.0.0.1:${server.httpServer.address().port}/benchmarks/camera-link.html`;
  for (let i = 0; i < 2; i++) {
    browsers.push(await chromium.launch({ channel: process.env.RBW_CHROMIUM_CHANNEL || 'msedge', headless: true }));
  }
  const processIds = await Promise.all(browsers.map(async browser => {
    const cdp = await browser.newBrowserCDPSession();
    return (await cdp.send('SystemInfo.getProcessInfo')).processInfo.find(info => info.type === 'browser')?.id;
  }));
  assert.ok(processIds.every(Number.isInteger) && new Set(processIds).size === 2, 'Two browser processes are required.');
  const results = [];
  for (const scenario of scenarios) results.push(await runScenario(scenario, url));
  console.log(JSON.stringify({
    schema: 1, source: 'synthetic-doorway', environment: 'separate Chromium processes on one host',
    browserVersion: browsers[0].version(), durationMs: clipLengthMs, results,
  }, null, 2));
} finally {
  await Promise.allSettled(browsers.map(browser => browser.close()));
  await server.close();
}
