import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import { extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';
import { scenarios, createSimulation, buildRecording, intervalMs } from './simulation.mjs';
import { environments, robotProfiles } from './environments.mjs';
import { loadRouteRecording, recordedRoute } from './route-replay.mjs';

const files = fileURLToPath(new URL('../dist/', import.meta.url));
const mime = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8' };

function json(response, status, data) {
  response.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
  response.end(JSON.stringify(data));
}

export function createBridge() {
  let clients = 0;
  const capturedRoute = loadRouteRecording().catch(error => {
    if (error.code !== 'ENOENT') console.warn('Recorded route unavailable:', error.message);
    return null;
  });
  return createServer(async (request, response) => {
    response.setHeader('X-Content-Type-Options', 'nosniff');
    if (!/^(?:localhost|127\.0\.0\.1)(?::\d+)?$/i.test(request.headers.host ?? '')) {
      return json(response, 403, { error: 'Loopback hostname required' });
    }
    if (request.method !== 'GET') return json(response, 405, { error: 'Read-only bridge' });
    const uri = new URL(request.url, 'http://127.0.0.1');
    if (uri.pathname === '/api/v1/config') {
      const playback = await capturedRoute;
      return json(response, 200, {
        version: 1, scenarios: playback ? [...scenarios, recordedRoute] : scenarios, environments, robotProfiles, intervalMs,
      });
    }
    if (uri.pathname === '/api/v1/replay' || uri.pathname === '/api/v1/world') {
      const scenario = uri.searchParams.get('scenario') ?? scenarios[0].id;
      const isRecorded = scenario === recordedRoute.id && uri.pathname.endsWith('/replay');
      if (!isRecorded && !scenarios.some(item => item.id === scenario)) return json(response, 400, { error: 'Unknown scenario' });
      if (uri.pathname.endsWith('/replay')) {
        const recording = isRecorded ? await capturedRoute : buildRecording(scenario);
        if (!recording) return json(response, 404, { error: 'Recording unavailable' });
        if (request.headers['accept-encoding']?.includes('gzip')) {
          response.writeHead(200, { 'Content-Type': 'application/json', 'Content-Encoding': 'gzip', 'Cache-Control': 'no-store' });
          return response.end(gzipSync(JSON.stringify(recording)));
        }
        return json(response, 200, recording);
      }
      if (clients >= 12) return json(response, 503, { error: 'Stream capacity reached' });
      clients++;
      const simulation = createSimulation(scenario, randomUUID());
      response.writeHead(200, {
        'Content-Type': 'text/event-stream', 'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no',
      });
      let writable = true;
      let blockedSince = 0;
      response.on('drain', () => { writable = true; });
      const tick = () => {
        const snapshot = simulation.next();
        if (writable) {
          writable = response.write(`event: world\ndata: ${JSON.stringify(snapshot)}\n\n`);
          if (!writable) blockedSince = performance.now();
        } else if (performance.now() - blockedSince > 5000) {
          response.destroy();
        }
      };
      tick();
      const timer = setInterval(tick, intervalMs);
      response.on('close', () => { clearInterval(timer); clients--; });
      return;
    }
    const asset = uri.pathname === '/' ? 'index.html' : /^\/assets\/[\w.-]+$/.test(uri.pathname) ? uri.pathname.slice(1) : null;
    if (asset) {
      const path = join(files, asset);
      const info = await stat(path).catch(() => null);
      if (info?.isFile()) {
        response.writeHead(200, { 'Content-Type': mime[extname(path)] ?? 'application/octet-stream', 'Content-Length': info.size });
        const stream = createReadStream(path);
        stream.on('error', () => response.destroy());
        return stream.pipe(response);
      }
    }
    return json(response, 404, { error: uri.pathname === '/' ? 'Build the console with npm run build first' : 'Not found' });
  });
}
