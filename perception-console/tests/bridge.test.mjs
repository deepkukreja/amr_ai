import assert from 'node:assert/strict';
import test from 'node:test';
import { request } from 'node:http';
import { createBridge } from '../server/bridge.mjs';

test('loopback bridge streams ordered world snapshots with explicit session and provenance', async (t) => {
  const server = createBridge();
  t.after(() => { server.closeAllConnections(); server.close(); });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  const response = await fetch(`${base}/api/v1/world?scenario=pedestrian`, { signal: AbortSignal.timeout(4000) });
  assert.match(response.headers.get('content-type'), /text\/event-stream/);
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let data = '';
  while (data.split('\n\n').length < 3) data += (await reader.read()).value;
  await reader.cancel();
  const frames = data.split('\n\n').slice(0, 2).map(message => JSON.parse(message.split('\ndata: ')[1]));
  assert.equal(frames[0].version, 1);
  assert.equal(frames[0].mode, 'SIMULATION');
  assert.equal(frames[0].frame, 'sim_warehouse');
  assert.equal(frames[0].session, frames[1].session);
  assert.ok(frames[1].sequence > frames[0].sequence);
  assert.ok(frames[1].stampMs > frames[0].stampMs);
  assert.equal((await fetch(`${base}/api/v1/world?scenario=invalid`)).status, 400);
  assert.equal((await fetch(`${base}/api/v1/world`, { method: 'POST' })).status, 405);
  assert.equal((await fetch(`${base}/api/v1/command`)).status, 404);
  const hostResponse = await new Promise((resolve, reject) => {
    request(`${base}/api/v1/config`, { headers: { Host: 'outside.example' } }, result => {
      resolve(result.statusCode);
      result.resume();
    }).on('error', reject).end();
  });
  assert.equal(hostResponse, 403);
});
