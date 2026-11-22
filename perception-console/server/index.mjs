import { createBridge } from './bridge.mjs';

const port = Number(process.env.RBW_PORT ?? 4180);
if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('RBW_PORT must be a valid TCP port.');
createBridge().listen(port, '127.0.0.1', () => {
  console.log(`Robot Buildworks perception console at http://127.0.0.1:${port}/`);
});
