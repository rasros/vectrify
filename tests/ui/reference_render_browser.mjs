import {spawn} from 'node:child_process';
import {createServer} from 'node:http';
import {mkdtemp, readFile, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

if (typeof WebSocket !== 'function') process.exit(77);
const root = fileURLToPath(new URL('../../', import.meta.url));
const profile = await mkdtemp(resolve(tmpdir(), 'vectrify-browser-test-'));
const server = createServer(async (request, response) => {
  try {
    const path = resolve(root, '.' + new URL(request.url, 'http://localhost').pathname);
    if (!path.startsWith(root)) throw new Error('Outside test root');
    response.setHeader('Content-Type', path.endsWith('.js') ? 'text/javascript' : 'text/html');
    response.end(await readFile(path));
  } catch { response.writeHead(404); response.end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let chrome, socket;
const timeout = (promise, ms) => {
  let timer;
  return Promise.race([promise, new Promise((_, reject) => {timer = setTimeout(() => reject(new Error('Browser test timed out')), ms);})]).finally(() => clearTimeout(timer));
};
try {
  chrome = spawn(process.argv[2], ['--headless', '--no-sandbox', '--disable-gpu', '--disable-extensions', '--disable-dev-shm-usage', '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank'], {stdio: 'ignore'});
  let port;
  for (let i = 0; i < 100; i++) {
    try { port = (await readFile(resolve(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0]; break; }
    catch { await new Promise(resolve => setTimeout(resolve, 50)); }
  }
  if (!port) throw new Error('Chrome did not start');
  const tabs = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
  const tab = tabs.find(tab => tab.type === 'page' && tab.url === 'about:blank');
  socket = new WebSocket(tab.webSocketDebuggerUrl);
  await timeout(new Promise((resolve, reject) => {socket.onopen = resolve; socket.onerror = reject;}), 5000);
  let id = 0; const waiting = new Map();
  socket.onmessage = event => {
    const reply = JSON.parse(event.data);
    if (!reply.id) return;
    const call = waiting.get(reply.id); waiting.delete(reply.id);
    reply.error ? call.reject(reply.error) : call.resolve(reply.result);
  };
  const call = (method, params = {}) => new Promise((resolve, reject) => {
    waiting.set(++id, {resolve, reject}); socket.send(JSON.stringify({id, method, params}));
  });
  await call('Page.enable');
  await call('Page.navigate', {url: `http://127.0.0.1:${server.address().port}/tests/ui/reference_render_browser.html`});
  const result = await timeout(call('Runtime.evaluate', {
    expression: "new Promise(resolve=>{const check=()=>{const el=document.getElementById('results');if(el && el.textContent!=='running')resolve(el.textContent);else setTimeout(check,10);};check();})",
    awaitPromise: true, returnByValue: true,
  }), 15000);
  if (result.result?.value !== 'ok') throw new Error(JSON.stringify(result));
  console.log('Native SVG, reference placement, clipping, alpha, and atomic frame swaps passed');
} finally {
  socket?.close();
  if (chrome && chrome.exitCode === null) {
    const exited = new Promise(resolve => chrome.once('exit', resolve)); chrome.kill(); await exited;
  }
  server.closeAllConnections(); server.close();
  await rm(profile, {recursive: true, force: true});
}
