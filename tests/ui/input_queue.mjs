// Checks that input during a running edit waits and runs in order, once;
// run by test_input_queue.py.
import assert from 'node:assert/strict';
import {HeldGesture, inputQueue} from '../../src/vectrify/ui/static/input.js';

// A fake editor: edits run for a while and the queue waits for them.
let running = 0, waiters = [];
const busy = () => running > 0;
const idle = () => running ? new Promise(resolve => waiters.push(resolve)) : Promise.resolve();
const tick = () => new Promise(resolve => setTimeout(resolve, 0));
function edit(log, name) {
  running++;
  return new Promise(resolve => setTimeout(() => {
    log.push(`edit ${name} done`);
    running--;
    if (!running) { const ready = waiters; waiters = []; ready.forEach(r => r()); }
    resolve();
  }, 5));
}

const log = [], queue = inputQueue(busy, idle);
// Nothing running: input runs at once.
queue.run(() => log.push('key 1'));
assert.deepEqual(log, ['key 1']);
// An edit starts; keys pressed meanwhile wait for it, in order.
const first = queue.run(() => edit(log, 'A'));
queue.run(() => log.push('key 2'));
const third = queue.run(async () => { log.push('key 3 starts edit B'); await edit(log, 'B'); });
queue.run(() => log.push('key 4'));
assert.deepEqual(log, ['key 1']);
assert.equal(queue.waiting, 3);
await first; await third; await tick();
// key 4 waited for edit B, which key 3 started after edit A.
assert.deepEqual(log, ['key 1', 'edit A done', 'key 2', 'key 3 starts edit B', 'edit B done', 'key 4']);
assert.equal(queue.waiting, 0);
// Each ran once; with nothing waiting, input runs at once again.
queue.run(() => log.push('key 5'));
assert.equal(log.at(-1), 'key 5');
// A failing input does not stop the ones after it.
edit(log, 'C');
const failing = queue.run(() => { throw new Error('refused'); });
const after = queue.run(() => log.push('key 6'));
await assert.rejects(failing, /refused/);
await after;
assert.deepEqual(log.slice(-2), ['edit C done', 'key 6']);

// A gesture pressed during an edit replays its events in order; one released
// before the replay does not go on live, one still held does.
const event = (type, x) => ({type, clientX: x, clientY: 0, button: 0, pointerId: 1, shiftKey: false, target: 'stale'});
const click = new HeldGesture(event('pointerdown', 1));
click.record(event('pointermove', 2));
click.record(event('pointerup', 3));
click.record(event('pointermove', 4));
const seen = [];
const handlers = {down: e => seen.push(`down ${e.clientX}`), move: e => seen.push(`move ${e.clientX}`), up: e => seen.push(`up ${e.clientX}`)};
assert.equal(click.replay(handlers), false);
assert.deepEqual(seen, ['down 1', 'move 2', 'up 3']);
// The recorded events carry no stale target: the replay finds it again.
assert.ok(click.events.every(e => !('target' in e)));
const held = new HeldGesture(event('pointerdown', 5));
held.record(event('pointermove', 6));
seen.length = 0;
assert.equal(held.replay(handlers), true);
assert.deepEqual(seen, ['down 5', 'move 6']);
// Escape drops a held gesture before it is replayed.
const dropped = new HeldGesture(event('pointerdown', 7));
dropped.record(event('pointermove', 8));
dropped.cancel();
dropped.record(event('pointerup', 9));
seen.length = 0;
assert.equal(dropped.replay(handlers), false);
assert.deepEqual(seen, []);
