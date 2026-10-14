// Input that arrives while an edit is still running waits for it: keys,
// clicks, commands and canvas gestures queue up and run in order, once each,
// on the state the edit leaves. Pure over callbacks, so it can be checked
// outside the browser; app.js says when the editor is busy.

// A queue of input. *busy* says whether an edit is running, *idle* resolves
// once none is. Input runs at once when nothing is running or waiting;
// otherwise after the running edits and the input queued before it.
//
// Input of one *kind* can coalesce instead: while the last input queued is
// of the same kind and has not started, a new one merges into it (*merge*
// combines their values) and *fn* runs once, with the total, when its turn
// comes. Held arrow keys so send one nudge per edit, not one per repeat.
export function inputQueue(busy, idle) {
  let chain = Promise.resolve(), waiting = 0, last = null;
  function run(fn) {
    last = null;
    if (!busy() && !waiting) return fn();
    waiting++;
    const result = chain.then(async () => { await idle(); return fn(); }).finally(() => { waiting--; });
    chain = result.catch(() => {});
    return result;
  }
  function coalesce(kind, value, merge, fn) {
    if (last && last.kind === kind && !last.started) { last.value = merge(last.value, value); return last.result; }
    const entry = {kind, value, started: false};
    entry.result = run(() => { entry.started = true; return fn(entry.value); });
    if (!entry.started) last = entry;
    return entry.result;
  }
  return {
    get waiting() { return waiting; },
    run,
    coalesce,
  };
}

// The fields of a pointer event a replay needs; its target is found again
// then, on what the edit left.
export function pointerRecord(event) {
  const {type, clientX, clientY, button, buttons, pointerId, shiftKey, ctrlKey, metaKey, altKey, timeStamp} = event;
  return {type, clientX, clientY, button, buttons, pointerId, shiftKey, ctrlKey, metaKey, altKey, timeStamp};
}

// A canvas gesture pressed while an edit runs: its events are recorded until
// the edit is done, then replayed through *handlers* {down, move, up,
// cancel}. A gesture still held when it is replayed goes on live afterwards.
export class HeldGesture {
  constructor(down) { this.events = [pointerRecord(down)]; this.released = false; }
  record(event) {
    if (this.released) return;
    this.events.push(pointerRecord(event));
    if (event.type === 'pointerup' || event.type === 'pointercancel') this.released = true;
  }
  // Escape drops a held gesture: nothing of it is replayed.
  cancel() { this.events = []; this.released = true; }
  replay(handlers) {
    const kinds = {pointerdown: 'down', pointermove: 'move', pointerup: 'up', pointercancel: 'cancel'};
    for (const event of this.events) handlers[kinds[event.type]]?.(event);
    return !this.released;
  }
}
