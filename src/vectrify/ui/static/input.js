// Input that arrives while an edit is still running waits for it: keys,
// clicks, commands and canvas gestures queue up and run in order, once each,
// on the state the edit leaves. Pure over callbacks, so it can be checked
// outside the browser; app.js says when the editor is busy.

// A queue of input. *busy* says whether an edit is running, *idle* resolves
// once none is. Input runs at once when nothing is running or waiting;
// otherwise after the running edits and the input queued before it.
export function inputQueue(busy, idle) {
  let chain = Promise.resolve(), waiting = 0;
  return {
    get waiting() { return waiting; },
    run(fn) {
      if (!busy() && !waiting) return fn();
      waiting++;
      const result = chain.then(async () => { await idle(); return fn(); }).finally(() => { waiting--; });
      chain = result.catch(() => {});
      return result;
    },
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
