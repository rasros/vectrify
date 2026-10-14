import {SAMPLE_STEP} from './pencil.js';

// Canvas gestures own their previews and completion. Hit testing, pointer
// capture and queued-input replay stay with the canvas event handlers.
export function canvasGestures({view, path, shapes, pencil, selection, nodes, resize, moving, rotate, gradient, knife, region, eyedropper, redraw, point, drawOverlay}) {
  const inspectPoints = () => { nodes.renderInspector(); drawOverlay(); };
  const restoreTransforms = gesture => {
    for (const member of gesture.members) {
      if (member.before) member.element.setAttribute('transform', member.before);
      else member.element.removeAttribute('transform');
    }
  };
  // One displacement in the overlay's (the document's) frame moves the whole
  // selection: each object takes it in its parent's frame, as previewed and
  // as sent, so objects in differently transformed groups move together.
  const moveSelection = (gesture, event) => {
    const {dx, dy} = moving.drag(event);
    for (const member of gesture.members) {
      // A displacement has no translation: only the frame's linear part maps it.
      const {a, b, c, d} = member.frame.inverse();
      member.offset = [a * dx + c * dy, b * dx + d * dy];
      member.element.setAttribute('transform', `translate(${member.offset.join(' ')}) ${member.before}`.trim());
    }
    drawOverlay();
  };
  return {
    pan: {
      press(gesture) { gesture.pan = {...view.pan()}; view.setPanning(true); },
      move(gesture, event) {
        view.setPan({x: gesture.pan.x + event.clientX - gesture.x, y: gesture.pan.y + event.clientY - gesture.y});
        view.update();
      },
      release() { view.setPanning(false); },
      cancel() { view.setPanning(false); },
    },
    drawPath: {
      press(gesture, event) {
        const p = point(event);
        gesture.anchor = {x: p.x, y: p.y};
        path.draft().push(gesture.anchor); path.setHover(null); drawOverlay();
      },
      move(gesture, event) {
        if (!gesture.moved) return;
        const p = point(event), anchor = gesture.anchor;
        anchor.out = {x: p.x, y: p.y};
        anchor.in = {x: 2 * anchor.x - p.x, y: 2 * anchor.y - p.y};
        drawOverlay();
      },
      release() { path.setHover(null); drawOverlay(); },
      cancel() { path.draft().pop(); },
    },
    closePath: {
      async release(gesture) { if (!gesture.moved) await path.finish(true); },
    },
    // A shape tool's drag, previewed as drawn. Shift and Alt are read on
    // every move and as they are pressed, so the preview follows them.
    shape: {
      press(gesture, event) {
        const p = point(event);
        gesture.start = {x: p.x, y: p.y}; gesture.end = gesture.start;
        gesture.square = event.shiftKey; gesture.centre = event.altKey;
      },
      move(gesture, event) {
        const p = point(event);
        gesture.end = {x: p.x, y: p.y}; gesture.square = event.shiftKey; gesture.centre = event.altKey;
        if (gesture.moved) { shapes.preview(gesture); drawOverlay(); }
      },
      modifiers(gesture, event) {
        gesture.square = event.shiftKey; gesture.centre = event.altKey;
        if (gesture.moved) { shapes.preview(gesture); drawOverlay(); }
      },
      async release(gesture) { drawOverlay(); await shapes.finish(gesture); },
      cancel: () => shapes.cancel(),
    },
    // A Pencil stroke: every pointer sample a screen pixel or more from the
    // last, those the browser coalesced between events included, so a fast
    // stroke is followed as closely as a slow one. The fit is previewed as
    // it is drawn.
    pencil: {
      press(gesture, event) {
        const p = point(event);
        gesture.points = [[p.x, p.y]];
      },
      move(gesture, event) {
        const samples = event.getCoalescedEvents?.() ?? [];
        for (const sample of samples.length ? samples : [event]) {
          const p = point(sample), last = gesture.points.at(-1);
          if (Math.hypot(p.x - last[0], p.y - last[1]) * view.zoom() >= SAMPLE_STEP) gesture.points.push([p.x, p.y]);
        }
        if (gesture.moved) { pencil.preview(gesture); drawOverlay(); }
      },
      async release(gesture) { drawOverlay(); await pencil.finish(gesture); },
      cancel: () => drawOverlay(),
    },
    knife: {
      press(gesture, event) {
        const p = point(event);
        gesture.start = {x: p.x, y: p.y}; gesture.end = {x: p.x, y: p.y};
      },
      move(gesture, event) { if (gesture.moved) { gesture.end = knife.end(event); drawOverlay(); } },
      async release(gesture) {
        if (!gesture.moved) return selection.pick(gesture);
        drawOverlay(); await knife.finish(gesture);
      },
    },
    // Region: a drag draws a rectangle (a click still selects); in polygon
    // shape each click places a corner.
    region: {
      press(gesture, event) {
        const p = point(event);
        gesture.start = {x: p.x, y: p.y}; gesture.end = {x: p.x, y: p.y};
      },
      move(gesture, event) {
        if (!gesture.moved) return;
        const p = point(event);
        gesture.end = {x: p.x, y: p.y}; drawOverlay();
      },
      async release(gesture) {
        if (!gesture.moved) return selection.pick(gesture);
        region.rectangle(gesture);
      },
    },
    // Eyedropper: the sample is taken where the press is released, the mark
    // following a drag; Escape before then cancels it, changing nothing.
    eyedropper: {
      press(gesture, event) { const p = point(event); gesture.at = {x: p.x, y: p.y}; drawOverlay(); },
      move(gesture, event) { const p = point(event); gesture.at = {x: p.x, y: p.y}; drawOverlay(); },
      async release(gesture) { drawOverlay(); await eyedropper.sample(gesture.at); },
      cancel: () => drawOverlay(),
    },
    regionCorner: {
      press(gesture, event) { const p = point(event); gesture.at = {x: p.x, y: p.y}; },
      release(gesture) { region.corner(gesture.at); },
    },
    redraw: {
      press(gesture, event) {
        const p = point(event);
        gesture.points = [[p.x, p.y]]; gesture.longWay = event.shiftKey;
        redraw.setHover(null); drawOverlay();
      },
      move(gesture, event) {
        const p = point(event), last = gesture.points.at(-1);
        gesture.longWay = event.shiftKey;
        if (Math.hypot(p.x - last[0], p.y - last[1]) * view.zoom() >= 1.5) gesture.points.push([p.x, p.y]);
        if (gesture.moved) drawOverlay();
      },
      async release(gesture) {
        if (!gesture.moved) return selection.pick(gesture);
        drawOverlay(); await redraw.finish(gesture);
      },
    },
    // Moving the selection. Shift, Alt and Ctrl/⌘ are read on every move and
    // as they are pressed, so the preview follows them.
    move: {
      press(gesture, event) { const p = point(event); gesture.start = {x: p.x, y: p.y}; },
      move(gesture, event) {
        gesture.pointer = {clientX: event.clientX, clientY: event.clientY};
        // A selection holding hidden objects stays put: they would move unseen.
        if (gesture.moved && gesture.refusal) {
          if (!gesture.warned) selection.refuse(gesture.refusal);
          gesture.warned = true; return;
        }
        if (gesture.moved) moveSelection(gesture, event);
      },
      modifiers(gesture, event) {
        if (!gesture.moved || !gesture.pointer || gesture.refusal) return;
        const {shiftKey, altKey, ctrlKey, metaKey} = event;
        moveSelection(gesture, {...gesture.pointer, shiftKey, altKey, ctrlKey, metaKey});
      },
      async release(gesture) {
        if (!gesture.moved) return selection.pick(gesture);
        if (gesture.refusal) return;
        // A drag that ends where it started, or snaps back there, edits nothing.
        if (gesture.members.every(member => !member.offset || (!member.offset[0] && !member.offset[1]))) {
          restoreTransforms(gesture); selection.renderDrawing(); drawOverlay(); return;
        }
        await selection.move(Object.fromEntries(gesture.members.map(member => [member.id, member.offset || [0, 0]])));
      },
      cancel: restoreTransforms,
    },
    box: {
      press(gesture, event) { gesture.end = {x: event.clientX, y: event.clientY}; },
      move(gesture, event) {
        gesture.end = {x: event.clientX, y: event.clientY};
        if (gesture.moved) drawOverlay();
      },
      async release(gesture) {
        if (!gesture.moved) return selection.pick(gesture);
        drawOverlay(); await selection.box(gesture);
      },
      // A completed selection box leaves the click cycle available.
      keepClickCycle: true,
    },
    node: {
      press: inspectPoints,
      move(gesture, event) {
        if (!gesture.moved) return;
        nodes.preview(nodes.snap(event)); drawOverlay(); nodes.renderNodeInspector();
      },
      async release(gesture) {
        if (gesture.moved) await nodes.finish(gesture);
        else if (gesture.toggle) await nodes.select(gesture.objects, gesture.toggle);
        else if (gesture.collapse) await nodes.select(gesture.objects, [gesture.key]);
      },
      cancel: gesture => nodes.restore(gesture),
    },
    'point-click': {
      press: inspectPoints,
    },
    resize: {
      move(gesture, event) {
        if (!gesture.moved) return;
        if (gesture.refusal) {
          if (!gesture.warned) resize.refuse(gesture.refusal);
          gesture.warned = true; return;
        }
        gesture.result = resize.result(event); resize.preview(); drawOverlay();
      },
      async release(gesture) {
        // Grabbing the frame only resizes on a drag; clicks still pick and
        // cycle the painted objects beneath it, including thin shapes.
        if (!gesture.moved) return selection.pick(gesture);
        if (gesture.refusal) return;
        const result = gesture.result;
        if (result && (result.sx !== 1 || result.sy !== 1)) await resize.finish(result);
        else { selection.renderDrawing(); drawOverlay(); }
      },
      cancel: restoreTransforms,
    },
    // Turning the selection by its handle about the pivot, previewed as the
    // server will compose it. A click on the handle turns nothing.
    rotate: {
      move(gesture, event) {
        if (!gesture.moved) return;
        gesture.result = rotate.result(event); rotate.preview(); drawOverlay();
      },
      async release(gesture) {
        const result = gesture.moved && gesture.result;
        if (result && Math.abs(result.angle) > 1e-9) await rotate.finish(result);
        else { selection.renderDrawing(); drawOverlay(); }
      },
      cancel: restoreTransforms,
    },
    // A click on the pivot picks what is under it, as a click in the frame
    // does, so the object there can still be picked (or made the key object).
    pivot: {
      move(gesture, event) { if (gesture.moved) { rotate.placePivot(event); drawOverlay(); } },
      release(gesture) { if (!gesture.moved) return selection.pick(gesture); drawOverlay(); },
      cancel: gesture => rotate.restorePivot(gesture),
    },
    // Moving a gradient's end by its handle in the Gradient tool, previewed
    // on the drawn gradient as the edit will set it. Shift, Alt and Ctrl/⌘
    // are read as they are pressed too. A click on a handle picks what is
    // under it, as a click elsewhere does.
    gradient: {
      move(gesture, event) {
        gesture.pointer = {clientX: event.clientX, clientY: event.clientY};
        if (!gesture.moved) return;
        if (gesture.refusal) {
          if (!gesture.warned) gradient.refuse(gesture.refusal);
          gesture.warned = true; return;
        }
        gesture.result = gradient.result(event); gradient.preview(); drawOverlay();
      },
      modifiers(gesture, event) {
        if (!gesture.moved || !gesture.pointer || gesture.refusal) return;
        const {shiftKey, altKey, ctrlKey, metaKey} = event;
        gesture.result = gradient.result({...gesture.pointer, shiftKey, altKey, ctrlKey, metaKey}); gradient.preview(); drawOverlay();
      },
      async release(gesture) {
        if (!gesture.moved) return selection.pick(gesture);
        if (gesture.result) await gradient.finish(gesture);
      },
      cancel: gesture => gradient.restore(gesture),
    },
  };
}
