// Canvas gestures own their previews and completion. Hit testing, pointer
// capture and queued-input replay stay with the canvas event handlers.
export function canvasGestures({view, path, selection, nodes, resize, knife, redraw, point, drawOverlay}) {
  const inspectPoints = () => { nodes.renderInspector(); drawOverlay(); };
  const restoreTransforms = gesture => {
    for (const member of gesture.members) {
      if (member.before) member.element.setAttribute('transform', member.before);
      else member.element.removeAttribute('transform');
    }
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
    move: {
      move(gesture, event) {
        if (!gesture.moved) return;
        for (const member of gesture.members) {
          const matrix = member.element.parentElement.getScreenCTM(); if (!matrix) continue;
          const inverse = matrix.inverse();
          const start = new DOMPoint(gesture.x, gesture.y).matrixTransform(inverse);
          const end = new DOMPoint(event.clientX, event.clientY).matrixTransform(inverse);
          member.offset = [end.x - start.x, end.y - start.y];
          member.element.setAttribute('transform', `translate(${member.offset.join(' ')}) ${member.before}`);
        }
        drawOverlay();
      },
      async release(gesture) {
        if (!gesture.moved) return selection.pick(gesture);
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
  };
}
