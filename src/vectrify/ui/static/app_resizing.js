import {CURSORS, frameHandle, nearestEdge, resizeScale} from './resize.js';
import {FRAME_REACH, SNAP_RADIUS} from './app_config.js';
import {selectionContains} from './selection.js';
import {boxLines, snapBox} from './snap.js';

// Resizing owns its controls; editor contains the live state shared with other controllers.
export function createResizingController(editor, {
  DOMMatrix = globalThis.DOMMatrix,
  DOMPoint = globalThis.DOMPoint,
  hitStack,
  localToOverlay,
  object,
  objectBounds,
  overlay,
  parents,
  point,
  resizeRefusal = () => selectionResizeRefusal(editor, {topSelection, object}),
  requestAnimationFrame = globalThis.requestAnimationFrame,
  rotationCursor = () => null,
  stage,
  startGesture,
  svgElement,
  topSelection,
} = {}) {
  let frameHover = 0;

  // Why the selection cannot be resized, or ''.
  // The part of the frame at the screen point (x, y): an edge, a corner,
  // 'inside' or null. A line's whole frame moves it; Select has no frame hits.
  function frameAt(x, y) {
    if (editor.tool !== 'move') return null;
    const matrix = editor.selectionBox && overlay.getScreenCTM();
    if (!matrix) return null;
    const a = new DOMPoint(editor.selectionBox.left, editor.selectionBox.top).matrixTransform(matrix), b = new DOMPoint(editor.selectionBox.right, editor.selectionBox.bottom).matrixTransform(matrix);
    const handle = frameHandle({left: a.x, top: a.y, right: b.x, bottom: b.y}, x, y, FRAME_REACH);
    return handle && Math.min(Math.abs(b.x - a.x), Math.abs(b.y - a.y)) <= 2 * FRAME_REACH ? 'inside' : handle;
  }

  // The cursor over the frame: a resize arrow on an edge or corner, and the
  // move cursor inside it, except over an unselected object, which a press
  // picks instead. The rotation handle and pivot have their own.
  function frameCursor(x, y) {
    const turning = editor.tool === 'move' && !editor.space && rotationCursor(x, y);
    if (turning) return turning;
    const handle = frameAt(x, y);
    if (!handle || editor.space) return 'default';
    if (handle !== 'inside') return resizeRefusal() ? 'not-allowed' : CURSORS[handle];
    const hit = hitStack(x, y)[0];
    return !hit || selectionContains(editor.state.selection.objects, hit, parents(), editor.state.root) ? CURSORS.inside : 'default';
  }

  function hoverResize(event) {
    if (frameHover) return;
    const x = event.clientX, y = event.clientY;
    frameHover = requestAnimationFrame(() => {
      frameHover = 0;
      if (editor.tool === 'move' && !editor.drag && editor.state) stage.style.cursor = frameCursor(x, y);
    });
  }

  // A press on an edge or corner of the frame starts resizing the selection.
  function pressFrame(event, common, handle) {
    const refusal = resizeRefusal();
    const members = topSelection().map(id => {
      const element = svgElement(id);
      return {id, element, before: object(id).attributes.transform || '', frame: element && localToOverlay(element.parentElement)};
    }).filter(member => member.element && member.frame);
    const box = {...editor.selectionBox}, grab = point(event);
    // The edge follows the pointer from where it was grabbed.
    const offset = {x: handle.includes('w') ? box.left - grab.x : handle.includes('e') ? box.right - grab.x : 0,
      y: handle.includes('n') ? box.top - grab.y : handle.includes('s') ? box.bottom - grab.y : 0};
    startGesture('resize', event, common, {handle, box, members, offset, refusal, result: null});
  }

  // The objects the selection can snap to: not the selected ones, what they
  // hold, nor the groups holding them, which move or grow with them.
  function snapCandidates() {
    const map = parents(), selected = new Set(editor.state.selection.objects), around = new Set();
    for (const id of selected) for (let at = map.get(id); at; at = map.get(at)) around.add(at);
    const inside = id => { for (let at = id; at; at = map.get(at)) if (selected.has(at)) return true; return false; };
    return editor.state.objects.filter(item => !item.resource && !['defs', 'clipPath'].includes(item.tag) && !around.has(item.id) && !inside(item.id));
  }

  // Other objects' bounds and the artboard's edges, which a dragged edge snaps
  // to, in the overlay's frame.
  function frameSnapTargets() {
    const [bx, by, bw, bh] = editor.state.bounds, x = [bx, bx + bw], y = [by, by + bh];
    const inverse = overlay.getScreenCTM()?.inverse();
    if (!inverse) return {x, y};
    for (const item of snapCandidates()) {
      const rect = svgElement(item.id)?.getBoundingClientRect();
      if (!rect || rect.width + rect.height === 0) continue;
      const a = new DOMPoint(rect.left, rect.top).matrixTransform(inverse), b = new DOMPoint(rect.right, rect.bottom).matrixTransform(inverse);
      x.push(a.x, b.x); y.push(a.y, b.y);
    }
    return {x, y};
  }

  // The lines a moved selection snaps to, in the overlay's frame: the edges
  // and centres of the artboard and of other visible objects' geometric
  // bounds, the same bounds the selection frame shows.
  function moveSnapLines() {
    const [bx, by, bw, bh] = editor.state.bounds, boxes = [{left: bx, top: by, right: bx + bw, bottom: by + bh}];
    for (const item of snapCandidates()) {
      const element = svgElement(item.id);
      if (!element || element.closest?.('defs, clipPath, mask, pattern, symbol, marker')) continue;
      if (element.checkVisibility && !element.checkVisibility({visibilityProperty: true})) continue;
      const box = objectBounds(item.id);
      if (box) boxes.push(box);
    }
    return boxLines(boxes);
  }

  // The displacement a move drag gives, in the overlay's frame: the pointer's
  // offset from its press. Shift keeps it to the axis the pointer has gone
  // furthest along. Unless Alt or Ctrl/⌘ is held, the frame's edges and centre
  // then snap to other objects' and the artboard's within SNAP_RADIUS screen
  // pixels, on the free axes only.
  function moveDrag(event) {
    const drag = editor.drag, p = point(event);
    let dx = p.x - drag.start.x, dy = p.y - drag.start.y;
    const lockedAxis = event.shiftKey ? (Math.abs(dx) >= Math.abs(dy) ? 'y' : 'x') : null;
    if (lockedAxis === 'x') dx = 0;
    if (lockedAxis === 'y') dy = 0;
    drag.guides = null;
    if (drag.box && !event.altKey && !event.ctrlKey && !event.metaKey) {
      drag.lines ??= moveSnapLines();
      const snap = snapBox(drag.box, dx, dy, drag.lines, SNAP_RADIUS / editor.zoom, lockedAxis);
      ({dx, dy} = snap); drag.guides = snap.guides;
    }
    return {dx, dy};
  }

  // The scale a resize drag gives: the dragged edge snaps to other objects'
  // bounds and the artboard's edges unless Ctrl/⌘ is held; Shift keeps the
  // aspect ratio and Alt resizes from the centre.
  function resizeDrag(event) {
    const p = point(event), at = {x: p.x + editor.drag.offset.x, y: p.y + editor.drag.offset.y};
    editor.drag.snapX = editor.drag.snapY = null;
    if (!event.ctrlKey && !event.metaKey) {
      editor.drag.edges ??= frameSnapTargets();
      if (/[we]/.test(editor.drag.handle)) editor.drag.snapX = nearestEdge(at.x, editor.drag.edges.x, SNAP_RADIUS / editor.zoom);
      if (/[ns]/.test(editor.drag.handle)) editor.drag.snapY = nearestEdge(at.y, editor.drag.edges.y, SNAP_RADIUS / editor.zoom);
      at.x = editor.drag.snapX ?? at.x; at.y = editor.drag.snapY ?? at.y;
    }
    return resizeScale(editor.drag.box, editor.drag.handle, at, {keepRatio: event.shiftKey, fromCentre: event.altKey, minimum: 1 / editor.zoom});
  }

  // The preview scales each object in its parent's frame, as the server will.
  function previewResize() {
    const {sx, sy, anchor: [ax, ay]} = editor.drag.result;
    const page = new DOMMatrix().translate(ax, ay).scale(sx, sy).translate(-ax, -ay);
    for (const member of editor.drag.members) {
      const local = member.frame.inverse().multiply(page).multiply(member.frame);
      member.element.setAttribute('transform', `${local} ${member.before}`.trim());
    }
  }

  const frameSize = box => [box.right - box.left, box.bottom - box.top].map(v => (+v.toFixed(1)).toLocaleString()).join(' × ');

  function wire() {
      return api;
  }
  const api = {resizeRefusal, frameAt, frameCursor, hoverResize, pressFrame, snapCandidates, frameSnapTargets, moveSnapLines, moveDrag, resizeDrag, previewResize, frameSize, wire};
  return api;
}

// Why the selection cannot be resized, or rotated or flipped (*verb*), or ''.
export function selectionResizeRefusal(editor, {topSelection, object}, verb = 'resize') {
  for (const id of topSelection()) {
    const item = object(id), lock = item?.inherited_locks.find(kind => kind === 'transform' || kind === 'geometry');
    if (lock) return `${item.label}: ${lock === 'transform' ? 'position' : 'geometry'} is locked; unlock it to ${verb}`;
  }
  return '';
}

