import {CURSORS, frameHandle, nearestEdge, resizeScale} from './resize.js';
import {FRAME_REACH, SNAP_RADIUS} from './app_config.js';
import {selectionContains} from './selection.js';

// Resizing owns its controls; editor contains the live state shared with other controllers.
export function createResizingController(editor, {
  DOMMatrix = globalThis.DOMMatrix,
  DOMPoint = globalThis.DOMPoint,
  hitStack,
  localToOverlay,
  object,
  overlay,
  parents,
  point,
  resizeRefusal = () => selectionResizeRefusal(editor, {topSelection, object}),
  requestAnimationFrame = globalThis.requestAnimationFrame,
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
  // picks instead.
  function frameCursor(x, y) {
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

  // Other objects' bounds and the artboard's edges, which a dragged edge snaps
  // to, in the overlay's frame.
  function frameSnapTargets() {
    const [bx, by, bw, bh] = editor.state.bounds, x = [bx, bx + bw], y = [by, by + bh];
    const inverse = overlay.getScreenCTM()?.inverse();
    if (!inverse) return {x, y};
    const map = parents(), selected = new Set(editor.state.selection.objects), around = new Set();
    for (const id of selected) for (let at = map.get(id); at; at = map.get(at)) around.add(at);
    const inside = id => { for (let at = id; at; at = map.get(at)) if (selected.has(at)) return true; return false; };
    for (const item of editor.state.objects) {
      if (item.resource || ['defs', 'clipPath'].includes(item.tag) || around.has(item.id) || inside(item.id)) continue;
      const rect = svgElement(item.id)?.getBoundingClientRect();
      if (!rect || rect.width + rect.height === 0) continue;
      const a = new DOMPoint(rect.left, rect.top).matrixTransform(inverse), b = new DOMPoint(rect.right, rect.bottom).matrixTransform(inverse);
      x.push(a.x, b.x); y.push(a.y, b.y);
    }
    return {x, y};
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
  const api = {resizeRefusal, frameAt, frameCursor, hoverResize, pressFrame, frameSnapTargets, resizeDrag, previewResize, frameSize, wire};
  return api;
}

export function selectionResizeRefusal(editor, {topSelection, object}) {
  for (const id of topSelection()) {
    const item = object(id), lock = item?.inherited_locks.find(kind => kind === 'transform' || kind === 'geometry');
    if (lock) return `${item.label}: ${lock === 'transform' ? 'position' : 'geometry'} is locked; unlock it to resize`;
  }
  return '';
}

