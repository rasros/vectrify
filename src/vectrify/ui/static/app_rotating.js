import {dragAngle, normalAngle, sharedAngle, turnPoint, turnedBox} from './transform.js';
import {FRAME_REACH, ROTATE_REACH, SNAP_RADIUS} from './app_config.js';
import {selectionResizeRefusal} from './app_resizing.js';
import {hiddenRefusal} from './visibility.js';

// Rotating owns the rotation handle, the pivot and the flips; editor contains
// the live state shared with other controllers.
//
// The pivot is the selection frame's centre unless dragged elsewhere. A
// dragged pivot, and the one a rotation turned about, stays for the same
// objects until the drawing changes some other way, so turns add up about
// one point. Rotations and flips go to the server as one edit each, which
// composes them in each object's parent frame as the preview does.
export function createRotatingController(editor, {
  DOMMatrix = globalThis.DOMMatrix,
  DOMPoint = globalThis.DOMPoint,
  action,
  drawOverlay,
  localToOverlay,
  object,
  overlay,
  point,
  selectionBounds,
  startGesture,
  svgElement,
  topSelection,
  xmlElement,
} = {}) {

  const selectionKey = () => [...editor.state.selection.objects].sort().join(' ');
  const centre = box => ({x: (box.left + box.right) / 2, y: (box.top + box.bottom) / 2});

  // Why the selection cannot be rotated or flipped (*verb*), or ''.
  function turnRefusal(verb = 'rotate') {
    if (!editor.state?.selection.objects.length) return 'Select objects first';
    const hidden = hiddenRefusal(topSelection(), object, verb);
    if (hidden) return hidden;
    if (!selectionBounds()) return 'The selection has no bounds to turn: select shapes or groups, not definitions or clipping boundaries';
    return selectionResizeRefusal(editor, {topSelection, object}, verb);
  }

  // The point rotations turn about, in document coordinates, or null.
  function pivot() {
    const placed = editor.pivot;
    if (placed && placed.key === selectionKey() && placed.revision === editor.state.revision) return {x: placed.x, y: placed.y};
    const box = editor.selectionBox || selectionBounds();
    return box && centre(box);
  }

  // The angle the selected objects are turned by on the page, or null when
  // they differ.
  function selectionAngle() {
    return sharedAngle(topSelection().map(id => localToOverlay(svgElement(id))).filter(Boolean));
  }

  // Rotate the selection by *angle* degrees about *about*, the pivot by
  // default, as one undoable edit. The pivot stays where it turned.
  async function rotateBy(angle, about = pivot()) {
    if (!about) return false;
    const key = selectionKey();
    const done = await action('rotate', {angle, pivot: [about.x, about.y]}, 'Rotating…');
    if (done && selectionKey() === key) { editor.pivot = {...about, key, revision: editor.state.revision}; drawOverlay(); }
    return done;
  }

  // Mirror the selection about its frame's centre.
  function flip(axis) {
    const box = selectionBounds();
    if (!box) return false;
    const {x, y} = centre(box);
    return action(`flip_${axis}`, {pivot: [x, y]}, 'Flipping…');
  }

  // Where the handle and the pivot are shown, in the overlay's frame: the
  // handle above the middle of the frame's top edge, a fixed screen distance
  // out. Neither shows outside Move or while the selection cannot turn.
  function handles() {
    const box = editor.selectionBox;
    if (editor.tool !== 'move' || !box || selectionResizeRefusal(editor, {topSelection, object}, 'rotate')) return null;
    return {box, handle: {x: (box.left + box.right) / 2, y: box.top - ROTATE_REACH / editor.zoom}, pivot: pivot()};
  }

  // The rotation handle or the pivot at the screen point (x, y), or null.
  // The pivot of a frame too small to grab apart from it is not grabbed, so
  // a small object still moves wherever it is pressed.
  function rotationAt(x, y) {
    const shown = handles(), matrix = shown && overlay.getScreenCTM();
    if (!matrix) return null;
    const near = p => { const s = new DOMPoint(p.x, p.y).matrixTransform(matrix); return Math.hypot(s.x - x, s.y - y) <= FRAME_REACH + 2; };
    if (near(shown.handle)) return 'rotate';
    const {box} = shown, small = Math.min(box.right - box.left, box.bottom - box.top) * editor.zoom <= 4 * FRAME_REACH;
    return !small && shown.pivot && near(shown.pivot) ? 'pivot' : null;
  }

  function rotationCursor(x, y) {
    const part = rotationAt(x, y);
    return part === 'rotate' ? 'grab' : part === 'pivot' ? 'crosshair' : null;
  }

  // A press on the handle starts turning the selection about the pivot; one
  // on the pivot starts moving it.
  function pressRotation(event, common, part) {
    if (part === 'pivot') {
      startGesture('pivot', event, common, {before: editor.pivot ?? null, box: {...editor.selectionBox}});
      return;
    }
    const members = topSelection().map(id => {
      const element = svgElement(id);
      return {id, element, before: object(id).attributes.transform || '', frame: element && localToOverlay(element.parentElement)};
    }).filter(member => member.element && member.frame);
    const p = point(event);
    startGesture('rotate', event, common, {members, box: {...editor.selectionBox}, pivot: pivot(), grab: {x: p.x, y: p.y}, result: null});
  }

  // How far the drag turns; Shift turns in 15° steps.
  function rotateDrag(event) {
    const p = point(event), {pivot: about, grab} = editor.drag;
    return {angle: dragAngle(about, grab, {x: p.x, y: p.y}, {snap: event.shiftKey}), pivot: about};
  }

  // The preview turns each object in its parent's frame, as the server will.
  function previewRotate() {
    const {angle, pivot: {x, y}} = editor.drag.result;
    const page = new DOMMatrix().translate(x, y).rotate(angle).translate(-x, -y);
    for (const member of editor.drag.members) {
      const local = member.frame.inverse().multiply(page).multiply(member.frame);
      member.element.setAttribute('transform', `${local} ${member.before}`.trim());
    }
  }

  const finishRotate = result => rotateBy(result.angle, result.pivot);

  // A dragged pivot snaps to the frame's corners, edge middles and centre,
  // unless Alt or Ctrl/⌘ is held.
  function placePivot(event) {
    let p = point(event);
    if (!event.altKey && !event.ctrlKey && !event.metaKey) {
      const {left, top, right, bottom} = editor.drag.box, xs = [left, (left + right) / 2, right], ys = [top, (top + bottom) / 2, bottom];
      let best = null;
      for (const x of xs) for (const y of ys) {
        const distance = Math.hypot(x - p.x, y - p.y);
        if (distance <= SNAP_RADIUS / editor.zoom && (!best || distance < best.distance)) best = {x, y, distance};
      }
      if (best) p = best;
    }
    editor.pivot = {x: p.x, y: p.y, key: selectionKey(), revision: editor.state.revision};
  }

  function restorePivot(gesture) { editor.pivot = gesture.before; }

  // The handle on a stem from the frame and the pivot as a ringed cross.
  // While turning, the frame turns with the selection about the pivot.
  function drawRotation() {
    const turning = editor.drag?.kind === 'rotate' && editor.drag.moved && editor.drag.result;
    const shown = turning ? {box: editor.drag.box, pivot: editor.drag.pivot} : handles();
    if (!shown?.pivot) return;
    const angle = turning ? editor.drag.result.angle : 0, {box} = shown, r = 4 / editor.zoom;
    const top = {x: (box.left + box.right) / 2, y: box.top};
    const stem = turnPoint(top, shown.pivot, angle), handle = turnPoint({x: top.x, y: top.y - ROTATE_REACH / editor.zoom}, shown.pivot, angle);
    const content = editor.overlayFrame.content;
    if (turning) content.append(xmlElement('polygon', {points: turnedBox(box, shown.pivot, angle).map(p => `${p.x},${p.y}`).join(' '), class: 'resize-frame'}));
    content.append(xmlElement('line', {x1: stem.x, y1: stem.y, x2: handle.x, y2: handle.y, class: 'rotate-stem'}),
      xmlElement('circle', {cx: handle.x, cy: handle.y, r, class: 'rotate-handle'}));
    // A dark halo keeps the pivot visible on light and dark paint alike.
    const {x, y} = shown.pivot, cross = `M${x - 2 * r} ${y}H${x + 2 * r}M${x} ${y - 2 * r}V${y + 2 * r}`;
    for (const part of ['rotate-pivot-halo', 'rotate-pivot'])
      content.append(xmlElement('circle', {cx: x, cy: y, r, class: part}), xmlElement('path', {d: cross, class: part}));
  }

  // The status line while turning.
  const rotationStatus = result => `Rotate · ${+normalAngle(result.angle).toFixed(1)}°`;

  function wire() {
    return api;
  }
  const api = {turnRefusal, pivot, selectionAngle, rotateBy, flip, rotationAt, rotationCursor, pressRotation, rotateDrag, previewRotate, finishRotate, placePivot, restorePivot, drawRotation, rotationStatus, wire};
  return api;
}
