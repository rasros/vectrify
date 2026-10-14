import {DOUBLE_CLICK, NUDGE_RUN, POINT_REACH} from './app_config.js';
import {pickTarget, pointKey, selectionActionLevel, splitKey} from './selection.js';
import {freeEnd} from './lines.js';
import {hiddenRefusal} from './visibility.js';
import {plural} from './util.js';

// What Select all leaves out: definitions, gradients and clipping paths,
// which are not drawn where they are.
const UNSELECTABLE = ['defs', 'clipPath', 'linearGradient', 'stop'];

// Selection owns its controls; editor contains the live state shared with other controllers.
export function createSelectionController(editor, {
  $,
  action,
  carryHandles,
  contourAt,
  document = globalThis.document,
  focusSelection,
  geometryNodes,
  later,
  laterMerged,
  level,
  localToOverlay,
  moveLockRefusal,
  nodeAt,
  now = () => globalThis.performance.now(),
  object,
  overlay,
  parents,
  pointPaths,
  pointTargetsNow,
  renderCommands,
  request,
  restoreValues,
  revealObject,
  runCommand,
  scheduleStrip,
  selectPoints,
  selectedPoints,
  sharingPaths,
  svgElement,
  toast,
  topSelection,
  valuesById,
} = {}) {

  const nearKey = target => target && `${target.dataset.object} ${target.dataset.node} ${target.dataset.part}`;

  function nearestPoint(x, y) {
    let best = null, distance = POINT_REACH;
    for (const circle of overlay.querySelectorAll('.node:not(.passive), .handle')) {
      const box = circle.getBoundingClientRect(), d = Math.hypot(box.x + box.width/2 - x, box.y + box.height/2 - y);
      // Handles come after points, so they win a tie with their own point.
      if (d <= distance) { best = circle; distance = d; }
    }
    return best;
  }

  // The selected points, by the contour they are on, for the line edits.
  function pointContours() {
    const contours = new Map();
    for (const key of selectedPoints()) {
      const contour = contourAt(key);
      if (!contour) continue;
      const id = `${splitKey(key)[0]} ${contour.id}`;
      if (!contours.has(id)) contours.set(id, {contour, ids: new Set()});
      contours.get(id).ids.add(splitKey(key)[1]);
    }
    return [...contours.values()];
  }

  async function selectObject(id, additive = false, focus = false, toggle = false) {
    editor.activeHandle = null;
    editor.clickCycle = null;
    const removing = (additive || toggle) && editor.state.selection.objects.includes(id);
    let selected = new Set(additive || removing ? editor.state.selection.objects : []);
    if (id) { if (removing) selected.delete(id); else selected.add(id); }
    // Picking an object in the tree enters the group it is in.
    if (focus && id && !additive) {
      const parent = object(id)?.parent;
      editor.scope = parent && parent !== editor.state.root ? parent : null;
    }
    editor.focusPoint = null;
    const success = await action('select', {objects: [...selected]}, 'Selecting…');
    if (success && selected.has(id)) revealObject(id);
    if (success && focus) focusSelection();
    return success;
  }

  function sameClickSpot(x, y, hits) {
    return editor.clickCycle && Math.hypot(x-editor.clickCycle.x, y-editor.clickCycle.y) <= 4 &&
      hits.length === editor.clickCycle.hits.length && hits.every((id, i) => id === editor.clickCycle.hits[i]);
  }

  // What clicks at a spot pick, front to back: the deepest painted shapes.
  function clickTargets(hits) {
    if (level() !== 'objects') return hits;
    const map = parents();
    // A click outside the entered group leaves it.
    if (editor.scope && pickTarget(hits[0], editor.scope, map, editor.state.root).scope !== editor.scope) editor.scope = null;
    return [...new Set(hits.map(hit => svgElement(hit)?.closest('clipPath') ? hit : pickTarget(hit, editor.scope, map, editor.state.root).id))];
  }

  async function selectAtPoint(finished) {
    const hits = finished.hits?.length ? clickTargets(finished.hits) : [];
    if (!hits.length) return clickEmpty(finished.shift);
    // The second click of a double-click keeps what the first one picked.
    const again = sameClickSpot(finished.x, finished.y, hits), double = again && finished.time - editor.clickCycle.time < DOUBLE_CLICK;
    const cycled = !finished.shift && again && !double;
    const index = double ? editor.clickCycle.index : cycled ? (editor.clickCycle.index+1)%hits.length : 0;
    // A double-click acts on what was picked before it: its first click may
    // have cycled on from a selected object to the next one under it.
    editor.lastPick = {id: hits[index], first: double ? editor.lastPick?.first ?? hits[index] : cycled ? hits[editor.clickCycle.index] : hits[index]};
    const success = await selectObject(hits[index], finished.shift, false, !double);
    if (success && !finished.shift) editor.clickCycle = {x:finished.x, y:finished.y, time:finished.time, hits, index};
  }

  // A click on empty canvas: in point tools it drops the points first.
  function clickEmpty(shift) {
    if (shift) return;
    if (selectedPoints().length) return selectPoints(editor.state.selection.objects, []);
    return selectObject(null);
  }

  // The points a point command acts on: [object, node] pairs.
  const pointPairs = () => selectedPoints().map(splitKey);

  // The one selected point, or the one clicked last among several.
  function focusedPoint() {
    const points = selectedPoints();
    return points.includes(editor.focusPoint) ? editor.focusPoint : points.length === 1 ? points[0] : null;
  }

  // A point's incoming and outgoing handle slots, wrapping at a closed join.
  function nodeHandles(key) {
    const node = nodeAt(key), contour = contourAt(key); if (!node) return [];
    const nodes = contour.nodes, i = nodes.indexOf(node), last = nodes.length - 1;
    const seam = contour.closed && last > 0 && (i === 0 || i === last) && nodes[last].values.at(-2) === nodes[0].values.at(-2) && nodes[last].values.at(-1) === nodes[0].values.at(-1);
    const incoming = seam ? nodes[last] : node, outgoing = nodes[seam ? 1 : i + 1];
    const handles = [];
    if (incoming.command === 'C') handles.push({node:incoming, offset:2});
    if (outgoing?.command === 'C') handles.push({node:outgoing, offset:0});
    return handles;
  }

  // Retracted handles do not count as handles in the inspector.
  function handleCount(key) {
    const node = nodeAt(key); if (!node) return 0;
    const point = node.values.slice(-2);
    const apart = handle => handle[0] !== point[0] || handle[1] !== point[1];
    return nodeHandles(key).filter(({node, offset}) => apart(node.values.slice(offset, offset + 2))).length;
  }

  // A handle selection belongs to its anchor, even when its control point is
  // stored on the following segment or the closing segment of a contour.
  function selectedHandle() {
    if (!editor.activeHandle || editor.tool !== 'nodes' || !selectedPoints().includes(editor.activeHandle.anchor)) return null;
    const anchor = nodeAt(editor.activeHandle.anchor);
    const slot = nodeHandles(editor.activeHandle.anchor).find(({node, offset}) => node.id === editor.activeHandle.node && offset === editor.activeHandle.offset);
    if (!slot || slot.node.values.slice(slot.offset, slot.offset + 2).every((v, i) => v === anchor.values.at(i - 2))) return null;
    return editor.activeHandle;
  }

  function straightenHandles() {
    const handle = selectedHandle();
    const points = handle ? [handle.anchor] : selectedPoints();
    const aligned = !points.every(key => nodeAt(key)?.handles_aligned);
    return action('straighten_handles', {points: points.map(splitKey), aligned, ...(handle ? {side: handle.offset} : {})}, 'Changing handle alignment…');
  }

  // The point section of the right panel and the Nodes strip.
  function renderNodeInspector(commands = true) {
    const paths = pointPaths(), points = selectedPoints(), onPoints = level() === 'points';
    const nodes = paths.flatMap(geometryNodes), loaded = paths.every(id => editor.geometries.has(id));
    const node = points.length === 1 && nodeAt(points[0]);
    const chosen = points.map(nodeAt).filter(Boolean), count = new Set(points.map(key => splitKey(key)[0])).size;
    const instances = pointTargetsNow().instances.length;
    $('point-section').hidden = false;
    $('point-selection').textContent = !onPoints ? 'Choose the Nodes tool (S) to edit points' : !chosen.length ? 'Select a point on the canvas' : chosen.length > 1 ? `${chosen.length} points in ${count} ${count === 1 ? 'path' : 'paths'}` : node.command === 'M' ? 'Start point' : node.command === 'C' ? 'Curve endpoint' : 'Line endpoint';
    $('node-count').textContent = nodes.length ? nodes.length.toLocaleString() : '';
    const hints = [selectedHandle() ? 'Handle selected: Delete removes this handle. Toggle Straighten to keep both handles in line while dragging.' : !loaded ? 'Loading path points…' : !chosen.length ? (editor.tool === 'nodes' ? 'Click a point, or drag a box around several; Shift adds. Hold Shift to move horizontally or vertically. Dragged points snap to others; hold Alt or Ctrl/⌘ to drag without snapping.' : 'Points selected in Nodes (S) stay selected here.') : chosen.length > 1 ? 'Drag one to move them together.' : node.pinned ? 'Pinned: unpin it to move or delete it.' : ''];
    const contour = node && contourAt(points[0]);
    if (contour && freeEnd(contour, node.id)) hints.push('This contour is open, so its endpoint can only have one handle. Select both ends and use Join points (5) to close it.');
    // Points of shared geometry are points of every path drawing it.
    const sharing = new Set(points.flatMap(key => sharingPaths(splitKey(key)[0])));
    if (sharing.size) hints.push(`Shared geometry: ${chosen.length === 1 ? 'this point is' : 'these points are'} also in ${plural(sharing.size, 'other path')}, marked faintly, and edits change ${sharing.size === 1 ? 'both' : 'them all'}. Detach (under Path) to edit one path alone.`);
    if (instances) hints.push(`${plural(instances, 'instance')} (use) ${instances === 1 ? 'has' : 'have'} no points of ${instances === 1 ? 'its' : 'their'} own here: Detach ${instances === 1 ? 'it' : 'them'} to edit ${instances === 1 ? 'its' : 'their'} points.`);
    $('node-hint').textContent = hints.filter(Boolean).join(' ');
    $('node-hint').hidden = !$('node-hint').textContent;
    $('point-properties').hidden = false;
    $('node-pin').disabled = !onPoints || !chosen.length;
    // Coordinates are shown for one point, in its path's own frame.
    for (const row of document.querySelectorAll('.point-coordinates')) {
      row.hidden = false;
      if (!node) {
        for (const field of row.querySelectorAll('input')) {
          field.value = ''; field.placeholder = chosen.length > 1 ? 'Multiple' : '—'; field.disabled = true;
        }
        continue;
      }
      for (const [selector, value] of [['.point-x', node.values.at(-2)], ['.point-y', node.values.at(-1)]]) {
        const field = row.querySelector(selector);
        if (document.activeElement !== field) field.value = +value.toFixed(4);
        field.placeholder = ''; field.disabled = !onPoints || !!node.pinned;
      }
    }
    const handles = node ? handleCount(points[0]) : null;
    for (const button of document.querySelectorAll('[data-command^="handles-"]')) button.setAttribute('aria-pressed', String(handles === Number(button.dataset.command.slice(-1))));
    const alignmentPoints = selectedHandle() ? [nodeAt(editor.activeHandle.anchor)] : chosen;
    const aligned = alignmentPoints.filter(n => n?.handles_aligned).length;
    for (const button of document.querySelectorAll('[data-command="straighten-handles"]')) button.setAttribute('aria-pressed', aligned && aligned === alignmentPoints.length ? 'true' : aligned ? 'mixed' : 'false');
    const pinned = chosen.filter(n => n.pinned).length;
    $('node-pin').checked = chosen.length > 0 && pinned === chosen.length; $('node-pin').indeterminate = pinned > 0 && pinned < chosen.length;
    // Points on holes, in one path or several, can fill the holes or make them
    // shapes; which they are on is read once per path.
    if (holeContours(points) === undefined) loadNodeHoles(points);
    if (commands) renderCommands();
    scheduleStrip();
  }

  function holesKey(id) { return `${editor.state.epoch}:${editor.state.revision}:${id}`; }

  function holeContours(points) {
    if (!points.length) return null;
    const contours = new Map();
    for (const key of points) {
      const [id] = splitKey(key), contour = contourAt(key)?.id;
      if (!contour) return null;
      contours.set(`${id} ${contour}`, [id, contour]);
    }
    let loading = false;
    for (const [id, contour] of contours.values()) {
      const holes = editor.pathHoles.get(holesKey(id));
      if (!holes) { loading = true; continue; }
      if (!holes.has(contour)) return null;
    }
    if (loading) return undefined;
    const pairs = [...contours.values()];
    return {contours: pairs, paths: new Set(pairs.map(([id]) => id)).size};
  }

  async function loadNodeHoles(points) {
    const ids = [...new Set(points.map(key => splitKey(key)[0]))].filter(id => !editor.pathHoles.has(holesKey(id)));
    if (!ids.length) return;
    const {epoch, revision} = editor.state;
    for (const id of ids) editor.pathHoles.set(holesKey(id), null);
    await Promise.all(ids.map(async id => {
      let found = new Set();
      try {
        const result = await request('/api/holes', {object: id, epoch, revision});
        found = new Set(result.holes.map(h => h.id));
      } catch { /* A path whose holes cannot be read simply offers none. */ }
      if (editor.state.epoch === epoch && editor.state.revision === revision) editor.pathHoles.set(holesKey(id), found);
    }));
    if (editor.state.epoch === epoch && editor.state.revision === revision) renderNodeInspector();
  }


  async function fillPointHoles() {
    const holes = holeContours(selectedPoints()); if (!holes) return;
    const count = holes.contours.length, where = holes.paths > 1 ? ` in ${plural(holes.paths, 'path')}` : '';
    if (await action('fill_holes', {contours: holes.contours}, count === 1 ? 'Filling hole…' : 'Filling holes…'))
      toast(`Filled ${count === 1 ? 'the hole' : `${count} holes`}${where}. Undo restores ${count === 1 ? 'it' : 'them'}.`);
  }

  async function pointHolesToShapes() {
    const holes = holeContours(selectedPoints()); if (!holes) return;
    const count = holes.contours.length;
    if (await action('holes_to_shapes', {contours: holes.contours}, count === 1 ? 'Making a shape…' : 'Making shapes…'))
      toast(count === 1 ? 'The hole is now its own shape, just above the path. Undo restores the hole.' : `The ${count} holes are now shapes of their own, each just above its path. Undo restores them.`);
  }

  // Point commands act on every selected point, in however many paths.
  function movePointTo(x, y) {
    const key = focusedPoint(), node = key && nodeAt(key); if (!node) return;
    const [id, nodeId] = splitKey(key);
    action('node', {object: id, node: nodeId, values: [...node.values.slice(0,-2), x, y]});
  }

  // Select all: in a point tool, every point of the selected paths; else
  // every object at the editing level, inside the entered group if any, that
  // is drawn: not definitions, gradients or clipping boundaries, nor hidden
  // objects, which the Objects tree selects.
  function selectAllTargets() {
    if (level() === 'points') return pointPaths().flatMap(id => geometryNodes(id).map(node => pointKey(id, node.id)));
    const container = editor.scope || editor.state.root;
    return editor.state.objects.filter(item => item.parent === container && !item.resource && !item.inherited_hidden && !UNSELECTABLE.includes(item.tag)).map(item => item.id);
  }

  function selectAllRefusal() {
    if (level() === 'points') return pointPaths().length ? '' : 'Select paths first: Select all then picks all their points';
    return selectAllTargets().length ? '' : editor.scope ? 'The entered group has nothing to select' : 'The drawing has nothing to select';
  }

  function selectAll() {
    const found = selectAllTargets();
    if (level() === 'points') return found.length ? selectPoints(editor.state.selection.objects, found) : undefined;
    editor.activeHandle = null; editor.clickCycle = null; editor.focusPoint = null;
    return action('select', {objects: found}, 'Selecting…');
  }

  // Arrow keys nudge the selection by *dx*, *dy* in document units: the
  // selected points when any are selected, else the selected objects. Nudges
  // pressed while an edit runs add up into one edit, sent once it is done,
  // so a held key never queues a backlog. A run of nudges of one selection,
  // each within NUDGE_RUN of the last, is one undo step (the server joins
  // them while nothing else changes the drawing in between).
  let nudgeRun = null;

  function nudge(dx, dy) {
    const {epoch, selection} = editor.state, key = JSON.stringify([epoch, selectionActionLevel(level(), selectedPoints(), selection.nodes), selection.objects, selection.nodes]), time = now();
    if (!nudgeRun || nudgeRun.key !== key || time - nudgeRun.time > NUDGE_RUN) nudgeRun = {id: `${epoch}:${time}:${Math.random().toString(36).slice(2)}`, key};
    nudgeRun.time = time;
    const run = nudgeRun.id;
    return laterMerged(`nudge ${run}`, [dx, dy], (a, b) => [a[0] + b[0], a[1] + b[1]], ([x, y]) => nudgeBy(x, y, run));
  }

  async function nudgeBy(dx, dy, run) {
    if (!editor.state || (!dx && !dy)) return false;
    const edit = selectionActionLevel(level(), selectedPoints(), editor.state.selection.nodes) === 'points' ? pointNudge(dx, dy) : objectNudge();
    if (!edit) return false;
    if (edit.refusal) { toast(edit.refusal, true); return false; }
    if (edit.changes) return action('move_nodes', {changes: edit.changes, nudge: run}, 'Moving points…');
    return action('move', {dx, dy, nudge: run}, 'Moving selection…');
  }

  // The objects move through the move command, which takes the document
  // offset into each one's own frame; locked positions refuse.
  function objectNudge() {
    const top = topSelection();
    if (!top.length) return null;
    if (top.some(id => UNSELECTABLE.includes(object(id)?.tag))) return {refusal: 'Definitions keep their place: select drawing objects to move'};
    const hidden = hiddenRefusal(top, object, 'move');
    if (hidden) return {refusal: hidden};
    const refusal = moveLockRefusal(top);
    return refusal ? {refusal} : {};
  }

  // The selected points move by the offset taken into each path's own frame,
  // their handles riding along, as a drag moves them: a point of geometry
  // shared by several shown paths moves once.
  function pointNudge(dx, dy) {
    const points = selectedPoints();
    if (!points.length) return null;
    if (points.some(key => nodeAt(key)?.pinned)) return {refusal: 'Pinned points stay put: unpin them to move them'};
    const locked = [...new Set(points.map(key => splitKey(key)[0]))].map(object).find(item => item?.inherited_locks.includes('geometry'));
    if (locked) return {refusal: `${locked.label}: geometry is locked; unlock it to move its points`};
    const once = new Set(), moving = new Map();
    for (const key of points) {
      const [id, nodeId] = splitKey(key), twin = `${editor.geometries.get(id)?.id} ${nodeId}`;
      if (!nodeAt(key) || once.has(twin)) continue;
      once.add(twin);
      if (!moving.has(id)) moving.set(id, new Set());
      moving.get(id).add(nodeId);
    }
    const changes = {};
    for (const [id, nodes] of moving) {
      const g = editor.geometries.get(id), matrix = localToOverlay(svgElement(id));
      if (!g || !matrix) continue;
      // The overlay is in document units: its inverse's linear part takes
      // the offset into the path's frame, through any nested transforms.
      const {a, b, c, d} = matrix.inverse(), x = a * dx + c * dy, y = b * dx + d * dy;
      const saved = valuesById(g);
      for (const node of geometryNodes(id)) if (nodes.has(node.id)) { const v = node.values; v[v.length-2] += x; v[v.length-1] += y; }
      carryHandles(g, saved);
      for (const node of geometryNodes(id)) {
        const before = saved.get(node.id);
        if (node.values.some((v, i) => v !== before[i])) (changes[id] ||= {})[node.id] = [...node.values];
      }
      restoreValues(g, saved);
    }
    return Object.keys(changes).length ? {changes} : null;
  }

  function wire() {
    // The point's coordinates stay in the Points inspector.
    for (const field of document.querySelectorAll('.point-x, .point-y')) {
      field.onchange = () => {
        const row = field.closest('.point-coordinates');
        const x = Number(row.querySelector('.point-x').value), y = Number(row.querySelector('.point-y').value);
        later(() => movePointTo(x, y));
      };
      field.onkeydown = event => { if (event.key === 'Enter') { event.preventDefault(); field.blur(); } };
    }

    $('node-pin').onchange=()=>runCommand('pin');
      return api;
  }
  const api = {nearKey, nearestPoint, pointContours, selectObject, sameClickSpot, clickTargets, selectAtPoint, clickEmpty, pointPairs, focusedPoint, nodeHandles, handleCount, selectedHandle, straightenHandles, renderNodeInspector, holesKey, holeContours, loadNodeHoles, fillPointHoles, pointHolesToShapes, movePointTo, selectAllTargets, selectAllRefusal, selectAll, nudge, nudgeBy, pointNudge, wire};
  return api;
}
