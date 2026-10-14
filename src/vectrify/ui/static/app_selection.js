import {DOUBLE_CLICK, POINT_REACH} from './app_config.js';
import {pickTarget, splitKey} from './selection.js';

// Selection owns its controls; editor contains the live state shared with other controllers.
export function createSelectionController(editor, {
  $,
  action,
  contourAt,
  document = globalThis.document,
  focusSelection,
  geometryNodes,
  later,
  level,
  nodeAt,
  object,
  overlay,
  parents,
  pointPaths,
  pointTargetsNow,
  renderCommands,
  request,
  revealObject,
  runCommand,
  scheduleStrip,
  selectPoints,
  selectedPoints,
  sharingPaths,
  svgElement,
  toast,
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
    $('point-section').hidden = !onPoints || (!paths.length && !instances);
    $('point-title').textContent = !chosen.length ? 'Points' : chosen.length > 1 ? `${chosen.length} points in ${count} ${count === 1 ? 'path' : 'paths'}` : node.command === 'M' ? 'Start point' : node.command === 'C' ? 'Curve endpoint' : 'Line endpoint';
    $('node-count').textContent = nodes.length ? nodes.length.toLocaleString() : '';
    const hints = [selectedHandle() ? 'Handle selected: Delete removes this handle. Toggle Straighten to keep both handles in line while dragging.' : !loaded ? 'Loading path points…' : !chosen.length ? (editor.tool === 'nodes' ? 'Click a point, or drag a box around several; Shift adds. Hold Shift to move horizontally or vertically. Dragged points snap to others; hold Alt or Ctrl/⌘ to drag without snapping.' : 'Points selected in Nodes (S) stay selected here.') : chosen.length > 1 ? 'Drag one to move them together.' : node.pinned ? 'Pinned: unpin it to move or delete it.' : ''];
    // Points of shared geometry are points of every path drawing it.
    const sharing = new Set(points.flatMap(key => sharingPaths(splitKey(key)[0])));
    if (sharing.size) hints.push(`Shared geometry: ${chosen.length === 1 ? 'this point is' : 'these points are'} also in ${plural(sharing.size, 'other path')}, marked faintly, and edits change ${sharing.size === 1 ? 'both' : 'them all'}. Detach (under Path) to edit one path alone.`);
    if (instances) hints.push(`${plural(instances, 'instance')} (use) ${instances === 1 ? 'has' : 'have'} no points of ${instances === 1 ? 'its' : 'their'} own here: Detach ${instances === 1 ? 'it' : 'them'} to edit ${instances === 1 ? 'its' : 'their'} points.`);
    $('node-hint').textContent = hints.filter(Boolean).join(' ');
    $('node-hint').hidden = !$('node-hint').textContent;
    $('point-properties').hidden = !chosen.length;
    // Coordinates are shown for one point, in its path's own frame.
    for (const row of document.querySelectorAll('.point-coordinates')) {
      row.hidden = !node;
      if (!node) continue;
      for (const [selector, value] of [['.point-x', node.values.at(-2)], ['.point-y', node.values.at(-1)]]) {
        const field = row.querySelector(selector);
        if (document.activeElement !== field) field.value = +value.toFixed(4);
        field.disabled = node.pinned;
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

  const plural = (count, one, many = `${one}s`) => `${count} ${count === 1 ? one : many}`;

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

  function wire() {
    // The point's coordinates, in the right panel and the Nodes strip.
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
  const api = {nearKey, nearestPoint, pointContours, selectObject, sameClickSpot, clickTargets, selectAtPoint, clickEmpty, pointPairs, focusedPoint, nodeHandles, handleCount, selectedHandle, straightenHandles, renderNodeInspector, holesKey, holeContours, loadNodeHoles, plural, fillPointHoles, pointHolesToShapes, movePointTo, wire};
  return api;
}
