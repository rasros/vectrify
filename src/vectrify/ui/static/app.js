import {pathEndpoints, snapIndex, snapPoint} from './snap.js';
import {dropIndex, dropRefusal, dropTarget, replayDrop} from './tree.js';
import {attach, contourLines, stretch} from './redraw.js';
import {matchCommands, moveHighlight} from './palette.js';
import {TOOL_LEVEL, boxSelect, clickPoint, clickPointPath, dragBox, escapeStep, instancePoints, pickTarget, pointInside, pointKey, pointOwners, pointTargets, pointerTarget, rectInside, scopeChain, selectionStatus, splitKey, switchTool} from './selection.js';
import {HeldGesture, inputQueue} from './input.js';
import {canvasGestures} from './gestures.js';
import {overflowLayout} from './strip.js';
import {CURSORS, frameHandle, nearestEdge, resizeScale} from './resize.js';
import {breakable, freeEnd, segmentAmong} from './lines.js';
const $ = id => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
let state, session, tool = 'select', zoom = 1, pan = {x: 0, y: 0}, drag = null;
let reference = null;
// Point tools show the points of every selected path: their geometry at this
// revision, the unselected path under the pointer, the point last clicked,
// and the points hidden while an object tool is active.
let geometries = new Map(), hoverPath = null, focusPoint = null, pointMemory = null;
// Which path each selected point was picked in: paths drawing one geometry
// share its node ids, and only the path a point was picked in shows it
// selected.
let owners = pointOwners([], () => null);
// The group entered by double-clicking it: object tools then pick within it.
let scope = null;
let clickCycle = null, lastPick = null;
let pathDraft = [], pathHover = null;
let joinContext = null;
let redrawHover = null;
const SNAP_RADIUS = 8;
// How near, in screen pixels, the selection's frame is grabbed to resize it.
const FRAME_REACH = 6;
// The selection's bounding box in the overlay's frame, while Select can
// resize it.
let selectionBox = null;
// Two clicks at one spot within this many milliseconds are a double-click.
const DOUBLE_CLICK = 400;
let pending = 0, queue = Promise.resolve(), dirty = false, space = false, toastTimer;
const drawing = $('drawing'), overlay = $('overlay'), stage = $('stage');
const names = {select: 'Select', nodes: 'Nodes', path: 'Draw path', knife: 'Knife', redraw: 'Redraw outline', hand: 'Pan'};
// A one-line hint for the tools with few controls of their own.
const hints = {select: '', nodes: '', path: 'Click for corners · Drag for curves · Click the first point to close · Enter finishes', knife: 'Drag a line across shapes to cut them, only the selected ones if any are · Shift snaps to 15°', redraw: 'Draw along the reference edge from a path\'s outline back to it, a selected path if any are · Shift replaces the longer way round · Escape cancels', hand: 'Drag to pan · Scroll to zoom'};

function toast(message, error = false) {
  clearTimeout(toastTimer); $('toast-message').textContent = message;
  $('toast').classList.toggle('error', error); $('toast').hidden = false;
  if (!error) toastTimer = setTimeout(() => $('toast').hidden = true, 4500);
}
$('toast-close').onclick = () => $('toast').hidden = true;
function setBusy(label, delta) {
  pending += delta; $('busy').hidden = pending === 0;
  if (label) $('busy-label').textContent = label;
  document.body.setAttribute('aria-busy', String(pending > 0 || input.waiting > 0));
}
// Input while an edit runs, or while a canvas drag is under way, waits for it:
// keys, clicks and commands run in order afterwards, on the state it leaves.
// Only event handlers queue input; what they run calls the editor directly.
const idle = () => new Promise(resolve => {
  const check = () => { if (!pending && !drag) resolve(); else setTimeout(check, 25); };
  check();
});
const input = inputQueue(() => pending > 0 || !!drag, idle);
function later(fn) {
  const result = input.run(fn);
  document.body.setAttribute('aria-busy', String(pending > 0 || input.waiting > 0));
  Promise.resolve(result).catch(() => {}).finally(() => document.body.setAttribute('aria-busy', String(pending > 0 || input.waiting > 0)));
  return result;
}
// In the desktop window (vectrify[desktop]) the page calls Python through
// pywebview's bridge instead of the local server; the bridge appears a moment
// after the page loads.
const desktop = new URLSearchParams(location.search).has('desktop');
const bridge = desktop ? new Promise(resolve => {
  if (window.pywebview?.api) resolve(window.pywebview.api);
  else window.addEventListener('pywebviewready', () => resolve(window.pywebview.api), {once: true});
}) : null;
async function request(path, data = {}) {
  if (bridge) {
    const {status, body} = await (await bridge).request(path, data, session || '');
    if (status >= 400) throw new Error(body.error || `Request failed (${status})`);
    return body;
  }
  const response = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Vectrify-Session': session || ''}, body: JSON.stringify(data)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`);
  return result;
}
function action(command, data = {}, label = 'Applying edit…') {
  // Busy from now, so input meanwhile waits for this edit.
  setBusy('', 1);
  queue = queue.then(async () => {
    if (command !== 'select') clickCycle = null;
    $('busy-label').textContent = label;
    try {
      const ids = command === 'undo' ? state.undo_ids?.slice(-1) : command === 'redo' ? state.redo_ids?.slice(0, 1) : undefined;
      const result = await request('/api/action', {command, ...data, ...(ids ? {ids} : {}), selection: state.selection, epoch: state.epoch, revision: state.revision});
      if (command !== 'select') dirty = true;
      await applyState(result);
      return true;
    } catch (error) {
      toast(error.message, true);
      // A genuine overlap can be refused. Catch up automatically so the
      // person can keep editing the live drawing without a manual refresh.
      try { await applyState(await request('/api/session', {session})); } catch { /* Keep the last state if disconnected. */ }
      // Discard optimistic dragging even when the backend rejects the command.
      if (state?.svg) renderDrawing();
      renderInspector(); drawOverlay();
      return false;
    } finally { setBusy('', -1); }
  });
  return queue;
}
let svgElements = new Map(), objectsById = new Map();
function svgElement(id) { return svgElements.get(id); }
function object(id) { return objectsById.get(id); }
function oneObject() { return state?.selection.objects.length === 1 ? object(state.selection.objects[0]) : null; }
function xmlElement(name, attrs = {}) {
  const element = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
  return element;
}
function renderDrawing() {
  const parsed = new DOMParser().parseFromString(state.svg, 'image/svg+xml');
  const root = document.importNode(parsed.documentElement, true);
  svgElements = new Map();
  // Isolate drawing IDs from editor controls while keeping local SVG references.
  for (const element of [root, ...root.querySelectorAll('*')]) {
    if (element.id) { svgElements.set(element.id, element); element.dataset.objectId = element.id; element.id = `art-${element.id}`; }
    for (const attribute of [...element.attributes]) {
      if (attribute.localName === 'href' && attribute.value.startsWith('#')) {
        element.setAttributeNS(attribute.namespaceURI, attribute.name, `#art-${attribute.value.slice(1)}`);
      } else if (['clip-path', 'fill', 'stroke'].includes(attribute.localName) && attribute.value.startsWith('url(#')) {
        element.setAttribute(attribute.localName, `url(#art-${attribute.value.slice(5,-1)})`);
      }
    }
  }
  root.setAttribute('width', state.bounds[2]); root.setAttribute('height', state.bounds[3]);
  root.setAttribute('viewBox', state.bounds.join(' ')); root.setAttribute('preserveAspectRatio', 'none');
  root.setAttribute('aria-label', state.name);
  drawing.replaceChildren(root);
}
async function applyState(next) {
  const changed = !state || next.epoch !== state.epoch || next.revision !== state.revision;
  const replaced = !state || next.epoch !== state.epoch;
  state = {...state, ...next};
  if (changed) objectsById = new Map(state.objects.map(item => [item.id, item]));
  $('filename').textContent = state.name; $('dirty').textContent = dirty ? '•' : '';
  if (next.svg) renderDrawing();
  const [x, y, w, h] = state.bounds;
  $('artboard').style.width = `${w}px`; $('artboard').style.height = `${h}px`;
  overlay.setAttribute('viewBox', `${x} ${y} ${w} ${h}`);
  overlay.setAttribute('preserveAspectRatio', 'none');
  $('dimensions').textContent = `${Math.round(w)} × ${Math.round(h)}`;
  $('undo').disabled = !state.undo.length; $('redo').disabled = !state.redo.length;
  $('undo').title = state.undo.length ? `Undo: ${state.undo.at(-1)}` : 'Nothing to undo';
  $('redo').title = state.redo.length ? `Redo: ${state.redo[0]}` : 'Nothing to redo';
  if (replaced) { focusPoint = null; pointMemory = null; scope = null; fit(); }
  if (changed) { geometries = new Map(); pathHoles.clear(); clickCycle = null; lastPick = null; }
  if (scope && object(scope)?.tag !== 'g') scope = null;
  // Selection does not change the drawing or the tree's labels and swatches.
  // Keep its rows: rebuilding them reads every path's paint and forces style
  // work on large drawings for every click.
  if (changed || next.svg) renderObjects();
  else updateObjectSelection();
  if (level() === 'points') {
    if (missingGeometries().length) renderInspector();
    await loadGeometries();
  }
  renderInspector(); drawOverlay();
}
const level = () => TOOL_LEVEL[tool];
const parents = () => new Map(state.objects.map(item => [item.id, item.parent]));
// The paths whose points the point tools show and edit: the selected ones and
// every path inside a selected group, and the instances among them, which
// have no points of their own. Worked out once per state.
let targetCache = {state: null};
function pointTargetsNow() {
  if (targetCache.state !== state) targetCache = {state, ...pointTargets(state.selection.objects, state.objects)};
  return targetCache;
}
const pointPaths = () => pointTargetsNow().paths;
const geometryIndexes = new WeakMap();
function geometryIndex(id) {
  const geometry = geometries.get(id);
  if (!geometry) return null;
  let index = geometryIndexes.get(geometry);
  if (!index) {
    const nodes = [], byId = new Map(), contours = new Map();
    for (const contour of geometry.subpaths) for (const node of contour.nodes) {
      nodes.push(node); byId.set(node.id, node); contours.set(node.id, contour);
    }
    index = {nodes, byId, contours};
    geometryIndexes.set(geometry, index);
  }
  return index;
}
function geometryNodes(id) { return geometryIndex(id)?.nodes || []; }
function nodeAt(key) { const [id, node] = splitKey(key); return geometryIndex(id)?.byId.get(node); }
function contourAt(key) { const [id, node] = splitKey(key); return geometryIndex(id)?.contours.get(node); }
// The selected points, from the node ids the server holds, while a point tool
// is active: in an object tool they are hidden. A node of geometry several
// shown paths draw is selected in the path it was picked in; the others show
// it faintly, as a twin, since editing it moves them too.
let shownCache = {};
function shownPoints() {
  if (level() !== 'points') return {strong: [], twins: []};
  const nodes = state.selection.nodes, cache = shownCache;
  if (cache.state !== state || cache.geometries !== geometries || cache.size !== geometries.size || cache.owners !== owners || cache.tool !== tool) {
    const paths = pointPaths().filter(id => geometries.has(id)).map(id => ({id, geometry: geometries.get(id).id, nodes: geometryNodes(id).map(n => n.id)}));
    shownCache = {state, geometries, size: geometries.size, owners, tool, ...(nodes.length ? instancePoints(paths, new Set(nodes), owners) : {strong: [], twins: []})};
  }
  return shownCache;
}
const selectedPoints = () => shownPoints().strong;
// The other paths drawing the geometry of a shown path.
function sharingPaths(id) { return (geometries.get(id)?.users || []).filter(user => user !== id); }
function selectPoints(objects, keys, focus = keys.at(-1)) {
  focusPoint = focus ?? null;
  owners = pointOwners(keys, id => geometries.get(id)?.id);
  return action('select', {objects: [...new Set(objects)], nodes: [...new Set(keys.map(key => splitKey(key)[1]))]}, 'Selecting…');
}
function missingGeometries() {
  const wanted = level() === 'points' ? [...pointPaths(), ...(hoverPath ? [hoverPath] : [])] : [];
  return [...new Set(wanted)].filter(id => !geometries.has(id));
}
async function loadGeometries() {
  const missing = missingGeometries();
  if (!missing.length) return;
  const {epoch, revision} = state;
  const result = await request('/api/nodes', {objects: missing, epoch, revision});
  if (state.epoch !== epoch || state.revision !== revision) return;
  if (result.epoch !== epoch || result.revision !== revision) { schedulePoll(0); return; }
  for (const [id, geometry] of Object.entries(result.geometries)) geometries.set(id, geometry);
}
function paintReference(element) {
  const href = element?.getAttribute('href') || element?.getAttributeNS('http://www.w3.org/1999/xlink', 'href');
  return element?.localName === 'use' && href?.startsWith('#') ? drawing.querySelector(`[id="${CSS.escape(href.slice(1))}"]`) : null;
}
function resolvedPaint(id, attr, fallback = '') {
  const element = svgElement(id);
  if (!element) return fallback;
  const target = paintReference(element);
  return getComputedStyle(target?.hasAttribute(attr) ? target : element).getPropertyValue(attr).trim() || fallback;
}
// Any CSS colour as the browser computes it, rgb(), which colorHex reads.
function cssColour(value) {
  const probe = document.createElement('span'); probe.style.color = value; document.body.append(probe);
  const colour = getComputedStyle(probe).color; probe.remove(); return colour;
}
function colorHex(value) {
  const rgb = value.match(/^rgba?\(([^)]+)\)$/i);
  if (!rgb) return /^#[a-f\d]{6}$/i.test(value) ? value : null;
  const channels = rgb[1].split(/[, /]+/).map(Number);
  if (channels.length < 3 || channels.some(n => !Number.isFinite(n))) return null;
  return '#' + channels.slice(0,3).map(n=>Math.round(n).toString(16).padStart(2,'0')).join('') +
    (channels.length > 3 && channels[3] < 1 ? Math.round(channels[3]*255).toString(16).padStart(2,'0') : '');
}
// A fill or stroke of url(#id) names a linear gradient in the drawing's defs.
function paintGradient(value) {
  const id = value?.match(/^url\(\s*["']?#([^"')\s]+)["']?\s*\)$/)?.[1];
  // Paint in the drawing names its ID with the art- prefix; a document ID has none.
  const element = id && (drawing.querySelector(`[id="${CSS.escape(id)}"]`) || svgElement(id));
  return element?.localName === 'linearGradient' ? element : null;
}
function gradientStops(gradient) {
  return [...gradient.querySelectorAll('stop')].map(stop => {
    const offset = stop.getAttribute('offset') || '0';
    const value = offset.endsWith('%') ? parseFloat(offset) / 100 : Number(offset);
    return {offset: Math.max(0, Math.min(1, value || 0)), colour: stop.getAttribute('stop-color') || 'black'};
  });
}
// The gradient as a left-to-right CSS gradient, for swatches.
function gradientCss(value) {
  const gradient = paintGradient(value); if (!gradient) return null;
  const stops = gradientStops(gradient);
  return stops.length ? `linear-gradient(90deg, ${stops.map(s => `${s.colour} ${(s.offset*100).toFixed(1)}%`).join(', ')})` : null;
}
// One colour where a single one is shown: a gradient's first stop.
function paintColour(value) {
  const gradient = paintGradient(value);
  return gradient ? gradientStops(gradient)[0]?.colour || 'none' : value;
}
function paintSource(id, kind) {
  let element = svgElement(id);
  const target = paintReference(element);
  if (target?.hasAttribute(kind)) return `From referenced ${object(target.dataset.objectId)?.label || 'shape'}`;
  while (element && drawing.contains(element)) {
    if (element.hasAttribute(kind)) {
      const source = element.dataset.objectId;
      return source === id ? 'Set on this object' : `Inherited from ${object(source)?.label || 'SVG root'}`;
    }
    element = element.parentElement;
  }
  return 'SVG default';
}
function swatchPaint(element) {
  const style = getComputedStyle(element);
  // A referenced path can supply its own paint over the use's inherited paint.
  const target = paintReference(element);
  const fill = target?.getAttribute('fill') || style.fill;
  const stroke = target?.getAttribute('stroke') || style.stroke;
  return {fill, stroke};
}
function paintSwatch(swatch, item) {
  const element = svgElement(item.id); if (!element) return;
  const own = item.tag === 'linearGradient' && gradientCss(`url(#${item.id})`);
  if (own) {
    swatch.style.background = own;
    swatch.title = 'Linear gradient: paints the shapes that use it';
    swatch.setAttribute('aria-hidden', 'true');
    return;
  }
  if (item.resource) {
    swatch.classList.add('resource-symbol');
    swatch.textContent = item.tag === 'defs' ? '◇' : element.closest('clipPath') ? '▧' : '⌁';
    swatch.title = 'Definition only — not painted on the canvas';
    swatch.setAttribute('aria-hidden', 'true');
    return;
  }
  const paint = swatchPaint(element);
  const display = color => colorHex(color) || color;
  if (item.tag === 'g') {
    const colors = [...new Set([...element.querySelectorAll('path,rect,circle,ellipse,line,polyline,polygon,use')]
      .filter(el=>!el.closest('defs,clipPath')).map(el=>{const p=swatchPaint(el);return paintColour(p.fill !== 'none' ? p.fill : p.stroke);}).filter(c=>c !== 'none'))];
    const shown = colors.slice(0,4);
    swatch.classList.add('group-swatch');
    swatch.style.background = shown.length === 1 ? shown[0] : shown.length ? `conic-gradient(${shown.map((c,i)=>`${c} ${i*100/shown.length}% ${(i+1)*100/shown.length}%`).join(',')})` : 'transparent';
    swatch.title = colors.length ? `Group colors: ${shown.map(display).join(', ')}${colors.length > shown.length ? ` (+${colors.length-shown.length} more)` : ''}` : 'No paint';
  } else {
    swatch.classList.toggle('no-paint', paint.fill === 'none' && paint.stroke === 'none');
    const ramp = gradientCss(paint.fill);
    if (ramp) swatch.style.background = ramp;
    else swatch.style.backgroundColor = paint.fill === 'none' ? 'transparent' : paint.fill;
    if (paint.stroke !== 'none') { swatch.style.borderColor = paintColour(paint.stroke); swatch.style.borderWidth = '3px'; }
    const named = color => paintGradient(color) ? 'linear gradient' : display(color);
    swatch.title = `Fill: ${named(paint.fill)} (${paintSource(item.id,'fill')}) · Stroke: ${named(paint.stroke)} (${paintSource(item.id,'stroke')})`;
  }
  swatch.setAttribute('aria-hidden', 'true');
}
function objectContext(item) {
  const element = svgElement(item.id);
  const source = object(paintReference(element)?.dataset.objectId);
  const clipId = item.attributes['clip-path']?.match(/^url\(#(.+)\)$/)?.[1];
  const clip = object(clipId);
  const inClip = !!element?.closest('clipPath');
  let role = '';
  if (item.tag === 'defs') role = 'Not drawn · reusable geometry';
  else if (item.tag === 'linearGradient') role = 'Linear gradient · paint, not drawn';
  else if (item.tag === 'stop') role = 'Gradient stop';
  else if (item.tag === 'clipPath') role = 'Clipping boundary · not drawn';
  else if (item.tag === 'use') role = `${inClip ? 'Clip contour' : item.resource ? 'Shared instance' : 'Instance'} of ${source?.label || 'missing source'}`;
  else if (inClip) role = 'Clip contour · not drawn';
  else if (item.resource) role = 'Shared geometry · not drawn';
  else if (clip) role = `Clipped by ${clip.label}`;
  return {source, clip, inClip, role};
}
let objectRows = new Map(), treeSelection = new Set();
function updateObjectSelection() {
  const selected = new Set(state.selection.objects);
  for (const id of new Set([...treeSelection, ...selected])) {
    if (treeSelection.has(id) === selected.has(id)) continue;
    const row = objectRows.get(id);
    if (!row) continue;
    row.classList.toggle('selected', selected.has(id));
    row.setAttribute('aria-selected', String(selected.has(id)));
  }
  treeSelection = selected;
}
function renderObjects() {
  const search = $('object-search').value.toLowerCase();
  const fragment = document.createDocumentFragment();
  objectRows = new Map();
  treeSelection = new Set(state.selection.objects);
  const contexts = new Map(state.objects.map(item => [item.id, objectContext(item)]));
  const visible = new Set();
  for (const item of state.objects) {
    if (!`${item.label} ${item.tag} ${contexts.get(item.id).role}`.toLowerCase().includes(search)) continue;
    // Retain parents in filtered results so indentation never implies a false parent.
    for (let ancestor = item; ancestor; ancestor = object(ancestor.parent)) visible.add(ancestor.id);
  }
  let count = 0;
  for (const item of state.objects) {
    if (!visible.has(item.id)) continue;
    const context = contexts.get(item.id);
    count++;
    const row = document.createElement('button'); row.className = 'object-row';
    row.classList.toggle('selected', treeSelection.has(item.id));
    row.classList.toggle('resource', item.resource); row.setAttribute('role', 'option');
    row.setAttribute('aria-selected', String(treeSelection.has(item.id)));
    row.dataset.object = item.id; row.title = `${item.tag} · ${item.id}`;
    row.style.paddingLeft = `${9 + item.depth * 9}px`;
    const swatch = document.createElement('span'); swatch.className = 'swatch';
    paintSwatch(swatch, item);
    row.title += ` · ${swatch.title}`;
    const label = document.createElement('span'); label.className = 'object-label';
    const name = document.createElement('span'); name.textContent = (item.tag === 'g' ? '▱ ' : '') + item.label;
    label.append(name);
    if (context.role) {
      row.classList.add('has-description');
      const detail = document.createElement('small'); detail.textContent = context.role;
      label.append(detail); row.title += ` · ${context.role}`;
    }
    row.append(swatch, label);
    if (item.inherited_locks.length) { const mark = document.createElement('span'); mark.className = 'lock-mark'; mark.textContent = '◆'; mark.title = `Locked: ${item.inherited_locks.map(lock => lock === 'transform' ? 'position' : lock).join(', ')}`; row.append(mark); }
    row.onclick = event => {
      const additive = event.shiftKey || event.ctrlKey || event.metaKey;
      if (!treeDragEnded) later(() => object(item.id) && selectObject(item.id, additive, true));
    };
    row.ondblclick = () => { if (!item.resource) later(() => object(item.id) && enterObject(item.id)); };
    row.onpointerdown = event => pressTreeRow(event, item);
    objectRows.set(item.id, row);
    fragment.append(row);
  }
  $('objects').replaceChildren(fragment, treeDropLine); $('object-count').textContent = count;
  // A drag held across an edit drops where the pointer is on the new rows.
  if (treeDrag?.active) {
    treeDrag.ids = new Set([...treeDrag.ids].filter(id => object(id)));
    treeDrag.revision = `${state.epoch}:${state.revision}`;
    if (treeDrag.point) Object.assign(treeDrag, treeDropAt(treeDrag.point, treeDrag.ids));
    showTreeDrop();
  }
}
// Dragging rows in the tree restacks them, or moves them into a group. The
// drag carries the whole selection when it starts on a selected row.
const TREE_INDENT = 9;
const treeDropLine = document.createElement('div'); treeDropLine.className = 'tree-drop-line'; treeDropLine.hidden = true;
let treeDrag = null, treeDragEnded = false;
function pressTreeRow(event, item) {
  if (event.button !== 0 || item.resource) return;
  treeDrag = {item: item.id, x: event.clientX, y: event.clientY, active: false, target: null};
}
function treeRows() {
  return [...$('objects').querySelectorAll('.object-row')].map(row => {
    const item = object(row.dataset.object), box = row.getBoundingClientRect();
    return {id: item.id, parent: item.parent, depth: item.depth, container: item.tag === 'g' && !item.resource, top: box.top, height: box.height, left: box.left};
  });
}
function showTreeDrop() {
  const {target, refusal} = treeDrag;
  for (const row of $('objects').querySelectorAll('.object-row')) {
    row.classList.toggle('drop-into', !refusal && !!target?.into && row.dataset.object === target.parent);
    row.classList.toggle('dragged', treeDrag.ids.has(row.dataset.object));
  }
  treeDropLine.hidden = !!refusal || !target?.line;
  $('objects').classList.toggle('drop-refused', !!refusal);
  $('objects').title = refusal;
  if (treeDropLine.hidden) return;
  const list = $('objects'), box = list.getBoundingClientRect();
  treeDropLine.style.top = `${target.line.y - box.top + list.scrollTop - 1}px`;
  treeDropLine.style.left = `${8 + TREE_INDENT * (1 + target.line.depth)}px`;
}
function moveTreeDrag(event) {
  if (!treeDrag) return;
  if (!treeDrag.active) {
    if (Math.hypot(event.clientX - treeDrag.x, event.clientY - treeDrag.y) < 5) return;
    const selected = state.selection.objects;
    const ids = selected.includes(treeDrag.item) ? selected.filter(id => !object(id)?.resource) : [treeDrag.item];
    treeDrag = {...treeDrag, active: true, ids: new Set(ids), revision: `${state.epoch}:${state.revision}`};
    $('objects').classList.add('dragging');
  }
  const list = $('objects'), box = list.getBoundingClientRect();
  // Scroll while the pointer is near the list's top or bottom edge.
  if (event.clientY < box.top + 24) list.scrollTop -= 12;
  else if (event.clientY > box.bottom - 24) list.scrollTop += 12;
  treeDrag.point = {x: event.clientX, y: event.clientY};
  Object.assign(treeDrag, treeDropAt(treeDrag.point, treeDrag.ids));
  showTreeDrop();
}
// Where rows dropped at *point* land, and why they cannot, if so.
function treeDropAt(point, ids) {
  const rows = treeRows(), box = $('objects').getBoundingClientRect();
  const target = dropTarget(rows, point.x - (rows[0]?.left ?? box.left) - TREE_INDENT, point.y, TREE_INDENT);
  const parents = new Map(state.objects.map(item => [item.id, item.parent]));
  const resources = new Set(state.objects.filter(item => item.resource).map(item => item.id));
  return {target, refusal: dropRefusal(target, ids, parents, resources)};
}
function endTreeDrag(drop) {
  const finished = treeDrag;
  treeDrag = null; treeDropLine.hidden = true;
  $('objects').classList.remove('dragging', 'drop-refused'); $('objects').title = '';
  for (const row of $('objects').querySelectorAll('.drop-into, .dragged')) row.classList.remove('drop-into', 'dragged');
  if (!finished?.active) return;
  // The pointer is released over a row: that is not a click on it.
  treeDragEnded = true; setTimeout(() => treeDragEnded = false);
  if (!drop) return;
  // Dropped while an edit runs, it lands once the edit is done, next to the
  // rows it was dropped by, if they are still there.
  later(() => {
    const {ids, target, refusal} = finished.refusal || finished.revision === `${state.epoch}:${state.revision}` ? finished : replayDrop(finished.target, finished.ids, state.objects);
    if (refusal) { toast(refusal, true); return; }
    const children = state.objects.filter(item => item.parent === target.parent).map(item => item.id);
    return action('move_objects', {objects: [...ids], parent: target.parent, index: dropIndex(target, children, ids)}, 'Moving objects…');
  });
}
window.addEventListener('pointermove', moveTreeDrag);
window.addEventListener('pointerup', () => endTreeDrag(true));
window.addEventListener('pointercancel', () => endTreeDrag(false));
function paintValue(attr, fallback = '') {
  const values = state.selection.objects.map(id => resolvedPaint(id, attr, fallback));
  return values.length && values.every(v => v === values[0]) ? values[0] : '';
}
function renderRelationships(item) {
  const section = $('object-relationships'), links = $('relationship-links');
  links.replaceChildren(); section.hidden = true;
  const context = item && objectContext(item);
  const clipOnly = state.selection.objects.some(id => {
    const selected = object(id);
    return selected?.tag === 'defs' || objectContext(selected).inClip;
  });
  $('paint-section').hidden = clipOnly;
  $('paint-heading').textContent = item?.resource ? 'Shared paint' : 'Paint';
  if (!item) return;
  $('selection-kind').textContent = item.tag === 'defs' ? 'Definitions' : context.inClip ? 'Clipping' : item.resource ? 'Shared geometry' : item.tag === 'use' ? 'Instance' : item.tag;
  const consumers = state.objects.filter(candidate => {
    const relation = objectContext(candidate);
    return relation.source?.id === item.id || relation.clip?.id === item.id;
  });
  if (!context.role && !consumers.length) return;
  section.hidden = false;
  let explanation = context.role;
  if (item.tag === 'defs') explanation = 'Reusable definitions. These entries do not draw anything by themselves.';
  else if (context.inClip) explanation = 'This defines a clipping boundary, not a painted shape. Shapes using this boundary are only visible inside it.';
  else if (item.resource) explanation = 'This geometry is stored for reuse, not drawn directly. Its instances supply the visible paint unless shared paint is set here.';
  else if (context.source) explanation = 'This draws an instance of the shared geometry below, using this instance’s paint and position.';
  else if (context.clip) explanation = 'This artwork is clipped: only the parts inside its clipping boundary are visible.';
  $('relationship-description').textContent = explanation;
  function link(label, target) {
    const button = document.createElement('button');
    button.textContent = `${label}: ${target.label}`;
    button.onclick = () => later(() => object(target.id) && selectObject(target.id, false, true));
    links.append(button);
  }
  if (context.source) link('Source geometry', context.source);
  if (context.clip) link('Clipping boundary', context.clip);
  if (context.inClip && item.tag !== 'clipPath') {
    const parent = object(svgElement(item.id).closest('clipPath').dataset.objectId);
    if (parent) link('Clipping boundary', parent);
  }
  for (const consumer of consumers) link(objectContext(consumer).inClip ? 'Used for clipping by' : consumer.tag === 'use' ? 'Drawn by' : 'Clips', consumer);
}
// Disable a control with the reason as its tooltip, or enable it with its own.
function enable(element, reason) {
  const button = typeof element === 'string' ? $(element) : element;
  button.dataset.title ??= button.title;
  button.disabled = !!reason;
  button.title = reason || button.dataset.title;
}
// Every command the selection offers, the tool strip, the Actions list, the
// context menu and the command palette run. *disabled* gives the reason it
// cannot run now, or ''; *level* limits it to object or point tools; a *rare*
// command's buttons hide while it cannot run.
const noSelection = () => !state?.selection.objects.length && 'Select objects first';
const noPoints = () => !selectedPoints().length && (level() === 'points' ? 'Select points first' : 'Select points in Nodes (A) first');
const visiblePaths = () => state.selection.objects.every(id => object(id)?.tag === 'path' && !object(id)?.resource);
const noReference = () => !state.reference && 'Load a reference image first, under Reference below the objects';
// Lines are paths that paint no fill; fills are paths that do.
const filledPath = id => resolvedPaint(id, 'fill', 'black') !== 'none';
const strokedPath = id => resolvedPaint(id, 'stroke', 'none') !== 'none';
const linePaths = () => joinCandidates().filter(item => !filledPath(item.id) && strokedPath(item.id));
const fillPaths = () => joinCandidates().filter(item => filledPath(item.id));
// How far apart, in screen pixels, two line ends may be for Join ends: zoom
// out to join wider gaps.
const JOIN_REACH = 12;
// The deepest zoom, 25600%, enough to place points within a hundredth of a
// pixel.
const MAX_ZOOM = 256;
// How near, in screen pixels, a press in Nodes must be to a point or handle
// to take it: the nearest one within reach wins, so a point need not be hit
// exactly. *nearPoint* is the one under the pointer, drawn larger.
const POINT_REACH = 10;
// The least length, in screen pixels, a handle is drawn at.
const HANDLE_SPREAD = 16;
let nearPoint = null, nearAnchor = null;
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
const twoEnds = () => selectedPoints().length === 2;
// Agents (an MCP client) may edit this drawing once the footer allows them.
// The editor then hosts the MCP server itself; the footer's popover shows its
// URL and the command that adds it to Claude Code. Agents' edits land in this
// session, so while allowed the page is told of each agent call as it happens
// (server-sent events from /api/events; in the desktop window, a script the
// window runs), redraws, and says in the footer what the agent did. A slow
// poll covers a dropped channel.
let agentStatus = {enabled: false, connected: false, last_action: null, changes: 0, touched: []}, agentSeen = 0, agentPoll = null, agentEvents = null;
const AGENT_POLL_MS = 5000, AGENT_RETRY_MS = 120;
// The objects an agent's change touched flash briefly once the page shows it:
// the person's selection stays theirs, never the agent's.
const FLASH_MS = 1500;
let flash = {ids: [], start: 0, timer: null, seen: null};
function flashTouched(touched) {
  const ids = [...new Set(touched.filter(t => t.change > flash.seen).flatMap(t => t.ids))];
  flash.seen = Math.max(flash.seen, ...touched.map(t => t.change));
  if (!ids.length) return;
  clearTimeout(flash.timer);
  flash = {...flash, ids, start: performance.now(), timer: setTimeout(() => { flash.ids = []; drawOverlay(); }, FLASH_MS)};
  drawOverlay();
}
function drawFlash() {
  if (!flash.ids.length) return;
  // Redrawing the overlay keeps the fade where it was.
  const delay = `${-Math.min(FLASH_MS, performance.now() - flash.start)}ms`;
  for (const id of flash.ids) {
    const element = svgElement(id), matrix = localToOverlay(element);
    if (!element?.getBBox || !matrix) continue;
    try {
      const b = element.getBBox();
      if (!b.width && !b.height) continue;
      const style = getComputedStyle(element);
      const pad = style.stroke === 'none' ? 0 : (parseFloat(style.strokeWidth) || 0) / 2;
      const pts = [[b.x-pad,b.y-pad], [b.x+b.width+pad,b.y-pad], [b.x+b.width+pad,b.y+b.height+pad], [b.x-pad,b.y+b.height+pad]].map(([x,y]) => new DOMPoint(x,y).matrixTransform(matrix));
      const shape = xmlElement('polygon', {points: pts.map(p => `${p.x},${p.y}`).join(' '), class: 'agent-flash', 'data-object': id, 'aria-hidden': 'true'});
      shape.style.animationDelay = delay;
      overlayFrame.content.append(shape);
    } catch { /* Resource elements have no display bounds. */ }
  }
}
function showAgent() {
  const s = agentStatus, button = $('agent-toggle');
  button.setAttribute('aria-pressed', String(s.enabled));
  button.classList.toggle('connected', s.enabled && s.connected);
  const text = !s.enabled ? 'Agents off' : s.connected ? `Agent connected${s.last_action ? ' · ' + s.last_action.replace(/^Agent: /, '') : ''}` : 'Agents allowed';
  $('agent-status').textContent = text;
  button.title = s.enabled ? 'Agents may edit this drawing: the MCP server and how to add it' : 'Allow agents to edit this drawing';
  $('agent-enabled').checked = s.enabled;
  $('agent-popover-status').textContent = !s.enabled ? 'Turn this on to let an MCP client such as Claude Code look at and edit this drawing. Each edit shows here. Undo and redo affect only your own changes.' : text + '.';
  $('agent-setup').hidden = !(s.enabled && s.apps);
  $('agent-mcp').hidden = $('agent-claude-code').hidden = !(s.enabled && s.mcp);
  const apps = s.apps || {};
  const fields = {'agent-url': s.mcp?.url, 'agent-command': s.mcp?.command, 'agent-codex': apps.codex, 'agent-codex-command': apps.codex_command, 'agent-claude-desktop': apps.claude_desktop};
  // Keep a selection the person is making.
  for (const [id, value] of Object.entries(fields)) if ($(id).value !== (value || '')) $(id).value = value || '';
  $('agent-codex-config').textContent = apps.codex_config || '~/.codex/config.toml';
  $('agent-claude-desktop-config').textContent = apps.claude_desktop_config || '';
  $('agent-mcp-error').hidden = !(s.enabled && s.mcp_error);
  $('agent-mcp-error').textContent = s.mcp_error || '';
  // The port clients were given was taken: they must be given the new URL.
  const moved = s.enabled ? s.mcp?.moved : null;
  $('agent-mcp-moved').hidden = !moved;
  $('agent-mcp-moved').textContent = moved ? `The MCP server's URL changed: ${moved.old} was taken, so it is at ${moved.new} now. Update clients added with the old URL (copy the command again).` : '';
  $('agent-toggle').classList.toggle('moved', !!moved);
  listenAgents();
}
// The push channel, open while agents are allowed.
function listenAgents() {
  if (desktop) return;
  if (agentStatus.enabled && !agentEvents && session) {
    agentEvents = new EventSource(`/api/events?session=${encodeURIComponent(session)}`);
    agentEvents.onmessage = event => { try { pulse(JSON.parse(event.data)); } catch { /* The poll catches up. */ } };
  } else if (!agentStatus.enabled && agentEvents) {
    agentEvents.close(); agentEvents = null;
  }
}
// The desktop window runs this with each pulse.
window.vectrifyPulse = result => { if (result?.session === session) pulse(result); };
// What the window shows, for an agent's view(): the visible part of the
// drawing in its own units [x, y, w, h], the zoom (screen pixels per unit),
// the canvas size, the tool, the entered group and the reference view.
function viewReport() {
  if (!state) return null;
  const w = stage.clientWidth, h = stage.clientHeight;
  if (!w || !h) return null;
  return {region: [state.bounds[0] - pan.x / zoom, state.bounds[1] - pan.y / zoom, w / zoom, h / zoom], zoom, pixels: [w, h], tool, entered: scope, reference_view: reference ? referenceView : null, reference_opacity: reference ? reference.opacity : null};
}
// A changed view is told to the session soon, not on every wheel step.
let viewTimer = null;
function reportView() {
  if (!agentStatus.enabled) return;
  clearTimeout(viewTimer);
  viewTimer = setTimeout(pollAgent, 250);
}
// One pulse at a time: one that comes while the page takes the last is taken
// after it.
let pulsing = false, nextPulse = null;
async function pulse(result) {
  if (!state) return;
  if (pulsing) { nextPulse = result; return; }
  agentStatus = result.agent; showAgent();
  // What agents did before the page opened is not news.
  if (flash.seen === null) flash.seen = agentStatus.changes;
  // An edit the page did not make: take the session's state as it is now,
  // once the person is not mid-gesture (until then, look again soon).
  const behind = agentStatus.changes !== agentSeen || result.epoch !== state.epoch || result.revision !== state.revision;
  if (!behind) return;
  if (drag || pending > 0) { schedulePoll(AGENT_RETRY_MS); return; }
  agentSeen = agentStatus.changes;
  const touched = agentStatus.touched || [];
  pulsing = true;
  queue = queue.then(async () => {
    const previous = JSON.stringify(state.reference);
    const next = await request('/api/session', {session});
    // The agent's edits are unsaved changes like the person's.
    if (next.epoch === state.epoch && next.revision !== state.revision) dirty = true;
    await applyState(next);
    if (JSON.stringify(state.reference) !== previous) await loadReference();
    flashTouched(touched);
  }).catch(error => toast(error.message, true));
  try { await queue; } finally {
    pulsing = false;
    const waiting = nextPulse; nextPulse = null;
    if (waiting) pulse(waiting);
  }
}
function schedulePoll(ms) {
  clearTimeout(agentPoll);
  if (agentStatus.enabled) agentPoll = setTimeout(pollAgent, ms);
}
async function pollAgent() {
  clearTimeout(agentPoll);
  // Looked again by now unless a pulse wants a sooner look.
  agentPoll = null;
  try { await pulse(await request('/api/poll', {view: viewReport()})); } catch { /* The next poll tries again. */ }
  if (agentPoll === null) schedulePoll(AGENT_POLL_MS);
}
async function setAgents(body) {
  try {
    agentStatus = await request('/api/agent', body);
    agentSeen = flash.seen = agentStatus.changes; showAgent();
    if (agentStatus.enabled) pollAgent();
  } catch (error) { toast(error.message, true); }
}
async function toggleAgents() {
  await setAgents({enabled: !agentStatus.enabled});
  if (agentStatus.enabled) openAgentPopover();
}
function openAgentPopover() {
  const popover = $('agent-popover'), rect = $('agent-toggle').getBoundingClientRect();
  popover.hidden = false;
  $('agent-toggle').setAttribute('aria-expanded', 'true');
  const width = popover.offsetWidth;
  popover.style.left = `${Math.max(16, Math.min(window.innerWidth - width - 16, rect.right - width))}px`;
  popover.style.bottom = `${window.innerHeight - rect.top + 8}px`;
  $(agentStatus.enabled && agentStatus.mcp ? 'agent-copy' : 'agent-enabled').focus();
}
function closeAgentPopover() {
  $('agent-popover').hidden = true;
  $('agent-toggle').setAttribute('aria-expanded', 'false');
}
$('agent-toggle').onclick = async () => {
  if (!$('agent-popover').hidden) { closeAgentPopover(); return; }
  if (!agentStatus.enabled) await setAgents({enabled: true});
  openAgentPopover();
};
$('agent-enabled').onchange = () => setAgents({enabled: $('agent-enabled').checked});
// Each app's setup is text to copy: the editor never edits an app's config.
for (const button of document.querySelectorAll('#agent-popover .agent-copy')) button.onclick = async () => {
  const field = $(button.dataset.copy);
  try { await navigator.clipboard.writeText(field.value); }
  catch { field.select(); document.execCommand('copy'); }
  toast(`Copied. Once ${button.dataset.app} has it and is restarted, it reaches this window whenever Agents is on.`);
};
$('agent-regenerate').onclick = () => setAgents({regenerate: true}).then(() => toast('New token. Clients added with the old one are refused: copy the command again.'));
for (const id of ['agent-url', 'agent-command', 'agent-codex', 'agent-codex-command', 'agent-claude-desktop']) $(id).onfocus = () => $(id).select();
$('agent-popover').addEventListener('keydown', event => { if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); closeAgentPopover(); $('agent-toggle').focus(); } });
document.addEventListener('pointerdown', event => { if (!$('agent-popover').hidden && !event.target.closest('#agent-popover, #agent-toggle')) closeAgentPopover(); });
const COMMANDS = [
  {id: 'tool-select', name: 'Select tool', group: 'Tools', keys: 'V', keywords: 'move arrow objects', run: () => setTool('select')},
  {id: 'tool-nodes', name: 'Nodes tool', group: 'Tools', keys: 'A', keywords: 'edit points handles', run: () => setTool('nodes')},
  {id: 'tool-path', name: 'Draw path tool', group: 'Tools', keys: 'D', keywords: 'pen draw path shape', run: () => setTool('path')},
  {id: 'tool-knife', name: 'Knife tool', group: 'Tools', keys: 'C', keywords: 'cut slice', run: () => setTool('knife')},
  {id: 'tool-redraw', name: 'Redraw outline tool', group: 'Tools', keys: 'R', keywords: 'lasso outline fix', run: () => setTool('redraw')},
  {id: 'tool-hand', name: 'Pan tool', group: 'Tools', keys: 'Hold Space', keywords: 'hand scroll', run: () => setTool('hand')},
  {id: 'open', name: 'Open…', group: 'File', keywords: 'svg project load', run: () => $('open-file').click()},
  {id: 'restore', name: 'Restore saved…', group: 'File', keywords: 'recovery browser copy', run: () => $('restore-saved').click()},
  {id: 'save', name: 'Save project', group: 'File', keys: 'Ctrl/⌘ S', keywords: 'download vectrify', run: () => download(true)},
  {id: 'export', name: 'Export SVG', group: 'File', keywords: 'download save', run: () => download(false)},
  {id: 'agents', name: 'Allow agents to edit', label: () => agentStatus.enabled ? 'Stop agents editing' : 'Allow agents to edit', group: 'File', keywords: 'mcp claude ai assistant agent live', run: toggleAgents},
  {id: 'help', name: 'Keyboard shortcuts', group: 'Help', keys: '?', keywords: 'help keys', run: () => $('help-dialog').showModal()},
  {id: 'undo', name: 'Undo', label: () => state?.undo.length ? `Undo ${state.undo.at(-1).toLowerCase()}` : 'Undo', group: 'Edit', keys: 'Ctrl/⌘ Z', run: () => action('undo', {}, 'Undoing…'), disabled: () => !state.undo.length && 'Nothing to undo'},
  {id: 'redo', name: 'Redo', label: () => state?.redo.length ? `Redo ${state.redo[0].toLowerCase()}` : 'Redo', group: 'Edit', keys: 'Ctrl/⌘ Shift Z', run: () => action('redo', {}, 'Redoing…'), disabled: () => !state.redo.length && 'Nothing to redo'},
  {id: 'fit', name: 'Fit the drawing', group: 'View', keys: 'F', keywords: 'zoom', run: fit},
  {id: 'zoom-selection', name: 'Zoom to selection', group: 'View', keys: 'Z', run: focusSelection, disabled: noSelection},
  {id: 'zoom-in', name: 'Zoom in', group: 'View', keys: 'Scroll', run: () => zoomAt(1.25)},
  {id: 'zoom-out', name: 'Zoom out', group: 'View', keys: 'Scroll', run: () => zoomAt(.8)},
  {id: 'finish-path', name: 'Finish path', group: 'Pen', keys: 'Enter', run: () => finishPath(false), disabled: () => (tool !== 'path' || pathDraft.length < 2) && 'Draw two or more points with the pen first'},
  {id: 'close-path', name: 'Close shape', group: 'Pen', run: () => finishPath(true), disabled: () => (tool !== 'path' || pathDraft.length < 3) && 'Draw three or more points with the pen first'},
  {id: 'fill', name: 'Edit fill', group: 'Properties', keywords: 'paint colour color', run: () => $('fill-value').focus(), disabled: noSelection},
  {id: 'stroke', name: 'Edit stroke', group: 'Properties', keywords: 'paint colour color outline', run: () => $('stroke-value').focus(), disabled: noSelection},
  {id: 'move-by', name: 'Move by…', group: 'Properties', keywords: 'offset position', run: () => $('move-x').select(), disabled: noSelection},
  {id: 'command-palette', name: 'Command palette', group: 'Help', keys: 'Ctrl/⌘ K', hidden: () => true, run: openPalette},
  {id: 'to-back', name: 'Send to back', group: 'Arrange', keys: 'Ctrl/⌘ Shift [', run: () => action('reorder', {to: 'back'}),
    disabled: () => noSelection() || (state.selection.objects.some(id => object(id)?.resource) && 'Definitions and clipping boundaries keep their place')},
  {id: 'backward', name: 'Send backward', group: 'Arrange', keys: 'Ctrl/⌘ [', run: () => action('reorder', {step: -1}), disabled: () => !oneObject() && 'Select one object to restack'},
  {id: 'forward', name: 'Bring forward', group: 'Arrange', keys: 'Ctrl/⌘ ]', run: () => action('reorder', {step: 1}), disabled: () => !oneObject() && 'Select one object to restack'},
  {id: 'to-front', name: 'Bring to front', group: 'Arrange', keys: 'Ctrl/⌘ Shift ]', run: () => action('reorder', {to: 'front'}),
    disabled: () => noSelection() || (state.selection.objects.some(id => object(id)?.resource) && 'Definitions and clipping boundaries keep their place')},
  {id: 'enter', name: 'Enter', label: () => oneObject()?.tag === 'g' ? 'Enter group' : 'Edit points', group: 'Select', keys: 'Double-click', keywords: 'group points nodes into',
    run: () => enterObject(oneObject().id), disabled: () => !(oneObject()?.tag === 'g' || (oneObject()?.tag === 'path' && !oneObject().resource)) && 'Select one group or visible path'},
  {id: 'step-up', name: 'Select one level up', group: 'Select', keys: 'Escape', keywords: 'leave exit group', run: stepUp, disabled: () => !state.selection.objects.length && !scope && 'Nothing to step up from'},
  {id: 'rename', name: 'Rename…', group: 'Object', keys: 'F2', run: renameObject, disabled: () => !oneObject() && 'Select one object to rename'},
  {id: 'group', name: 'Group', group: 'Actions', keys: 'Ctrl/⌘ G', keywords: 'combine', run: () => action('group'), disabled: () => state.selection.objects.length < 2 && 'Select two or more objects to group'},
  {id: 'ungroup', name: 'Ungroup', group: 'Actions', keys: 'Ctrl/⌘ Shift G', run: () => action('ungroup'),
    disabled: () => noSelection() || (!state.selection.objects.every(id => object(id)?.tag === 'g') && 'Select one or more groups')},
  {id: 'join', name: 'Join', label: () => joinOpensDialog() ? 'Join…' : 'Join', group: 'Actions', keys: 'Ctrl/⌘ J', keywords: 'merge union combine connect ends dashed broken lines strokes gaps close points', run: join,
    disabled: () => !twoEnds() && (level() === 'points' ? 'Select the two points to join, or paths in Select'
      : joinCandidates().length < 2 && !linePaths().length && 'Select two or more paths, or lines whose ends to join')},
  {id: 'convert-lines', name: 'Convert line/fill', label: () => ({fills: 'Fill to line', lines: 'Line to fill'})[!linePaths().length ? fillPaths().length && 'fills' : !fillPaths().length && 'lines'] || 'Convert line/fill', group: 'Actions',
    keywords: 'fill to line, line to fill, centreline centerline stroke outline expand skeleton thin', run: () => action('convert_lines', {}, 'Converting…'),
    disabled: () => !fillPaths().length && !linePaths().length && 'Select filled paths or stroked lines'},
  {id: 'split-parts', name: 'Split parts', group: 'Actions', run: splitParts,
    disabled: () => noSelection() || (oneObject()?.tag === 'use' ? 'Detach this instance to an editable path first' : !visiblePaths() && 'Only visible paths can be split')},
  {id: 'cut-hole', name: 'Cut out as hole', group: 'Actions', run: cutHole,
    disabled: () => (state.selection.objects.length !== 2 || !visiblePaths()) && 'Select two visible paths, one inside or overlapping the other'},
  {id: 'snap-edges', name: 'Snap edges…', group: 'Actions', run: openSnapEdges,
    disabled: () => state.selection.objects.length < 2 ? 'Select two or more paths' : !visiblePaths() && 'Every object must be a visible path'},
  {id: 'cleanup', name: 'Clean up…', group: 'Actions', keywords: 'duplicate vertices merge tidy', run: openCleanup, disabled: noSelection},
  {id: 'detach', name: 'Detach', rare: true, label: () => oneObject()?.tag === 'use' ? 'Detach to editable path' : 'Detach shared geometry', group: 'Actions', run: () => action('detach'),
    disabled: () => !(oneObject()?.tag === 'use' || oneObject()?.shared) && 'Select one instance, or a path that shares its geometry'},
  {id: 'delete', name: 'Delete', group: 'Actions', keys: 'Delete', keywords: 'remove points', run: deleteSelection,
    disabled: () => level() === 'points' ? noPoints() || (selectedPoints().some(key => nodeAt(key)?.pinned) && 'Unpin the points to delete them') : noSelection()},
  {id: 'load-reference', name: 'Load reference…', label: () => state?.reference ? 'Replace reference…' : 'Load reference…', group: 'Reference', run: () => $('reference-file').click()},
  {id: 'remove-reference', name: 'Remove reference', group: 'Reference', run: removeReference, disabled: noReference},
  {id: 'toggle-overlay', name: 'Cycle the view: drawing, overlay, reference only', group: 'Reference', keys: 'W', run: () => cycleReference(), disabled: noReference},
  {id: 'cycle-view-backward', name: 'Cycle the view backward', group: 'Reference', keys: 'Shift W', run: () => cycleReference(-1), disabled: noReference},
  {id: 'drawing-only', name: 'Show the drawing only', group: 'Reference', run: () => setView('drawing'), disabled: noReference},
  {id: 'generate', name: 'Generate from reference…', group: 'Reference', run: openGenerate, disabled: noReference},
  {id: 'tidy', name: 'Tidy…', group: 'Reference', keywords: 'simplify snap fit shape optimize nodes points', run: openTidy,
    disabled: () => !tidyTargets() && !state.reference && 'Select one or more paths, or groups that contain them, or add a reference to tidy what is in view'},
  {id: 'fit-colours', name: 'Fit colours…', group: 'Reference', run: () => openColours('flat'), disabled: () => noReference() || noSelection()},
  {id: 'fit-gradient', name: 'Fit gradient…', group: 'Reference', keywords: 'gradient ramp shading linear fill colour color', run: () => openColours('linear'), disabled: () => noReference() || noSelection()},
  {id: 'handles-0', name: 'No handles', group: 'Points', keys: '1', run: () => pointHandles(0), disabled: noPoints},
  {id: 'handles-1', name: 'One handle', group: 'Points', keys: '2', run: () => pointHandles(1), disabled: noPoints},
  {id: 'handles-2', name: 'Two handles', group: 'Points', keys: '3', run: () => pointHandles(2), disabled: noPoints},
  {id: 'pin', name: 'Pin points', label: () => selectedPoints().length && selectedPoints().every(key => nodeAt(key)?.pinned) ? 'Unpin points' : 'Pin points', group: 'Points',
    run: () => action('pin', {points: pointPairs(), pinned: !selectedPoints().every(key => nodeAt(key)?.pinned)}), disabled: noPoints},
  {id: 'split-edge', name: 'Add node', group: 'Points', keywords: 'split edge insert point vertex', run: () => action('split', {points: pointPairs()}),
    disabled: () => noPoints() || (selectedPoints().every(key => nodeAt(key)?.command === 'M' && !contourAt(key)?.closed) && 'A start point has no edge leading into it')},
  {id: 'delete-contour', name: 'Delete contour', group: 'Points', keys: 'Shift Delete', run: () => action('delete_contour', {points: pointPairs()}, 'Deleting contours…'),
    disabled: () => noPoints() || (selectedPoints().some(key => contourAt(key)?.nodes.some(n => n.pinned)) && 'Unpin the contour\'s points to delete it')},
  // Break cuts at the points, or with the two ends of a segment picked,
  // deletes that segment.
  {id: 'break-points', name: 'Break', group: 'Points', keywords: 'cut split disconnect open loop delete segment remove edge gap', run: breakPoints,
    disabled: () => noPoints() || (!segmentPicked() && !selectedPoints().some(key => breakable(contourAt(key), splitKey(key)[1]))
      && 'Pick a point between a line\'s ends or on a closed contour, or the two points at a segment\'s ends')},
  {id: 'fill-hole', name: 'Fill hole', group: 'Points', keywords: 'holes remove', run: fillPointHoles, disabled: () => noPoints() || (!holeContours(selectedPoints()) && 'Select points on holes')},
  {id: 'hole-to-shape', name: 'Hole to shape', group: 'Points', keywords: 'holes shapes', run: pointHolesToShapes, disabled: () => noPoints() || (!holeContours(selectedPoints()) && 'Select points on holes')},
];
const commandById = new Map(COMMANDS.map(command => [command.id, command]));
const commandName = command => command.label?.() || command.name;
function commandRefusal(command) {
  if (!state) return 'The drawing is still opening';
  return command.disabled?.() || '';
}
// Run a command now; event handlers queue it with later() instead, so it runs
// once any edit under way is done.
function runCommand(id) {
  const command = commandById.get(id), reason = command && commandRefusal(command);
  if (!command) return;
  if (reason) { toast(reason, true); return; }
  return command.run();
}
function pointHandles(count) { return action('node_handles', {points: pointPairs(), count}, 'Changing handles…'); }
function renameObject() { $('object-name').focus(); $('object-name').select(); }
// Buttons naming a command in data-command take its state and run it.
function renderCommands() {
  const actions = $('action-list');
  for (const button of document.querySelectorAll('[data-command]')) {
    const command = commandById.get(button.dataset.command); if (!command) continue;
    if (!button.dataset.title) button.dataset.title = button.title || `${commandName(command)}${command.keys ? ` (${command.keys})` : ''}`;
    const reason = commandRefusal(command);
    enable(button, reason);
    // Selection actions only show what applies, just like the context menu.
    if (actions.contains(button) || command.rare) button.hidden = !!reason;
    if (button.dataset.label === 'command') button.firstChild.textContent = commandName(command);
  }
  actions.closest('.inspector-section').hidden = !actions.querySelector('button:not([hidden])');
}
document.addEventListener('click', event => {
  const button = event.target.closest('[data-command]');
  if (button && !button.closest('#context-menu')) later(() => runCommand(button.dataset.command));
});
// The right panel shows the selection's properties: with points selected their
// coordinates and handles first, then the objects' name, paint and locks, and
// the actions on them.
function renderInspector() {
  const selected = state.selection.objects, item = oneObject();
  $('object-name-section').hidden = !item;
  $('object-name').value = item?.name || '';
  $('object-name').placeholder = item?.label || 'Automatic name';
  $('object-name').dataset.objectId = item?.id || '';
  $('empty-inspector').hidden = !!selected.length;
  $('properties').hidden = !selected.length;
  $('selection-kind').textContent = item?.tag || (selected.length ? 'Multiple' : 'Drawing');
  for (const controls of document.querySelectorAll('[data-tools]')) controls.hidden = !controls.dataset.tools.split(' ').includes(tool);
  renderNodeInspector(false);
  $('empty-reference-hint').hidden = !!state.reference;
  renderRelationships(item);
  // Empty selections have no paint to resolve; keep the remaining controls reset.
  for (const kind of selected.length ? ['fill', 'stroke'] : []) {
    const value = paintValue(kind, kind === 'fill' ? 'black' : 'none');
    // A gradient shows as its ramp; picking a colour makes the paint flat.
    const ramp = gradientCss(value);
    const hex = colorHex(ramp ? cssColour(paintColour(value)) : value);
    $(`${kind}-value`).value = ramp ? '' : hex || value;
    $(`${kind}-value`).placeholder = ramp ? 'Linear gradient' : selected.length ? 'Mixed colors' : '';
    const picker = $(`${kind}-color`);
    picker.value = hex?.slice(0,7) || '#000000';
    picker.classList.toggle('no-paint', value === 'none');
    picker.classList.toggle('mixed-paint', !value);
    picker.classList.toggle('gradient-paint', !!ramp);
    picker.style.background = ramp || '';
    const sources = [...new Set(selected.map(id=>paintSource(id,kind)))];
    const source = !value ? 'Mixed colors' : sources.length === 1 ? sources[0] : 'Different paint sources';
    $(`${kind}-source`).textContent = ramp ? ` ${paintGradient(value)?.getAttribute('data-vectrify-paint-owner') ? 'Private linear gradient' : 'Shared linear gradient'} · ${source}. Pick a colour to make it flat.`.trim() : source;
    picker.title = `${kind === 'fill' ? 'Fill' : 'Stroke'}: ${ramp ? 'linear gradient' : hex || value || 'mixed'} · ${source}`;
  }
  renderFillGradient(item);
  const strokeWidth = paintValue('stroke-width', '1');
  $('stroke-width').value = strokeWidth ? parseFloat(strokeWidth) : '';
  const opacity = paintValue('opacity', '1'); $('opacity').value = opacity === '' ? '' : Math.round(Number(opacity) * 100);
  document.querySelectorAll('[data-lock]').forEach(input => {
    input.disabled = !item; input.checked = item?.locks.includes(input.dataset.lock) || false;
    input.parentElement.title = item ? '' : 'Select one object to change its locks';
  });
  renderCommands();
  scheduleStrip();
}
// Private ramps are edited with the owning shape's fill properties.
function renderFillGradient(item) {
  const gradient = item?.fill_gradient;
  // Fit Color makes ordinary user-space ramps. Other imported paints keep
  // their SVG coordinate system until the user fits a new ramp.
  const editable = gradient?.private && gradient.attributes.gradientUnits === 'userSpaceOnUse' && !gradient.attributes.gradientTransform;
  $('fill-gradient').hidden = !editable;
  if (!editable) return;
  for (const key of ['x1', 'y1', 'x2', 'y2']) $('gradient-' + key).value = gradient.attributes[key] || 0;
  const container = $('gradient-stops'); container.replaceChildren();
  gradient.stops.forEach((stop, index) => {
    const row = document.createElement('div'); row.className = 'gradient-stop';
    const offset = document.createElement('input'); offset.type = 'number'; offset.min = 0; offset.max = 100; offset.step = 'any'; offset.value = stop.offset?.endsWith('%') ? parseFloat(stop.offset) : 100 * Number(stop.offset || 0); offset.setAttribute('aria-label', `Stop ${index + 1} position %`);
    const colour = document.createElement('input'); colour.type = 'color'; colour.value = colorHex(cssColour(stop['stop-color'] || 'black'))?.slice(0, 7) || '#000000'; colour.setAttribute('aria-label', `Stop ${index + 1} colour`);
    const opacity = document.createElement('input'); opacity.type = 'number'; opacity.min = 0; opacity.max = 100; opacity.value = 100 * Number(stop['stop-opacity'] || 1); opacity.setAttribute('aria-label', `Stop ${index + 1} opacity %`);
    const remove = document.createElement('button'); remove.textContent = '×'; remove.title = `Remove stop ${index + 1}`; remove.disabled = gradient.stops.length === 1;
    for (const input of [offset, colour, opacity]) input.onchange = saveFillGradient;
    remove.onclick = () => { row.remove(); saveFillGradient(); };
    row.append(offset, colour, opacity, remove); container.append(row);
  });
}
function fillGradientValue() {
  const stops = [...$('gradient-stops').children].map(row => {
    const [offset, colour, opacity] = row.querySelectorAll('input');
    return {offset: Number(offset.value) / 100, colour: colour.value, opacity: Number(opacity.value) / 100};
  }).sort((a, b) => a.offset - b.offset);
  return {start: ['x1', 'y1'].map(key => Number($('gradient-' + key).value)), end: ['x2', 'y2'].map(key => Number($('gradient-' + key).value)), stops};
}
function saveFillGradient() { paint({fill: fillGradientValue()}); }
for (const key of ['x1', 'y1', 'x2', 'y2']) $('gradient-' + key).onchange = saveFillGradient;
$('gradient-add-stop').onclick = () => {
  const fill = fillGradientValue(); fill.stops.push({offset: 0.5, colour: '#808080', opacity: 1}); fill.stops.sort((a, b) => a.offset - b.offset); paint({fill});
};

// The context menu offers the selection's commands where the pointer is.
const MENU_GROUPS = {objects: ['Select', 'Arrange', 'Object', 'Actions'], points: ['Points', 'Actions']};
function openContextMenu(x, y) {
  const menu = $('context-menu'), groups = MENU_GROUPS[level() === 'points' ? 'points' : 'objects'];
  menu.replaceChildren();
  for (const group of groups) {
    const commands = COMMANDS.filter(command => command.group === group && (!command.level || command.level === level()));
    // Only what applies here: the palette lists the rest, with the reason.
    const shown = commands.filter(command => command.id !== 'step-up' && !commandRefusal(command));
    if (!shown.length) continue;
    if (menu.children.length) menu.append(Object.assign(document.createElement('div'), {className: 'menu-divider', role: 'separator'}));
    for (const command of shown) {
      const item = document.createElement('button');
      item.className = 'menu-item'; item.setAttribute('role', 'menuitem'); item.dataset.command = command.id;
      const reason = commandRefusal(command);
      item.disabled = !!reason; item.title = reason;
      const name = document.createElement('span'); name.textContent = commandName(command);
      const keys = document.createElement('kbd'); keys.textContent = command.keys || '';
      item.append(name, keys);
      item.onclick = () => { closeContextMenu(); later(() => runCommand(command.id)); };
      menu.append(item);
    }
  }
  if (!menu.children.length) return;
  menu.hidden = false;
  const box = menu.getBoundingClientRect();
  menu.style.left = `${Math.min(x, innerWidth - box.width - 8)}px`;
  menu.style.top = `${Math.min(y, innerHeight - box.height - 8)}px`;
  menu.querySelector('.menu-item:not(:disabled)')?.focus();
}
function closeContextMenu() { $('context-menu').hidden = true; }
// The command palette (Ctrl/⌘ K) finds any command by name and runs it; a
// command that cannot run now says why.
let paletteRows = [], paletteIndex = 0;
function openPalette() {
  if (!state || document.querySelector('dialog[open]')) return;
  closeContextMenu();
  $('palette-input').value = '';
  renderPalette();
  $('palette').showModal();
  $('palette-input').focus();
}
function renderPalette() {
  paletteRows = matchCommands(COMMANDS.filter(command => !command.hidden?.()), $('palette-input').value);
  paletteIndex = Math.min(paletteIndex, Math.max(0, paletteRows.length - 1));
  if ($('palette-input').dataset.query !== $('palette-input').value) paletteIndex = 0;
  $('palette-input').dataset.query = $('palette-input').value;
  const list = document.createDocumentFragment();
  paletteRows.forEach((command, i) => {
    const row = document.createElement('div');
    row.className = 'palette-row'; row.id = `palette-row-${i}`; row.setAttribute('role', 'option');
    row.setAttribute('aria-selected', String(i === paletteIndex));
    const reason = commandRefusal(command);
    row.classList.toggle('disabled', !!reason); row.setAttribute('aria-disabled', String(!!reason));
    const text = document.createElement('span'); text.className = 'palette-text';
    const name = document.createElement('span'); name.className = 'palette-name'; name.textContent = commandName(command);
    const detail = document.createElement('small'); detail.textContent = reason || command.group;
    text.append(name, detail);
    const keys = document.createElement('kbd'); keys.textContent = command.keys || ''; keys.hidden = !command.keys;
    row.append(text, keys);
    row.onpointermove = () => { if (paletteIndex !== i) { paletteIndex = i; highlightPalette(); } };
    row.onclick = () => runPaletteRow(i);
    list.append(row);
  });
  if (!paletteRows.length) list.append(Object.assign(document.createElement('p'), {className: 'palette-empty', textContent: 'No command matches.'}));
  $('palette-list').replaceChildren(list);
  highlightPalette();
}
function highlightPalette() {
  for (const row of $('palette-list').querySelectorAll('.palette-row')) row.setAttribute('aria-selected', String(row.id === `palette-row-${paletteIndex}`));
  const row = $(`palette-row-${paletteIndex}`);
  $('palette-input').setAttribute('aria-activedescendant', row?.id || '');
  row?.scrollIntoView({block: 'nearest'});
}
async function runPaletteRow(i) {
  const command = paletteRows[i]; if (!command) return;
  const reason = commandRefusal(command);
  // A command that cannot run keeps the palette open, with its reason shown.
  if (reason) return;
  $('palette').close();
  later(() => runCommand(command.id));
}
$('palette-input').addEventListener('input', renderPalette);
$('palette-input').addEventListener('keydown', event => {
  if (['ArrowDown', 'ArrowUp'].includes(event.key)) {
    event.preventDefault();
    paletteIndex = moveHighlight(paletteIndex, event.key === 'ArrowDown' ? 1 : -1, paletteRows.length);
    highlightPalette();
  } else if (event.key === 'Enter') { event.preventDefault(); runPaletteRow(paletteIndex); }
});
$('palette').addEventListener('click', event => { if (event.target === $('palette')) $('palette').close(); });

window.addEventListener('pointerdown', event => { if (!event.target.closest('#context-menu')) closeContextMenu(); }, true);
window.addEventListener('blur', closeContextMenu);
$('context-menu').addEventListener('keydown', event => {
  const items = [...$('context-menu').querySelectorAll('.menu-item:not(:disabled)')], i = items.indexOf(document.activeElement);
  if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); closeContextMenu(); stage.focus(); }
  if (['ArrowDown', 'ArrowUp'].includes(event.key) && items.length) { event.preventDefault(); items[(i + (event.key === 'ArrowDown' ? 1 : items.length - 1)) % items.length].focus(); }
});
// Right-clicking something unselected selects it first, as a click would.
stage.addEventListener('contextmenu', event => {
  event.preventDefault();
  if (!state) return;
  const x = event.clientX, y = event.clientY;
  later(async () => {
    if (level() === 'objects' || level() === 'points') {
      // Found again now: an edit before it may have changed what is there.
      const hits = hitStack(x, y), targets = hits.length ? clickTargets(hits) : [];
      const at = document.elementFromPoint(x, y);
      const node = at?.dataset?.node && pointKey(at.dataset.object, at.dataset.node);
      if (node && level() === 'points' && !selectedPoints().includes(node)) await selectPoints(clickPointPath(state.selection.objects, pointPaths(), splitKey(node)[0], false), [node]);
      else if (!node && targets.length && !targets.some(id => state.selection.objects.includes(id) || pointPaths().includes(id))) await selectObject(targets[0]);
    }
    await queue;
    openContextMenu(x, y);
  });
});
$('objects').addEventListener('contextmenu', event => {
  const row = event.target.closest('.object-row'); if (!row) return;
  event.preventDefault();
  if (!state) return;
  const id = row.dataset.object, x = event.clientX, y = event.clientY;
  later(async () => {
    if (!object(id)) return;
    if (!state.selection.objects.includes(id)) await selectObject(id);
    await queue;
    openContextMenu(x, y);
  });
});
async function selectObject(id, additive = false, focus = false) {
  clickCycle = null;
  let selected = new Set(additive ? state.selection.objects : []);
  if (id) { if (additive && selected.has(id)) selected.delete(id); else selected.add(id); }
  // Picking an object in the tree enters the group it is in.
  if (focus && id && !additive) {
    const parent = object(id)?.parent;
    scope = parent && parent !== state.root ? parent : null;
  }
  focusPoint = null;
  const success = await action('select', {objects: [...selected]}, 'Selecting…');
  if (success && selected.has(id)) revealObject(id);
  if (success && focus) focusSelection();
  return success;
}
function focusSelection() {
  if (!state?.selection.objects.length) return;
  const targets = new Set(state.selection.objects.flatMap(selectionTargets));
  const points = [];
  for (const element of targets) {
    const matrix = localToOverlay(element);
    if (!matrix || !element.getBBox) continue;
    try {
      const b = element.getBBox();
      const style = getComputedStyle(element);
      const stroke = style.stroke === 'none' ? 0 : (parseFloat(style.strokeWidth) || 0) / 2;
      const x1 = b.x-stroke, y1 = b.y-stroke, x2 = b.x+b.width+stroke, y2 = b.y+b.height+stroke;
      points.push(...[[x1,y1],[x2,y1],[x2,y2],[x1,y2]].map(p => new DOMPoint(...p).matrixTransform(matrix)));
    } catch { /* Unrendered resources may have no display bounds. */ }
  }
  if (!points.length || points.some(p => !Number.isFinite(p.x) || !Number.isFinite(p.y))) return;
  const xs = points.map(p=>p.x), ys = points.map(p=>p.y);
  const left = Math.min(...xs), right = Math.max(...xs), top = Math.min(...ys), bottom = Math.max(...ys);
  const w = stage.clientWidth, h = stage.clientHeight;
  // Keep context around small details and frame the whole additive selection.
  zoom = Math.max(.005, Math.min(MAX_ZOOM, Math.max(1,w-80)/Math.max(60,(right-left)*1.2), Math.max(1,h-80)/Math.max(60,(bottom-top)*1.2)));
  pan = {x:w/2-((left+right)/2-state.bounds[0])*zoom,
         y:h/2-((top+bottom)/2-state.bounds[1])*zoom};
  updateView();
}
function revealObject(id) {
  let row = $('objects').querySelector(`[data-object="${CSS.escape(id)}"]`);
  if (!row && $('object-search').value) {
    $('object-search').value = ''; renderObjects();
    row = $('objects').querySelector(`[data-object="${CSS.escape(id)}"]`);
  }
  row?.scrollIntoView({block: 'nearest', inline: 'nearest'});
}
function hitStack(x, y) {
  // Browser hit testing respects SVG fill/stroke, clipping, transforms and use
  // instances, including painted shapes hidden behind another painted shape.
  return [...new Set(document.elementsFromPoint(x, y).map(targetId).filter(id => {
    const item = object(id);
    return item && !item.resource && !['g', 'svg', 'defs', 'clipPath'].includes(item.tag);
  }))];
}
function sameClickSpot(x, y, hits) {
  return clickCycle && Math.hypot(x-clickCycle.x, y-clickCycle.y) <= 4 &&
    hits.length === clickCycle.hits.length && hits.every((id, i) => id === clickCycle.hits[i]);
}
// What clicks at a spot pick, front to back: in object tools the outermost
// objects inside the entered group, in the others the painted shapes.
function clickTargets(hits) {
  if (level() !== 'objects') return hits;
  const map = parents();
  // A click outside the entered group leaves it.
  if (scope && pickTarget(hits[0], scope, map, state.root).scope !== scope) scope = null;
  return [...new Set(hits.map(hit => pickTarget(hit, scope, map, state.root).id))];
}
async function selectAtPoint(finished) {
  const hits = finished.hits?.length ? clickTargets(finished.hits) : [];
  if (!hits.length) return clickEmpty(finished.shift);
  // The second click of a double-click keeps what the first one picked.
  const again = sameClickSpot(finished.x, finished.y, hits), double = again && finished.time - clickCycle.time < DOUBLE_CLICK;
  const cycled = !finished.shift && again && !double;
  const index = double ? clickCycle.index : cycled ? (clickCycle.index+1)%hits.length : 0;
  // A double-click acts on what was picked before it: its first click may
  // have cycled on from a selected object to the next one under it.
  lastPick = {id: hits[index], first: double ? lastPick?.first ?? hits[index] : cycled ? hits[clickCycle.index] : hits[index]};
  const success = await selectObject(hits[index], finished.shift);
  if (success && !finished.shift) clickCycle = {x:finished.x, y:finished.y, time:finished.time, hits, index};
}
// A click on empty canvas: in point tools it drops the points first.
function clickEmpty(shift) {
  if (shift) return;
  if (selectedPoints().length) return selectPoints(state.selection.objects, []);
  return selectObject(null);
}
// The points a point command acts on: [object, node] pairs.
const pointPairs = () => selectedPoints().map(splitKey);
// The one selected point, or the one clicked last among several.
function focusedPoint() {
  const points = selectedPoints();
  return points.includes(focusPoint) ? focusPoint : points.length === 1 ? points[0] : null;
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
// The point section of the right panel and the Nodes strip.
function renderNodeInspector(commands = true) {
  const paths = pointPaths(), points = selectedPoints(), onPoints = level() === 'points';
  const nodes = paths.flatMap(geometryNodes), loaded = paths.every(id => geometries.has(id));
  const node = points.length === 1 && nodeAt(points[0]);
  const chosen = points.map(nodeAt).filter(Boolean), count = new Set(points.map(key => splitKey(key)[0])).size;
  const instances = pointTargetsNow().instances.length;
  $('point-section').hidden = !onPoints || (!paths.length && !instances);
  $('point-title').textContent = !chosen.length ? 'Points' : chosen.length > 1 ? `${chosen.length} points in ${count} ${count === 1 ? 'path' : 'paths'}` : node.command === 'M' ? 'Start point' : node.command === 'C' ? 'Curve endpoint' : 'Line endpoint';
  $('node-count').textContent = nodes.length ? nodes.length.toLocaleString() : '';
  const hints = [!loaded ? 'Loading path points…' : !chosen.length ? (tool === 'nodes' ? 'Click a point, or drag a box around several; Shift adds. Dragged points snap to others; hold Alt or Ctrl/⌘ to drag freely.' : 'Points selected in Nodes (A) stay selected here.') : chosen.length > 1 ? 'Drag one to move them together.' : node.pinned ? 'Pinned: unpin it to move or delete it.' : ''];
  // Points of shared geometry are points of every path drawing it.
  const sharing = new Set(points.flatMap(key => sharingPaths(splitKey(key)[0])));
  if (sharing.size) hints.push(`Shared geometry: ${chosen.length === 1 ? 'this point is' : 'these points are'} also in ${plural(sharing.size, 'other path')}, marked faintly, and edits change ${sharing.size === 1 ? 'both' : 'them all'}. Detach (Actions) to edit one path alone.`);
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
  const pinned = chosen.filter(n => n.pinned).length;
  $('node-pin').checked = chosen.length > 0 && pinned === chosen.length; $('node-pin').indeterminate = pinned > 0 && pinned < chosen.length;
  // Points on holes, in one path or several, offer to fill the holes or make
  // them shapes.
  const holes = holeContours(points);
  if (holes === undefined) loadNodeHoles(points);
  $('strip-hole').hidden = !holes;
  if (commands) renderCommands();
  scheduleStrip();
}
// The hole contours the selected points are on, as [path, contour] pairs,
// when every point is on a hole, in however many paths: null when they are
// not, undefined while the paths' holes are being found out.
const pathHoles = new Map();
function holesKey(id) { return `${state.epoch}:${state.revision}:${id}`; }
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
    const holes = pathHoles.get(holesKey(id));
    if (!holes) { loading = true; continue; }
    if (!holes.has(contour)) return null;
  }
  if (loading) return undefined;
  const pairs = [...contours.values()];
  return {contours: pairs, paths: new Set(pairs.map(([id]) => id)).size};
}
async function loadNodeHoles(points) {
  const ids = [...new Set(points.map(key => splitKey(key)[0]))].filter(id => !pathHoles.has(holesKey(id)));
  if (!ids.length) return;
  const {epoch, revision} = state;
  for (const id of ids) pathHoles.set(holesKey(id), null);
  await Promise.all(ids.map(async id => {
    let found = new Set();
    try {
      const result = await request('/api/holes', {object: id, epoch, revision});
      found = new Set(result.holes.map(h => h.id));
    } catch { /* A path whose holes cannot be read simply offers none. */ }
    if (state.epoch === epoch && state.revision === revision) pathHoles.set(holesKey(id), found);
  }));
  if (state.epoch === epoch && state.revision === revision) renderNodeInspector();
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

// Matrix reads share one overlay frame during a redraw. Outside it, dragging
// and hit testing read the live frames so optimistic transforms stay current.
let overlayFrame = null;
function localToOverlay(element) {
  // Agent updates include paint resources such as gradients, which have no
  // canvas frame. Only graphics elements can be positioned in the overlay.
  if (typeof element?.getScreenCTM !== 'function') return null;
  if (overlayFrame?.matrices.has(element)) return overlayFrame.matrices.get(element);
  const from = element?.getScreenCTM(), to = overlayFrame ? overlayFrame.to : overlay.getScreenCTM();
  if (!from || !to) return null;
  try {
    const matrix = DOMMatrix.fromMatrix(to.inverse().multiply(from));
    overlayFrame?.matrices.set(element, matrix);
    return matrix;
  } catch { return null; }
}
function selectionContour(source, seen = new Set(), inheritedPaint = null) {
  if (seen.has(source) || ['defs', 'clipPath'].includes(source.localName)) return null;
  const branch = new Set(seen); branch.add(source);
  const style = inheritedPaint || getComputedStyle(source);
  const paint = {
    fill: source.getAttribute('fill') || style.fill,
    fillRule: source.getAttribute('fill-rule') || style.fillRule,
  };
  // Expand instances so explicit paint on the source cannot hide the highlight.
  const clone = source.localName === 'use' ? xmlElement('g') : source.cloneNode(false);
  if (source.localName === 'use') {
    for (const attr of source.attributes) clone.setAttributeNS(attr.namespaceURI, attr.name, attr.value);
    const href = source.getAttribute('href') || source.getAttributeNS('http://www.w3.org/1999/xlink', 'href');
    const target = href?.startsWith('#') ? drawing.querySelector(`#${CSS.escape(href.slice(1))}`) : null;
    const child = target && selectionContour(target, branch, paint);
    if (child) {
      const offset = xmlElement('g', {transform: `translate(${source.getAttribute('x') || 0} ${source.getAttribute('y') || 0})`});
      offset.append(child); clone.append(offset);
    }
  } else {
    for (const child of source.children) {
      const contour = selectionContour(child, branch, paint);
      if (contour) clone.append(contour);
    }
  }
  clone.removeAttribute('id'); clone.removeAttribute('data-object-id');
  clone.removeAttribute('class'); clone.removeAttribute('style');
  clone.removeAttribute('opacity');
  if (['path', 'rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon'].includes(source.localName)) {
    clone.setAttribute('class', `selection-contour${paint.fill !== 'none' && source.localName !== 'line' ? ' selection-filled' : ''}`);
    clone.setAttribute('fill-rule', paint.fillRule || 'nonzero');
  }
  return clone;
}
function selectionTargets(id) {
  const element = svgElement(id);
  if (!element) return [];
  const clip = element.closest('clipPath');
  if (clip) {
    return [...drawing.querySelectorAll('[clip-path]')].filter(item =>
      !item.closest('defs') && item.getAttribute('clip-path') === `url(#${clip.id})`);
  }
  // Definitions are not painted themselves; highlight their visible instances.
  if (element.closest('defs')) {
    return [...drawing.querySelectorAll('use')].filter(use => !use.closest('defs') &&
      (use.getAttribute('href') || use.getAttributeNS('http://www.w3.org/1999/xlink', 'href')) === `#${element.id}`);
  }
  return [element];
}
function clipSelection(group, element) {
  // A selected child must keep the clipping inherited from its drawing groups.
  for (let ancestor = element.parentElement; ancestor && ancestor !== drawing; ancestor = ancestor.parentElement) {
    const clip = ancestor.getAttribute('clip-path'), matrix = localToOverlay(ancestor);
    if (!clip || !matrix) continue;
    const wrapper = xmlElement('g', {transform: matrix.toString(), 'clip-path': clip});
    const content = xmlElement('g', {transform: matrix.inverse().toString()});
    content.append(group);
    // Preserve the whole ancestor's bounds for objectBoundingBox clips.
    const b = ancestor.getBBox();
    wrapper.append(xmlElement('rect', {x:b.x,y:b.y,width:b.width,height:b.height,fill:'none',stroke:'none'}), content);
    group = wrapper;
  }
  return group;
}
function drawOverlay() {
  // The artboard zoom is a CSS transform, outside SVG's non-scaling-stroke.
  overlay.style.setProperty('--selection-scale', 1 / zoom);
  if (!state) { overlay.replaceChildren(); return; }
  // Read the drawing while all new overlay shapes are detached, then publish
  // them together. Appending each shape before the next path's CTM or bounds
  // read forced layout repeatedly during selection, node display and dragging.
  const content = document.createDocumentFragment();
  overlayFrame = {content, to: overlay.getScreenCTM(), matrices: new Map()};
  try { drawOverlayContent(); overlay.replaceChildren(content); }
  finally { overlayFrame = null; }
}
function drawOverlayContent() {
  const targets = new Set(state.selection.objects.flatMap(selectionTargets));
  selectionBox = selectionFrame();
  for (const element of targets) {
    const matrix = localToOverlay(element);
    if (!element?.getBBox || !matrix) continue;
    try {
      const contour = selectionContour(element);
      if (contour) {
        // The screen matrix already includes the selected object's transform.
        contour.removeAttribute('transform');
        let group = xmlElement('g', {transform: matrix.toString(), 'aria-hidden': 'true'});
        group.append(contour);
        group = clipSelection(group, element);
        const halo = group.cloneNode(true); halo.setAttribute('class', 'selection-halo');
        overlayFrame.content.append(halo, group);
      }
      // Point tools show the points instead of the bounds, and one object's
      // frame stands for its bounds.
      if (level() === 'points' || (selectionBox && targets.size === 1)) continue;
      const b = element.getBBox();
      const pts = [[b.x,b.y], [b.x+b.width,b.y], [b.x+b.width,b.y+b.height], [b.x,b.y+b.height]].map(([x,y]) => new DOMPoint(x,y).matrixTransform(matrix));
      overlayFrame.content.append(xmlElement('polygon', {points: pts.map(p => `${p.x},${p.y}`).join(' '), class: 'selection-box'}));
    } catch { /* Resource elements may have no display bounds. */ }
  }
  drawFrame();
  drawPathDraft();
  drawKnife();
  drawRedraw();
  drawSnap();
  drawBox();
  if (level() === 'points') drawPoints();
  drawFlash();
  renderStatus();
}
// Select resizes the selection like a window: its frame is the bounding box
// of the selected objects, with small ticks at the corners; the cursor shows
// what a drag does. Null when nothing that can be resized is selected.
function selectionFrame() {
  if (tool !== 'select' || !state?.selection.objects.length) return null;
  const ids = topSelection();
  if (ids.some(id => !object(id) || object(id).resource || ['defs', 'clipPath', 'svg'].includes(object(id).tag))) return null;
  const xs = [], ys = [];
  for (const id of ids) {
    const element = svgElement(id), matrix = localToOverlay(element);
    if (!element?.getBBox || !matrix) continue;
    try {
      const b = element.getBBox();
      if (!b.width && !b.height) continue;
      for (const [x, y] of [[b.x,b.y], [b.x+b.width,b.y], [b.x+b.width,b.y+b.height], [b.x,b.y+b.height]]) {
        const p = new DOMPoint(x, y).matrixTransform(matrix); xs.push(p.x); ys.push(p.y);
      }
    } catch { /* Unrendered objects have no bounds. */ }
  }
  if (!xs.length) return null;
  return {left: Math.min(...xs), top: Math.min(...ys), right: Math.max(...xs), bottom: Math.max(...ys)};
}
// Why the selection cannot be resized, or ''.
function resizeRefusal() {
  for (const id of topSelection()) {
    const item = object(id), lock = item?.inherited_locks.find(kind => kind === 'transform' || kind === 'geometry');
    if (lock) return `${item.label}: ${lock === 'transform' ? 'position' : 'geometry'} is locked; unlock it to resize`;
  }
  return '';
}
function drawFrame() {
  if (!selectionBox) return;
  const {left, top, right, bottom} = selectionBox, locked = !!resizeRefusal();
  overlayFrame.content.append(xmlElement('rect', {x: left, y: top, width: right - left, height: bottom - top, class: `resize-frame${locked ? ' locked' : ''}`}));
  if (locked) return;
  const r = 2.5 / zoom;
  for (const [x, y] of [[left, top], [right, top], [right, bottom], [left, bottom]]) overlayFrame.content.append(xmlElement('rect', {x: x - r, y: y - r, width: 2 * r, height: 2 * r, class: 'resize-tick'}));
}
// The part of the frame at the screen point (x, y): an edge, a corner,
// 'inside' or null.
function frameAt(x, y) {
  const matrix = selectionBox && overlay.getScreenCTM();
  if (!matrix) return null;
  const a = new DOMPoint(selectionBox.left, selectionBox.top).matrixTransform(matrix), b = new DOMPoint(selectionBox.right, selectionBox.bottom).matrixTransform(matrix);
  return frameHandle({left: a.x, top: a.y, right: b.x, bottom: b.y}, x, y, FRAME_REACH);
}
// The cursor over the frame: a resize arrow on an edge or corner, and the
// move cursor inside it, except over an unselected object, which a press
// picks instead.
function frameCursor(x, y) {
  const handle = frameAt(x, y);
  if (!handle || space) return 'default';
  if (handle !== 'inside') return resizeRefusal() ? 'not-allowed' : CURSORS[handle];
  const hit = hitStack(x, y)[0], picked = hit && pickTarget(hit, scope, parents(), state.root).id;
  return !hit || state.selection.objects.includes(picked) ? CURSORS.inside : 'default';
}
let frameHover = 0;
function hoverResize(event) {
  if (frameHover) return;
  const x = event.clientX, y = event.clientY;
  frameHover = requestAnimationFrame(() => {
    frameHover = 0;
    if (tool === 'select' && !drag && state) stage.style.cursor = frameCursor(x, y);
  });
}
// A press on an edge or corner of the frame starts resizing the selection.
function pressFrame(event, common, handle) {
  const refusal = resizeRefusal();
  if (refusal) { toast(refusal, true); return; }
  const members = topSelection().map(id => {
    const element = svgElement(id);
    return {id, element, before: object(id).attributes.transform || '', frame: element && localToOverlay(element.parentElement)};
  }).filter(member => member.element && member.frame);
  const box = {...selectionBox}, grab = point(event);
  // The edge follows the pointer from where it was grabbed.
  const offset = {x: handle.includes('w') ? box.left - grab.x : handle.includes('e') ? box.right - grab.x : 0,
    y: handle.includes('n') ? box.top - grab.y : handle.includes('s') ? box.bottom - grab.y : 0};
  startGesture('resize', event, common, {handle, box, members, offset, result: null});
}
// Other objects' bounds and the artboard's edges, which a dragged edge snaps
// to, in the overlay's frame.
function frameSnapTargets() {
  const [bx, by, bw, bh] = state.bounds, x = [bx, bx + bw], y = [by, by + bh];
  const inverse = overlay.getScreenCTM()?.inverse();
  if (!inverse) return {x, y};
  const map = parents(), selected = new Set(state.selection.objects), around = new Set();
  for (const id of selected) for (let at = map.get(id); at; at = map.get(at)) around.add(at);
  const inside = id => { for (let at = id; at; at = map.get(at)) if (selected.has(at)) return true; return false; };
  for (const item of state.objects) {
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
  const p = point(event), at = {x: p.x + drag.offset.x, y: p.y + drag.offset.y};
  drag.snapX = drag.snapY = null;
  if (!event.ctrlKey && !event.metaKey) {
    drag.edges ??= frameSnapTargets();
    if (/[we]/.test(drag.handle)) drag.snapX = nearestEdge(at.x, drag.edges.x, SNAP_RADIUS / zoom);
    if (/[ns]/.test(drag.handle)) drag.snapY = nearestEdge(at.y, drag.edges.y, SNAP_RADIUS / zoom);
    at.x = drag.snapX ?? at.x; at.y = drag.snapY ?? at.y;
  }
  return resizeScale(drag.box, drag.handle, at, {keepRatio: event.shiftKey, fromCentre: event.altKey, minimum: 1 / zoom});
}
// The preview scales each object in its parent's frame, as the server will.
function previewResize() {
  const {sx, sy, anchor: [ax, ay]} = drag.result;
  const page = new DOMMatrix().translate(ax, ay).scale(sx, sy).translate(-ax, -ay);
  for (const member of drag.members) {
    const local = member.frame.inverse().multiply(page).multiply(member.frame);
    member.element.setAttribute('transform', `${local} ${member.before}`.trim());
  }
}
const frameSize = box => [box.right - box.left, box.bottom - box.top].map(v => (+v.toFixed(1)).toLocaleString()).join(' × ');
// The points of every selected path and of the paths in selected groups, and
// of the unselected path under the pointer in Nodes, faintly, so a click on
// one adds its path. A selected point of shared geometry is marked faintly in
// the other paths drawing it.
function drawPoints() {
  const stageBox = stage.getBoundingClientRect(), {strong, twins} = shownPoints(), chosen = new Set(strong), twin = new Set(twins);
  const selected = new Set(pointPaths()), editable = tool === 'nodes';
  const paths = [...pointPaths(), ...(editable && hoverPath && !selected.has(hoverPath) ? [hoverPath] : [])];
  // Limit handles in dense drawings by screen-space spacing, without dropping
  // geometry. Zooming in exposes the original nodes at their full resolution.
  const occupied = new Set(); let count = 0, total = 0;
  const unpicked = !chosen.size && !twin.size;
  for (const id of paths) {
    const element = svgElement(id), matrix = localToOverlay(element), screen = element?.getScreenCTM();
    if (!matrix || !screen || !geometries.has(id)) continue;
    const ghost = !selected.has(id);
    const nodes = geometryNodes(id); total += nodes.length;
    for (const node of nodes) {
      // Nothing after the marker cap can be drawn unless it is selected or
      // shared with a selected point. Still count every original node above.
      if (unpicked && count >= 1200) break;
      const key = pointKey(id, node.id), picked = chosen.has(key), mirrored = twin.has(key);
      const x = node.values.at(-2), y = node.values.at(-1);
      const sx = screen.a*x + screen.c*y + screen.e, sy = screen.b*x + screen.d*y + screen.f;
      if (sx < stageBox.left || sx > stageBox.right || sy < stageBox.top || sy > stageBox.bottom) continue;
      const cell = `${Math.floor(sx/10)},${Math.floor(sy/10)}`;
      if (!picked && !mirrored && (occupied.has(cell) || count >= 1200)) continue;
      occupied.add(cell); count++;
      const p = {x: matrix.a*x + matrix.c*y + matrix.e, y: matrix.b*x + matrix.d*y + matrix.f};
      const near = editable && nearPoint === `${id} ${node.id} endpoint`;
      const circle = xmlElement('circle', {cx:p.x, cy:p.y, r: (picked ? 4.8 : mirrored ? 4.2 : 3.3)*(near ? 1.5 : 1)/zoom, class:`node${picked ? ' selected' : ''}${mirrored ? ' twin' : ''}${node.pinned ? ' pinned' : ''}${ghost ? ' ghost' : ''}${near ? ' near' : ''}${editable ? '' : ' passive'}`});
      circle.dataset.object = id; circle.dataset.node = node.id; circle.dataset.part = 'endpoint';
      overlayFrame.content.append(circle);
    }
  }
  // The handles of the selected points, up to a few hundred of them, and of
  // the point under the pointer, to show what it has before it is picked.
  if (editable) for (const key of new Set([...[...chosen].slice(0, 300), ...(nearAnchor ? [nearAnchor] : [])])) drawHandles(key);
  $('node-count').textContent = `${count.toLocaleString()} / ${total.toLocaleString()}`;
}
function drawHandles(key) {
  const [id] = splitKey(key), node = nodeAt(key), matrix = localToOverlay(svgElement(id));
  if (!node || !matrix) return;
  for (const handle of nodeHandles(key)) {
    let p = new DOMPoint(...handle.node.values.slice(handle.offset, handle.offset+2)).matrixTransform(matrix);
    const anchor = new DOMPoint(...node.values.slice(-2)).matrixTransform(matrix);
    // A handle on its point is retracted: none. One closer than
    // HANDLE_SPREAD is drawn that far out along its direction, on a dashed
    // line, so it shows clear of the point; a drag puts it at the pointer.
    const length = Math.hypot(p.x - anchor.x, p.y - anchor.y) * zoom;
    if (length < 1e-6) continue;
    const short = length < HANDLE_SPREAD;
    if (short) p = new DOMPoint(anchor.x + (p.x - anchor.x) * HANDLE_SPREAD / length, anchor.y + (p.y - anchor.y) * HANDLE_SPREAD / length);
    overlayFrame.content.append(xmlElement('line', {x1:anchor.x,y1:anchor.y,x2:p.x,y2:p.y,class:`handle-line${short ? ' short' : ''}`}));
    const near = nearPoint === `${id} ${handle.node.id} ${handle.offset}`;
    const circle = xmlElement('circle', {cx:p.x,cy:p.y,r:3.8*(near ? 1.5 : 1)/zoom,class:`handle${near ? ' near' : ''}`});
    circle.dataset.object = id; circle.dataset.node = handle.node.id; circle.dataset.part = String(handle.offset); overlayFrame.content.append(circle);
  }
}
// The rubber band of a box select, in the overlay's frame.
function drawBox() {
  if (drag?.kind !== 'box' || !drag.moved) return;
  const inverse = overlay.getScreenCTM()?.inverse(); if (!inverse) return;
  const a = new DOMPoint(drag.x, drag.y).matrixTransform(inverse), b = new DOMPoint(drag.end.x, drag.end.y).matrixTransform(inverse);
  overlayFrame.content.append(xmlElement('rect', {x:Math.min(a.x,b.x), y:Math.min(a.y,b.y), width:Math.abs(a.x-b.x), height:Math.abs(a.y-b.y), class:'select-box'}));
}
// The tool strip keeps to one row: the active tool's controls that do not fit
// go, least important first, into the "⋯" menu, keeping the status line.
const parked = new Map();
let stripFrame = 0, stripSignature = '';
function scheduleStrip() {
  if (!stripFrame) stripFrame = requestAnimationFrame(() => { stripFrame = 0; layoutStrip(); });
}
function layoutStrip(force = false) {
  const strip = $('tool-strip'), more = $('strip-more'), menu = $('strip-more-menu');
  const controls = strip.querySelector('.tool-controls:not([hidden])');
  const all = controls ? [...controls.querySelectorAll('[data-priority]'), ...[...parked.keys()].filter(item => controls.contains(parked.get(item)))] : [];
  const signature = [strip.clientWidth, tool, $('selection-level').textContent, $('scope-trail').textContent, $('canvas-hint').textContent,
    ...all.map(item => `${item.hidden}${item.textContent.length}`)].join('|');
  if (!force && signature === stripSignature) return;
  stripSignature = signature;
  for (const [item, mark] of parked) mark.replaceWith(item);
  parked.clear();
  more.hidden = false;
  const moreWidth = more.getBoundingClientRect().width;
  more.hidden = true;
  const items = controls ? [...controls.querySelectorAll('[data-priority]')].filter(item => !item.hidden) : [];
  const style = getComputedStyle(strip), gap = parseFloat(style.columnGap) || 0;
  const inner = parseFloat(getComputedStyle(controls || strip).columnGap) || 0;
  // What the strip holds besides the controls, and the gaps between them all.
  const others = [...strip.children].filter(child => child !== controls && child !== more && child.id !== 'canvas-hint' && !child.hidden && getComputedStyle(child).display !== 'none');
  const fixed = others.reduce((sum, child) => sum + child.getBoundingClientRect().width + gap, 0) + ($('canvas-hint').textContent ? gap + 40 : 0);
  const available = strip.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight) - fixed;
  const widths = items.map(item => ({width: item.getBoundingClientRect().width + inner, priority: Number(item.dataset.priority)}));
  const hidden = overflowLayout(widths, available, moreWidth + gap);
  for (const index of hidden) {
    const item = items[index], mark = document.createComment('parked');
    item.before(mark); menu.append(item); parked.set(item, mark);
  }
  more.hidden = !hidden.length;
  if (!hidden.length) closeStripMenu();
  else if (!menu.hidden) placeStripMenu();
}
// A "⋯" button and the menu of what did not fit beside it. A command run from
// the menu closes it; fields and choices keep it open.
function moreMenu(button, menu) {
  const place = () => {
    const box = button.getBoundingClientRect(), size = menu.getBoundingClientRect();
    menu.style.left = `${Math.max(8, Math.min(box.left, innerWidth - size.width - 8))}px`;
    menu.style.top = `${box.bottom + 6}px`;
  };
  const close = () => { menu.hidden = true; button.setAttribute('aria-expanded', 'false'); };
  button.onclick = () => {
    if (!menu.hidden) { close(); return; }
    menu.hidden = false; button.setAttribute('aria-expanded', 'true');
    place();
    menu.querySelector('button:not(:disabled), input, select')?.focus();
  };
  menu.addEventListener('click', event => { if (event.target.closest('button')) close(); });
  menu.addEventListener('keydown', event => { if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); close(); button.focus(); } });
  window.addEventListener('pointerdown', event => { if (!menu.contains(event.target) && !button.contains(event.target)) close(); }, true);
  return {place, close};
}
const stripMenu = moreMenu($('strip-more'), $('strip-more-menu'));
const placeStripMenu = stripMenu.place, closeStripMenu = stripMenu.close;
new ResizeObserver(() => scheduleStrip()).observe($('tool-strip'));
// The top bar keeps to one row too: the title shrinks, then the file actions
// that do not fit go into its "⋯" menu, least important first.
const topbarPriority = {'export-svg': 1, 'palette-open': 1, 'save-project': 2, 'open-file': 3};
const topbarMenu = moreMenu($('topbar-more'), $('topbar-more-menu'));
const topbarParked = new Map();
let topbarSignature = '';
function layoutTopbar() {
  const bar = document.querySelector('.topbar'), nav = bar.querySelector('nav'), more = $('topbar-more'), menu = $('topbar-more-menu');
  const signature = `${bar.clientWidth}|${$('filename').textContent}`;
  if (signature === topbarSignature) return;
  topbarSignature = signature;
  for (const [item, mark] of topbarParked) mark.replaceWith(item);
  topbarParked.clear();
  more.hidden = false;
  const moreWidth = more.getBoundingClientRect().width;
  more.hidden = true;
  const style = getComputedStyle(bar), gap = parseFloat(style.columnGap) || 0, inner = parseFloat(getComputedStyle(nav).columnGap) || 0;
  const title = bar.querySelector('.document-title'), name = $('filename');
  // The title keeps room for a short name while it is shown.
  const titleRoom = getComputedStyle(title).display === 'none' ? 0 : Math.min(name.scrollWidth + 30, 140) + gap;
  const available = bar.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight)
    - bar.querySelector('.brand').getBoundingClientRect().width - titleRoom - gap;
  const items = [...nav.children].filter(item => getComputedStyle(item).display !== 'none');
  const widths = items.map(item => ({width: item.getBoundingClientRect().width + inner, priority: topbarPriority[item.id] || 4}));
  const hidden = overflowLayout(widths, available + inner, moreWidth + gap);
  for (const index of hidden) {
    const item = items[index], mark = document.createComment('parked');
    item.before(mark); menu.append(item); topbarParked.set(item, mark);
  }
  more.hidden = !hidden.length;
  if (!hidden.length) topbarMenu.close();
  else if (!menu.hidden) topbarMenu.place();
}
new ResizeObserver(() => layoutTopbar()).observe(document.querySelector('.topbar'));
new MutationObserver(() => layoutTopbar()).observe($('filename'), {childList: true, characterData: true, subtree: true});
document.fonts?.ready.then(() => { topbarSignature = ''; layoutTopbar(); });
// The level and count of the selection, and the entered group.
function renderStatus() {
  reportView();
  if (!state) return;
  scheduleStrip();
  const points = selectedPoints(), paths = level() === 'points' ? pointPaths() : state.selection.objects;
  $('selection-level').textContent = selectionStatus(level() === 'points' ? 'points' : 'objects', paths, points);
  // Select gives the selection's size, and the new size while resizing.
  if (drag?.kind === 'resize' && drag.result) $('selection-level').textContent = `Resize · ${frameSize(drag.result.box)}`;
  else if (selectionBox) $('selection-level').textContent += ` · ${frameSize(selectionBox)}`;
  const trail = $('scope-trail');
  trail.replaceChildren();
  trail.hidden = !scope;
  if (!scope) return;
  const crumb = (label, id) => {
    const button = document.createElement('button'); button.className = 'crumb'; button.textContent = label;
    button.title = id ? `Pick within ${label}` : 'Leave the entered groups';
    button.onclick = () => { scope = id; clickCycle = null; renderStatus(); };
    trail.append(button);
  };
  crumb('Drawing', null);
  for (const id of scopeChain(scope, parents(), state.root)) { trail.append(' › '); crumb(object(id)?.label || id, id); }
}
function draftPathData(points, closed=false) {
  if (!points.length) return '';
  let d=`M${points[0].x} ${points[0].y}`;
  function edge(a,b) {
    if (a.out || b.in) {const u=a.out||a,v=b.in||b;return ` C${u.x} ${u.y} ${v.x} ${v.y} ${b.x} ${b.y}`;}
    return ` L${b.x} ${b.y}`;
  }
  for(let i=1;i<points.length;i++)d+=edge(points[i-1],points[i]);
  if(closed){if(points.at(-1).out||points[0].in)d+=edge(points.at(-1),points[0]);d+=' Z';}
  return d;
}
function drawPathDraft() {
  $('path-controls').hidden=tool!=='path';
  $('path-finish').disabled=pathDraft.length<2;
  $('path-close').disabled=pathDraft.length<3;
  $('path-cancel').disabled=!pathDraft.length;
  if(tool!=='path'||!pathDraft.length)return;
  const group=xmlElement('g',{'pointer-events':'none','aria-hidden':'true'});
  const points=pathHover?[...pathDraft,pathHover]:pathDraft;
  group.append(xmlElement('path',{d:draftPathData(points),fill:'none',stroke:'#052b3a','stroke-width':4/zoom}),
    xmlElement('path',{d:draftPathData(points),fill:'none',stroke:'#5cdeff','stroke-width':2/zoom}));
  pathDraft.forEach((p,i)=>{
    for(const h of [p.in,p.out].filter(Boolean)){
      group.append(xmlElement('line',{x1:p.x,y1:p.y,x2:h.x,y2:h.y,stroke:'#5cdeff','stroke-width':1/zoom}),
        xmlElement('circle',{cx:h.x,cy:h.y,r:3/zoom,fill:'#5cdeff'}));
    }
    group.append(xmlElement('circle',{cx:p.x,cy:p.y,r:(i===0?5:3.5)/zoom,fill:i===0?'#fff':'#5cdeff',stroke:'#052b3a','stroke-width':1.5/zoom}));
  });
  overlayFrame.content.append(group);
}
function drawKnife() {
  if(drag?.kind!=='knife'||!drag.moved)return;
  const {start:a,end:b}=drag, line={x1:a.x,y1:a.y,x2:b.x,y2:b.y,'pointer-events':'none','aria-hidden':'true'};
  overlayFrame.content.append(xmlElement('line',{...line,stroke:'#052b3a','stroke-width':4/zoom}),
    xmlElement('line',{...line,stroke:'#ff8a5c','stroke-width':2/zoom,'stroke-dasharray':`${6/zoom} ${4/zoom}`}));
}
// Snapping while dragging points or a handle: onto the on-curve points of
// every visible path and the artboard's edges and corners, within a fixed
// screen distance. Alt or Ctrl/⌘ drags freely.
function snapTargets(moving, start) {
  const points = [];
  for (const path of drawing.querySelectorAll('path')) {
    if (path.closest('defs, clipPath, mask, pattern, symbol, marker')) continue;
    if (path.checkVisibility && !path.checkVisibility({visibilityProperty: true})) continue;
    const matrix = localToOverlay(path); if (!matrix) continue;
    // The points being dragged are no targets; their paths' others are.
    const id = path.dataset.objectId, shown = drag.saved.has(id) && geometries.get(id);
    const local = shown ? geometryNodes(id).filter(n => !moving.has(pointKey(id, n.id))).map(n => drag.saved.get(id).get(n.id).slice(-2)) : pathEndpoints(path.getAttribute('d') || '');
    for (const [x, y] of local) {
      const p = new DOMPoint(x, y).matrixTransform(matrix);
      // Where the dragged point started is no target: it would hold it there.
      if (Math.hypot(p.x - start.x, p.y - start.y) > 1e-9) points.push([p.x, p.y]);
    }
  }
  return snapIndex(points, SNAP_RADIUS / zoom);
}
// Where the dragged point or handle goes, in the overlay's frame.
function snappedDrag(event) {
  let p = point(event);
  drag.snap = null;
  if (!event.altKey && !event.ctrlKey && !event.metaKey) {
    drag.snaps ??= snapTargets(new Set(drag.part === 'endpoint' ? drag.moving : []), drag.start);
    drag.snap = snapPoint(p.x, p.y, drag.snaps, state.bounds, SNAP_RADIUS / zoom);
    if (drag.snap) p = new DOMPoint(drag.snap.x, drag.snap.y);
  }
  return p;
}
function drawSnap() {
  if (drag?.kind === 'resize' && drag.moved) {
    const [x, y, w, h] = state.bounds;
    if (drag.snapX !== null) overlayFrame.content.append(xmlElement('line', {x1: drag.snapX, y1: y, x2: drag.snapX, y2: y + h, class: 'snap-edge'}));
    if (drag.snapY !== null) overlayFrame.content.append(xmlElement('line', {x1: x, y1: drag.snapY, x2: x + w, y2: drag.snapY, class: 'snap-edge'}));
    return;
  }
  const snap = drag?.kind === 'node' && drag.moved ? drag.snap : null;
  if (!snap) return;
  const [x, y, w, h] = state.bounds, r = 6 / zoom;
  if (snap.target.x !== undefined) overlayFrame.content.append(xmlElement('line', {x1: snap.target.x, y1: y, x2: snap.target.x, y2: y + h, class: 'snap-edge'}));
  if (snap.target.y !== undefined) overlayFrame.content.append(xmlElement('line', {x1: x, y1: snap.target.y, x2: x + w, y2: snap.target.y, class: 'snap-edge'}));
  overlayFrame.content.append(xmlElement('rect', {x: snap.x - r, y: snap.y - r, width: 2 * r, height: 2 * r, transform: `rotate(45 ${snap.x} ${snap.y})`, class: 'snap-target'}));
}
function knifeEnd(event) {
  const p=point(event), a=drag.start;
  if(!event.shiftKey)return {x:p.x,y:p.y};
  const step=Math.PI/12, angle=Math.round(Math.atan2(p.y-a.y,p.x-a.x)/step)*step, length=Math.hypot(p.x-a.x,p.y-a.y);
  return {x:a.x+Math.cos(angle)*length,y:a.y+Math.sin(angle)*length};
}
async function cutWithKnife({start, end}) {
  // With nothing selected the knife cuts what it crosses, in the entered group.
  if(await action('knife',{start:[start.x,start.y],end:[end.x,end.y],within:scope},'Cutting…')) {
    const pieces = state.selection.objects.length, lines = state.selection.objects.every(id => !filledPath(id));
    toast(lines ? `Cut the line${pieces > 1 ? ` into ${pieces} paths` : ' open'} where the knife crosses it; Join ends joins it again.` : `Cut into ${pieces} pieces. They meet exactly along the cut; Join merges them again.`);
  }
}
// Redraw outline: the stroke, where its ends attach to a path's outline and
// the stretch it will replace, all in the overlay's frame. The path is a
// selected one, or with nothing selected the one under the pointer.
function redrawLines() {
  const lines = [];
  for (const id of state.selection.objects.length ? pointPaths() : hoverPath ? [hoverPath] : []) {
    const matrix = geometries.has(id) && localToOverlay(svgElement(id));
    if (!matrix) continue;
    for (const line of contourLines(geometries.get(id), ([x, y]) => { const p = new DOMPoint(x, y).matrixTransform(matrix); return [p.x, p.y]; }))
      lines.push({...line, object: id});
  }
  return lines.length ? lines : null;
}
// The stroke starts on whichever selected path it touches and ends on the
// same contour.
function redrawPlan(points, longWay) {
  const lines = points.length ? redrawLines() : null;
  if (!lines) return null;
  const start = attach(lines, ...points[0], 1/zoom);
  const line = start && lines.find(l => l.id === start.contour);
  const own = line && lines.filter(l => l.object === line.object);
  const end = start && points.length > 1 ? attach(own, ...points.at(-1), 1/zoom, start.contour) : null;
  return {object: line?.object, start, end, replaced: end ? stretch(line, start, end, longWay) : null};
}
function drawRedraw() {
  if (tool !== 'redraw') return;
  const stroke = drag?.kind === 'redraw' ? drag.points : null;
  const plan = redrawPlan(stroke || (redrawHover ? [redrawHover] : []), drag?.longWay);
  const group = xmlElement('g', {'pointer-events': 'none', 'aria-hidden': 'true'});
  if (plan?.replaced) {
    group.append(xmlElement('polyline', {points: plan.replaced.join(' '), fill: 'none', stroke: '#ff8a5c', 'stroke-width': 3/zoom, 'stroke-dasharray': `${6/zoom} ${4/zoom}`}));
  }
  if (stroke && drag.moved) {
    group.append(xmlElement('polyline', {points: stroke.join(' '), fill: 'none', stroke: '#052b3a', 'stroke-width': 4/zoom}),
      xmlElement('polyline', {points: stroke.join(' '), fill: 'none', stroke: '#5cdeff', 'stroke-width': 2/zoom}));
  }
  for (const end of [plan?.start, plan?.end]) {
    if (end) group.append(xmlElement('circle', {cx: end.x, cy: end.y, r: 5/zoom, fill: '#fff', stroke: '#052b3a', 'stroke-width': 1.5/zoom}));
  }
  overlayFrame.content.append(group);
}
async function redrawOutline({points, longWay}) {
  const plan = redrawPlan(points, longWay);
  if (!redrawLines()) {toast(state.selection.objects.length ? 'The selection has no path to redraw.' : 'Start on a path\'s outline, then draw along the edge it should follow.', true);return;}
  if (!plan?.start || !plan.end) {toast(`Start and end the stroke on the same outline of ${state.selection.objects.length ? 'a selected' : 'one'} path.`, true);return;}
  await action('redraw_outline', {object: plan.object, points, pixel: 1/zoom, long_way: Boolean(longWay)}, state.reference ? 'Fitting to the reference…' : 'Redrawing outline…');
}
async function finishPath(closed) {
  if(pathDraft.length<(closed?3:2))return;
  const draft=pathDraft; pathDraft=[];pathHover=null;
  if(await action('add_path',{d:draftPathData(draft,closed),stroke_width:2/zoom},'Drawing path…'))await setTool('select');
  else {pathDraft=draft;drawOverlay();}
}
$('path-finish').onclick=()=>later(()=>finishPath(false));
$('path-close').onclick=()=>later(()=>finishPath(true));
$('path-cancel').onclick=()=>{pathDraft=[];pathHover=null;drawOverlay();};
function updateView() {
  reportView();
  clickCycle = null;
  $('artboard').style.transform = `translate(${pan.x}px,${pan.y}px) scale(${zoom})`;
  // Transparency shows as a checkerboard of 8px squares at any zoom.
  $('artboard').style.setProperty('--checker', `${16/zoom}px`);
  $('zoom-percent').textContent = `${Math.round(zoom*100)}%`;
  $('canvas-label').style.left = `${pan.x}px`; $('canvas-label').style.top = `${pan.y-22}px`;
  drawOverlay();
}
function fit() {
  if (!state) return;
  const w = stage.clientWidth, h = stage.clientHeight;
  zoom = Math.max(.01, Math.min((w-90)/state.bounds[2], (h-100)/state.bounds[3]));
  pan = {x:(w-state.bounds[2]*zoom)/2,y:(h-state.bounds[3]*zoom)/2}; updateView();
}
function zoomAt(factor, x=stage.clientWidth/2, y=stage.clientHeight/2) {
  const next = Math.max(.005, Math.min(MAX_ZOOM, zoom*factor)), ratio = next/zoom;
  pan = {x:x-(x-pan.x)*ratio, y:y-(y-pan.y)*ratio}; zoom = next; updateView();
}
function point(event, element = overlay) {
  const matrix = element.getScreenCTM();
  if (!matrix) throw new Error('This object has no editable canvas position');
  return new DOMPoint(event.clientX,event.clientY).matrixTransform(matrix.inverse());
}
// Switching tools keeps the selected objects. Leaving a point tool hides the
// points and remembers them; coming back shows them again if the objects are
// the same.
async function setTool(value) {
  if (!state) return;
  const from = tool;
  const switched = switchTool({objects: state.selection.objects, points: selectedPoints(), memory: pointMemory}, from, value);
  clickCycle = null; lastPick = null; pathDraft=[]; pathHover=null; redrawHover=null; hoverPath = null; tool=value; pointMemory = switched.memory;
  reportView();
  document.querySelectorAll('.tool[data-tool]').forEach(button => {
    const active = button.dataset.tool === tool;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });
  $('active-tool').dataset.tool = tool;
  $('tool-strip').dataset.tool = tool;
  $('tool-shortcut').textContent = {select: 'V', nodes: 'A', path: 'D', knife: 'C', redraw: 'R', hand: 'Space'}[tool];
  $('tool-name').textContent = tool === 'select' ? 'Select objects' : tool === 'nodes' ? 'Edit points' : names[tool];
  $('canvas-hint').textContent=hints[tool];
  stage.style.cursor = tool === 'hand' ? 'grab' : ['path','knife','redraw'].includes(tool) ? 'crosshair' : 'default';
  renderInspector();
  const nodes = [...new Set(switched.points.map(key => splitKey(key)[1]))];
  if (level() === 'points') {
    setBusy('Loading path nodes…',1);
    try { await loadGeometries(); } catch(error) {toast(error.message,true);} finally {setBusy('',-1);}
    // Restored points show in the paths they were picked in.
    if (switched.points.length) owners = pointOwners(switched.points, id => geometries.get(id)?.id);
  }
  if (nodes.length || state.selection.nodes.length) {
    focusPoint = switched.points.includes(focusPoint) ? focusPoint : switched.points.at(-1) ?? null;
    await action('select', {objects: state.selection.objects, nodes}, 'Selecting…');
  }
  renderInspector(); drawOverlay();
}
function targetId(target) {
  if (!drawing.contains(target)) return null;
  for(let element=target; element && element !== drawing; element=element.parentElement) if (element.dataset.objectId && object(element.dataset.objectId)) return element.dataset.objectId;
  return null;
}
function topSelection() {
  return state.selection.objects.filter(id => {
    let parent = object(id)?.parent;
    while (parent) { if (state.selection.objects.includes(parent)) return false; parent = object(parent)?.parent; }
    return true;
  });
}
function pathData(g) { return g.subpaths.map(s => s.nodes.map(n => n.command+n.values.join(' ')).join(' ') + (s.closed ? ' Z' : '')).join(' '); }
// A point drag previews what the server will do: the points' handles ride along.
function valuesById(g) { return new Map(g.subpaths.flatMap(s => s.nodes.map(n => [n.id, [...n.values]]))); }
function restoreValues(g, saved) { for (const s of g.subpaths) for (const n of s.nodes) n.values = [...saved.get(n.id)]; }
function carryHandles(g, saved) {
  // Mirrors the server: a moved point takes along the handles the edit left
  // alone, and a closed contour ending on its moveto moves as one point.
  for (let changed = true; changed;) {
    changed = false;
    for (const s of g.subpaths) {
      const nodes = s.nodes, olds = nodes.map(n => saved.get(n.id)), last = nodes.length - 1;
      const twins = s.closed && last > 1 && olds[last].at(-2) === olds[0].at(-2) && olds[last].at(-1) === olds[0].at(-1);
      nodes.forEach((node, i) => {
        const dx = node.values.at(-2) - olds[i].at(-2), dy = node.values.at(-1) - olds[i].at(-1);
        if (!dx && !dy) return;
        const slots = node.command === 'C' ? [[i, 2]] : [];
        if (i < last && nodes[i+1].command === 'C') slots.push([i+1, 0]);
        if (twins && (i === 0 || i === last)) slots.push([last-i, nodes[last-i].values.length-2]);
        for (const [j, k] of slots) {
          const v = nodes[j].values;
          if (v[k] !== olds[j][k] || v[k+1] !== olds[j][k+1]) continue;
          v[k] += dx; v[k+1] += dy; changed = true;
        }
      });
    }
  }
}
// Move the dragged points by the pointer's offset in the overlay, each in its
// own path's frame, or the dragged handle to the pointer.
function previewPointDrag(target) {
  const [grabbedPath, grabbed] = splitKey(drag.key);
  for (const [id, saved] of drag.saved) {
    const g = geometries.get(id), element = svgElement(id), matrix = localToOverlay(element);
    if (!g || !matrix) continue;
    restoreValues(g, saved);
    const inverse = matrix.inverse();
    if (drag.part === 'endpoint') {
      const from = drag.start.matrixTransform(inverse), to = target.matrixTransform(inverse), dx = to.x - from.x, dy = to.y - from.y;
      for (const node of geometryNodes(id)) {
        if (!drag.moving.includes(pointKey(id, node.id))) continue;
        const v = node.values; v[v.length-2] += dx; v[v.length-1] += dy;
      }
    } else if (id === grabbedPath) {
      const node = geometryNodes(id).find(n => n.id === grabbed), local = target.matrixTransform(inverse), offset = Number(drag.part);
      node.values[offset] = local.x; node.values[offset+1] = local.y;
    }
    carryHandles(g, saved);
    element.setAttribute('d', pathData(g));
    // The other paths drawing this geometry follow it.
    for (const other of sharingPaths(id)) if (!drag.saved.has(other)) svgElement(other)?.setAttribute('d', pathData(g));
  }
}
// Press on a point: select it (with Shift, add or remove it), selecting its
// path when it is not (with Shift, adding it), and start dragging the selected
// points.
function pressPoint(event, common) {
  const id = event.target.dataset.object, nodeId = event.target.dataset.node, part = event.target.dataset.part;
  const key = pointKey(id, nodeId), current = selectedPoints();
  const objects = clickPointPath(state.selection.objects, pointPaths(), id, common.shift);
  let points = current;
  if (part === 'endpoint') {
    points = current.includes(key) && !common.shift ? current : clickPoint(current, key, common.shift);
    if (points !== current || objects !== state.selection.objects) selectPoints(objects, points, key);
    else focusPoint = key;
  }
  // A handle moves alone; a point moves with the other selected points, and a
  // point of shared geometry picked in two paths moves once, as grabbed.
  const once = new Set();
  const moving = part === 'endpoint' ? [key, ...points.filter(k => k !== key)].filter(k => {
    const node = nodeAt(k), twin = `${geometries.get(splitKey(k)[0])?.id} ${splitKey(k)[1]}`;
    if (!node || node.pinned || once.has(twin)) return false;
    once.add(twin); return true;
  }) : [key];
  if (part === 'endpoint' && (!points.includes(key) || nodeAt(key)?.pinned)) { startGesture('point-click', event, common); return; }
  const paths = new Set(moving.map(k => splitKey(k)[0]));
  const saved = new Map([...paths].filter(p => geometries.has(p)).map(p => [p, valuesById(geometries.get(p))]));
  const node = nodeAt(key), offset = part === 'endpoint' ? node.values.length - 2 : Number(part);
  const start = new DOMPoint(node.values[offset], node.values[offset+1]).matrixTransform(localToOverlay(svgElement(id)));
  // A click on one of several selected points, without dragging, selects it alone.
  const collapse = part === 'endpoint' && !common.shift && current.includes(key) && current.length > 1;
  startGesture('node', event, common, {key, part, moving, saved, start, collapse, objects});
}
// Box select: objects wholly inside the box at the entered group's level, or
// in point tools the points of the selected paths.
async function finishBox(finished) {
  const box = dragBox({x: finished.x, y: finished.y}, finished.end);
  if (level() === 'points') {
    const found = [];
    for (const id of pointPaths()) {
      const screen = svgElement(id)?.getScreenCTM(); if (!screen) continue;
      for (const node of geometryNodes(id)) {
        const p = new DOMPoint(...node.values.slice(-2)).matrixTransform(screen);
        if (pointInside(p.x, p.y, box)) found.push(pointKey(id, node.id));
      }
    }
    return selectPoints(state.selection.objects, boxSelect(selectedPoints(), found, finished.shift));
  }
  const container = scope || state.root;
  const found = state.objects.filter(item => item.parent === container && !item.resource && !['defs', 'clipPath'].includes(item.tag)).filter(item => {
    const rect = svgElement(item.id)?.getBoundingClientRect();
    return rect && rect.width + rect.height > 0 && rectInside(rect, box);
  }).map(item => item.id);
  focusPoint = null;
  return action('select', {objects: boxSelect(state.selection.objects, found, finished.shift)}, 'Selecting…');
}
// Each gesture owns its press, preview, completion and cancellation. The
// canvas retains hit testing, pointer capture and the shared input queue.
const gestures = canvasGestures({
  point, drawOverlay,
  view: {
    pan: () => pan, zoom: () => zoom,
    setPan: value => { pan = value; }, update: updateView,
    setPanning: value => stage.classList.toggle('panning', value),
  },
  path: {draft: () => pathDraft, setHover: value => { pathHover = value; }, finish: finishPath},
  selection: {
    pick: selectAtPoint, box: finishBox, renderDrawing,
    move: offsets => action('move', {dx: 0, dy: 0, offsets}, 'Moving selection…'),
  },
  nodes: {
    preview: previewPointDrag, snap: snappedDrag, finish: finishPointDrag,
    select: selectPoints, renderInspector, renderNodeInspector,
    restore: gesture => {
      for (const [id, saved] of gesture.saved) if (geometries.has(id)) restoreValues(geometries.get(id), saved);
    },
  },
  resize: {
    result: resizeDrag, preview: previewResize,
    finish: result => action('resize', {anchor: result.anchor, scale: [result.sx, result.sy]}, 'Resizing…'),
  },
  knife: {end: knifeEnd, finish: cutWithKnife},
  redraw: {setHover: value => { redrawHover = value; }, finish: redrawOutline},
});
function startGesture(kind, event, common, details = {}) {
  const controller = gestures[kind];
  drag = {...common, kind, ...details};
  controller.press?.(drag, event);
}
// A press on the canvas while an edit runs, or input waits, is held: its
// events are recorded and replayed once the edit is done, finding again what
// is under the pointer then, so the gesture acts on the state the edit left.
let holding = null;
stage.addEventListener('pointerdown', event => {
  if (!state || ![0,1].includes(event.button)) return;
  if (event.button === 1) event.preventDefault();
  if (drag || (holding && !holding.released)) return;
  if (pending || input.waiting) { holdGesture(event); return; }
  pressStage(event);
});
function holdGesture(event) {
  const gesture = new HeldGesture(event);
  holding = gesture;
  try { stage.setPointerCapture(event.pointerId); } catch { /* The pointer may be gone already. */ }
  later(async () => {
    let released;
    const again = e => ({...e, target: document.elementFromPoint(e.clientX, e.clientY) || stage, preventDefault() {}});
    const live = gesture.replay({down: e => pressStage(again(e)), move: e => moveStage(again(e)), up: e => { released = releaseStage(again(e)); }, cancel: cancelStage});
    // Still pressed: the rest of the gesture goes on live.
    if (live && holding === gesture) holding = null;
    await released;
  });
}
function pressStage(event) {
  if (!state || drag) return;
  const middle = event.button === 1;
  stage.focus({preventScroll:true});
  try { stage.setPointerCapture(event.pointerId); } catch { /* A replayed press may be released already. */ }
  const common = {x:event.clientX,y:event.clientY,time:event.timeStamp ?? performance.now(),shift:event.shiftKey || event.ctrlKey || event.metaKey,moved:false};
  const bounds = $('artboard').getBoundingClientRect();
  common.deselectOutside = !middle && !space && (event.clientX < bounds.left || event.clientX > bounds.right ||
    event.clientY < bounds.top || event.clientY > bounds.bottom);
  if (middle || space || tool === 'hand') {
    startGesture('pan', event, common); return;
  }
  if (tool === 'path' && !common.deselectOutside) {
    const p=point(event), first=pathDraft[0];
    if (first && pathDraft.length>=3 && Math.hypot(p.x-first.x,p.y-first.y)*zoom<8) {
      startGesture('closePath', event, common); return;
    }
    startGesture('drawPath', event, common); return;
  }
  const near = tool === 'nodes' && !middle && !space ? nearestPoint(event.clientX, event.clientY) : null;
  if (near) { pressPoint({target: near}, common); return; }
  const handle = tool === 'select' ? frameAt(event.clientX, event.clientY) : null;
  if (handle && handle !== 'inside') { pressFrame(event, common, handle); return; }
  const hits = hitStack(event.clientX, event.clientY);
  common.hits = hits;
  const id = hits[0] || null;
  if (tool === 'knife') {
    startGesture('knife', event, common, {id}); return;
  }
  if (tool === 'redraw') {
    startGesture('redraw', event, common, {id}); return;
  }
  // In Select, dragging a selected object, or the empty canvas inside the
  // selection's frame, moves the selection; a click picks what is under the
  // pointer, and any other drag selects what lies inside its box. Knife only
  // selects.
  if (tool === 'select' && (id || handle === 'inside')) {
    const targets = id ? clickTargets(hits) : [], picked = targets[0];
    const selectedHit = !id || state.selection.objects.includes(picked) ||
      (sameClickSpot(event.clientX, event.clientY, targets) && state.selection.objects.includes(targets[clickCycle.index]));
    if (selectedHit && !common.shift) {
      const members=topSelection().map(oid => ({id:oid,element:svgElement(oid),before:object(oid).attributes.transform || ''})).filter(m=>m.element);
      startGesture('move', event, common, {id, members}); return;
    }
  }
  startGesture('box', event, common, {id});
}
// In Nodes the unselected path under the pointer shows its points faintly.
let hoverFrame = 0;
function hoverPoints(event) {
  if (hoverFrame) return;
  const x = event.clientX, y = event.clientY;
  hoverFrame = requestAnimationFrame(async () => {
    hoverFrame = 0;
    if (!['nodes', 'redraw'].includes(tool) || drag || pending || holding) return;
    let next;
    if (tool === 'redraw') {
      // With nothing selected, the stroke redraws the path it starts on: the
      // last one the pointer was over, since the outline's edge is easy to miss.
      if (state.selection.objects.length) return;
      next = pointerTarget(hitStack(x, y), state.objects, [], scope) ?? hoverPath;
    } else {
      // Near one of the hovered path's points, it stays hovered off its body.
      const near = nearestPoint(x, y), key = nearKey(near) ?? null;
      const hit = near?.dataset.object || hitStack(x, y).find(id => object(id)?.tag === 'path');
      next = hit && !pointPaths().includes(hit) ? hit : null;
      // The point under the pointer shows its handles, kept on its handles.
      const anchor = near?.dataset.part === 'endpoint' ? pointKey(near.dataset.object, near.dataset.node) : near ? nearAnchor : null;
      if (key !== nearPoint || anchor !== nearAnchor) {
        nearPoint = key; nearAnchor = anchor;
        if (next === hoverPath) { drawOverlay(); return; }
      }
    }
    if (next === hoverPath) return;
    hoverPath = next;
    if (next && !geometries.has(next)) { try { await loadGeometries(); } catch { return; } }
    drawOverlay();
  });
}
stage.addEventListener('pointerleave', () => {
  if (!nearPoint && !nearAnchor) return;
  nearPoint = nearAnchor = null; if (!drag) drawOverlay();
});
stage.addEventListener('pointermove', event => {
  if (holding && !holding.released) { holding.record(event); return; }
  moveStage(event);
});
function moveStage(event) {
  if (!drag) {
    if (tool==='path' && pathDraft.length) {pathHover=point(event);drawOverlay();}
    if (tool==='redraw' && redrawLines()) {const p=point(event);redrawHover=[p.x,p.y];drawOverlay();}
    if (['nodes', 'redraw'].includes(tool) && state) hoverPoints(event);
    if (tool==='select' && state) hoverResize(event);
    return;
  }
  drag.moved ||= Math.hypot(event.clientX-drag.x,event.clientY-drag.y)>3;
  gestures[drag.kind].move?.(drag, event);
}
// A point drag sends the moved values: one point or handle as a node edit the
// server carries the handles of, several points with their handles moved.
async function finishPointDrag(finished) {
  const changes = {};
  for (const [id, saved] of finished.saved) {
    const g = geometries.get(id); if (!g) continue;
    for (const node of geometryNodes(id)) {
      const before = saved.get(node.id);
      if (node.values.some((v, i) => v !== before[i])) (changes[id] ||= {})[node.id] = [...node.values];
    }
    restoreValues(g, saved);
  }
  if (!Object.keys(changes).length) { renderDrawing(); drawOverlay(); return; }
  const [id, nodeId] = splitKey(finished.key);
  if (finished.part !== 'endpoint' || finished.moving.length === 1) {
    const values = changes[id]?.[nodeId];
    if (values) await action('node', {object: id, node: nodeId, values}, 'Updating contour…');
    else { renderDrawing(); drawOverlay(); }
  } else await action('move_nodes', {changes}, 'Moving points…');
}
stage.addEventListener('pointerup', event => {
  if (holding && !holding.released) { holding.record(event); holding = null; return; }
  releaseStage(event);
});
async function releaseStage(event) {
  if (!drag) return;
  const finished = drag, controller = gestures[finished.kind];
  drag = null; stage.classList.remove('panning');
  if (stage.hasPointerCapture(event.pointerId)) stage.releasePointerCapture(event.pointerId);
  // A click outside the artboard clears the selection in every tool.
  if (finished.deselectOutside && !finished.moved) { await clickEmpty(false); return; }
  if (finished.moved && !controller.keepClickCycle) clickCycle = null;
  await controller.release?.(finished);
}
// Double-click a group to pick within it, or a path to edit its points.
stage.addEventListener('dblclick', event => {
  if (!state || space) return;
  const x = event.clientX, y = event.clientY;
  later(async () => {
    await queue;
    if (level() !== 'objects') return;
    const id = lastPick?.first, item = object(id);
    if (!item || item.resource) return;
    await enterObject(id, hitStack(x, y));
  });
});
async function enterObject(id, hits = []) {
  const item = object(id);
  if (item?.tag === 'g') {
    scope = id; clickCycle = null; lastPick = null;
    const inner = hits.length ? pickTarget(hits[0], scope, parents(), state.root) : null;
    await action('select', {objects: [inner?.scope === id ? inner.id : id]}, 'Selecting…');
  } else if (item?.tag === 'path') {
    if (state.selection.objects.length !== 1 || state.selection.objects[0] !== id) await action('select', {objects: [id]}, 'Selecting…');
    await setTool('nodes');
  } else if (item?.tag === 'use') toast('Detach this instance to edit its points (Detach, under Actions).');
}
// Escape steps up one level: points to their paths, objects to their group,
// then to nothing, and out of an entered group.
async function stepUp() {
  const hadPoints = selectedPoints().length > 0;
  const next = escapeStep({objects: state.selection.objects, points: selectedPoints(), scope}, parents(), state.root);
  scope = next.scope; clickCycle = null; focusPoint = null;
  if (next.objects.join() !== state.selection.objects.join() || hadPoints) await action('select', {objects: next.objects}, 'Selecting…');
  renderStatus();
}
stage.addEventListener('pointercancel', event => {
  if (holding && !holding.released) { holding.record(event); holding = null; return; }
  cancelStage();
});
function cancelStage() {
  if (drag) gestures[drag.kind].cancel?.(drag);
  stage.classList.remove('panning'); clickCycle = null; drag = null; geometries = new Map();
  if (state) { renderDrawing(); loadGeometries().then(drawOverlay).catch(() => {}); }
  drawOverlay();
}
stage.addEventListener('auxclick',event=>{if(event.button===1)event.preventDefault();});
stage.addEventListener('lostpointercapture', () => {
  if (drag?.kind === 'pan') { gestures.pan.cancel(drag); drag = null; }
});
stage.addEventListener('wheel',event=>{event.preventDefault();if(!state)return;const b=stage.getBoundingClientRect();zoomAt(Math.exp(-event.deltaY*.0015),event.clientX-b.left,event.clientY-b.top);},{passive:false});
new ResizeObserver(()=>{if(state)updateView();}).observe(stage);
$('fit').onclick=fit; $('zoom-in').onclick=()=>zoomAt(1.25); $('zoom-out').onclick=()=>zoomAt(.8);
document.querySelectorAll('.tool[data-tool]').forEach(button=>button.onclick=()=>later(()=>setTool(button.dataset.tool)));
$('object-search').oninput=renderObjects;
$('undo').onclick=()=>later(()=>action('undo',{},'Undoing…')); $('redo').onclick=()=>later(()=>action('redo',{},'Redoing…'));
// Paint and offsets apply in order with other input, after any edit under way.
const paint = changes => later(() => action('paint', {changes}));
for (const kind of ['fill','stroke']) {
  $(`${kind}-value`).onchange=event=>paint({[kind]:event.target.value.trim()||null});
  $(`${kind}-color`).onchange=event=>paint({[kind]:event.target.value});
  $(`${kind}-none`).onclick=()=>paint({[kind]:'none'});
}
$('stroke-width').onchange=event=>paint({'stroke-width':event.target.value||null});
$('opacity').onchange=event=>paint({opacity:String(Number(event.target.value)/100)});
$('move-apply').onclick=()=>{
  const dx=Number($('move-x').value), dy=Number($('move-y').value);
  $('move-x').value='0';$('move-y').value='0';
  later(()=>action('move',{dx,dy}));
};
// Point commands act on every selected point, in however many paths.
function movePointTo(x, y) {
  const key = focusedPoint(), node = key && nodeAt(key); if (!node) return;
  const [id, nodeId] = splitKey(key);
  action('node', {object: id, node: nodeId, values: [...node.values.slice(0,-2), x, y]});
}
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
function joinSelectionKey() { return JSON.stringify([state.epoch,state.revision,state.selection.objects]); }
function joinCandidates() {
  const selected = new Set(state.selection.objects), candidates = [];
  for (const item of state.objects) {
    let included = selected.has(item.id);
    for (let parent = object(item.parent); !included && parent; parent = object(parent.parent)) included = selected.has(parent.id);
    if (!included) continue;
    if (item.resource || !['g','path'].includes(item.tag)) return [];
    if (item.tag === 'path') candidates.push(item);
  }
  return candidates;
}
function openJoin() {
  if (pending) return;
  const candidates = joinCandidates();
  // Lines join at their ends, not by area.
  if (candidates.length > 1 && linePaths().length === candidates.length) return joinEnds();
  if (candidates.length < 2) {
    toast('Select paths or groups containing at least two paths in total.',true); return;
  }
  joinContext = {key:joinSelectionKey(), candidates};
  $('join-summary').textContent = `Combine ${candidates.length} paths into one geometry.`;
  $('join-error').hidden = true;
  const options = document.createDocumentFragment();
  function option(value, title, detail, item = null) {
    const label = document.createElement('label'); label.className = 'join-color-option';
    const radio = document.createElement('input'); radio.type = 'radio'; radio.name = 'join-colors'; radio.value = value; radio.checked = value === 'mix';
    label.append(radio);
    if (item) { const swatch = document.createElement('span'); swatch.className = 'swatch'; paintSwatch(swatch,item); label.append(swatch); }
    const text = document.createElement('span'), name = document.createElement('strong'), description = document.createElement('small');
    name.textContent = title; description.textContent = detail; text.append(name,description); label.append(text); options.append(label);
  }
  option('mix','Area-weighted mix','Larger shapes contribute more to the mixed colors.');
  const colorChoices = new Map();
  for (const item of candidates) {
    const fill = resolvedPaint(item.id,'fill','black'), stroke = resolvedPaint(item.id,'stroke','none');
    const fillAlpha = resolvedPaint(item.id,'fill-opacity','1'), strokeAlpha = resolvedPaint(item.id,'stroke-opacity','1');
    const key = JSON.stringify([fill,stroke,fillAlpha,strokeAlpha]);
    if (colorChoices.has(key)) colorChoices.get(key).count++;
    else colorChoices.set(key,{item,fill,stroke,fillAlpha,strokeAlpha,count:1});
  }
  for (const {item,fill,stroke,fillAlpha,strokeAlpha,count} of colorChoices.values()) {
    const alpha = value => Number(value) < 1 ? ` at ${Math.round(Number(value)*100)}%` : '';
    const detail = `Fill ${colorHex(fill)||fill}${alpha(fillAlpha)} · Stroke ${colorHex(stroke)||stroke}${alpha(strokeAlpha)}` +
      (count > 1 ? ` · ${count} paths share these colors` : '');
    option(item.id,`Use ${item.label} colors`,detail,item);
  }
  $('join-color-options').replaceChildren(options);
  $('join-dialog').showModal();
}
$('join-cancel').onclick = () => $('join-dialog').close();
$('join-dialog').addEventListener('close',()=>joinContext=null);
$('join-dialog').addEventListener('cancel',event=>{if(pending)event.preventDefault();});
$('join-confirm').onclick = async () => {
  if(pending || !joinContext) return;
  if(joinSelectionKey() !== joinContext.key) { $('join-error').textContent = 'The selection changed. Close this dialog and open Join again.'; $('join-error').hidden = false; return; }
  const choice = $('join-color-options').querySelector('input:checked')?.value;
  if(!choice) return;
  const source = joinContext.candidates.find(item=>item.id === choice);
  const options = choice === 'mix' ? {colors:'mix'} : {colors:'source',color_source:choice};
  $('join-confirm').disabled = true; $('join-cancel').disabled = true; $('join-error').hidden = true;
  try {
    if(await action('join_paths',{options},'Joining outlines…')) {
      $('join-dialog').close();
      toast(source ? `Joined using ${source.label}'s colors at the frontmost selected position.` : 'Joined at the frontmost selected position with area-weighted paint.');
    } else { $('join-error').textContent = $('toast-message').textContent; $('join-error').hidden = false; }
  } finally { $('join-confirm').disabled = false; $('join-cancel').disabled = false; }
};
// Join: two selected points join each other; lines join the ends that
// continue them nearby; filled paths merge by area, after choosing colours.
const joinOpensDialog = () => !twoEnds() && joinCandidates().length > 1 && linePaths().length < joinCandidates().length;
function join() {
  if (twoEnds()) return action('join_two_ends', {points: pointPairs()}, 'Joining ends…');
  return joinOpensDialog() ? openJoin() : joinEnds();
}
// Every selected line joins the ends that continue it nearby.
async function joinEnds() {
  const lines = linePaths().length;
  if (await action('join_ends', {reach: JOIN_REACH / zoom}, 'Joining line ends…')) toast(`Joined the ends of ${plural(lines, 'line path')} that continue each other within ${JOIN_REACH} screen pixels; zoom out to reach wider gaps.`);
}
async function splitParts() {
  const before = state.selection.objects.length;
  if (await action('split_disconnected', {}, 'Splitting disconnected parts…')) {
    toast(state.selection.objects.length > before ? `Split into ${state.selection.objects.length} independently editable paths.` : 'No disconnected parts found. Holes and touching contours stay together.');
  }
}
// Delete works at the current level: the selected points in a point tool,
// else the selected objects.
function deleteSelection() {
  return level() === 'points' ? action('delete_node', {points: pointPairs()}, 'Deleting points…') : action('delete');
}
const segmentPicked = () => pointContours().some(({contour, ids}) => segmentAmong(contour, ids));
function breakPoints() {
  return segmentPicked() ? action('delete_segment', {points: pointPairs()}, 'Deleting segments…') : action('break_points', {points: pointPairs()}, 'Breaking lines…');
}
async function cutHole() {
  if (await action('cut_hole', {}, 'Cutting out the hole…')) toast('Cut the shape out as a hole. Undo restores both paths.');
}
document.querySelectorAll('[data-lock]').forEach(box=>box.onchange=()=>{
  const item=oneObject(), locks=[...document.querySelectorAll('[data-lock]:checked')].map(el=>el.dataset.lock);
  if(item)later(()=>object(item.id)&&action('locks',{object:item.id,locks}));
});
$('palette-open').onclick=openPalette;
$('help').onclick=()=>$('help-dialog').showModal();$('help-close').onclick=()=>$('help-dialog').close();
// Keep a local copy of saved projects so a backend restart can restore this tab.
function recoveryStore(mode, key, value) {
  return new Promise((resolve, reject) => {
    const opening = indexedDB.open('vectrify-recovery', 1);
    opening.onupgradeneeded = () => opening.result.createObjectStore('projects');
    opening.onerror = () => reject(opening.error);
    opening.onsuccess = () => {
      const db = opening.result, tx = db.transaction('projects', mode), store = tx.objectStore('projects');
      const request = mode === 'readonly' ? (key === undefined ? store.getAll() : store.get(key)) : store.put(value, key);
      tx.oncomplete = () => {resolve(request.result);db.close();};
      tx.onerror = () => {reject(tx.error);db.close();};
      tx.onabort = () => {reject(tx.error);db.close();};
    };
  });
}
$('restore-saved').onclick=async()=>{
  try {
    const saved=await recoveryStore('readonly');
    $('recovery-list').replaceChildren();
    const unique=[...new Map(saved.map(item=>[item.source,item])).values()];
    unique.sort((a,b)=>(b.savedAt||0)-(a.savedAt||0));
    for(const item of unique){
      const data=JSON.parse(item.source), doc=data.document||data;
      let objects=0; const walk=e=>{objects++;(e.children||[]).forEach(walk);};walk(doc.root);
      const nodes=(doc.geometries||[]).reduce((total,g)=>total+g.subpaths.reduce((n,s)=>n+s.nodes.length,0),0);
      const button=document.createElement('button');button.className='wide-button';
      button.textContent=`${item.name} · ${objects-1} objects · ${nodes.toLocaleString()} points${item.savedAt?' · '+new Date(item.savedAt).toLocaleString():''}`;
      button.onclick=async()=>{
        if(dirty&&!window.confirm('Replace this tab’s drawing with the saved copy?'))return;
        if(await action('open',{name:item.name,source:item.source},'Restoring saved project…')){
          await recoveryStore('readwrite',session,item);dirty=false;$('dirty').textContent='';
          $('recovery-dialog').close();await loadReference();fit();
        }
      };
      $('recovery-list').append(button);
    }
    if(!unique.length)$('recovery-list').textContent='No saved browser recovery copies yet.';
    $('recovery-dialog').showModal();
  }catch(error){toast(error.message,true);}
};
$('recovery-close').onclick=()=>$('recovery-dialog').close();
async function download(project) {
  await queue; setBusy(project?'Saving project…':'Exporting SVG…',1);
  try {
    const result=await request('/api/export',{project,epoch:state.epoch,revision:state.revision});
    const filename=state.name.replace(/\.(svg|json|vectrify)$/i,'')+(project?'.vectrify':'.svg');
    if(bridge){
      // A native save dialog; cancelling it saves nothing.
      if(!await (await bridge).save(filename,result.content))return;
    }else{
      const blob=new Blob([result.content],{type:project?'application/json':'image/svg+xml'}),url=URL.createObjectURL(blob);
      const link=document.createElement('a');link.href=url;link.download=filename;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    }
    if(project){
      await recoveryStore('readwrite',session,{source:result.content,name:state.name,savedAt:Date.now()});
      dirty=false;$('dirty').textContent='';
    }
    toast(project?'Project saved, with a browser recovery copy.':'SVG exported.');
  }catch(error){toast(error.message,true);}finally{setBusy('',-1);}
}
$('save-project').onclick=()=>download(true);$('export-svg').onclick=()=>download(false);
$('open-file').onclick=()=>$('svg-file').click();
$('svg-file').onchange=async event=>{
  const file=event.target.files[0];event.target.value='';if(!file)return;
  if(dirty&&!window.confirm('Open another drawing? Save your project first if you want to keep these edits.'))return;
  const success=await action('open',{name:file.name,source:await file.text()},'Opening drawing…');
  if(success){dirty=false;$('dirty').textContent='';await loadReference();fit();}
};
// The Reference panel, below the objects, holds the reference image, the
// view and the tools that match the drawing to it. The view is the drawing
// alone, the reference over it, or the reference alone at full strength in
// place of the drawing.
const VIEWS = ['drawing', 'overlay', 'reference'];
let referenceView = 'overlay';
function showReference() {
  reportView();
  $('reference-controls').hidden=!reference; $('remove-reference').hidden=!reference;
  $('reference-name').textContent=reference?.name || 'No reference';
  $('reference-name').title=reference?.name || '';
  $('add-reference').textContent=reference ? 'Replace…' : 'Load…';
  $('reference-thumb').hidden=!reference; $('reference-thumb-empty').hidden=!!reference;
  const only = !!reference && referenceView === 'reference', enabled = !!reference && referenceView !== 'drawing';
  const visible = enabled && (only || reference.opacity > 0);
  $('reference-image').hidden=!enabled;
  $('drawing').style.visibility = only ? 'hidden' : '';
  const toggle = $('reference-toggle');
  toggle.hidden=!reference;
  toggle.setAttribute('aria-pressed', String(enabled));
  toggle.title=(only ? 'Show the drawing only' : enabled ? 'Show the reference only' : 'Show the reference overlay') + ' (W; Shift+W cycles backward)';
  const status = only ? 'Reference only' : enabled ? `Reference overlay · ${Math.round(reference.opacity*100)}%` : 'Drawing only';
  $('reference-status').textContent=status;
  $('reference-state').textContent=reference ? {drawing: 'Hidden', overlay: `${Math.round(reference.opacity*100)}%`, reference: 'Alone'}[referenceView] : 'None';
  for (const button of document.querySelectorAll('[data-view]')) button.setAttribute('aria-pressed', String(button.dataset.view === referenceView));
  stage.classList.toggle('reference-visible', visible);
  if(reference){$('reference-image').src=reference.data_url;$('reference-thumb').src=reference.data_url;$('reference-image').style.opacity=only ? 1 : reference.opacity;$('reference-opacity').value=Math.round(reference.opacity*100);$('reference-percent').textContent=`${Math.round(reference.opacity*100)}%`;}
  else{$('reference-image').removeAttribute('src');$('reference-thumb').removeAttribute('src');}
  scheduleStrip();
}
function setView(view) { referenceView = view; showReference(); }
// W cycles forward through the views; Shift+W cycles backward.
function cycleReference(direction = 1) {
  if (reference) setView(VIEWS[(VIEWS.indexOf(referenceView) + direction + VIEWS.length) % VIEWS.length]);
}
for (const button of document.querySelectorAll('[data-view]')) button.onclick = () => setView(button.dataset.view);
// The panel folds to its heading; a newly loaded reference opens it again.
function foldReference(folded) {
  $('reference-panel').classList.toggle('folded', folded);
  $('reference-body').hidden = folded;
  $('reference-fold').setAttribute('aria-expanded', String(!folded));
  try { localStorage.setItem('vectrify-reference-folded', folded ? '1' : ''); } catch { /* Folding is only remembered where storage works. */ }
}
try { if (localStorage.getItem('vectrify-reference-folded')) foldReference(true); } catch { /* Open by default. */ }
$('reference-fold').onclick = () => foldReference(!$('reference-body').hidden);
$('reference-toggle').onclick=()=>cycleReference();
async function loadReference(){const result=await request('/api/reference');reference=result.reference;showReference();}
$('add-reference').onclick=()=>$('reference-file').click();
$('reference-file').onchange=async event=>{
  const file=event.target.files[0];event.target.value='';if(!file)return;
  const reader=new FileReader();reader.onload=async()=>{const value={name:file.name,data_url:reader.result,opacity:.5};if(await action('reference',{reference:value},'Loading reference…')){
    // The editor fits the image to the artboard: it may pad it, or size an
    // empty artboard to it, so show what it kept.
    referenceView='overlay';foldReference(false);await loadReference();fit();}};reader.readAsDataURL(file);
};
async function removeReference(){if(await action('reference',{reference:null})){reference=null;showReference();renderInspector();}}
$('remove-reference').onclick=removeReference;
$('reference-opacity').oninput=event=>{if(reference){reference.opacity=Number(event.target.value)/100;if(referenceView==='drawing')referenceView='overlay';showReference();}};
$('reference-opacity').onchange=()=>{if(reference)action('reference',{reference});};
$('object-name').onchange = event => {
  const input = event.target, item = object(input.dataset.objectId);
  const name = input.value;
  if (item && name.trim() !== item.name) later(() => object(item.id) && action('rename', {object:item.id, name}, 'Renaming object…'));
};
$('object-name').onkeydown = event => {
  if (event.key === 'Enter') {event.preventDefault();event.target.blur();}
  if (event.key === 'Escape') {event.preventDefault();event.target.value=oneObject()?.name || '';event.target.blur();}
};
// The tool keys, all under the left hand; Space held pans.
const TOOL_KEYS = {v: 'select', a: 'nodes', d: 'path', c: 'knife', r: 'redraw'};
// A tool strip's numbered controls, by digit.
function stripKeys(name) {
  const keys = {};
  for (const button of document.querySelectorAll('[data-key][data-strip]'))
    if (button.dataset.strip.split(' ').includes(name)) keys[button.dataset.key] = button;
  return keys;
}
// Cancel the gesture under way: a drag (put back as it was), a path being
// drawn or a held gesture. Whether there was one.
function cancelGesture() {
  const busy = !!(drag || pathDraft.length || (holding && !holding.released));
  if (holding && !holding.released) { holding.cancel(); holding = null; }
  if (drag) {
    gestures[drag.kind].cancel?.(drag);
    drag = null; if (state) renderDrawing();
  }
  pathDraft=[];pathHover=null;stage.classList.remove('panning');
  if (state) renderInspector();
  drawOverlay();
  return busy;
}
// Each numbered strip control remembers its strip, since the overflow menu
// may move it out of the strip.
for (const controls of document.querySelectorAll('.tool-controls[data-tools]'))
  for (const button of controls.querySelectorAll('[data-key]')) {
    button.dataset.strip = controls.dataset.tools;
    button.title = `${button.title || button.getAttribute('aria-label') || ''} (${button.dataset.key})`.trim();
  }
// Keys that edit wait for any edit under way, and run in order once it is
// done; keys that only change the view act at once.
window.addEventListener('keydown',event=>{
  const typing=['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName), dialog=document.querySelector('dialog[open]');
  const mod=event.ctrlKey||event.metaKey, key=event.key.toLowerCase();
  if(event.key==='F2'&&!typing&&!dialog&&oneObject()){event.preventDefault();later(()=>{if(oneObject())renameObject();});return;}
  if(event.key==='Escape'&&treeDrag){event.preventDefault();endTreeDrag(false);return;}
  if(mod&&!event.altKey&&key==='k'){event.preventDefault();if(!dialog)openPalette();return;}
  if(typing||dialog)return;
  if(event.code==='Space'){event.preventDefault();space=true;return;}
  if(event.key==='Escape'){
    // Escape cancels what is under way at once, else steps the selection up
    // a level once any edit is done.
    const busy = cancelGesture();
    if (!busy && state && !event.repeat) later(stepUp);
    return;
  }
  // Held keys repeat; while an edit runs the repeats would pile up.
  if(event.repeat&&(pending||input.waiting)&&!['arrowleft','arrowright','arrowup','arrowdown'].includes(key)){event.preventDefault();return;}
  if (tool==='path' && pathDraft.length) {
    if (event.key==='Enter') {event.preventDefault();later(()=>finishPath(false));return;}
    if (event.key==='Backspace' || event.key==='Delete' || (mod&&key==='z')) {
      event.preventDefault();if(!drag){pathDraft.pop();pathHover=null;drawOverlay();}return;
    }
  }
  if(mod&&key==='s'){event.preventDefault();download(true);return;}
  if(mod&&key==='z'){event.preventDefault();const redo=event.shiftKey;later(()=>action(redo?'redo':'undo',{},redo?'Redoing…':'Undoing…'));return;}
  // Ctrl/⌘ G groups (with Shift, ungroups) and J joins.
  if(mod&&!event.altKey&&['g','j'].includes(key)){
    event.preventDefault();
    const command=key==='j'?'join':event.shiftKey?'ungroup':'group';
    later(()=>runCommand(command));return;
  }
  if(mod&&['BracketLeft','BracketRight'].includes(event.code)){
    // Ctrl/⌘ ] and [ step forward and backward; with Shift, to the front and back.
    event.preventDefault();
    const command=event.shiftKey?(event.code==='BracketRight'?'to-front':'to-back'):(event.code==='BracketRight'?'forward':'backward');
    later(()=>{if(state?.selection.objects.length)runCommand(command);});return;
  }
  if(mod||event.altKey)return;
  if(event.key==='?'){event.preventDefault();$('help-dialog').showModal();return;}
  // Tools and the view are on the left hand. A tool key switches at once,
  // even while an edit runs, and cancels a gesture under way.
  if(TOOL_KEYS[key]&&!event.shiftKey){event.preventDefault();if(!event.repeat){cancelGesture();setTool(TOOL_KEYS[key]);}return;}
  if(key==='w'){event.preventDefault();if(!event.repeat)cycleReference(event.shiftKey?-1:1);return;}
  if(key==='f'&&!event.shiftKey){fit();return;}
  if(key==='z'&&!event.shiftKey){event.preventDefault();if(state?.selection.objects.length)focusSelection();return;}
  // 1-9 press the tool strip's controls, as numbered on them.
  if(/^[1-9]$/.test(event.key)){
    // A command's control runs it (saying why when it cannot); another
    // control is pressed, as a click would.
    const button=stripKeys(tool)[event.key];
    if(button){event.preventDefault();const command=button.dataset.command;later(()=>command?runCommand(command):button.disabled||button.click());}
    return;
  }
  if(event.key==='Delete'||event.key==='Backspace'){
    event.preventDefault();
    const contours=event.shiftKey;
    later(()=>{
      if(!state?.selection.objects.length)return;
      if(level()!=='points') { if(!contours) runCommand('delete'); return; }
      if(selectedPoints().length) runCommand(contours ? 'delete-contour' : 'delete');
    });
  }
});
window.addEventListener('keyup',event=>{if(event.code==='Space')space=false;});
window.addEventListener('blur',()=>space=false);
window.addEventListener('beforeunload',event=>{if(dirty||pathDraft.length){event.preventDefault();event.returnValue='';}});
async function start(){
  setBusy('Opening editor…',1);
  try {
    const previous = sessionStorage.getItem('vectrify-session');
    let result = await request('/api/session',{session:previous}); session = result.session;
    if (previous && session !== previous) {
      const saved = await recoveryStore('readonly',previous);
      if (saved) {
        result = await request('/api/action',{command:'open',...saved,epoch:result.epoch,revision:result.revision});
        await recoveryStore('readwrite',session,saved);
        toast('Restored your saved project after the server restart. Earlier undo history is unavailable.');
      }
    }
    sessionStorage.setItem('vectrify-session',session);
    await applyState(result);await loadReference();fit();
    pollAgent();
  }
  catch(error){toast(error.message,true);}finally{setBusy('',-1);}
}
start();

// Automated operations share one job API: start, status, stop, apply, discard.
const operation = (command, body) => request('/api/operation', {command, ...body});


// Tidy: a quick clean-up of the selected paths that mixes snapping, simplifying
// and fitting (the operation's method is still called nodes).
const NODE_STEPS = ['shape', 'snap', 'detail', 'simplify'];
const STEP_NAMES = {shape:'fit', snap:'snap', simplify:'simplify'};
const nodeSteps = () => Object.fromEntries(NODE_STEPS.map(step => [step, $('nodes-'+step).checked]));
function syncNodeSteps() {
  const steps = nodeSteps();
  $('nodes-detail').disabled = !steps.snap || !state.reference;
  $('nodes-detail-gain').disabled = !steps.snap || !steps.detail;
  // A step that is off keeps its options visible but dimmed.
  for (const card of document.querySelectorAll('#nodes-settings .step-card[data-step]')) {
    const on = steps[card.dataset.step];
    card.classList.toggle('off', !on);
    for (const input of card.querySelectorAll('.two-fields input')) if (input.id !== 'nodes-detail' && input.id !== 'nodes-detail-gain') input.disabled = !on;
  }
  const customRun = $('nodes-custom-run').checked;
  for (const id of ['nodes-rounds', 'nodes-workers']) $(id).disabled = !customRun;
  $('nodes-steps').disabled = !customRun || !steps.shape;
  $('nodes-apply').hidden = true; $('nodes-previews').hidden = true;
}
const nodesDialog = jobDialog('nodes', {
  start: () => {
    const steps = nodeSteps();
    const customRun = $('nodes-custom-run').checked;
    const region = $('nodes-in-view').checked ? viewReport()?.region : null;
    return {action:'improve', method:'nodes', scope:'selection',
      permissions:{geometry:true, structure:(steps.snap && steps.detail) || steps.simplify, paint:true},
      settings:{...steps, tolerance:Number($('nodes-tolerance').value),
        movement:Number($('nodes-movement').value),
        ...(customRun ? {steps:Number($('nodes-steps').value), workers:Number($('nodes-workers').value)} : {}),
        detail_gain:Number($('nodes-detail-gain').value), gain:Number($('nodes-gain').value), margin:Number($('nodes-margin').value),
        allowance:Number($('nodes-allowance').value), budget:Number($('nodes-budget').value),
        shared:$('nodes-shared').checked,
        seconds:Number($('nodes-seconds').value), ...(region ? {region} : {})},
      budget:{steps:customRun ? Number($('nodes-rounds').value) : 1}};
  },
  describe: ({changed, metrics}) => {
    if (!changed) return 'No step improved the paths within these settings. They are unchanged.';
    const points = `${metrics.before.nodes.toLocaleString()} → ${metrics.after.nodes.toLocaleString()} points`;
    const fit = metrics.reference ? `difference from the reference ${errorChange(metrics, 'difference')}` : 'the look is kept within the tolerance';
    const order = metrics.steps.map(step => STEP_NAMES[step] || step).join(' → ');
    const skipped = Object.values(metrics.skipped || {});
    const note = skipped.length ? ` Some paths were not fitted: ${[...new Set(skipped)].join('; ')}.` : '';
    const late = metrics.out_of_time ? ' Stopped at the time limit.' : '';
    const followed = metrics.followed ? ` ${metrics.followed} neighbouring path${metrics.followed === 1 ? '' : 's'} moved along shared edges.` : '';
    return `${points} · ${fit} · ${order}.${followed}${late}${note} Apply keeps this result as one undoable edit.`;
  },
  applied: 'Paths tidied. Undo restores them.',
}).wire();
for (const step of NODE_STEPS) $('nodes-'+step).addEventListener('change', syncNodeSteps);
$('nodes-custom-run').addEventListener('change', syncNodeSteps);
for (const id of ['nodes-tolerance', 'nodes-rounds', 'nodes-workers', 'nodes-steps', 'nodes-movement', 'nodes-detail-gain', 'nodes-gain', 'nodes-margin', 'nodes-seconds', 'nodes-allowance', 'nodes-budget', 'nodes-in-view', 'nodes-shared']) $(id).addEventListener('input', () => { $('nodes-apply').hidden = true; $('nodes-previews').hidden = true; });
// Whether paths, or groups that may hold them, are selected for Tidy.
const tidyTargets = () => state.selection.objects.some(id => ['path', 'g'].includes(object(id)?.tag));
async function openTidy() {
  await queue;
  const reference = Boolean(state.reference);
  // With nothing selected Tidy works on what is in view.
  const targets = tidyTargets();
  if (!targets) $('nodes-in-view').checked = true;
  $('nodes-in-view').disabled = !targets;
  // Snap comes back on with the reference, as it is by default.
  if (reference && $('nodes-snap').disabled) $('nodes-snap').checked = true;
  for (const step of ['shape', 'snap']) {
    $('nodes-'+step).disabled = !reference;
    if (!reference) $('nodes-'+step).checked = false;
  }
  if (!reference) { $('nodes-detail').checked = false; $('nodes-simplify').checked = true; }
  // With a reference the error budget decides and the tolerance is a cap;
  // without one the tolerance decides alone, so its default is tighter.
  const tolerance = $('nodes-tolerance');
  if (tolerance.value === (reference ? '1' : '3')) tolerance.value = reference ? '3' : '1';
  $('nodes-reference-caption').textContent = reference ? 'Reference' : 'Original';
  $('nodes-description').textContent = reference
    ? 'Clean up paths against the reference. Preview the result before applying.'
    : 'Simplify paths while keeping their shape. Add a reference to enable snapping and fitting.';
  syncNodeSteps();
  nodesDialog.open(targets ? selectionSummary() : 'The paths painting in view.');
}

function simplifyBounds() {
  const points = [];
  for (const id of state.selection.objects) {
    const element = svgElement(id), matrix = element && localToOverlay(element);
    if (!matrix) continue;
    const b = element.getBBox();
    for (const [x,y] of [[b.x,b.y],[b.x+b.width,b.y],[b.x+b.width,b.y+b.height],[b.x,b.y+b.height]]) points.push(new DOMPoint(x,y).matrixTransform(matrix));
  }
  if (!points.length) return state.bounds;
  const left=Math.min(...points.map(p=>p.x)),right=Math.max(...points.map(p=>p.x)),top=Math.min(...points.map(p=>p.y)),bottom=Math.max(...points.map(p=>p.y));
  const padding=Math.max(8,Math.max(right-left,bottom-top)*.06);
  return [left-padding,top-padding,Math.max(1,right-left)+2*padding,Math.max(1,bottom-top)+2*padding];
}
const contactDialog = jobDialog('contact', {
  start: () => ({action:'snap', method:'edges', bounds:simplifyBounds(), permissions:{geometry:true, structure:true},
    settings:{tolerance:Number($('contact-distance').value)}}),
  describe: ({changed, metrics}) => changed
    ? `${metrics.edges} touching edge spans. Apply snaps their nodes and curve handles together as one undoable edit.`
    : metrics.edges ? 'These edges already meet exactly.' : 'No touching edges within this distance. Try a larger contact distance.',
  applied: 'Edges snapped. The paths stay independent: editing one no longer moves the other.',
}).wire();
async function openSnapEdges() { await queue; contactDialog.open(); }
// Changing a setting invalidates the preview shown for the old one.
for (const [prefix, ids] of [['contact', ['contact-distance']]]) {
  for (const id of ids) $(id).addEventListener('input', () => {
    $(prefix+'-apply').hidden = true; $(prefix+'-previews').hidden = true;
  });
}

// Generate: new shapes from the reference, placed as one group.
const generateSettings = {
  cel: () => ({regions:Number($('cel-regions').value), tolerance:Number($('cel-tolerance').value), line_width:Number($('cel-line-width').value), strokes:$('cel-strokes').checked, outline:$('cel-outline').checked, fit_colours:$('cel-fit-colours').checked, gradients:$('cel-gradients').checked}),
  'colour-regions': () => {
    const outlines = $('regions-outlines').value;
    return {colours:Number($('regions-colours').value), min_pixels:Number($('regions-min-pixels').value), tolerance:Number($('regions-tolerance').value),
      preserve_outlines:outlines !== 'none', outline_style:outlines === 'none' ? 'preserve' : outlines, geometry_cleanup:$('regions-cleanup').checked};
  },
};
function showGenerateMethod() {
  for (const panel of document.querySelectorAll('[data-generate]')) panel.hidden = panel.dataset.generate !== $('generate-method').value;
}
$('generate-method').onchange = showGenerateMethod;
const generateDialog = jobDialog('generate', {
  start: () => {
    const name = $('generate-method').value;
    return {action:'generate', method:name, scope:$('generate-scope').value, permissions:{structure:true}, settings:generateSettings[name]()};
  },
  describe: ({changed, metrics}) => changed ? `${metrics.shapes.toLocaleString()} shapes · reference error ${errorChange(metrics)}. Apply adds them as one undoable edit.` : 'Nothing was generated. Try other settings.',
  applied: 'Generated shapes added. Undo removes them.',
  choiceLabel: (result, index) => `Drawing ${index + 1} · error ${result.metrics.after.error.toFixed(5)}`,
}).wire();
async function openGenerate() {
  await queue;
  const group = oneObject()?.tag === 'g';
  $('generate-scope').value = group ? 'selection' : 'drawing';
  $('generate-scope').options[1].disabled = !group;
  showGenerateMethod(); generateDialog.open('');
}

// A dialog around one operation job: start, poll, stop, preview, choose, apply.
// Elements are found by id as `${prefix}-name`; missing optional ones are skipped.
function jobDialog(prefix, {start, describe, applied, choiceLabel = null}) {
  let context = null, poll = null, runLabel = '';
  const show = id => $(prefix+'-'+id);
  const fail = error => { show('error').textContent = error.message; show('error').hidden = false; };
  const ready = () => { show('settings').disabled = false; show('run').disabled = false; show('stop').hidden = true; };
  // A run that failed leaves nothing to discard; show only the error.
  const failedRun = error => { show('progress').hidden = true; show('close').textContent = 'Cancel'; fail(error); ready(); };
  function choose(index) {
    const result = context?.results[index]; if (!result) return;
    context.choice = index;
    for (const [key, url] of Object.entries(result.previews)) { const img = show(key); if (img) img.src = url; }
    show('metrics').textContent = describe(result);
    show('apply').hidden = !result.changed;
  }
  function showChoices() {
    const box = show('choices'); if (!box) return;
    box.replaceChildren();
    const results = context.results;
    if (!choiceLabel || results.length < 2) return;
    results.forEach((result, index) => {
      if (index && !result.changed) return;
      const label = document.createElement('label'); label.className = 'toggle';
      const input = document.createElement('input');
      input.type = 'radio'; input.name = prefix+'-choice'; input.checked = index === 0;
      input.onchange = () => choose(index);
      label.append(input, ' '+choiceLabel(result, index));
      box.append(label);
    });
  }
  async function refresh() {
    const current = context; if (!current?.job) return;
    try {
      const job = await operation('status', {job:current.job, preview:true});
      if (current !== context) return;
      show('meter').max = job.steps || 1; show('meter').value = job.step; show('status').textContent = job.message;
      if (job.status === 'running') { poll = setTimeout(refresh, 700); return; }
      ready();
      if (job.status === 'failed') throw new Error(job.error);
      if (job.status !== 'ready') return;
      current.results = [job.result, ...(job.alternatives || [])];
      showChoices(); choose(0);
      show('previews').hidden = false;
      show('close').textContent = current.results.some(result => result.changed) ? 'Discard' : 'Close';
      show('run').textContent = `${runLabel} again`;
    } catch (error) { if (current === context) failedRun(error); }
  }
  return {
    open(summary) {
      context = {epoch:state.epoch, revision:state.revision, job:null, results:[], choice:0};
      if (show('summary') && summary !== undefined) show('summary').textContent = summary;
      for (const id of ['error','previews','progress','apply','stop']) show(id).hidden = true;
      show('run').textContent = runLabel; show('close').textContent = 'Cancel'; ready();
      show('dialog').showModal();
    },
    async run() {
      const current = context; if (!current) return;
      for (const id of ['error','apply','previews']) show(id).hidden = true;
      show('settings').disabled = true; show('run').disabled = true;
      // Quick methods finish inside the start request; say so while it runs.
      show('progress').hidden = false; show('meter').removeAttribute('value'); show('status').textContent = 'Working…';
      try {
        if (current.job) { await operation('discard', {job:current.job}); current.job = null; }
        const job = await operation('start', {epoch:current.epoch, revision:current.revision, selection:current.selection, ...start()});
        if (current !== context) { await operation('discard', {job:job.id}); return; }
        current.job = job.id; show('progress').hidden = false; show('stop').hidden = job.status !== 'running';
        show('close').textContent = 'Cancel & discard';
        await refresh();
      } catch (error) { failedRun(error); }
    },
    wire() {
      runLabel = show('run').textContent;
      show('run').onclick = () => this.run();
      show('stop').onclick = async () => { try { await operation('stop', {job:context.job}); show('stop').hidden = true; } catch (error) { fail(error); } };
      show('apply').onclick = async () => {
        show('apply').disabled = true;
        try { const result = await operation('apply', {job:context.job, choice:context.choice}); context.job = null; dirty = true; await applyState(result); show('dialog').close(); toast(applied); }
        catch (error) { fail(error); } finally { show('apply').disabled = false; }
      };
      show('close').onclick = () => show('dialog').close();
      show('dialog').addEventListener('close', () => {
        clearTimeout(poll); const job = context?.job; context = null;
        if (job) operation('discard', {job}).catch(error => toast(error.message, true));
      });
      return this;
    },
  };
}
function errorChange(metrics, key = 'error') {
  const before = metrics.before[key], after = metrics.after[key];
  const change = before > 0 ? 100*(before-after)/before : 0;
  return change >= 0 ? `reduced ${change.toFixed(1)}%` : `increased ${(-change).toFixed(1)}%`;
}
const selectionSummary = () => oneObject()?.label || `${state.selection.objects.length} selected objects`;
const coloursDialog = jobDialog('colours', {
  start: () => ({action:'improve', method:'colours', permissions:{paint:true},
    settings:{passes:Number($('colours-passes').value), resolution:Number($('colours-resolution').value), fill:$('colours-fill').value}}),
  describe: result => {
    const m = result.metrics, gradients = m.gradients ? ` (${m.gradients} as ${m.gradients === 1 ? 'a gradient' : 'gradients'})` : '';
    return result.changed ? `${m.objects} of ${m.considered} fills changed${gradients} · reference error ${errorChange(m)}. Apply keeps it as one undoable edit.` : 'The colours already fit the reference.';
  },
  applied: 'Fills fitted. Undo restores the previous fills.',
}).wire();
// Fit colours and Fit gradient are one dialog, opened with its fill kind chosen.
const showColoursFill = () => { $('colours-title').textContent = $('colours-fill').value === 'linear' ? 'Fit gradient' : 'Fit colours'; };
$('colours-fill').onchange = showColoursFill;
async function openColours(fill = 'flat') { await queue; $('colours-fill').value = fill; showColoursFill(); coloursDialog.open(selectionSummary()); }
const cleanupDialog = jobDialog('cleanup', {
  start: () => ({action:'simplify', method:'cleanup', bounds:simplifyBounds(), permissions:{geometry:true, structure:true}}),
  describe: result => {
    const c = result.metrics.cleanup;
    return result.changed ? `${c.paths_before} → ${c.paths_after} paths (${c.paths_merged} merged, ${c.duplicate_paths_removed + c.empty_paths_removed} removed) · ${c.vertices_removed} redundant vertices removed. Apply keeps it as one undoable edit.` : 'Nothing to clean up in the selection.';
  },
  applied: 'Geometry cleaned up. Undo restores the original paths.',
}).wire();
async function openCleanup() { await queue; cleanupDialog.open(selectionSummary()); }
