import {pathEndpoints, snapIndex, snapPoint} from './snap.js';
import {dropIndex, dropRefusal, dropTarget} from './tree.js';
import {attach, contourLines, stretch} from './redraw.js';
import {TOOL_LEVEL, boxSelect, clickPoint, dragBox, escapeStep, pickTarget, pointInside, pointKey, rectInside, scopeChain, selectionStatus, splitKey, switchTool} from './selection.js';
const $ = id => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
let state, session, tool = 'select', zoom = 1, pan = {x: 0, y: 0}, drag = null;
let reference = null;
// Point tools show the points of every selected path: their geometry at this
// revision, the unselected path under the pointer, the point last clicked,
// and the points hidden while an object tool is active.
let geometries = new Map(), hoverPath = null, focusPoint = null, pointMemory = null;
// The group entered by double-clicking it: object tools then pick within it.
let scope = null;
let clickCycle = null, lastPick = null;
let pathDraft = [], pathHover = null;
let joinContext = null;
let holePlan = null, chosenHoles = new Set(), chosenCleanup = new Set();
let nodeHoles = null;
let redrawHover = null;
const SNAP_RADIUS = 8;
let pending = 0, queue = Promise.resolve(), dirty = false, space = false, toastTimer;
const drawing = $('drawing'), overlay = $('overlay'), stage = $('stage');
const names = {select: 'Select', nodes: 'Nodes', path: 'Draw path', knife: 'Knife', redraw: 'Redraw outline', hand: 'Pan'};
const hints = {select: 'Click to select · Double-click a group to enter it, a path to edit its points · Drag elsewhere to box select, a selected object to move it', nodes: 'Click or box select points, Shift to add · Drag to move them · Alt or Ctrl/⌘ drags without snapping', path: 'Click for corners · Drag for curves · Click the first point to close · Enter finishes', knife: 'Drag a line across selected shapes to cut them · Shift snaps to 15°', redraw: 'Draw along the reference edge from the selected path\'s outline back to it · Shift replaces the longer way round · Escape cancels', hand: 'Drag to pan · Scroll to zoom'};

function toast(message, error = false) {
  clearTimeout(toastTimer); $('toast-message').textContent = message;
  $('toast').classList.toggle('error', error); $('toast').hidden = false;
  if (!error) toastTimer = setTimeout(() => $('toast').hidden = true, 4500);
}
$('toast-close').onclick = () => $('toast').hidden = true;
function setBusy(label, delta) {
  pending += delta; $('busy').hidden = pending === 0;
  $('hole-controls').disabled = pending > 0;
  if (label) $('busy-label').textContent = label;
  document.body.setAttribute('aria-busy', String(pending > 0));
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
  queue = queue.then(async () => {
    if (command !== 'select') clickCycle = null;
    setBusy(label, 1);
    try {
      const result = await request('/api/action', {command, ...data, epoch: state.epoch, revision: state.revision});
      if (command !== 'select') dirty = true;
      await applyState(result);
      return true;
    } catch (error) {
      toast(error.message, true);
      // Discard optimistic dragging even when the backend rejects the command.
      if (state?.svg) renderDrawing();
      renderInspector(); drawOverlay();
      return false;
    } finally { setBusy('', -1); }
  });
  return queue;
}
function svgElement(id) { return drawing.querySelector(`[data-object-id="${CSS.escape(id)}"]`); }
function object(id) { return state?.objects.find(item => item.id === id); }
function oneObject() { return state?.selection.objects.length === 1 ? object(state.selection.objects[0]) : null; }
function xmlElement(name, attrs = {}) {
  const element = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
  return element;
}
function renderDrawing() {
  const parsed = new DOMParser().parseFromString(state.svg, 'image/svg+xml');
  const root = document.importNode(parsed.documentElement, true);
  // Isolate drawing IDs from editor controls while keeping local SVG references.
  for (const element of [root, ...root.querySelectorAll('*')]) {
    if (element.id) { element.dataset.objectId = element.id; element.id = `art-${element.id}`; }
    for (const attribute of [...element.attributes]) {
      if (attribute.localName === 'href' && attribute.value.startsWith('#')) {
        element.setAttributeNS(attribute.namespaceURI, attribute.name, `#art-${attribute.value.slice(1)}`);
      } else if (attribute.localName === 'clip-path' && attribute.value.startsWith('url(#')) {
        element.setAttribute('clip-path', `url(#art-${attribute.value.slice(5,-1)})`);
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
  if (changed) { geometries = new Map(); clickCycle = null; lastPick = null; }
  if (scope && object(scope)?.tag !== 'g') scope = null;
  if (holePlan && (changed || oneObject()?.id !== holePlan.object)) holePlan = null;
  renderObjects(); renderInspector();
  if (level() === 'points') await loadGeometries();
  renderInspector(); drawOverlay();
}
const level = () => TOOL_LEVEL[tool];
const parents = () => new Map(state.objects.map(item => [item.id, item.parent]));
// The paths whose points the point tools show and edit: the selected ones.
function pointPaths() {
  return state.selection.objects.filter(id => { const item = object(id); return item?.tag === 'path' && !item.resource; });
}
function geometryNodes(id) { return geometries.get(id)?.subpaths.flatMap(s => s.nodes) || []; }
function nodeAt(key) { const [id, node] = splitKey(key); return geometryNodes(id).find(n => n.id === node); }
function contourAt(key) { const [id, node] = splitKey(key); return geometries.get(id)?.subpaths.find(s => s.nodes.some(n => n.id === node)); }
// The selected points, from the node ids the server holds, while a point tool
// is active: in an object tool they are hidden.
function selectedPoints() {
  if (level() !== 'points') return [];
  const nodes = new Set(state.selection.nodes), keys = [];
  if (!nodes.size) return keys;
  for (const id of pointPaths()) for (const node of geometryNodes(id)) if (nodes.has(node.id)) keys.push(pointKey(id, node.id));
  return keys;
}
function selectPoints(objects, keys, focus = keys.at(-1)) {
  focusPoint = focus ?? null;
  return action('select', {objects: [...new Set(objects)], nodes: [...new Set(keys.map(key => splitKey(key)[1]))]}, 'Selecting…');
}
async function loadGeometries() {
  const wanted = level() === 'points' ? [...pointPaths(), ...(hoverPath ? [hoverPath] : [])] : [];
  const missing = [...new Set(wanted)].filter(id => !geometries.has(id));
  if (!missing.length) return;
  const {epoch, revision} = state;
  const result = await request('/api/nodes', {objects: missing, epoch, revision});
  if (state.epoch !== epoch || state.revision !== revision) return;
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
function colorHex(value) {
  const rgb = value.match(/^rgba?\(([^)]+)\)$/i);
  if (!rgb) return /^#[a-f\d]{6}$/i.test(value) ? value : null;
  const channels = rgb[1].split(/[, /]+/).map(Number);
  if (channels.length < 3 || channels.some(n => !Number.isFinite(n))) return null;
  return '#' + channels.slice(0,3).map(n=>Math.round(n).toString(16).padStart(2,'0')).join('') +
    (channels.length > 3 && channels[3] < 1 ? Math.round(channels[3]*255).toString(16).padStart(2,'0') : '');
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
      .filter(el=>!el.closest('defs,clipPath')).map(el=>{const p=swatchPaint(el);return p.fill !== 'none' ? p.fill : p.stroke;}).filter(c=>c !== 'none'))];
    const shown = colors.slice(0,4);
    swatch.classList.add('group-swatch');
    swatch.style.background = shown.length === 1 ? shown[0] : shown.length ? `conic-gradient(${shown.map((c,i)=>`${c} ${i*100/shown.length}% ${(i+1)*100/shown.length}%`).join(',')})` : 'transparent';
    swatch.title = colors.length ? `Group colors: ${shown.map(display).join(', ')}${colors.length > shown.length ? ` (+${colors.length-shown.length} more)` : ''}` : 'No paint';
  } else {
    swatch.classList.toggle('no-paint', paint.fill === 'none' && paint.stroke === 'none');
    swatch.style.backgroundColor = paint.fill === 'none' ? 'transparent' : paint.fill;
    if (paint.stroke !== 'none') { swatch.style.borderColor = paint.stroke; swatch.style.borderWidth = '3px'; }
    swatch.title = `Fill: ${display(paint.fill)} (${paintSource(item.id,'fill')}) · Stroke: ${display(paint.stroke)} (${paintSource(item.id,'stroke')})`;
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
  else if (item.tag === 'clipPath') role = 'Clipping boundary · not drawn';
  else if (item.tag === 'use') role = `${inClip ? 'Clip contour' : item.resource ? 'Shared instance' : 'Instance'} of ${source?.label || 'missing source'}`;
  else if (inClip) role = 'Clip contour · not drawn';
  else if (item.resource) role = 'Shared geometry · not drawn';
  else if (clip) role = `Clipped by ${clip.label}`;
  return {source, clip, inClip, role};
}
function renderObjects() {
  const search = $('object-search').value.toLowerCase();
  const fragment = document.createDocumentFragment();
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
    row.classList.toggle('selected', state.selection.objects.includes(item.id));
    row.classList.toggle('resource', item.resource); row.setAttribute('role', 'option');
    row.setAttribute('aria-selected', String(state.selection.objects.includes(item.id)));
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
    row.onclick = event => { if (!treeDragEnded) selectObject(item.id, event.shiftKey || event.ctrlKey || event.metaKey, true); };
    row.ondblclick = () => { if (!item.resource) queue.then(() => enterObject(item.id)); };
    row.onpointerdown = event => pressTreeRow(event, item);
    fragment.append(row);
  }
  $('objects').replaceChildren(fragment, treeDropLine); $('object-count').textContent = count;
  if (treeDrag?.active) showTreeDrop();
}
// Dragging rows in the tree restacks them, or moves them into a group. The
// drag carries the whole selection when it starts on a selected row.
const TREE_INDENT = 9;
const treeDropLine = document.createElement('div'); treeDropLine.className = 'tree-drop-line'; treeDropLine.hidden = true;
let treeDrag = null, treeDragEnded = false;
function pressTreeRow(event, item) {
  if (event.button !== 0 || pending || item.resource) return;
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
    treeDrag = {...treeDrag, active: true, ids: new Set(ids)};
    $('objects').classList.add('dragging');
  }
  const list = $('objects'), box = list.getBoundingClientRect();
  // Scroll while the pointer is near the list's top or bottom edge.
  if (event.clientY < box.top + 24) list.scrollTop -= 12;
  else if (event.clientY > box.bottom - 24) list.scrollTop += 12;
  const rows = treeRows();
  const target = dropTarget(rows, event.clientX - (rows[0]?.left ?? box.left) - TREE_INDENT, event.clientY, TREE_INDENT);
  const parents = new Map(state.objects.map(item => [item.id, item.parent]));
  const resources = new Set(state.objects.filter(item => item.resource).map(item => item.id));
  treeDrag.target = target;
  treeDrag.refusal = dropRefusal(target, treeDrag.ids, parents, resources);
  showTreeDrop();
}
function endTreeDrag(drop) {
  const finished = treeDrag;
  treeDrag = null; treeDropLine.hidden = true;
  $('objects').classList.remove('dragging', 'drop-refused'); $('objects').title = '';
  for (const row of $('objects').querySelectorAll('.drop-into, .dragged')) row.classList.remove('drop-into', 'dragged');
  if (!finished?.active) return;
  // The pointer is released over a row: that is not a click on it.
  treeDragEnded = true; setTimeout(() => treeDragEnded = false);
  if (!drop || finished.refusal) { if (drop) toast(finished.refusal, true); return; }
  const {target, ids} = finished;
  const children = state.objects.filter(item => item.parent === target.parent).map(item => item.id);
  action('move_objects', {objects: [...ids], parent: target.parent, index: dropIndex(target, children, ids)}, 'Moving objects…');
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
    button.onclick = () => selectObject(target.id, false, true);
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
// Disable a button with the reason as its tooltip, or enable it with its own.
function enable(id, reason) {
  const button = $(id);
  button.dataset.title ??= button.title;
  button.disabled = !!reason;
  button.title = reason || button.dataset.title;
}
function renderInspector() {
  const selected = state.selection.objects, item = oneObject();
  $('object-name-section').hidden = !item;
  $('object-name').value = item?.name || '';
  $('object-name').placeholder = item?.label || 'Automatic name';
  $('object-name').dataset.objectId = item?.id || '';
  const editingNodes = tool === 'nodes';
  $('inspector-title').textContent = editingNodes ? 'Edit nodes' : 'Inspector';
  $('node-inspector').hidden = !editingNodes;
  $('empty-inspector').hidden = editingNodes || !!selected.length;
  $('object-inspector').hidden = editingNodes || !selected.length;
  $('selection-kind').textContent = item?.tag || (selected.length ? 'Multiple' : 'Drawing');
  $('selected-id').textContent = item?.label || `${selected.length} objects`;
  $('selection-status').textContent = selected.length ? `${selected.length} selected` : 'Nothing selected';
  renderNodeInspector();
  const paths = selected.every(id => object(id)?.tag === 'path' && !object(id)?.resource);
  const noReference = !state.reference && 'Add a reference image first (Reference, left panel)';
  enable('snap-edges', selected.length < 2 ? 'Select two or more paths' : !paths && 'Every object must be a visible path');
  $('empty-reference-hint').hidden = !!state.reference;
  if (editingNodes) return;
  renderRelationships(item);
  enable('nodes-open', !selected.some(id => ['path','g'].includes(object(id)?.tag)) && 'Select one or more paths, or groups that contain them');
  enable('llm-open', noReference);
  enable('colours-open', noReference);
  enable('retrace', noReference || ((!selected.length || !paths) && 'Select one or more visible paths'));
  $('optimize-hint').textContent = state.reference ? 'Compare with the reference image. Retrace applies at once as one undoable edit; the other tools show a preview first.' :'These tools compare the drawing with a reference image. Add one under Reference on the left to use them.';
  // Empty selections have no paint to resolve; keep the remaining controls reset.
  for (const kind of selected.length ? ['fill', 'stroke'] : []) {
    const value = paintValue(kind, kind === 'fill' ? 'black' : 'none');
    const hex = colorHex(value);
    $(`${kind}-value`).value = hex || value;
    $(`${kind}-value`).placeholder = selected.length ? 'Mixed colors' : '';
    const picker = $(`${kind}-color`);
    picker.value = hex?.slice(0,7) || '#000000';
    picker.classList.toggle('no-paint', value === 'none');
    picker.classList.toggle('mixed-paint', !value);
    const sources = [...new Set(selected.map(id=>paintSource(id,kind)))];
    $(`${kind}-source`).textContent = !value ? 'Mixed colors' : sources.length === 1 ? sources[0] : 'Different paint sources';
    picker.title = `${kind === 'fill' ? 'Fill' : 'Stroke'}: ${hex || value || 'mixed'} · ${$(`${kind}-source`).textContent}`;

  }
  const strokeWidth = paintValue('stroke-width', '1');
  $('stroke-width').value = strokeWidth ? parseFloat(strokeWidth) : '';
  const opacity = paintValue('opacity', '1'); $('opacity').value = opacity === '' ? '' : Math.round(Number(opacity) * 100);
  document.querySelectorAll('[data-lock]').forEach(input => {
    input.disabled = !item; input.checked = item?.locks.includes(input.dataset.lock) || false;
    input.parentElement.title = item ? '' : 'Select one object to change its locks';
  });
  enable('group', selected.length < 2 && 'Select two or more objects to group');
  enable('ungroup', !selected.every(id => object(id)?.tag === 'g') && 'Select one or more groups');
  for (const id of ['backward', 'forward']) enable(id, !item && 'Select one object to restack');
  for (const id of ['to-back', 'to-front']) enable(id, selected.some(id => object(id)?.resource) && 'Definitions and clipping boundaries keep their place');
  enable('join_paths', joinCandidates().length < 2 && 'Select at least two paths, or groups that contain them');
  enable('split_disconnected', item?.tag === 'use' ? 'Detach this instance to an editable path first' : !paths && 'Only visible paths can be split');
  enable('cut_hole', (selected.length !== 2 || !paths) && 'Select two visible paths, one inside or overlapping the other');
  enable('detach', (!item || !['path', 'use'].includes(item.tag)) && 'Select one path or instance');
  $('detach').textContent = item?.tag === 'use' ? 'Detach to editable path' : 'Detach shared geometry';
  $('hole-section').hidden = !item || item.tag !== 'path' || item.resource;
  renderHoles();
}
async function selectObject(id, additive = false, focus = false) {
  clickCycle = null;
  if (pending) return;
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
  zoom = Math.max(.005, Math.min(32, Math.max(1,w-80)/Math.max(60,(right-left)*1.2), Math.max(1,h-80)/Math.max(60,(bottom-top)*1.2)));
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
  const cycled = !finished.shift && sameClickSpot(finished.x, finished.y, hits);
  const index = cycled ? (clickCycle.index+1)%hits.length : 0;
  // A double-click acts on what its first click picked, not on the next
  // shape its second click cycled to.
  lastPick = {id: hits[index], first: cycled ? hits[clickCycle.index] : hits[index]};
  const success = await selectObject(hits[index], finished.shift);
  if (success && !finished.shift) clickCycle = {x:finished.x, y:finished.y, hits, index};
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
function renderNodeInspector() {
  const item = oneObject(), paths = pointPaths(), points = selectedPoints();
  const nodes = paths.flatMap(geometryNodes), loaded = paths.every(id => geometries.has(id));
  const single = points.length === 1 ? points[0] : null, node = single && nodeAt(single);
  const chosen = points.map(nodeAt).filter(Boolean);
  $('node-path-name').textContent = paths.length > 1 ? `${paths.length} paths` : item?.label || (state?.selection.objects.length ? 'Multiple objects selected' : 'No path selected');
  const contours = paths.reduce((total, id) => total + (geometries.get(id)?.subpaths.length || 0), 0);
  $('node-path-stats').textContent = paths.length && loaded ? `${nodes.length.toLocaleString()} points · ${contours.toLocaleString()} ${contours===1?'contour':'contours'}` : '';
  $('node-detach').hidden = item?.tag !== 'use';
  $('node-properties').hidden = !chosen.length;
  $('node-count').textContent = nodes.length ? nodes.length.toLocaleString() : '';
  $('node-hint').textContent = item?.tag === 'use' ? 'This is a shared instance. Detach it to edit its points independently.' : !paths.length ? 'Select one or more paths on the canvas or in the object tree.' : !loaded ? 'Loading path nodes…' : chosen.length > 1 ? `${chosen.length} points in ${new Set(points.map(key => splitKey(key)[0])).size} paths. Drag one to move them together.` : node ? (node.pinned ? 'This endpoint is pinned. Unpin it to move or delete it.' : node.command === 'C' ? 'Drag the blue handles to adjust this curve.' : 'Drag this point or enter its coordinates below.') : 'Click a point, or drag a box around several.';
  if (chosen.length) {
    $('node-type').textContent = !node ? `${chosen.length} points` : node.command === 'M' ? 'Start point' : node.command === 'C' ? 'Curve endpoint' : 'Line endpoint';
    $('node-coordinates').hidden = !node;
    if (node) { $('node-x').value = +node.values.at(-2).toFixed(4); $('node-y').value = +node.values.at(-1).toFixed(4); }
    for (const id of ['node-x', 'node-y', 'node-apply']) $(id).disabled = !node || node.pinned;
    const pinned = chosen.filter(n => n.pinned).length;
    $('node-pin').checked = pinned === chosen.length; $('node-pin').indeterminate = pinned > 0 && pinned < chosen.length;
    enable('node-delete', pinned && 'Unpin the points to delete them');
    enable('node-delete-contour', points.some(key => contourAt(key)?.nodes.some(n => n.pinned)) && 'Unpin the contour\'s points to delete it');
    enable('node-split', points.every(key => nodeAt(key)?.command === 'M' && !contourAt(key)?.closed) && 'A start point has no edge leading into it');
  }
  // Points on holes of one path offer to fill the holes or make them shapes.
  const holes = holeContours(points);
  if (holes === undefined) loadNodeHoles(points);
  $('node-hole').hidden = !holes;
}
// The hole contours the selected points are on, when they are all on holes of
// one path: null when they are not, undefined while that is being found out.
function holeContours(points) {
  const objects = new Set(points.map(key => splitKey(key)[0]));
  if (!points.length || objects.size !== 1) return null;
  const [id] = objects;
  if (nodeHoles?.key !== holesKey(id)) return undefined;
  const contours = new Set(points.map(key => contourAt(key)?.id));
  return nodeHoles.ids && [...contours].every(c => nodeHoles.ids.has(c)) ? {object: id, holes: [...contours]} : null;
}
function holesKey(id) { return `${state.epoch}:${state.revision}:${id}`; }
async function loadNodeHoles(points) {
  const id = splitKey(points[0])[0], key = holesKey(id);
  if (nodeHoles?.key === key) return;
  nodeHoles = {key, ids: null};
  let ids = new Set();
  try {
    const result = await request('/api/holes', {object: id, epoch: state.epoch, revision: state.revision});
    ids = new Set(result.holes.map(h => h.id));
  } catch { /* A path whose holes cannot be read simply offers none. */ }
  if (nodeHoles.key !== key) return;
  nodeHoles.ids = ids; renderNodeInspector();
}
$('node-hole-fill').onclick=async()=>{
  const holes = holeContours(selectedPoints()); if (!holes) return;
  if (await action('fill_holes',{object:holes.object,holes:holes.holes},'Filling hole…')) toast(`Filled the ${holes.holes.length === 1 ? 'hole' : 'holes'}. Undo restores ${holes.holes.length === 1 ? 'it' : 'them'}.`);
};
$('node-hole-shape').onclick=async()=>{
  const holes = holeContours(selectedPoints()); if (!holes) return;
  if (await action('holes_to_shapes',{object:holes.object,holes:holes.holes},'Making a shape…')) toast('The hole is now its own shape, just above the path. Undo restores the hole.');
};
$('node-select-tool').onclick=()=>setTool('select');
$('node-detach').onclick=()=>action('detach');

function localToOverlay(element) {
  const from = element?.getScreenCTM(), to = overlay.getScreenCTM();
  if (!from || !to) return null;
  try { return DOMMatrix.fromMatrix(to.inverse().multiply(from)); } catch { return null; }
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
  overlay.replaceChildren();
  // The artboard zoom is a CSS transform, outside SVG's non-scaling-stroke.
  overlay.style.setProperty('--selection-scale', 1 / zoom);
  if (!state) return;
  const targets = new Set(state.selection.objects.flatMap(selectionTargets));
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
        overlay.append(halo, group);
      }
      const b = element.getBBox();
      const pts = [[b.x,b.y], [b.x+b.width,b.y], [b.x+b.width,b.y+b.height], [b.x,b.y+b.height]].map(([x,y]) => new DOMPoint(x,y).matrixTransform(matrix));
      overlay.append(xmlElement('polygon', {points: pts.map(p => `${p.x},${p.y}`).join(' '), class: 'selection-box'}));
    } catch { /* Resource elements may have no display bounds. */ }
  }
  drawHoles();
  drawPathDraft();
  drawKnife();
  drawRedraw();
  drawSnap();
  drawBox();
  if (!holePlan && level() === 'points') drawPoints();
  renderStatus();
}
// The points of every selected path, and of the unselected path under the
// pointer in Nodes, faintly, so a click on one adds its path.
function drawPoints() {
  const stageBox = stage.getBoundingClientRect(), chosen = new Set(selectedPoints());
  const selected = new Set(state.selection.objects), editable = tool === 'nodes';
  const paths = [...pointPaths(), ...(editable && hoverPath && !selected.has(hoverPath) ? [hoverPath] : [])];
  // Limit handles in dense drawings by screen-space spacing, without dropping
  // geometry. Zooming in exposes the original nodes at their full resolution.
  const occupied = new Set(); let shown = 0, total = 0;
  for (const id of paths) {
    const element = svgElement(id), matrix = localToOverlay(element), screen = element?.getScreenCTM();
    if (!matrix || !screen || !geometries.has(id)) continue;
    const ghost = !selected.has(id);
    for (const node of geometryNodes(id)) {
      total++;
      const key = pointKey(id, node.id), picked = chosen.has(key);
      const x = node.values.at(-2), y = node.values.at(-1), pos = new DOMPoint(x,y).matrixTransform(screen);
      if (pos.x < stageBox.left || pos.x > stageBox.right || pos.y < stageBox.top || pos.y > stageBox.bottom) continue;
      const cell = `${Math.floor(pos.x/10)},${Math.floor(pos.y/10)}`;
      if (!picked && (occupied.has(cell) || shown >= 1200)) continue;
      occupied.add(cell); shown++;
      const p = new DOMPoint(x,y).matrixTransform(matrix);
      const circle = xmlElement('circle', {cx:p.x, cy:p.y, r: (picked ? 4.8 : 3.3)/zoom, class:`node${picked ? ' selected' : ''}${node.pinned ? ' pinned' : ''}${ghost ? ' ghost' : ''}${editable ? '' : ' passive'}`});
      circle.dataset.object = id; circle.dataset.node = node.id; circle.dataset.part = 'endpoint';
      overlay.append(circle);
    }
  }
  // The handles of the selected points, up to a few hundred of them.
  if (editable) for (const key of [...chosen].slice(0, 300)) drawHandles(key);
  $('node-count').textContent = `${shown.toLocaleString()} / ${total.toLocaleString()}`;
}
function drawHandles(key) {
  const [id] = splitKey(key), node = nodeAt(key), subpath = contourAt(key), matrix = localToOverlay(svgElement(id));
  if (!node || !matrix) return;
  const i = subpath.nodes.indexOf(node), handles = [];
  if (node.command === 'C') handles.push({node, offset:2, anchor:node.values.slice(-2)});
  const next = subpath.nodes[i+1];
  if (next?.command === 'C') handles.push({node:next, offset:0, anchor:node.values.slice(-2)});
  for (const handle of handles) {
    const p = new DOMPoint(...handle.node.values.slice(handle.offset, handle.offset+2)).matrixTransform(matrix);
    const anchor = new DOMPoint(...handle.anchor).matrixTransform(matrix);
    overlay.append(xmlElement('line', {x1:anchor.x,y1:anchor.y,x2:p.x,y2:p.y,class:'handle-line'}));
    const circle = xmlElement('circle', {cx:p.x,cy:p.y,r:3.8/zoom,class:'handle'});
    circle.dataset.object = id; circle.dataset.node = handle.node.id; circle.dataset.part = String(handle.offset); overlay.append(circle);
  }
}
// The rubber band of a box select, in the overlay's frame.
function drawBox() {
  if (drag?.kind !== 'box' || !drag.moved) return;
  const inverse = overlay.getScreenCTM()?.inverse(); if (!inverse) return;
  const a = new DOMPoint(drag.x, drag.y).matrixTransform(inverse), b = new DOMPoint(drag.end.x, drag.end.y).matrixTransform(inverse);
  overlay.append(xmlElement('rect', {x:Math.min(a.x,b.x), y:Math.min(a.y,b.y), width:Math.abs(a.x-b.x), height:Math.abs(a.y-b.y), class:'select-box'}));
}
// The level and count of the selection, and the entered group.
function renderStatus() {
  if (!state) return;
  const points = selectedPoints(), paths = level() === 'points' ? pointPaths() : state.selection.objects;
  $('selection-level').textContent = selectionStatus(level() === 'points' ? 'points' : 'objects', paths, points);
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
  $('path-finish').disabled=pending>0||pathDraft.length<2;
  $('path-close').disabled=pending>0||pathDraft.length<3;
  $('path-cancel').disabled=pending>0||!pathDraft.length;
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
  overlay.append(group);
}
function drawKnife() {
  if(drag?.kind!=='knife'||!drag.moved)return;
  const {start:a,end:b}=drag, line={x1:a.x,y1:a.y,x2:b.x,y2:b.y,'pointer-events':'none','aria-hidden':'true'};
  overlay.append(xmlElement('line',{...line,stroke:'#052b3a','stroke-width':4/zoom}),
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
  const snap = drag?.kind === 'node' && drag.moved ? drag.snap : null;
  if (!snap) return;
  const [x, y, w, h] = state.bounds, r = 6 / zoom;
  if (snap.target.x !== undefined) overlay.append(xmlElement('line', {x1: snap.target.x, y1: y, x2: snap.target.x, y2: y + h, class: 'snap-edge'}));
  if (snap.target.y !== undefined) overlay.append(xmlElement('line', {x1: x, y1: snap.target.y, x2: x + w, y2: snap.target.y, class: 'snap-edge'}));
  overlay.append(xmlElement('rect', {x: snap.x - r, y: snap.y - r, width: 2 * r, height: 2 * r, transform: `rotate(45 ${snap.x} ${snap.y})`, class: 'snap-target'}));
}
function knifeEnd(event) {
  const p=point(event), a=drag.start;
  if(!event.shiftKey)return {x:p.x,y:p.y};
  const step=Math.PI/12, angle=Math.round(Math.atan2(p.y-a.y,p.x-a.x)/step)*step, length=Math.hypot(p.x-a.x,p.y-a.y);
  return {x:a.x+Math.cos(angle)*length,y:a.y+Math.sin(angle)*length};
}
async function cutWithKnife({start, end}) {
  if(!state.selection.objects.length){toast('Select the shapes to cut, then drag the knife across them.',true);return;}
  if(await action('knife',{start:[start.x,start.y],end:[end.x,end.y]},'Cutting…'))
    toast(`Cut into ${state.selection.objects.length} pieces. They meet exactly along the cut; Join paths merges them again.`);
}
// Redraw outline: the stroke, where its ends attach to a selected path's
// outline and the stretch it will replace, all in the overlay's frame.
function redrawLines() {
  const lines = [];
  for (const id of pointPaths()) {
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
  overlay.append(group);
}
async function redrawOutline({points, longWay}) {
  const plan = redrawPlan(points, longWay);
  if (!redrawLines()) {toast('Select a path, then draw along the edge it should follow.', true);return;}
  if (!plan?.start || !plan.end) {toast('Start and end the stroke on the same outline of a selected path.', true);return;}
  await action('redraw_outline', {object: plan.object, points, pixel: 1/zoom, long_way: Boolean(longWay)}, state.reference ? 'Fitting to the reference…' : 'Redrawing outline…');
}
async function finishPath(closed) {
  if(pending||pathDraft.length<(closed?3:2))return;
  const draft=pathDraft; pathDraft=[];pathHover=null;
  if(await action('add_path',{d:draftPathData(draft,closed),stroke_width:2/zoom},'Drawing path…'))await setTool('select');
  else {pathDraft=draft;drawOverlay();}
}
$('path-finish').onclick=()=>finishPath(false);
$('path-close').onclick=()=>finishPath(true);
$('path-cancel').onclick=()=>{pathDraft=[];pathHover=null;drawOverlay();};
function updateView() {
  clickCycle = null;
  $('artboard').style.transform = `translate(${pan.x}px,${pan.y}px) scale(${zoom})`;
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
  const next = Math.max(.005, Math.min(32, zoom*factor)), ratio = next/zoom;
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
  if (!state || pending) return;
  const from = tool;
  const switched = switchTool({objects: state.selection.objects, points: selectedPoints(), memory: pointMemory}, from, value);
  clickCycle = null; lastPick = null; pathDraft=[]; pathHover=null; redrawHover=null; hoverPath = null; tool=value; pointMemory = switched.memory;
  if (!['select','hand'].includes(value)) holePlan = null;
  document.querySelectorAll('[data-tool]').forEach(button => button.classList.toggle('active', button.dataset.tool === tool));
  $('tool-name').textContent=names[tool]; $('canvas-hint').textContent=hints[tool];
  stage.style.cursor = tool === 'hand' ? 'grab' : ['path','knife','redraw'].includes(tool) ? 'crosshair' : 'default';
  renderInspector();
  const nodes = [...new Set(switched.points.map(key => splitKey(key)[1]))];
  if (level() === 'points') {
    setBusy('Loading path nodes…',1);
    try { await loadGeometries(); } catch(error) {toast(error.message,true);} finally {setBusy('',-1);}
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
  }
}
// Press on a point: select it (with Shift, add or remove it), adding its path
// when it is not selected, and start dragging the selected points.
function pressPoint(event, common) {
  const id = event.target.dataset.object, nodeId = event.target.dataset.node, part = event.target.dataset.part;
  const key = pointKey(id, nodeId), current = selectedPoints();
  const objects = state.selection.objects.includes(id) ? state.selection.objects : [...state.selection.objects, id];
  let points = current;
  if (part === 'endpoint') {
    points = current.includes(key) && !common.shift ? current : clickPoint(current, key, common.shift);
    if (points !== current || objects !== state.selection.objects) selectPoints(objects, points, key);
    else focusPoint = key;
  }
  // A handle moves alone; a point moves with the other selected points.
  const moving = part === 'endpoint' ? points.filter(k => nodeAt(k) && !nodeAt(k).pinned) : [key];
  if (part === 'endpoint' && (!points.includes(key) || nodeAt(key)?.pinned)) { drag = {...common, kind: 'point-click'}; renderInspector(); drawOverlay(); return; }
  const paths = new Set(moving.map(k => splitKey(k)[0]));
  const saved = new Map([...paths].filter(p => geometries.has(p)).map(p => [p, valuesById(geometries.get(p))]));
  const node = nodeAt(key), offset = part === 'endpoint' ? node.values.length - 2 : Number(part);
  const start = new DOMPoint(node.values[offset], node.values[offset+1]).matrixTransform(localToOverlay(svgElement(id)));
  drag = {...common, kind: 'node', key, part, moving, saved, start};
  renderInspector(); drawOverlay();
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
stage.addEventListener('pointerdown', event => {
  if (!state || pending || drag || ![0,1].includes(event.button)) return;
  const middle = event.button === 1;
  if (middle) event.preventDefault();
  const holeId = event.target.dataset?.hole;
  if (holeId && !middle && !space && tool !== 'hand') { toggleHole(holeId); return; }
  stage.focus({preventScroll:true}); stage.setPointerCapture(event.pointerId);
  const common = {x:event.clientX,y:event.clientY,shift:event.shiftKey || event.ctrlKey || event.metaKey,moved:false};
  const bounds = $('artboard').getBoundingClientRect();
  common.deselectOutside = !middle && !space && (event.clientX < bounds.left || event.clientX > bounds.right ||
    event.clientY < bounds.top || event.clientY > bounds.bottom);
  if (middle || space || tool === 'hand') {
    drag = {...common,kind:'pan',pan:{...pan}}; stage.classList.add('panning'); return;
  }
  if (tool === 'path' && !common.deselectOutside) {
    const p=point(event), first=pathDraft[0];
    if (first && pathDraft.length>=3 && Math.hypot(p.x-first.x,p.y-first.y)*zoom<8) {
      drag={...common,kind:'closePath'}; return;
    }
    const anchor={x:p.x,y:p.y}; pathDraft.push(anchor); pathHover=null;
    drag={...common,kind:'drawPath',anchor}; drawOverlay(); return;
  }
  if (tool === 'nodes' && event.target.dataset?.node) { pressPoint(event, common); return; }
  const hits = hitStack(event.clientX, event.clientY);
  common.hits = hits;
  const id = hits[0] || null;
  if (tool === 'knife') {
    const p=point(event);
    drag={...common,kind:'knife',id,start:{x:p.x,y:p.y},end:{x:p.x,y:p.y}}; return;
  }
  if (tool === 'redraw') {
    const p=point(event);
    drag={...common,kind:'redraw',id,points:[[p.x,p.y]],longWay:event.shiftKey}; redrawHover=null; drawOverlay(); return;
  }
  // Dragging a selected object moves the selection; a click picks what is
  // under the pointer, and any other drag selects what lies inside its box.
  if (level() === 'objects' && id) {
    const targets = clickTargets(hits), picked = targets[0];
    const selectedHit = state.selection.objects.includes(picked) ||
      (sameClickSpot(event.clientX, event.clientY, targets) && state.selection.objects.includes(targets[clickCycle.index]));
    if (selectedHit && !common.shift) {
      const members=topSelection().map(oid => ({id:oid,element:svgElement(oid),before:object(oid).attributes.transform || ''})).filter(m=>m.element);
      drag={...common,kind:'move',id,members}; return;
    }
  }
  drag = {...common, kind: 'box', id, end: {x: event.clientX, y: event.clientY}};
});
// In Nodes the unselected path under the pointer shows its points faintly.
let hoverFrame = 0;
function hoverPoints(event) {
  if (hoverFrame) return;
  const x = event.clientX, y = event.clientY;
  hoverFrame = requestAnimationFrame(async () => {
    hoverFrame = 0;
    if (tool !== 'nodes' || drag || pending) return;
    const hit = event.target.dataset?.object || hitStack(x, y).find(id => object(id)?.tag === 'path');
    const next = hit && !state.selection.objects.includes(hit) ? hit : null;
    if (next === hoverPath) return;
    hoverPath = next;
    if (next && !geometries.has(next)) { try { await loadGeometries(); } catch { return; } }
    drawOverlay();
  });
}
stage.addEventListener('pointermove', event => {
  if (!drag) {
    if (tool==='path' && pathDraft.length) {pathHover=point(event);drawOverlay();}
    if (tool==='redraw' && redrawLines()) {const p=point(event);redrawHover=[p.x,p.y];drawOverlay();}
    if (tool==='nodes' && state) hoverPoints(event);
    return;
  }
  drag.moved ||= Math.hypot(event.clientX-drag.x,event.clientY-drag.y)>3;
  if (drag.kind === 'box') { drag.end = {x: event.clientX, y: event.clientY}; if (drag.moved) drawOverlay(); }
  if (drag.kind === 'knife' && drag.moved) { drag.end=knifeEnd(event); drawOverlay(); }
  if (drag.kind === 'redraw') {
    const p=point(event), last=drag.points.at(-1);
    drag.longWay=event.shiftKey;
    if (Math.hypot(p.x-last[0],p.y-last[1])*zoom>=1.5) drag.points.push([p.x,p.y]);
    if (drag.moved) drawOverlay();
  }
  if (drag.kind === 'pan') { pan={x:drag.pan.x+event.clientX-drag.x,y:drag.pan.y+event.clientY-drag.y}; updateView(); }
  if (drag.kind === 'drawPath' && drag.moved) {
    const p=point(event), a=drag.anchor;
    a.out={x:p.x,y:p.y}; a.in={x:2*a.x-p.x,y:2*a.y-p.y}; drawOverlay();
  }
  if (drag.kind === 'node' && drag.moved) { previewPointDrag(snappedDrag(event)); drawOverlay(); renderNodeInspector(); }
  if (drag.kind === 'move' && drag.moved) {
    for (const member of drag.members) {
      const matrix=member.element.parentElement.getScreenCTM(); if (!matrix) continue;
      const inverse=matrix.inverse(), start=new DOMPoint(drag.x,drag.y).matrixTransform(inverse), end=new DOMPoint(event.clientX,event.clientY).matrixTransform(inverse);
      member.offset=[end.x-start.x,end.y-start.y];
      member.element.setAttribute('transform',`translate(${member.offset.join(' ')}) ${member.before}`);
    }
    drawOverlay();
  }
});
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
stage.addEventListener('pointerup', async event => {
  if (!drag) return;
  const finished=drag; drag=null; stage.classList.remove('panning');
  if (stage.hasPointerCapture(event.pointerId)) stage.releasePointerCapture(event.pointerId);
  if (finished.kind === 'box' && finished.moved) { drawOverlay(); await finishBox(finished); return; }
  // A click outside the artboard clears the selection in every tool.
  if (finished.deselectOutside && !finished.moved) { await clickEmpty(false); return; }
  if (['move','knife','redraw','box'].includes(finished.kind) && !finished.moved) await selectAtPoint(finished);
  if (finished.moved) clickCycle = null;
  if (finished.kind === 'knife' && finished.moved) {drawOverlay();await cutWithKnife(finished);return;}
  if (finished.kind === 'redraw' && finished.moved) {drawOverlay();await redrawOutline(finished);return;}
  if (finished.kind === 'drawPath') {pathHover=null;drawOverlay();return;}
  if (finished.kind === 'closePath') {if (!finished.moved) await finishPath(true);return;}
  if (finished.kind === 'node' && finished.moved) await finishPointDrag(finished);
  if (finished.kind === 'move' && finished.moved) await action('move',{dx:0,dy:0,offsets:Object.fromEntries(finished.members.map(m=>[m.id,m.offset||[0,0]]))},'Moving selection…');
});
// Double-click a group to pick within it, or a path to edit its points.
stage.addEventListener('dblclick', async event => {
  if (!state || level() !== 'objects' || space) return;
  await queue;
  const id = lastPick?.first, item = object(id);
  if (!item || item.resource) return;
  await enterObject(id, hitStack(event.clientX, event.clientY));
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
  } else if (item?.tag === 'use') toast('Detach this instance to edit its points (Detach in the Structure section).');
}
// Escape steps up one level: points to their paths, objects to their group,
// then to nothing, and out of an entered group.
async function stepUp() {
  const hadPoints = selectedPoints().length > 0;
  const next = escapeStep({objects: state.selection.objects, points: selectedPoints(), scope}, parents(), state.root);
  scope = next.scope; clickCycle = null; focusPoint = null;
  if (next.objects.join() !== state.selection.objects.join() || hadPoints) await action('select', {objects: next.objects}, 'Selecting…');
  // A group has no points of its own: stepping up to it leaves the point tool.
  if (!hadPoints && level() === 'points') await setTool('select');
  renderStatus();
}
stage.addEventListener('pointercancel',()=>{if(drag?.kind==='drawPath')pathDraft.pop();stage.classList.remove('panning');clickCycle=null;drag=null;geometries=new Map();if(state){renderDrawing();loadGeometries().then(drawOverlay).catch(()=>{});}drawOverlay();});
stage.addEventListener('auxclick',event=>{if(event.button===1)event.preventDefault();});
stage.addEventListener('lostpointercapture',()=>{if(drag?.kind==='pan'){drag=null;stage.classList.remove('panning');}});
stage.addEventListener('wheel',event=>{event.preventDefault();if(!state)return;const b=stage.getBoundingClientRect();zoomAt(Math.exp(-event.deltaY*.0015),event.clientX-b.left,event.clientY-b.top);},{passive:false});
new ResizeObserver(()=>{if(state)updateView();}).observe(stage);
$('fit').onclick=fit; $('zoom-in').onclick=()=>zoomAt(1.25); $('zoom-out').onclick=()=>zoomAt(.8);
document.querySelectorAll('[data-tool]').forEach(button=>button.onclick=()=>setTool(button.dataset.tool));
$('object-search').oninput=renderObjects;
$('undo').onclick=()=>action('undo',{},'Undoing…'); $('redo').onclick=()=>action('redo',{},'Redoing…');
for (const kind of ['fill','stroke']) {
  $(`${kind}-value`).onchange=event=>action('paint',{changes:{[kind]:event.target.value.trim()||null}});
  $(`${kind}-color`).onchange=event=>action('paint',{changes:{[kind]:event.target.value}});
  $(`${kind}-none`).onclick=()=>action('paint',{changes:{[kind]:'none'}});
}
$('stroke-width').onchange=event=>action('paint',{changes:{'stroke-width':event.target.value||null}});
$('opacity').onchange=event=>action('paint',{changes:{opacity:String(Number(event.target.value)/100)}});
$('move-apply').onclick=async()=>{await action('move',{dx:Number($('move-x').value),dy:Number($('move-y').value)});$('move-x').value='0';$('move-y').value='0';};
// Point commands act on every selected point, in however many paths.
function movePointTo(x, y) {
  const key = focusedPoint(), node = key && nodeAt(key); if (!node) return;
  const [id, nodeId] = splitKey(key);
  action('node', {object: id, node: nodeId, values: [...node.values.slice(0,-2), x, y]});
}
$('node-apply').onclick=()=>movePointTo(Number($('node-x').value), Number($('node-y').value));
$('node-pin').onchange=event=>action('pin',{points:pointPairs(),pinned:event.target.checked});
$('node-split').onclick=()=>action('split',{points:pointPairs()});
for(const count of [0,1,2]) $(`node-handles-${count}`).onclick=()=>action('node_handles',{points:pointPairs(),count},'Changing handles…');
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
$('join_paths').onclick = () => {
  if (pending) return;
  const candidates = joinCandidates();
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
};
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
$('split_disconnected').onclick = async () => {
  const before = state.selection.objects.length;
  if (await action('split_disconnected', {}, 'Splitting disconnected parts…')) {
    toast(state.selection.objects.length > before ? `Split into ${state.selection.objects.length} independently editable paths.` : 'No disconnected parts found. Holes and touching contours stay together.');
  }
};
function renderHoles() {
  $('inspect-holes').hidden = !!holePlan;
  $('hole-controls').hidden = !holePlan;
  if (!holePlan) return;
  $('hole-total').textContent = `(${chosenHoles.size} / ${holePlan.holes.length})`;
  $('hole-all').checked = holePlan.holes.length > 0 && chosenHoles.size === holePlan.holes.length;
  $('hole-all').indeterminate = chosenHoles.size > 0 && chosenHoles.size < holePlan.holes.length;
  $('hole-fill').disabled = !chosenHoles.size;
  $('hole-shapes').disabled = !chosenHoles.size;
  $('hole-enclosed').disabled = !chosenHoles.size;
  $('hole-fill').textContent = chosenCleanup.size ? `Fill holes & delete ${chosenCleanup.size} ${chosenCleanup.size === 1 ? 'shape' : 'shapes'}` : 'Fill selected holes';
  const list = document.createDocumentFragment();
  for (const [i, hole] of holePlan.holes.entries()) {
    const row = document.createElement('div'); row.className = 'hole-row';
    const label = document.createElement('label'); label.className = 'toggle';
    const check = document.createElement('input'); check.type = 'checkbox'; check.checked = chosenHoles.has(hole.id);
    check.onchange = () => toggleHole(hole.id);
    label.append(check, `Hole ${i+1} · ${hole.area.toLocaleString(undefined,{maximumFractionDigits:1})} px²`);
    const focus = document.createElement('button'); focus.textContent = 'View'; focus.title = `Zoom to hole ${i+1}`;
    focus.onclick = () => focusHole(hole);
    row.append(label, focus); list.append(row);
  }
  if (!holePlan.holes.length) list.append('No removable holes found.');
  const scroll = $('hole-list').scrollTop;
  $('hole-list').replaceChildren(list); $('hole-list').scrollTop = scroll;
  const enclosed = document.createDocumentFragment();
  if (holePlan.enclosedChecked && !holePlan.enclosed.length) enclosed.append('No shapes fully inside the chosen holes.');
  if (holePlan.enclosed.length) {
    const note = document.createElement('p'); note.className = 'muted'; note.textContent = 'Choose shapes to delete with the fill:'; enclosed.append(note);
    for (const id of holePlan.enclosed) {
      const label = document.createElement('label'); label.className = 'toggle';
      const check = document.createElement('input'); check.type = 'checkbox'; check.checked = chosenCleanup.has(id);
      check.onchange = () => {if(check.checked) chosenCleanup.add(id); else chosenCleanup.delete(id); renderHoles(); drawOverlay();};
      label.append(check, object(id)?.label || id); enclosed.append(label);
    }
  }
  $('hole-enclosed-list').replaceChildren(enclosed);
}
function holesChanged() {
  chosenCleanup.clear(); holePlan.enclosed = []; holePlan.enclosedChecked = false;
  renderHoles(); drawOverlay();
}
function toggleHole(id) {
  if (pending || !holePlan) return;
  if (chosenHoles.has(id)) chosenHoles.delete(id); else chosenHoles.add(id);
  holesChanged();
}
function drawHoles() {
  if (!holePlan) return;
  const matrix = localToOverlay(svgElement(holePlan.object)); if (!matrix) return;
  // Draw smallest holes last so a nested hole remains independently clickable.
  for (const hole of [...holePlan.holes].reverse()) {
    const path = xmlElement('path', {d:hole.d, transform:matrix.toString(), class:`hole-preview${chosenHoles.has(hole.id) ? ' chosen' : ''}`, 'data-hole':hole.id});
    overlay.append(path);
  }
  for (const id of chosenCleanup) {
    const element = svgElement(id), contour = selectionContour(element), matrix = localToOverlay(element);
    if (!contour || !matrix) continue;
    contour.removeAttribute('transform');
    const group = xmlElement('g',{transform:matrix.toString(),class:'hole-delete-preview'}); group.append(contour); overlay.append(group);
  }
}
function focusHole(hole) {
  const matrix = localToOverlay(svgElement(holePlan.object)); if (!matrix) return;
  const [x1,y1,x2,y2] = hole.bounds;
  const points = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]].map(p => new DOMPoint(...p).matrixTransform(matrix));
  const xs = points.map(p=>p.x), ys = points.map(p=>p.y);
  const width = Math.max(...xs)-Math.min(...xs), height = Math.max(...ys)-Math.min(...ys);
  const bounds = stage.getBoundingClientRect();
  zoom = Math.min(32, bounds.width/Math.max(60,width*2), bounds.height/Math.max(60,height*2));
  pan = {x:bounds.width/2-((Math.min(...xs)+Math.max(...xs))/2-state.bounds[0])*zoom,
         y:bounds.height/2-((Math.min(...ys)+Math.max(...ys))/2-state.bounds[1])*zoom};
  updateView();
}
$('inspect-holes').onclick = async () => {
  if (pending || !oneObject()) return;
  const id = oneObject().id; await setTool('select'); setBusy('Finding holes…',1);
  try {
    holePlan = await request('/api/holes',{object:id,epoch:state.epoch,revision:state.revision});
    chosenHoles.clear(); chosenCleanup.clear(); renderHoles(); drawOverlay();
  } catch(error) {toast(error.message,true);} finally {setBusy('',-1);}
};
$('hole-all').onchange = event => {chosenHoles = new Set(event.target.checked ? holePlan.holes.map(h=>h.id) : []); holesChanged();};
$('hole-clear').onclick = () => {chosenHoles.clear(); holesChanged();};
$('hole-small').onclick = () => {
  const max = Number($('hole-max-area').value);
  if (!Number.isFinite(max) || max < 0) {toast('Enter a non-negative area.',true);return;}
  chosenHoles = new Set(holePlan.holes.filter(h=>h.area <= max).map(h=>h.id)); holesChanged();
};
$('hole-close').onclick = () => {holePlan = null; renderHoles(); drawOverlay();};
$('hole-enclosed').onclick = async () => {
  if(pending || !chosenHoles.size) return;
  setBusy('Finding enclosed shapes…',1);
  try {
    const result = await request('/api/holes',{object:holePlan.object,holes:[...chosenHoles],find_enclosed:true,epoch:state.epoch,revision:state.revision});
    holePlan.enclosed = result.enclosed; holePlan.enclosedChecked = true; chosenCleanup.clear(); renderHoles();
  } catch(error) {toast(error.message,true);} finally {setBusy('',-1);}
};
$('hole-fill').onclick = async () => {
  if(pending || !chosenHoles.size) return;
  const count = chosenHoles.size, deleted = chosenCleanup.size;
  if(await action('fill_holes',{object:holePlan.object,holes:[...chosenHoles],delete_objects:[...chosenCleanup]},'Filling holes…'))
    toast(`Filled ${count} chosen holes${deleted ? ` and deleted ${deleted} enclosed ${deleted === 1 ? 'shape' : 'shapes'}` : ''}. Undo restores both.`);
};
$('hole-shapes').onclick = async () => {
  if(pending || !chosenHoles.size) return;
  if(await action('holes_to_shapes',{object:holePlan.object,holes:[...chosenHoles]},'Making shapes…'))
    toast(`Turned the chosen holes into ${state.selection.objects.length} ${state.selection.objects.length === 1 ? 'shape' : 'shapes'} above the path. Undo restores the holes.`);
};
$('cut_hole').onclick = async () => {
  if (await action('cut_hole', {}, 'Cutting out the hole…')) toast('Cut the shape out as a hole. Undo restores both paths.');
};
$('node-delete').onclick=()=>action('delete_node',{points:pointPairs()},'Deleting points…');
$('node-delete-contour').onclick=()=>action('delete_contour',{points:pointPairs()},'Deleting contours…');
document.querySelectorAll('[data-lock]').forEach(input=>input.onchange=()=>{const item=oneObject();if(item)action('locks',{object:item.id,locks:[...document.querySelectorAll('[data-lock]:checked')].map(el=>el.dataset.lock)});});
for (const command of ['group','ungroup','delete','detach']) $(command).onclick=()=>action(command);
$('backward').onclick=()=>action('reorder',{step:-1});$('forward').onclick=()=>action('reorder',{step:1});
$('to-back').onclick=()=>action('reorder',{to:'back'});$('to-front').onclick=()=>action('reorder',{to:'front'});
const KEY_PROVIDERS=['openai','anthropic','gemini','local'];
const keyRemovals=new Set();
const HOSTED=['openai','anthropic','gemini'];
function showKeys({api_keys, local, models, defaults}){
  keyRemovals.clear();
  $('local-url').value=local.base_url;
  $('local-model').value=local.model;
  for(const name of HOSTED){
    $(`model-${name}`).value=models[name].model;
    $(`model-${name}`).placeholder=defaults.models[name];
    $(`reasoning-${name}`).value=models[name].reasoning;
    $(`reasoning-${name}`).options[0].textContent=`Default (${defaults.reasoning})`;
  }
  for(const name of KEY_PROVIDERS){
    const tail=api_keys[name];
    $(`key-${name}`).value='';
    $(`key-${name}`).placeholder=tail?`Saved (…${tail}) · type to replace`:'Not set';
    $(`key-${name}-remove`).hidden=!tail;
  }
}
async function openSettings(){
  try{
    showKeys(await request('/api/settings'));
    $('settings-error').hidden=true;
    $('settings-dialog').showModal();
  }catch(error){toast(error.message,true);}
}
$('settings-open').onclick=openSettings;
document.querySelectorAll('[data-open-settings]').forEach(button=>button.onclick=openSettings);
for(const name of KEY_PROVIDERS) $(`key-${name}-remove`).onclick=()=>{
  keyRemovals.add(name);
  $(`key-${name}`).value='';
  $(`key-${name}`).placeholder='Removed when you save';
  $(`key-${name}-remove`).hidden=true;
};
$('settings-save').onclick=async()=>{
  const api_keys={};
  for(const name of keyRemovals) api_keys[name]='';
  for(const name of KEY_PROVIDERS){const key=$(`key-${name}`).value.trim(); if(key) api_keys[name]=key;}
  const local={base_url:$('local-url').value.trim(), model:$('local-model').value.trim()};
  const models=Object.fromEntries(HOSTED.map(name=>[name,{model:$(`model-${name}`).value.trim(), reasoning:$(`reasoning-${name}`).value}]));
  try{
    showKeys(await request('/api/settings',{api_keys, local, models}));
    $('settings-dialog').close();
    toast('Settings saved');
  }catch(error){$('settings-error').textContent=error.message;$('settings-error').hidden=false;}
};
$('settings-close').onclick=()=>$('settings-dialog').close();
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
function showReference() {
  $('reference-empty').hidden=!!reference; $('reference-controls').hidden=!reference;
  const enabled = !!reference && $('reference-visible').checked;
  const visible = enabled && reference.opacity > 0;
  $('reference-image').hidden=!enabled;
  const toggle = $('reference-toggle');
  toggle.hidden=!reference;
  toggle.setAttribute('aria-pressed', String(enabled));
  toggle.title=enabled ? 'Hide reference overlay (O)' : 'Show reference overlay (O)';
  $('reference-status').textContent=enabled ? `Reference overlay · ${Math.round(reference.opacity*100)}%` : 'Overlay hidden';
  stage.classList.toggle('reference-visible', visible);
  if(reference){$('reference-image').src=reference.data_url;$('reference-image').style.opacity=reference.opacity;$('reference-name').textContent=reference.name;$('reference-opacity').value=Math.round(reference.opacity*100);$('reference-percent').textContent=`${Math.round(reference.opacity*100)}%`;}
  else{$('reference-image').removeAttribute('src');}
}
function toggleReference() {
  if (!reference) return;
  $('reference-visible').checked = !$('reference-visible').checked;
  showReference();
}
$('reference-toggle').onclick=toggleReference;
async function loadReference(){const result=await request('/api/reference');reference=result.reference;showReference();}
for(const id of ['add-reference','add-reference-empty'])$(id).onclick=()=>$('reference-file').click();
$('reference-file').onchange=async event=>{
  const file=event.target.files[0];event.target.value='';if(!file)return;
  const reader=new FileReader();reader.onload=async()=>{const value={name:file.name,data_url:reader.result,opacity:.5};if(await action('reference',{reference:value},'Loading reference…')){reference=value;$('reference-visible').checked=true;showReference();}};reader.readAsDataURL(file);
};
$('remove-reference').onclick=async()=>{if(await action('reference',{reference:null})){reference=null;showReference();}};
$('reference-visible').onchange=showReference;
$('reference-opacity').oninput=event=>{if(reference){reference.opacity=Number(event.target.value)/100;showReference();}};
$('reference-opacity').onchange=()=>{if(reference)action('reference',{reference});};
$('object-name').onchange = event => {
  const input = event.target, item = object(input.dataset.objectId);
  if (item && input.value.trim() !== item.name) action('rename', {object:item.id, name:input.value}, 'Renaming object…');
};
$('object-name').onkeydown = event => {
  if (event.key === 'Enter') {event.preventDefault();event.target.blur();}
  if (event.key === 'Escape') {event.preventDefault();event.target.value=oneObject()?.name || '';event.target.blur();}
};
window.addEventListener('keydown',event=>{
  if(event.key==='F2' && oneObject() && !['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName) && !document.querySelector('dialog[open]')) {event.preventDefault();$('object-name').focus();$('object-name').select();return;}
  if(event.key==='Escape'&&treeDrag){event.preventDefault();endTreeDrag(false);return;}
  if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)||document.querySelector('dialog[open]'))return;
  if (tool==='path' && pathDraft.length && !pending) {
    if (event.key==='Enter') {event.preventDefault();if(!drag)finishPath(false);return;}
    if (event.key==='Backspace' || event.key==='Delete' || ((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='z')) {
      event.preventDefault();if(!drag){pathDraft.pop();pathHover=null;drawOverlay();}return;
    }
  }
  if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='s'){event.preventDefault();download(true);return;}
  if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='z'){event.preventDefault();if(!pending)action(event.shiftKey?'redo':'undo');return;}
  if(event.code==='Space'){event.preventDefault();space=true;return;}
  if(event.key==='Escape'){
    // Escape cancels what is under way, else steps the selection up a level.
    const busy = drag || pathDraft.length || holePlan;
    pathDraft=[];pathHover=null;stage.classList.remove('panning');holePlan=null;
    if (drag) {
      if (drag.saved) for (const [id, saved] of drag.saved) if (geometries.has(id)) restoreValues(geometries.get(id), saved);
      drag = null; if (state) renderDrawing();
    }
    if (state) renderInspector();
    drawOverlay();
    if (!busy && state && !pending) stepUp();
    return;
  }
  if((event.ctrlKey||event.metaKey)&&['BracketLeft','BracketRight'].includes(event.code)){
    // Ctrl/⌘ ] and [ step forward and backward; with Shift, to the front and back.
    event.preventDefault();
    const button=$(event.shiftKey?(event.code==='BracketRight'?'to-front':'to-back'):(event.code==='BracketRight'?'forward':'backward'));
    if(!pending&&state?.selection.objects.length&&!button.disabled)button.click();return;
  }
  if(event.ctrlKey||event.metaKey||event.altKey)return;
  // Shift+R retraces the selected paths; R alone is the Redraw tool.
  if(event.shiftKey&&event.key.toLowerCase()==='r'){event.preventDefault();if(!event.repeat)retraceShapes();return;}
  const tools={v:'select',n:'nodes',p:'path',k:'knife',r:'redraw',h:'hand'};const key=event.key.toLowerCase();
  if(key==='o'){event.preventDefault();if(!event.repeat)toggleReference();return;}
  if(event.key==='?'){event.preventDefault();$('help-dialog').showModal();return;}
  if(tools[key])setTool(tools[key]);if(key==='f')fit();
  // 1, 2, 3: the selected point gets no handle, one or both.
  if(tool==='nodes'&&['1','2','3'].includes(event.key)&&!pending&&selectedPoints().length){
    const button=$(`node-handles-${Number(event.key)-1}`);
    if(button&&!button.disabled){event.preventDefault();button.click();}
    return;
  }
  if((event.key==='Delete'||event.key==='Backspace')&&!pending&&state?.selection.objects.length){
    event.preventDefault();
    if(level()==='points') {
      if(selectedPoints().length && !$('node-delete').disabled) $('node-delete').click();
    } else action('delete');
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
  }
  catch(error){toast(error.message,true);}finally{setBusy('',-1);}
}
start();

// Automated operations share one job API: start, status, stop, apply, discard.
const operation = (command, body) => request('/api/operation', {command, ...body});


// Improve: Optimize nodes, a quick tidy that mixes snapping, simplifying and fitting.
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
  $('nodes-apply').hidden = true; $('nodes-previews').hidden = true;
}
const nodesDialog = jobDialog('nodes', {
  start: () => {
    const steps = nodeSteps();
    return {action:'improve', method:'nodes', scope:'selection',
      permissions:{geometry:true, structure:(steps.snap && steps.detail) || steps.simplify},
      settings:{...steps, tolerance:Number($('nodes-tolerance').value), steps:Number($('nodes-steps').value),
        movement:Number($('nodes-movement').value), workers:Number($('nodes-workers').value),
        detail_gain:Number($('nodes-detail-gain').value), gain:Number($('nodes-gain').value), margin:Number($('nodes-margin').value),
        seconds:Number($('nodes-seconds').value)},
      budget:{steps:Number($('nodes-rounds').value)}};
  },
  describe: ({changed, metrics}) => {
    if (!changed) return 'No step improved the paths within these settings. They are unchanged.';
    const points = `${metrics.before.nodes.toLocaleString()} → ${metrics.after.nodes.toLocaleString()} points`;
    const fit = metrics.reference ? `difference from the reference ${errorChange(metrics, 'difference')}` : 'the look is kept within the tolerance';
    const order = metrics.steps.map(step => STEP_NAMES[step] || step).join(' → ');
    const skipped = Object.values(metrics.skipped || {});
    const note = skipped.length ? ` Some paths were not fitted: ${[...new Set(skipped)].join('; ')}.` : '';
    const late = metrics.out_of_time ? ' Stopped at the time limit.' : '';
    return `${points} · ${fit} · ${order}.${late}${note} Apply keeps this result as one undoable edit.`;
  },
  applied: 'Paths optimized. Undo restores them.',
}).wire();
for (const step of NODE_STEPS) $('nodes-'+step).addEventListener('change', syncNodeSteps);
for (const id of ['nodes-tolerance', 'nodes-rounds', 'nodes-workers', 'nodes-steps', 'nodes-movement', 'nodes-detail-gain', 'nodes-gain', 'nodes-margin', 'nodes-seconds']) $(id).addEventListener('input', () => { $('nodes-apply').hidden = true; $('nodes-previews').hidden = true; });
$('nodes-open').onclick = async () => {
  await queue;
  const reference = Boolean(state.reference);
  // Snap comes back on with the reference, as it is by default.
  if (reference && $('nodes-snap').disabled) $('nodes-snap').checked = true;
  for (const step of ['shape', 'snap']) {
    $('nodes-'+step).disabled = !reference;
    if (!reference) $('nodes-'+step).checked = false;
  }
  if (!reference) { $('nodes-detail').checked = false; $('nodes-simplify').checked = true; }
  $('nodes-reference-caption').textContent = reference ? 'Reference' : 'Original';
  syncNodeSteps();
  nodesDialog.open(selectionSummary());
};

// Retrace: a quick job applied as soon as it is ready, as one undoable edit.
async function retraceShapes() {
  await queue;
  if ($('retrace').disabled) { toast($('retrace').title, true); return; }
  const count = state.selection.objects.length;
  setBusy(`Retracing ${count === 1 ? 'the shape' : `${count} shapes`}…`, 1);
  let job = null;
  try {
    job = await operation('start', {epoch:state.epoch, revision:state.revision, action:'improve', method:'retrace',
      permissions:{geometry:true, structure:true}, settings:{mode:$('retrace-mode').value}});
    while (job.status === 'running') {
      await new Promise(resolve => setTimeout(resolve, 200));
      job = await operation('status', {job:job.id});
      $('busy-label').textContent = job.message;
    }
    if (job.status === 'failed') throw new Error(job.error);
    if (job.status !== 'ready') return;
    const {metrics} = job.result;
    if (!job.result.changed) { toast(job.message, true); return; }
    const result = await operation('apply', {job:job.id}); job = null; dirty = true;
    await applyState(result);
    const skipped = Object.keys(metrics.skipped).length;
    const colour = metrics.mode === 'colour' && $('retrace-mode').value === 'sam' ? ' by colour, since SAM needs a GPU' : metrics.mode === 'colour' ? ' by colour' : '';
    toast(`Retraced ${metrics.paths === 1 ? 'the shape' : `${metrics.paths} shapes`}${colour} · reference error ${errorChange(metrics)}${skipped ? ` · ${skipped} left as they were: ${[...new Set(Object.values(metrics.skipped))].join('; ')}` : ''}. Undo restores the outline.`);
  } catch (error) { toast(error.message, true); }
  finally {
    if (job) operation('discard', {job:job.id}).catch(() => {});
    setBusy('', -1);
  }
}
$('retrace').onclick = retraceShapes;

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
$('snap-edges').onclick = async () => { await queue; contactDialog.open(); };
// Changing a setting invalidates the preview shown for the old one.
for (const [prefix, ids] of [['contact', ['contact-distance']]]) {
  for (const id of ids) $(id).addEventListener('input', () => {
    $(prefix+'-apply').hidden = true; $(prefix+'-previews').hidden = true;
  });
}

// Generate: new shapes from the reference, placed as one group.
const generateSettings = {
  samvg: () => ({max_layers:Number($('samvg-max-layers').value), max_side:Number($('samvg-max-side').value), model:$('samvg-model').value}),
  cel: () => ({regions:Number($('cel-regions').value), tolerance:Number($('cel-tolerance').value), line_width:Number($('cel-line-width').value), strokes:$('cel-strokes').checked}),
  llm: () => ({provider:$('gen-llm-provider').value,
    candidates:Number($('gen-llm-candidates').value), instruction:$('gen-llm-instruction').value}),
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
$('generate-open').onclick = async () => {
  await queue;
  const group = oneObject()?.tag === 'g';
  $('generate-scope').value = group ? 'selection' : 'drawing';
  $('generate-scope').options[1].disabled = !group;
  showGenerateMethod(); generateDialog.open('');
};

// Improve dialogs that search or ask a model over the selection or drawing.
function openOnScope(dialog, prefix) {
  const count = state.selection.objects.length;
  $(prefix+'-scope').value = count ? 'selection' : 'drawing';
  $(prefix+'-scope').options[0].disabled = !count;
  dialog.open(count ? selectionSummary() : 'Whole drawing');
}
const llmDialog = jobDialog('llm', {
  start: () => ({action:'improve', method:'llm', scope:$('llm-scope').value,
    permissions:{geometry:$('llm-geometry').checked, paint:$('llm-paint').checked, structure:$('llm-structure').checked},
    settings:{instruction:$('llm-instruction').value, provider:$('llm-provider').value, candidates:Number($('llm-candidates').value)}}),
  describe: ({changed, metrics: {edits, skipped, ...metrics}}) => {
    const left = skipped ? ` · ${skipped} change(s) outside the scope were left out` : '';
    return changed ? `${edits} edit(s) · reference error ${errorChange(metrics)}${left}. Apply keeps it as one undoable edit.` : `The reply changed nothing that is allowed${left}.`;
  },
  applied: 'LLM edit applied. Undo restores the previous drawing.',
  choiceLabel: (result, index) => `Reply ${index + 1} · error ${result.metrics.after.error.toFixed(5)}`,
}).wire();
$('llm-open').onclick = async () => { await queue; openOnScope(llmDialog, 'llm'); };

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
        const job = await operation('start', {epoch:current.epoch, revision:current.revision, ...start()});
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
    settings:{passes:Number($('colours-passes').value), resolution:Number($('colours-resolution').value)}}),
  describe: result => result.changed ? `${result.metrics.objects} of ${result.metrics.considered} fills changed · reference error ${errorChange(result.metrics)}. Apply keeps it as one undoable edit.` : 'The colours already fit the reference.',
  applied: 'Colours fitted. Undo restores the previous fills.',
}).wire();
$('colours-open').onclick = async () => { await queue; coloursDialog.open(selectionSummary()); };
const cleanupDialog = jobDialog('cleanup', {
  start: () => ({action:'simplify', method:'cleanup', bounds:simplifyBounds(), permissions:{geometry:true, structure:true}}),
  describe: result => {
    const c = result.metrics.cleanup;
    return result.changed ? `${c.paths_before} → ${c.paths_after} paths (${c.paths_merged} merged, ${c.duplicate_paths_removed + c.empty_paths_removed} removed) · ${c.vertices_removed} redundant vertices removed. Apply keeps it as one undoable edit.` : 'Nothing to clean up in the selection.';
  },
  applied: 'Geometry cleaned up. Undo restores the original paths.',
}).wire();
$('cleanup-open').onclick = async () => { await queue; cleanupDialog.open(selectionSummary()); };
