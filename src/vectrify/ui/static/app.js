const $ = id => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
let state, session, tool = 'select', zoom = 1, pan = {x: 0, y: 0}, drag = null;
let geometry = null, geometryObject = null, activeNode = null, reference = null;
let clickCycle = null;
let pathDraft = [], pathHover = null;
let joinContext = null;
let holePlan = null, chosenHoles = new Set(), chosenCleanup = new Set();
let pending = 0, queue = Promise.resolve(), dirty = false, space = false, toastTimer;
const drawing = $('drawing'), overlay = $('overlay'), stage = $('stage');
const names = {select: 'Select', nodes: 'Nodes', path: 'Draw path', hand: 'Pan'};
const hints = {select: 'Click to select · Click again to cycle · Ctrl/Shift to add · Drag to move', nodes: 'Drag points or blue handles · Pin endpoints to keep them fixed', path: 'Click for corners · Drag for curves · Click the first point to close · Enter finishes', hand: 'Drag to pan · Scroll to zoom'};

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
  if (replaced) { geometry = null; activeNode = null; fit(); }
  if (changed) { geometry = null; clickCycle = null; }
  if (holePlan && (changed || oneObject()?.id !== holePlan.object)) holePlan = null;
  renderObjects(); renderInspector();
  if (tool === 'nodes') await loadNodes();
  drawOverlay();
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
    row.onclick = event => selectObject(item.id, event.shiftKey || event.ctrlKey || event.metaKey, true);
    fragment.append(row);
  }
  $('objects').replaceChildren(fragment); $('object-count').textContent = count;
}
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
  enable('share-boundaries', selected.length !== 2 ? 'Select exactly two paths' : !paths && 'Both objects must be visible paths');
  const shared = selected.reduce((sum,id)=>sum+(object(id)?.shared_edges || 0),0);
  $('contact-hint').textContent = shared ? 'Linked nodes move both regions. Unlink before moving a region separately.' : 'Shift-click two paths to snap their touching edges together.';
  $('unlink-boundaries').hidden = !shared;
  $('empty-reference-hint').hidden = !!state.reference;
  if (editingNodes) return;
  renderRelationships(item);
  enable('nodes-open', !selected.some(id => ['path','g'].includes(object(id)?.tag)) && 'Select one or more paths, or groups that contain them');
  enable('llm-open', noReference);
  enable('colours-open', noReference);
  $('optimize-hint').textContent = state.reference ? 'Compare with the reference image. Each tool shows a preview before anything changes.' :'These tools compare the drawing with a reference image. Add one under Reference on the left to use them.';
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
  enable('join_paths', joinCandidates().length < 2 && 'Select at least two paths, or groups that contain them');
  enable('split_disconnected', item?.tag === 'use' ? 'Detach this instance to an editable path first' : !paths && 'Only visible paths can be split');
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
  activeNode = null; geometry = null;
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
async function selectAtPoint(finished) {
  const hits = finished.hits;
  if (!hits?.length) return selectObject(null, finished.shift);
  const index = !finished.shift && sameClickSpot(finished.x, finished.y, hits) ? (clickCycle.index+1)%hits.length : 0;
  const success = await selectObject(hits[index], finished.shift);
  if (success && !finished.shift) clickCycle = {x:finished.x, y:finished.y, hits, index};
}
async function loadNodes() {
  const item = oneObject();
  if (!item || item.tag !== 'path') { geometry = null; geometryObject = null; renderNodeInspector(); return; }
  if (geometry && geometryObject === item.id) return;
  const result = await request('/api/nodes', {object: item.id, epoch: state.epoch, revision: state.revision});
  geometry = result.geometry; geometryObject = item.id;
  if (!nodeById(activeNode)) activeNode = null;
  renderNodeInspector();
}
function allNodes() { return geometry?.subpaths.flatMap(s => s.nodes) || []; }
function nodeById(id) { return allNodes().find(node => node.id === id); }
function renderNodeInspector() {
  const item = oneObject();
  const current = item?.tag === 'path' && geometryObject === item.id ? geometry : null;
  const nodes = current?.subpaths.flatMap(s=>s.nodes) || [];
  const node = nodes.find(n=>n.id===activeNode);
  $('node-path-name').textContent = item?.label || (state?.selection.objects.length ? 'Multiple objects selected' : 'No path selected');
  $('node-path-stats').textContent = current ? `${nodes.length.toLocaleString()} points · ${current.subpaths.length.toLocaleString()} ${current.subpaths.length===1?'contour':'contours'}` : '';
  if (item?.shared_edges) $('node-path-stats').textContent += ` · ${item.shared_edges} shared edges (edits also move linked regions)`;
  $('node-detach').hidden = item?.tag !== 'use';
  $('node-properties').hidden = !node;
  $('node-count').textContent = nodes.length ? nodes.length.toLocaleString() : '';
  $('node-hint').textContent = item?.tag === 'use' ? 'This is a shared instance. Detach it to edit its points independently.' : item?.tag !== 'path' ? 'Select one path on the canvas or in the object tree.' : !current ? 'Loading path nodes…' : node ? (node.pinned ? 'This endpoint is pinned. Unpin it to move or delete it.' : node.command === 'C' ? 'Drag the blue handles to adjust this curve.' : 'Drag this point or enter its coordinates below.') : 'Click a point on the canvas to edit it.';
  if (node) {
    $('node-type').textContent = node.command === 'M' ? 'Start point' : node.command === 'C' ? 'Curve endpoint' : 'Line endpoint';
    $('node-x').value = +node.values.at(-2).toFixed(4); $('node-y').value = +node.values.at(-1).toFixed(4);
    $('node-x').disabled = node.pinned; $('node-y').disabled = node.pinned; $('node-apply').disabled = node.pinned;
    $('node-pin').checked = node.pinned; $('node-delete').disabled = node.command === 'M' || node.pinned;
    $('node-split').disabled = node.command === 'M' && !current.subpaths.find(s => s.nodes.includes(node))?.closed;
  }
}
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
  if (holePlan || tool !== 'nodes' || !geometry || geometryObject !== oneObject()?.id) return;
  const element = svgElement(geometryObject), matrix = localToOverlay(element);
  if (!matrix) return;
  const stageBox = stage.getBoundingClientRect(), screen = element.getScreenCTM();
  if (!screen) return;
  // Limit handles in dense drawings by screen-space spacing, without dropping
  // geometry. Zooming in exposes the original nodes at their full resolution.
  const occupied = new Set(); let shown = 0;
  for (const node of allNodes()) {
    const x = node.values.at(-2), y = node.values.at(-1), pos = new DOMPoint(x,y).matrixTransform(screen);
    if (pos.x < stageBox.left || pos.x > stageBox.right || pos.y < stageBox.top || pos.y > stageBox.bottom) continue;
    const cell = `${Math.floor(pos.x/10)},${Math.floor(pos.y/10)}`;
    if (node.id !== activeNode && (occupied.has(cell) || shown >= 1200)) continue;
    occupied.add(cell); shown++;
    const p = new DOMPoint(x,y).matrixTransform(matrix);
    const circle = xmlElement('circle', {cx:p.x, cy:p.y, r: (node.id === activeNode ? 4.8 : 3.3)/zoom, class:`node${node.id === activeNode ? ' selected' : ''}${node.pinned ? ' pinned' : ''}`});
    circle.dataset.node = node.id; circle.dataset.part = 'endpoint';
    overlay.append(circle);
  }
  const node = nodeById(activeNode);
  if (node) {
    const subpath = geometry.subpaths.find(s => s.nodes.includes(node)), i = subpath.nodes.indexOf(node);
    const handles = [];
    if (node.command === 'C') handles.push({node, offset:2, anchor:node.values.slice(-2)});
    const next = subpath.nodes[i+1];
    if (next?.command === 'C') handles.push({node:next, offset:0, anchor:node.values.slice(-2)});
    for (const handle of handles) {
      const p = new DOMPoint(...handle.node.values.slice(handle.offset, handle.offset+2)).matrixTransform(matrix);
      const anchor = new DOMPoint(...handle.anchor).matrixTransform(matrix);
      overlay.append(xmlElement('line', {x1:anchor.x,y1:anchor.y,x2:p.x,y2:p.y,class:'handle-line'}));
      const circle = xmlElement('circle', {cx:p.x,cy:p.y,r:3.8/zoom,class:'handle'});
      circle.dataset.node = handle.node.id; circle.dataset.part = String(handle.offset); overlay.append(circle);
    }
  }
  $('node-count').textContent = `${shown.toLocaleString()} / ${allNodes().length.toLocaleString()}`;
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
async function setTool(value) {
  if (!state || pending) return;
  clickCycle = null; pathDraft=[]; pathHover=null; tool=value;
  if (!['select','hand'].includes(value)) holePlan = null;
  document.querySelectorAll('[data-tool]').forEach(button => button.classList.toggle('active', button.dataset.tool === tool));
  $('tool-name').textContent=names[tool]; $('canvas-hint').textContent=hints[tool];
  stage.style.cursor = tool === 'hand' ? 'grab' : tool === 'path' ? 'crosshair' : 'default';
  renderInspector();
  if (tool === 'nodes') {
    setBusy('Loading path nodes…',1);
    try { await loadNodes(); } catch(error) {toast(error.message,true);} finally {setBusy('',-1);}
  }
  drawOverlay();
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
function pathData() { return geometry.subpaths.map(s => s.nodes.map(n => n.command+n.values.join(' ')).join(' ') + (s.closed ? ' Z' : '')).join(' '); }
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
  const nodeId = event.target.dataset?.node;
  if (tool === 'nodes' && nodeId) {
    const node=nodeById(nodeId), part=event.target.dataset.part;
    if (part === 'endpoint') activeNode=nodeId;
    drag={...common,kind:'node',nodeId,part,before:[...node.values],element:svgElement(geometryObject),object:geometryObject};
    renderNodeInspector(); drawOverlay(); return;
  }
  const hits = hitStack(event.clientX, event.clientY);
  common.hits = hits;
  const id = hits[0] || null;
  const selectedHit = id && (state.selection.objects.includes(id) ||
    (sameClickSpot(event.clientX, event.clientY, hits) && state.selection.objects.includes(hits[clickCycle.index])));
  if (tool === 'select' && selectedHit && !common.shift) {
    const members=topSelection().map(oid => ({id:oid,element:svgElement(oid),before:object(oid).attributes.transform || ''})).filter(m=>m.element);
    drag={...common,kind:'move',id,members}; return;
  }
  drag={...common,kind:'click',id};
});
stage.addEventListener('pointermove', event => {
  if (!drag) {
    if (tool==='path' && pathDraft.length) {pathHover=point(event);drawOverlay();}
    return;
  }
  drag.moved ||= Math.hypot(event.clientX-drag.x,event.clientY-drag.y)>3;
  if (drag.kind === 'pan') { pan={x:drag.pan.x+event.clientX-drag.x,y:drag.pan.y+event.clientY-drag.y}; updateView(); }
  if (drag.kind === 'drawPath' && drag.moved) {
    const p=point(event), a=drag.anchor;
    a.out={x:p.x,y:p.y}; a.in={x:2*a.x-p.x,y:2*a.y-p.y}; drawOverlay();
  }
  if (drag.kind === 'node' && drag.moved) {
    const node=nodeById(drag.nodeId);
    if (drag.part === 'endpoint' && node.pinned) return;
    const pos=point(event,drag.element), offset=drag.part === 'endpoint' ? node.values.length-2 : Number(drag.part);
    node.values[offset]=pos.x; node.values[offset+1]=pos.y;
    drag.element.setAttribute('d',pathData()); drawOverlay(); renderNodeInspector();
  }
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
stage.addEventListener('pointerup', async event => {
  if (!drag) return;
  const finished=drag; drag=null; stage.classList.remove('panning');
  if (stage.hasPointerCapture(event.pointerId)) stage.releasePointerCapture(event.pointerId);
  // A workspace click clears selection in every tool; dragging keeps its normal behavior.
  if (finished.deselectOutside && !finished.moved) { await selectObject(null); return; }
  if ((finished.kind === 'click' || finished.kind === 'move') && !finished.moved) await selectAtPoint(finished);
  else if (finished.kind === 'click') await selectObject(finished.id,finished.shift);
  if (finished.moved) clickCycle = null;
  if (finished.kind === 'drawPath') {pathHover=null;drawOverlay();return;}
  if (finished.kind === 'closePath') {if (!finished.moved) await finishPath(true);return;}
  if (finished.kind === 'node' && finished.moved) {
    const node=nodeById(finished.nodeId), values=[...node.values];
    node.values=finished.before;
    if (values.some((v,i)=>v !== finished.before[i])) await action('node',{object:finished.object,node:finished.nodeId,values},'Updating contour…');
    geometry=null; if (tool==='nodes') {try {await loadNodes(); drawOverlay();} catch(error){toast(error.message,true);}}
  }
  if (finished.kind === 'move' && finished.moved) await action('move',{dx:0,dy:0,offsets:Object.fromEntries(finished.members.map(m=>[m.id,m.offset||[0,0]]))},'Moving selection…');
});
stage.addEventListener('pointercancel',()=>{if(drag?.kind==='drawPath')pathDraft.pop();stage.classList.remove('panning');clickCycle=null;drag=null;geometry=null;if(state)renderDrawing();drawOverlay();});
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
$('node-apply').onclick=()=>{const node=nodeById(activeNode);if(node)action('node',{object:geometryObject,node:activeNode,values:[...node.values.slice(0,-2),Number($('node-x').value),Number($('node-y').value)]});};
$('node-pin').onchange=event=>action('pin',{object:geometryObject,node:activeNode,pinned:event.target.checked});
$('node-split').onclick=()=>action('split',{object:geometryObject,node:activeNode});
for(const count of [0,1,2]) $(`node-handles-${count}`).onclick=()=>action('node_handles',{object:geometryObject,node:activeNode,count},'Changing handles…');
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
$('node-delete').onclick=()=>action('delete_node',{object:geometryObject,node:activeNode});
document.querySelectorAll('[data-lock]').forEach(input=>input.onchange=()=>{const item=oneObject();if(item)action('locks',{object:item.id,locks:[...document.querySelectorAll('[data-lock]:checked')].map(el=>el.dataset.lock)});});
for (const command of ['group','ungroup','delete','detach']) $(command).onclick=()=>action(command);
$('backward').onclick=()=>action('reorder',{step:-1});$('forward').onclick=()=>action('reorder',{step:1});
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
    pathDraft=[];pathHover=null;stage.classList.remove('panning');holePlan=null;
    drag=null;activeNode=null;geometry=null;
    if(state){renderDrawing();renderInspector();}
    if(tool==='nodes')loadNodes().then(drawOverlay).catch(error=>toast(error.message,true));
    drawOverlay();return;
  }
  if(event.ctrlKey||event.metaKey||event.altKey)return;
  const tools={v:'select',n:'nodes',p:'path',h:'hand'};const key=event.key.toLowerCase();
  if(key==='o'){event.preventDefault();if(!event.repeat)toggleReference();return;}
  if(event.key==='?'){event.preventDefault();$('help-dialog').showModal();return;}
  if(tools[key])setTool(tools[key]);if(key==='f')fit();
  if((event.key==='Delete'||event.key==='Backspace')&&!pending&&state?.selection.objects.length){
    event.preventDefault();
    if(tool==='nodes') {
      if(geometryObject===oneObject()?.id && nodeById(activeNode) && !$('node-delete').disabled) $('node-delete').click();
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


// Improve: Optimize nodes, on the GPU path fit where it can run, else the CPU search.
const NODE_MOVES = ['shape', 'detail', 'simplify', 'strokes', 'position', 'snap'];
let gpuNote = '', enginePicked = null;
const nodeMoves = () => Object.fromEntries(NODE_MOVES.map(move => [move, $('nodes-'+move).checked]));
const gpuChosen = () => $('nodes-engine-gpu').checked && !$('nodes-engine-gpu').disabled;
function syncNodeEngine() {
  const moves = nodeMoves();
  const cpuOnly = NODE_MOVES.filter(move => move !== 'shape' && moves[move]).map(move => $('nodes-'+move).parentElement.textContent.trim());
  const gpu = $('nodes-engine-gpu');
  gpu.disabled = Boolean(gpuNote) || cpuOnly.length > 0;
  // The GPU fit is the default wherever it can run, unless CPU was picked.
  if (gpu.disabled) $('nodes-engine-cpu').checked = true;
  else if (enginePicked !== 'cpu') gpu.checked = true;
  $('nodes-engine-note').textContent = gpuNote ? `GPU fit unavailable: ${gpuNote}.`
    : cpuOnly.length ? `${cpuOnly.join(', ')} need${cpuOnly.length === 1 ? 's' : ''} the CPU search.` : 'The GPU fit is faster; the CPU search also handles strokes, open and grouped paths.';
  for (const block of document.querySelectorAll('#nodes-settings [data-engine]')) block.hidden = block.dataset.engine !== (gpuChosen() ? 'gpu' : 'cpu');
  $('nodes-tolerance-row').hidden = !moves.simplify;
  $('nodes-apply').hidden = true; $('nodes-previews').hidden = true;
}
const nodesDialog = jobDialog('nodes', {
  start: () => {
    if (gpuChosen()) return {action:'improve', method:'path-fit', permissions:{geometry:true},
      settings:{nodes:true, handles:true, color:false, displacement:Number($('nodes-movement').value)}, budget:{steps:Number($('nodes-steps').value)}};
    const moves = nodeMoves();
    return {action:'improve', method:'nodes', scope:'selection',
      permissions:{geometry:true, structure:moves.detail || moves.simplify, paint:moves.strokes},
      settings:{...moves, tolerance:Number($('nodes-tolerance').value), workers:Number($('nodes-workers').value)},
      budget:{steps:Number($('nodes-tasks').value)}};
  },
  describe: ({changed, metrics}) => {
    if (!changed) return 'Nothing improved the paths within these settings. They are unchanged.';
    if (metrics.size) return `Reference error ${errorChange(metrics)} · ${metrics.size.join(' × ')} px crop. Apply keeps this result as one undoable edit.`;
    const points = `${metrics.before.nodes.toLocaleString()} → ${metrics.after.nodes.toLocaleString()} points`;
    const fit = metrics.reference ? `difference from the reference ${errorChange(metrics, 'difference')}` : 'the look is kept within the tolerance';
    const after = metrics.tasks ? `after ${metrics.snapped ? 'snapping and ' : ''}${metrics.tasks.toLocaleString()} tries` : 'after snapping';
    return `${points} · ${fit} ${after}. Apply keeps this result as one undoable edit.`;
  },
  applied: 'Paths optimized. Undo restores them.',
}).wire();
for (const move of NODE_MOVES) $('nodes-'+move).addEventListener('change', syncNodeEngine);
for (const id of ['nodes-engine-gpu', 'nodes-engine-cpu']) $(id).addEventListener('change', event => { enginePicked = event.target.value; syncNodeEngine(); });
for (const id of ['nodes-tolerance', 'nodes-tasks', 'nodes-workers', 'nodes-steps', 'nodes-movement']) $(id).addEventListener('input', () => { $('nodes-apply').hidden = true; $('nodes-previews').hidden = true; });
$('nodes-open').onclick = async () => {
  await queue;
  const reference = Boolean(state.reference);
  $('nodes-detail').disabled = !reference;
  $('nodes-detail').parentElement.title = reference ? 'Split segments and move the new point, where the reference needs more detail' : 'Adding detail needs a reference image';
  $('nodes-snap').disabled = !reference;
  $('nodes-snap').parentElement.title = reference ? "Move the points onto the reference's edges before searching; with Add detail it may also split segments" : 'Snapping needs a reference image';
  if (!reference) { $('nodes-detail').checked = false; $('nodes-snap').checked = false; $('nodes-simplify').checked = true; }
  $('nodes-reference-caption').textContent = reference ? 'Reference' : 'Original';
  try {
    const check = await operation('check', {epoch:state.epoch, revision:state.revision, action:'improve', method:'path-fit',
      permissions:{geometry:true}, settings:{nodes:true, handles:true, color:false}});
    gpuNote = check.ok ? '' : check.error.replace(/\.$/, '').replace(/^./, c => c.toLowerCase());
  } catch (error) { gpuNote = error.message; }
  enginePicked = null;
  syncNodeEngine();
  nodesDialog.open(selectionSummary());
};

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
  start: () => ({action:'link', method:'boundaries', bounds:simplifyBounds(), permissions:{geometry:true, structure:true},
    settings:{tolerance:Number($('contact-distance').value)}}),
  describe: ({changed, metrics}) => changed
    ? `${metrics.edges} shared edge spans. Apply links their nodes and curve handles as one undoable edit.`
    : 'No touching edges within this distance. Try a larger contact distance.',
  applied: 'Shared boundaries linked. Editing a shared node moves both regions.',
}).wire();
$('share-boundaries').onclick = async () => { await queue; contactDialog.open(); };
$('unlink-boundaries').onclick=()=>action('unlink_boundaries');
// Changing a setting invalidates the preview shown for the old one.
for (const [prefix, ids] of [['contact', ['contact-distance']]]) {
  for (const id of ids) $(id).addEventListener('input', () => {
    $(prefix+'-apply').hidden = true; $(prefix+'-previews').hidden = true;
  });
}

// Generate: new shapes from the reference, placed as one group.
const generateSettings = {
  samvg: () => ({max_layers:Number($('samvg-max-layers').value), segments:Number($('samvg-segments').value), model:$('samvg-model').value,
    fill_holes:$('samvg-fill-holes').checked, min_width:Number($('samvg-min-width').value), flatten:$('samvg-flatten').checked}),
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
