import {revealInspectorField} from './inspector.js';


import {ENDPOINTS, ENDPOINT_DEFAULTS, boxPoints, constrainEndpoint, endpointNumber, endpointText, isPercent, unitsMatrix, unitsOf} from './gradient.js';
import {FRAME_REACH, SNAP_RADIUS} from './app_config.js';

// The shapes whose fill the canvas handles place: those that paint an area.
const PLACED = ['path', 'rect', 'circle', 'ellipse', 'polygon', 'polyline'];

// Gradients owns its controls; editor contains the live state shared with other controllers.
//
// The Gradient tool (G) shows the selected shape's linear-gradient fill on
// the canvas: a square start handle and a round end handle on a line, dashed
// when the gradient is shared. Only that tool shows them, so they never
// compete with the Move tool's frame, rotation handle and dragging, and a
// press anywhere else selects as Select does. A drag previews by setting the
// endpoint on the drawn gradient, as the edit will, so every shape using it
// changes at once; the release sends one gradient edit with just those
// attributes, and Escape puts them back.
export function createGradientsController(editor, {
  $,
  DOMMatrix = globalThis.DOMMatrix,
  DOMPoint = globalThis.DOMPoint,
  action,
  colorHex,
  cssColour,
  document = globalThis.document,
  drawOverlay = () => {},
  gradientCss,
  later,
  localToOverlay = () => null,
  oneObject,
  overlay,
  paintColour,
  paintGradient,
  paintValue,
  point,
  renderInspector,
  resolvedPaint,
  setTool,
  startGesture,
  svgElement,
  toast,
  xmlElement,
} = {}) {

  function gradientRefusal() {
    const item = oneObject();
    if (!item || item.resource || !['path', 'rect', 'circle', 'ellipse', 'line'].includes(item.tag)) return 'Select one path or basic shape to create a gradient';
    if (item.inherited_locks.includes('paint')) return 'This shape’s paint is locked';
    return '';
  }

  // getBBox is in the shape's own coordinates, also when a parent transforms it.
  function gradientEndpoints(bounds, direction = 'horizontal') {
    const {x, y, width, height} = bounds;
    if (direction === 'vertical') return {start: [x + width / 2, y], end: [x + width / 2, y + (height || 1)]};
    if (direction === 'diagonal') return {start: [x, y], end: [x + (width || 1), y + (height || 1)]};
    return {start: [x, y + height / 2], end: [x + (width || 1), y + height / 2]};
  }

  function createFillGradient() {
    const reason = gradientRefusal();
    if (reason) { toast(reason, true); renderInspector(); return; }
    revealInspectorField(document, $('fill-type'));
    const item = oneObject(), bounds = svgElement(item.id).getBBox();
    const value = resolvedPaint(item.id, 'fill', 'black');
    const hex = value === 'none' ? '#000000' : colorHex(cssColour(paintColour(value))) || '#000000';
    const colour = hex.slice(0, 7), opacity = hex.length === 9 ? parseInt(hex.slice(7), 16) / 255 : 1;
    // Keep the current colour at the first stop, with a contrasting second stop.
    const light = [1, 3, 5].map(i => parseInt(colour.slice(i, i + 2), 16)).reduce((a, b) => a + b) > 600;
    const fill = {...gradientEndpoints(bounds), stops: [{offset: 0, colour, opacity}, {offset: 1, colour: light ? '#000000' : '#ffffff', opacity}]};
    return action('paint', {changes: {fill}}, 'Creating gradient…');
  }

  function setFillType(type) {
    if (type === 'linear') return createFillGradient();
    const value = paintValue('fill', 'black');
    const fill = type === 'none' ? 'none' : value === 'none' ? '#000000' : colorHex(cssColour(paintColour(value))) || '#000000';
    return action('paint', {changes: {fill}});
  }

  // Edit the existing paint server, whether it belongs to one shape or is shared.
  function renderFillGradient(item) {
    const gradient = item?.fill_gradient;
    const resource = item?.tag === 'linearGradient';
    $('fill-controls').hidden = resource;
    $('stroke-controls').hidden = resource;
    const value = editor.state.selection.objects.length ? paintValue('fill', 'black') : '';
    const ramp = gradientCss(resource ? `url(#${item.id})` : value);
    $('fill-type').value = !value ? '' : value === 'none' ? 'none' : paintGradient(value) ? 'linear' : 'solid';
    $('fill-type').querySelector('[value="linear"]').disabled = !!gradientRefusal();
    $('fill-type').title = gradientRefusal() || 'Choose a solid colour, linear gradient or no fill';
    $('fill-gradient-preview').style.backgroundImage = ramp ? `${ramp}, conic-gradient(#d4d4d4 25%, #fff 0 50%, #d4d4d4 0 75%, #fff 0)` : '';
    $('fill-gradient').hidden = !gradient;
    if (!gradient) return;
    $('gradient-scope').textContent = gradient.private ? 'Private gradient · changes apply to this shape.' : `Shared gradient · changes apply to all ${gradient.users} ${gradient.users === 1 ? 'shape' : 'shapes'} using it.`;
    const onCanvas = handlesRefusal(item);
    $('gradient-on-canvas').disabled = !!onCanvas || editor.tool === 'gradient';
    $('gradient-on-canvas').title = onCanvas || (editor.tool === 'gradient' ? 'Drag the square start and round end handles on the canvas' : 'Drag the endpoints on the canvas with the Gradient tool (G)');
    const defaults = {x1: '0%', y1: '0%', x2: '100%', y2: '0%'};
    for (const key of ['x1', 'y1', 'x2', 'y2']) $('gradient-' + key).value = gradient.attributes[key] ?? defaults[key];
    $('gradient-units').value = gradient.attributes.gradientUnits || 'objectBoundingBox';
    $('gradient-spread').value = gradient.attributes.spreadMethod || 'pad';
    $('gradient-transform').value = gradient.attributes.gradientTransform || '';
    const container = $('gradient-stops'); container.replaceChildren();
    let previous = 0;
    gradient.stops.forEach((stop, index) => {
      const row = document.createElement('div'); row.className = 'gradient-stop';
      const position = stop.offset?.endsWith('%') ? parseFloat(stop.offset) / 100 : Number(stop.offset || 0);
      previous = Math.max(previous, Math.min(1, position));
      const offset = document.createElement('input'); offset.type = 'number'; offset.min = 0; offset.max = 100; offset.step = 'any'; offset.value = 100 * previous; offset.setAttribute('aria-label', `Stop ${index + 1} position %`);
      const hex = colorHex(cssColour(stop['stop-color'] || 'black')) || '#000000';
      const colour = document.createElement('input'); colour.type = 'color'; colour.value = hex.slice(0, 7); colour.setAttribute('aria-label', `Stop ${index + 1} colour`);
      const alpha = hex.length === 9 ? parseInt(hex.slice(7), 16) / 255 : 1;
      const opacity = document.createElement('input'); opacity.type = 'number'; opacity.min = 0; opacity.max = 100; opacity.step = 'any'; opacity.value = 100 * Number(stop['stop-opacity'] ?? 1) * alpha; opacity.setAttribute('aria-label', `Stop ${index + 1} opacity %`);
      const remove = document.createElement('button'); remove.textContent = '×'; remove.title = `Remove stop ${index + 1}`; remove.setAttribute('aria-label', remove.title); remove.disabled = gradient.stops.length === 1;
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
    const coordinate = key => { const value = $('gradient-' + key).value.trim(); return value.endsWith('%') ? value : Number(value); };
    return {start: ['x1', 'y1'].map(coordinate), end: ['x2', 'y2'].map(coordinate), stops};
  }

  function gradientEdit(fill) {
    return {id: oneObject().fill_gradient.id, attributes: {
      x1: String(fill.start[0]), y1: String(fill.start[1]), x2: String(fill.end[0]), y2: String(fill.end[1]),
      gradientUnits: $('gradient-units').value, spreadMethod: $('gradient-spread').value,
      gradientTransform: $('gradient-transform').value.trim() || null,
    }, stops: fill.stops};
  }

  function saveGradient(fill) {
    const gradient = gradientEdit(fill);
    later(() => action('paint', {gradient}, 'Editing gradient…'));
  }

  function saveFillGradient() { saveGradient(fillGradientValue()); }

  function wire() {
    $('fill-type').onchange = event => {
      const type = event.target.value;
      later(() => setFillType(type));
    };

    for (const key of ['x1', 'y1', 'x2', 'y2']) $('gradient-' + key).onchange = saveFillGradient;

    for (const key of ['units', 'spread', 'transform']) $('gradient-' + key).onchange = saveFillGradient;

    $('gradient-add-stop').onclick = () => {
      const fill = fillGradientValue(); fill.stops.push({offset: 0.5, colour: '#808080', opacity: 1}); fill.stops.sort((a, b) => a.offset - b.offset); saveGradient(fill);
    };

    for (const button of document.querySelectorAll('[data-gradient-direction]')) button.onclick = () => later(() => {
      const item = oneObject(); if (!item?.fill_gradient) return;
      const fill = fillGradientValue();
      const bounds = $('gradient-units').value === 'objectBoundingBox' ? {x: 0, y: 0, width: 1, height: 1} : item.tag === 'linearGradient' ? {x: editor.state.bounds[0], y: editor.state.bounds[1], width: editor.state.bounds[2], height: editor.state.bounds[3]} : svgElement(item.id).getBBox();
      Object.assign(fill, gradientEndpoints(bounds, button.dataset.gradientDirection));
      return action('paint', {gradient: gradientEdit(fill)}, 'Editing gradient…');
    });

    $('gradient-reverse').onclick = () => {
      const fill = fillGradientValue();
      fill.stops = fill.stops.reverse().map(stop => ({...stop, offset: 1 - stop.offset}));
      saveGradient(fill);
    };

    $('gradient-on-canvas').onclick = () => later(() => setTool('gradient'));
      return api;
  }

  // Why the canvas cannot show handles for *item*, or ''.
  function handlesRefusal(item = oneObject()) {
    if (!item?.fill_gradient) return 'Select one shape with a linear gradient fill';
    if (item.tag === 'linearGradient') return 'Select a shape using this gradient to place it on the canvas';
    if (!PLACED.includes(item.tag)) return 'Gradient handles place the fill of one path or basic shape';
    return '';
  }

  // Why the endpoints of *gradient* cannot move: a paint lock on it or on a
  // shape it paints, which the server refuses too. '' when they can.
  function lockRefusal(gradient) {
    const keys = ['paint', 'x1', 'y1', 'x2', 'y2'];
    const users = editor.state.objects.filter(o => o.id === gradient.id || o.fill_gradient?.id === gradient.id);
    const locked = users.find(o => o.inherited_locks.some(lock => keys.includes(lock)));
    if (!locked) return '';
    return gradient.private ? 'This shape’s paint is locked' : `This shared gradient cannot move: ${locked.label}’s paint is locked`;
  }

  // The handles' placement in the Gradient tool, in the overlay's frame, or
  // null: the gradient as drawn, the matrix from its coordinates to the
  // overlay (the shape's frame, its bounds for objectBoundingBox, then the
  // gradientTransform) and both ends there.
  function placement() {
    if (editor.tool !== 'gradient' || !editor.state) return null;
    const item = oneObject();
    if (handlesRefusal(item)) return null;
    const shape = svgElement(item.id), element = svgElement(item.fill_gradient.id);
    const frame = shape && localToOverlay(shape);
    if (!frame || element?.localName !== 'linearGradient') return null;
    const units = unitsOf(element.getAttribute('gradientUnits'));
    let box;
    try { box = shape.getBBox(); } catch { return null; /* An unrendered shape has no bounds. */ }
    const bounds = unitsMatrix(units, box);
    if (!bounds) return null;
    const own = element.gradientTransform?.baseVal.consolidate()?.matrix;
    const matrix = frame.multiply(new DOMMatrix([bounds.a, bounds.b, bounds.c, bounds.d, bounds.e, bounds.f]))
      .multiply(own ? new DOMMatrix([own.a, own.b, own.c, own.d, own.e, own.f]) : new DOMMatrix());
    if (Math.abs(matrix.a * matrix.d - matrix.b * matrix.c) < 1e-12) return null;
    const [, , width, height] = editor.state.bounds;
    const attributes = Object.fromEntries(Object.keys(ENDPOINT_DEFAULTS).map(key => [key, element.getAttribute(key)]));
    const end = name => {
      const [kx, ky] = ENDPOINTS[name], value = key => attributes[key] ?? ENDPOINT_DEFAULTS[key];
      const x = endpointNumber(value(kx), units, width), y = endpointNumber(value(ky), units, height);
      return Number.isFinite(x) && Number.isFinite(y) ? new DOMPoint(x, y).matrixTransform(matrix) : null;
    };
    const start = end('start'), finish = end('end');
    if (!start || !finish) return null;
    return {item, gradient: item.fill_gradient, element, units, matrix, frame, box, attributes, start, end: finish, refusal: lockRefusal(item.fill_gradient)};
  }

  // The end whose handle is at the screen point (x, y), the nearer if both
  // are, or null.
  function handleAt(x, y) {
    const shown = placement(), screen = shown && overlay.getScreenCTM();
    if (!screen) return null;
    let best = null;
    for (const name of ['end', 'start']) {
      const p = new DOMPoint(shown[name].x, shown[name].y).matrixTransform(screen), distance = Math.hypot(p.x - x, p.y - y);
      if (distance <= FRAME_REACH + 2 && (!best || distance < best.distance)) best = {name, distance};
    }
    return best?.name ?? null;
  }

  function handleCursor(x, y) {
    if (editor.drag) return null;
    const name = handleAt(x, y);
    return !name ? 'default' : placement().refusal ? 'not-allowed' : 'move';
  }

  // A press on a handle starts moving that end. A locked gradient refuses
  // once dragged; a click still picks what is under the handle.
  function pressHandle(event, common, name) {
    const shown = placement(), p = point(event), at = shown[name];
    startGesture('gradient', event, common, {
      name, refusal: shown.refusal, gradient: shown.gradient, private: shown.gradient.private, users: shown.gradient.users,
      element: shown.element, units: shown.units, matrix: shown.matrix, saved: shown.attributes,
      snaps: boxPoints(shown.box).map(q => new DOMPoint(q.x, q.y).matrixTransform(shown.frame)),
      fixed: shown[name === 'start' ? 'end' : 'start'], grab: {x: p.x - at.x, y: p.y - at.y}, result: null,
    });
  }

  // Where the dragged end goes and its attributes, as previewed and sent.
  // It snaps to the shape's corners, edge middles and centre unless Alt or
  // Ctrl/⌘ is held; Shift turns it about the other end in 15° steps.
  function handleDrag(event) {
    const drag = editor.drag, p = point(event);
    let at = {x: p.x - drag.grab.x, y: p.y - drag.grab.y}, snap = null;
    if (event.shiftKey) at = constrainEndpoint(drag.fixed, at);
    else if (!event.altKey && !event.ctrlKey && !event.metaKey) {
      for (const target of drag.snaps) {
        const distance = Math.hypot(target.x - at.x, target.y - at.y);
        if (distance <= SNAP_RADIUS / editor.zoom && (!snap || distance < snap.distance)) snap = {x: target.x, y: target.y, distance};
      }
      if (snap) at = {x: snap.x, y: snap.y};
    }
    const local = new DOMPoint(at.x, at.y).matrixTransform(drag.matrix.inverse()), [kx, ky] = ENDPOINTS[drag.name];
    return {name: drag.name, snap, attributes: {
      [kx]: endpointText(local.x, drag.units, isPercent(drag.saved, kx)),
      [ky]: endpointText(local.y, drag.units, isPercent(drag.saved, ky)),
    }};
  }

  // The drawn gradient takes the attributes the edit will set, so every
  // shape using it shows the result, and the Endpoints fields follow.
  function previewHandle() {
    for (const [key, value] of Object.entries(editor.drag.result.attributes)) {
      editor.drag.element.setAttribute(key, value);
      $('gradient-' + key).value = value;
    }
  }

  function restoreHandle(gesture) {
    for (const [key, value] of Object.entries(gesture.saved)) {
      if (value === null) gesture.element.removeAttribute(key);
      else gesture.element.setAttribute(key, value);
    }
    renderFillGradient(oneObject());
  }

  // One edit of the gradient's endpoint; nothing when it ends where it was.
  function finishHandle(gesture) {
    const {attributes} = gesture.result;
    if (Object.entries(attributes).every(([key, value]) => value === (gesture.saved[key] ?? ENDPOINT_DEFAULTS[key]))) { restoreHandle(gesture); drawOverlay(); return false; }
    return action('paint', {gradient: {id: gesture.gradient.id, attributes}}, 'Moving gradient…');
  }

  // The line between the ends, a halo under it, and the handles. The line
  // is dashed for a shared gradient; locked handles are dimmed.
  function drawHandles() {
    const shown = placement();
    if (!shown) return;
    const content = editor.overlayFrame.content, r = 4 / editor.zoom, {start, end} = shown;
    const line = {x1: start.x, y1: start.y, x2: end.x, y2: end.y};
    const state = `${shown.gradient.private ? '' : ' shared'}${shown.refusal ? ' locked' : ''}`;
    content.append(xmlElement('line', {...line, class: 'gradient-line-halo'}), xmlElement('line', {...line, class: `gradient-line${state}`}),
      xmlElement('rect', {x: start.x - r, y: start.y - r, width: 2 * r, height: 2 * r, class: `gradient-handle gradient-start${state}`}),
      xmlElement('circle', {cx: end.x, cy: end.y, r: 1.15 * r, class: `gradient-handle gradient-end${state}`}));
    const snap = editor.drag?.kind === 'gradient' && editor.drag.moved && editor.drag.result?.snap;
    if (snap) {
      const s = 6 / editor.zoom;
      content.append(xmlElement('rect', {x: snap.x - s, y: snap.y - s, width: 2 * s, height: 2 * s, transform: `rotate(45 ${snap.x} ${snap.y})`, class: 'snap-target'}));
    }
  }

  // The status line in the Gradient tool: whose paint a drag changes.
  function handlesStatus() {
    if (editor.tool !== 'gradient') return '';
    const drag = editor.drag?.kind === 'gradient' ? editor.drag : null, gradient = drag?.gradient ?? placement()?.gradient;
    if (!gradient) return handlesRefusal();
    const scope = gradient.private ? 'Private gradient: changes this shape' : `Shared gradient: changes all ${gradient.users} ${gradient.users === 1 ? 'shape' : 'shapes'} using it`;
    return drag?.moved && drag.result ? `Gradient ${drag.name} · ${scope}` : scope;
  }

  const api = {gradientRefusal, gradientEndpoints, createFillGradient, setFillType, renderFillGradient, fillGradientValue, gradientEdit, saveGradient, saveFillGradient, handlesRefusal, placement, handleAt, handleCursor, pressHandle, handleDrag, previewHandle, restoreHandle, finishHandle, drawHandles, handlesStatus, wire};
  return api;
}
