

// Gradients owns its controls; editor contains the live state shared with other controllers.
export function createGradientsController(editor, {
  $,
  action,
  colorHex,
  cssColour,
  document = globalThis.document,
  gradientCss,
  later,
  oneObject,
  paintColour,
  paintGradient,
  paintValue,
  renderInspector,
  resolvedPaint,
  svgElement,
  toast,
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
      return api;
  }
  const api = {gradientRefusal, gradientEndpoints, createFillGradient, setFillType, renderFillGradient, fillGradientValue, gradientEdit, saveGradient, saveFillGradient, wire};
  return api;
}
