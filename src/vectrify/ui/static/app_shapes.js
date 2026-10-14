import {drawable, roundPoint, shapeFrom, shapeSize, shown, sizedEnd} from './shapes.js';

const STORAGE = 'vectrify-drawing';

// What the drawing tools start with: filled in Draw path's green, without a
// stroke, two units wide when one is turned on; and the size a click places.
export const DRAWING_DEFAULTS = Object.freeze({
  fill: '#83b899', fillOn: true, stroke: '#2f5e43', strokeOn: false, width: 2, radius: 0,
  size: Object.freeze({width: 100, height: 100, length: 100, angle: 0}),
});

const COLOUR = /^#[0-9a-f]{6}$/i;

// The remembered settings, with anything missing or broken in *saved* as by
// default.
export function drawingSettings(saved = {}) {
  const d = DRAWING_DEFAULTS, positive = v => Number.isFinite(v) && v > 0;
  const size = saved.size || {};
  const settings = {
    fill: COLOUR.test(saved.fill) ? saved.fill : d.fill,
    fillOn: typeof saved.fillOn === 'boolean' ? saved.fillOn : d.fillOn,
    stroke: COLOUR.test(saved.stroke) ? saved.stroke : d.stroke,
    strokeOn: typeof saved.strokeOn === 'boolean' ? saved.strokeOn : d.strokeOn,
    width: positive(saved.width) ? saved.width : d.width,
    radius: Number.isFinite(saved.radius) && saved.radius >= 0 ? saved.radius : d.radius,
    size: {
      width: positive(size.width) ? size.width : d.size.width,
      height: positive(size.height) ? size.height : d.size.height,
      length: positive(size.length) ? size.length : d.size.length,
      angle: Number.isFinite(size.angle) ? size.angle : d.size.angle,
    },
  };
  // Something always paints.
  if (!settings.fillOn && !settings.strokeOn) settings.fillOn = true;
  return settings;
}

// The paint *settings* give what is drawn: a closed outline their fill and
// stroke; a line or an open path their stroke, or the fill colour without one.
export function drawingPaint(settings, closed) {
  const stroke_width = settings.width;
  if (!closed) return {fill: 'none', stroke: settings.strokeOn ? settings.stroke : settings.fill, stroke_width};
  return {fill: settings.fillOn ? settings.fill : 'none', stroke: settings.strokeOn ? settings.stroke : 'none', stroke_width};
}

// Shapes owns the shape tools and the drawing paint every drawing tool uses;
// editor contains the live state shared with other controllers.
export function createShapesController(editor, {
  $,
  action,
  document = globalThis.document,
  localStorage = globalThis.localStorage,
  toast,
  xmlElement,
} = {}) {
  let settings = drawingSettings();
  try { settings = drawingSettings(JSON.parse(localStorage.getItem(STORAGE) || '{}')); } catch { /* Defaults where storage does not work. */ }

  function remember() {
    try { localStorage.setItem(STORAGE, JSON.stringify(settings)); } catch { /* Only remembered where storage works. */ }
  }

  const paint = closed => drawingPaint(settings, closed);

  // New objects go into the entered group, else the top level.
  const container = () => editor.scope ? {parent: editor.scope} : {};

  // The shape a drag draws, from its press to the pointer.
  function dragShape(gesture) {
    const start = roundPoint(gesture.start, editor.zoom), end = roundPoint(gesture.end, editor.zoom);
    return shapeFrom(gesture.tool, start, end, {square: gesture.square, centre: gesture.centre, radius: settings.radius});
  }

  // A click places a shape at the strip's size, from its corner or with Alt
  // around it.
  function clickShape(gesture) {
    const start = roundPoint(gesture.start, editor.zoom), end = sizedEnd(gesture.tool, start, settings.size, gesture.centre);
    return shapeFrom(gesture.tool, start, end, {centre: gesture.centre, radius: settings.radius});
  }

  // The strip's size fields show the remembered size, or the size being
  // dragged out; a field being typed in keeps what is typed.
  function showSize(size = settings.size) {
    for (const [id, value] of [['shape-width', size.width], ['shape-height', size.height], ['shape-length', size.length], ['shape-angle', size.angle]]) {
      const field = $(id);
      if (value !== undefined && document.activeElement !== field) field.value = shown(value);
    }
  }

  function renderControls() {
    $('draw-fill').value = settings.fill; $('draw-fill-on').checked = settings.fillOn;
    $('draw-stroke').value = settings.stroke; $('draw-stroke-on').checked = settings.strokeOn;
    $('draw-width').value = shown(settings.width); $('shape-radius').value = shown(settings.radius);
    showSize();
  }

  // A colour for the drawing paint's fill or stroke, as its picker sets it:
  // turned on and remembered.
  function setColour(kind, colour) {
    if (!COLOUR.test(colour)) return;
    settings[kind] = colour.toLowerCase(); settings[`${kind}On`] = true;
    remember(); renderControls();
  }

  function preview(gesture) {
    gesture.shape = dragShape(gesture);
    showSize(shapeSize(gesture.shape));
  }

  function cancel() { showSize(); }

  // The shape being dragged out, painted as it will be, with a thin outline
  // that shows on any background.
  function drawShape() {
    const drag = editor.drag;
    if (drag?.kind !== 'shape' || !drag.moved || !drag.shape) return;
    const {tag, attributes} = drag.shape, {fill, stroke, stroke_width} = paint(tag !== 'line');
    const group = xmlElement('g', {'pointer-events': 'none', 'aria-hidden': 'true'});
    group.append(xmlElement(tag, {...attributes, fill, stroke, 'stroke-width': stroke_width}),
      xmlElement(tag, {...attributes, fill: 'none', stroke: '#5cdeff', 'stroke-width': 1/editor.zoom, 'stroke-dasharray': `${4/editor.zoom} ${3/editor.zoom}`}));
    editor.overlayFrame.content.append(group);
  }

  // One undoable edit draws the shape into the drawing container and selects
  // it; the tool stays on for the next.
  async function finish(gesture) {
    const shape = gesture.moved ? gesture.shape ?? dragShape(gesture) : clickShape(gesture);
    if (!drawable(shape)) {
      toast(shape.tag === 'line' ? 'Drag out a line with some length.' : 'Drag out a width and a height, or click to place the size in the tool strip.', true);
      showSize(); return false;
    }
    if (gesture.moved) { settings.size = {...settings.size, ...shapeSize(shape)}; remember(); }
    showSize();
    const noun = {rect: 'rectangle', circle: 'circle', ellipse: 'ellipse', line: 'line'}[shape.tag];
    return action('add_shape', {shape: shape.tag, ...shape.attributes, ...paint(shape.tag !== 'line'), ...container()}, `Drawing ${noun}…`);
  }

  // A size field takes a positive number (an angle any); anything else puts
  // it back.
  function sizeField(id, apply, test = v => v > 0) {
    const field = $(id);
    field.onchange = () => {
      const value = Number(field.value);
      if (field.value.trim() !== '' && Number.isFinite(value) && test(value)) { apply(value); remember(); }
      renderControls();
    };
    field.onkeydown = event => {
      if (event.key === 'Enter') { event.preventDefault(); field.blur(); }
      else if (event.key === 'Escape') { event.preventDefault(); renderControls(); field.blur(); }
    };
  }

  function wire() {
    renderControls();
    sizeField('shape-width', v => { settings.size.width = v; });
    sizeField('shape-height', v => { settings.size.height = v; });
    sizeField('shape-length', v => { settings.size.length = v; });
    sizeField('shape-angle', v => { settings.size.angle = ((v + 180) % 360 + 360) % 360 - 180; }, () => true);
    sizeField('shape-radius', v => { settings.radius = v; }, v => v >= 0);
    sizeField('draw-width', v => { settings.width = v; });
    $('draw-fill').oninput = () => { settings.fill = $('draw-fill').value; settings.fillOn = true; remember(); renderControls(); };
    $('draw-stroke').oninput = () => { settings.stroke = $('draw-stroke').value; settings.strokeOn = true; remember(); renderControls(); };
    // Something always paints: turning one off with the other off turns
    // that one on.
    $('draw-fill-on').onchange = () => {
      settings.fillOn = $('draw-fill-on').checked;
      if (!settings.fillOn && !settings.strokeOn) settings.strokeOn = true;
      remember(); renderControls();
    };
    $('draw-stroke-on').onchange = () => {
      settings.strokeOn = $('draw-stroke-on').checked;
      if (!settings.fillOn && !settings.strokeOn) settings.fillOn = true;
      remember(); renderControls();
    };
    return api;
  }
  const api = {settings: () => settings, paint, setColour, container, dragShape, clickShape, showSize, renderControls, preview, cancel, drawShape, finish, wire};
  return api;
}
