import {SMOOTHING, fitStroke, smoothingDistance, strokePath} from './pencil.js';
import {shown} from './shapes.js';

const STORAGE = 'vectrify-pencil';

// The remembered Pencil settings, with anything missing or broken in *saved*
// as by default: the smoothing distance in screen pixels, and whether strokes
// close back to their start.
export function pencilSettings(saved = {}) {
  return {
    smoothing: Number.isFinite(saved.smoothing) ? smoothingDistance(saved.smoothing) : SMOOTHING.initial,
    closed: typeof saved.closed === 'boolean' ? saved.closed : false,
  };
}

// The on-curve points of path data from strokePath, in order.
function strokePoints(d) {
  return [...d.matchAll(/[MC]([^MCZ]*)/g)].map(([, numbers]) => numbers.trim().split(/\s+/).map(Number).slice(-2));
}

// Pencil owns the Pencil tool's settings, its preview and completion; editor
// contains the live state shared with other controllers. A stroke is fitted
// to curves as it is drawn, and the path it adds is the fit previewed.
export function createPencilController(editor, {
  $,
  action,
  drawOverlay,
  drawingContainer,
  drawingPaint,
  localStorage = globalThis.localStorage,
  toast,
  xmlElement,
} = {}) {
  let settings = pencilSettings();
  try { settings = pencilSettings(JSON.parse(localStorage.getItem(STORAGE) || '{}')); } catch { /* Defaults where storage does not work. */ }

  function remember() {
    try { localStorage.setItem(STORAGE, JSON.stringify(settings)); } catch { /* Only remembered where storage works. */ }
  }

  // The stroke's fit as path data, or '' while it is too short to draw. The
  // preview and the edit read the same fit: it is redone only when the
  // stroke, the zoom or the settings change.
  function fitted(gesture) {
    const key = `${gesture.points.length} ${editor.zoom} ${settings.smoothing} ${settings.closed}`;
    if (gesture.fitKey !== key) {
      gesture.fitKey = key;
      gesture.d = strokePath(fitStroke(gesture.points, {zoom: editor.zoom, smoothing: settings.smoothing, closed: settings.closed}), editor.zoom);
    }
    return gesture.d;
  }

  function renderControls() {
    $('pencil-close').setAttribute('aria-pressed', String(settings.closed));
    $('pencil-smoothing').value = shown(settings.smoothing);
  }

  function preview(gesture) { fitted(gesture); }

  // The fitted stroke, painted as it will be, with a thin outline and its
  // points, which show on any background.
  function drawPencil() {
    const drag = editor.drag;
    if (drag?.kind !== 'pencil' || !drag.moved) return;
    const d = fitted(drag);
    if (!d) return;
    const {fill, stroke, stroke_width} = drawingPaint(settings.closed), z = editor.zoom;
    const group = xmlElement('g', {'pointer-events': 'none', 'aria-hidden': 'true'});
    group.append(xmlElement('path', {d, fill, stroke, 'stroke-width': stroke_width, 'stroke-linecap': 'round', 'stroke-linejoin': 'round'}),
      xmlElement('path', {d, fill: 'none', stroke: '#5cdeff', 'stroke-width': 1/z, 'stroke-dasharray': `${4/z} ${3/z}`}));
    for (const [x, y] of strokePoints(d)) group.append(xmlElement('circle', {cx: x, cy: y, r: 2/z, fill: '#5cdeff', stroke: '#052b3a', 'stroke-width': 1/z}));
    editor.overlayFrame.content.append(group);
  }

  // One undoable edit adds the fitted path into the drawing container, with
  // the drawing paint, and selects it; the tool stays on for the next. A
  // click or a stroke too short to fit draws nothing.
  async function finish(gesture) {
    if (!gesture.moved) { toast('Drag to draw a freehand path.'); return false; }
    const d = fitted(gesture);
    if (!d) { toast('Draw a longer stroke: this one is too short to make a path.', true); return false; }
    const closed = d.endsWith('Z');
    return action('add_path', {d, ...drawingPaint(closed), ...drawingContainer()}, 'Drawing freehand path…');
  }

  function wire() {
    renderControls();
    // Close shape changes a stroke under way too, pressed with its key.
    $('pencil-close').onclick = () => {
      settings.closed = !settings.closed; remember(); renderControls();
      if (editor.drag?.kind === 'pencil') drawOverlay();
    };
    const field = $('pencil-smoothing');
    // A smoothing distance out of range is kept to it; anything not a number
    // puts the field back.
    field.onchange = () => {
      const value = Number(field.value);
      if (field.value.trim() !== '' && Number.isFinite(value)) { settings.smoothing = smoothingDistance(value); remember(); }
      renderControls();
    };
    field.onkeydown = event => {
      if (event.key === 'Enter') { event.preventDefault(); field.blur(); }
      else if (event.key === 'Escape') { event.preventDefault(); field.blur(); renderControls(); }
    };
    return api;
  }
  const api = {settings: () => settings, fitted, renderControls, preview, drawPencil, finish, wire};
  return api;
}
