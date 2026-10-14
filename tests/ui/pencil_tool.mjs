// The Pencil: freehand strokes fitted to curves within a smoothing distance
// in screen pixels at any zoom and input speed, never invalid geometry; the
// preview, the path added, paint and container, Close shape, cancellation
// and queued input, without a browser DOM.
import assert from 'node:assert/strict';

import {SMOOTHING, fitStroke, smoothingDistance, strokeNodes, strokePath, strokeStatus} from '../../src/vectrify/ui/static/pencil.js';
import {createPencilController, pencilSettings} from '../../src/vectrify/ui/static/app_pencil.js';
import {DRAWING_DEFAULTS, drawingPaint, drawingSettings} from '../../src/vectrify/ui/static/app_shapes.js';
import {canvasGestures} from '../../src/vectrify/ui/static/gestures.js';
import {HeldGesture} from '../../src/vectrify/ui/static/input.js';
import {TOOL_LEVEL} from '../../src/vectrify/ui/static/selection.js';
import {createEditorState} from '../../src/vectrify/ui/static/app_state.js';
import {createKeysController} from '../../src/vectrify/ui/static/app_keys.js';
import {createCanvasController} from '../../src/vectrify/ui/static/app_canvas.js';
import {hints, names} from '../../src/vectrify/ui/static/app_config.js';

// A seeded random number in [-0.5, 0.5), so failures repeat.
let seed = 7;
const random = () => { seed = seed * 16807 % 2147483647; return seed / 2147483647 - 0.5; };

// Path data from strokePath as its moveto and cubics, checking its form: a
// moveto, cubics and an optional Z, every number finite.
function parsed(d) {
  assert.match(d, /^M-?[\d.e+-]+ -?[\d.e+-]+( C(-?[\d.e+-]+ ){5}-?[\d.e+-]+)+( Z)?$/, d);
  const commands = [...d.matchAll(/([MC])([^MCZ]*)/g)].map(([, command, numbers]) => [command, numbers.trim().split(' ').map(Number)]);
  for (const [, values] of commands) assert.ok(values.every(Number.isFinite), d);
  return commands;
}

// The curve's points, densely, in screen pixels at *zoom*.
function traced(d, zoom) {
  const commands = parsed(d), points = [];
  let at = commands[0][1];
  for (const [, [x1, y1, x2, y2, x, y]] of commands.slice(1)) {
    for (let k = 0; k <= 200; k++) {
      const t = k / 200, u = 1 - t;
      points.push([0, 1].map(j => (u*u*u*at[j] + 3*u*u*t*[x1, y1][j] + 3*u*t*t*[x2, y2][j] + t*t*t*[x, y][j]) * zoom));
    }
    at = [x, y];
  }
  return points;
}

// The farthest any stroke point lies from the fitted curve, in screen pixels.
function stray(stroke, d, zoom) {
  const curve = traced(d, zoom);
  let most = 0;
  for (const [x, y] of stroke) {
    let near = Infinity;
    for (const q of curve) near = Math.min(near, Math.hypot(x * zoom - q[0], y * zoom - q[1]));
    most = Math.max(most, near);
  }
  return most;
}

// A shaky three-quarter circle of radius *r* screen pixels, sampled every
// *step* pixels, in document units at *zoom*.
function arc(zoom, {r = 120, step = 2, shake = 1.2} = {}) {
  const count = Math.ceil(1.5 * Math.PI * r / step), points = [];
  for (let i = 0; i <= count; i++) {
    const a = 1.5 * Math.PI * i / count;
    points.push([(200 + r * Math.cos(a) + shake * random()) / zoom, (200 + r * Math.sin(a) + shake * random()) / zoom]);
  }
  return points;
}

// Within the smoothing distance of the stroke (allowing for its shake), with
// a handful of points, alike at every zoom.
const counts = new Set();
for (const zoom of [0.05, 0.5, 1, 3, 40]) {
  seed = 7;
  const stroke = arc(zoom), d = strokePath(fitStroke(stroke, {zoom}), zoom);
  assert.ok(stray(stroke, d, zoom) <= SMOOTHING.initial + 1, `zoom ${zoom}: ${stray(stroke, d, zoom)} px`);
  assert.ok(strokeNodes(d) <= 10, `zoom ${zoom}: ${strokeNodes(d)} points`);
  assert.equal(d.endsWith('Z'), false);
  counts.add(strokeNodes(d));
}
assert.equal(counts.size, 1, 'the same stroke on screen fits alike at any zoom');

// More smoothing never adds points, and each stays within its distance.
seed = 11;
const wavy = [];
for (let i = 0; i <= 600; i++) wavy.push([i, 40 * Math.sin(i / 35) + 3 * Math.sin(i / 4) + 0.8 * random()]);
let previous = Infinity, most;
for (const smoothing of [1, 2, 4, 8, 20]) {
  const d = strokePath(fitStroke(wavy, {smoothing}), 1), count = strokeNodes(d);
  assert.ok(count <= previous, `${smoothing} px: ${count} points after ${previous}`);
  assert.ok(stray(wavy, d, 1) <= smoothing + 1, `${smoothing} px strays ${stray(wavy, d, 1)}`);
  previous = count; most ??= count;
}
// A small ripple on the wave is followed at 1 px and smoothed away at 20.
assert.ok(previous <= 8 && previous * 5 < most, `${most} points at 1 px, ${previous} at 20`);

// Fast and slow input over the same arc both fit it, the fast one's sparse
// samples as closely as the slow one's dense ones.
seed = 3;
const slow = arc(1, {step: 1, shake: 0}), fast = arc(1, {step: 25, shake: 0});
for (const stroke of [slow, fast]) {
  const d = strokePath(fitStroke(stroke, {zoom: 1}), 1);
  assert.ok(stray(stroke, d, 1) <= SMOOTHING.initial + 0.5);
  assert.ok(strokeNodes(d) <= 12, `${strokeNodes(d)} points`);
}

// A sharp turn stays a corner: the zigzag's tips are points of the path.
const zigzag = [];
for (let i = 0; i <= 300; i++) zigzag.push([i, 50 - Math.abs(i % 100 - 50)]);
const tips = parsed(strokePath(fitStroke(zigzag), 1)).map(([, v]) => v.slice(-2));
for (const tip of [[50, 50], [100, 0], [150, 50], [200, 0], [250, 50]]) assert.ok(tips.some(p => Math.hypot(p[0] - tip[0], p[1] - tip[1]) < 1.5), `a point at ${tip}`);

// Closed, the last curve ends exactly on the first point, then Z.
const loop = parsed(strokePath(fitStroke(arc(2), {zoom: 2, closed: true}), 2));
assert.deepEqual(loop.at(-1)[1].slice(-2), loop[0][1]);
assert.ok(strokePath(fitStroke(arc(2), {zoom: 2, closed: true}), 2).endsWith(' Z'));
// Closing a straight stroke makes a valid, if flat, shape.
parsed(strokePath(fitStroke([[0, 0], [10, 0]], {closed: true}), 1));

// Nothing to draw: a click, a stroke shorter than two pixels at the zoom, no
// numbers, repeats of one point. Not numbers in a stroke are left out.
assert.equal(fitStroke([[5, 5]]), null);
assert.equal(fitStroke([[5, 5], [5, 5], [5, 5]]), null);
assert.equal(fitStroke([[0, 0], [1, 1]], {zoom: 1}), null);
assert.notEqual(fitStroke([[0, 0], [1, 1]], {zoom: 2}), null, 'two pixels at a closer zoom');
assert.equal(fitStroke([[NaN, 0], [Infinity, 2]]), null);
assert.equal(fitStroke([[0, 0], [10, 0]], {zoom: 0}), null);
assert.equal(strokePath(null), '');
assert.equal(strokePath(fitStroke([[0, 0], [NaN, 4], [10, 10]]), 1), 'M0 0 C3.3 3.3 6.7 6.7 10 10');

// Numbers come out as round as the pointer places them: a tenth of a screen
// pixel.
assert.equal(strokePath(fitStroke([[0, 0], [30.0000001, 0]]), 1), 'M0 0 C10 0 20 0 30 0');
assert.equal(strokePath(fitStroke([[0, 0], [0.30000001, 0]], {zoom: 100}), 100), 'M0 0 C0.1 0 0.2 0 0.3 0');

// Random strokes at any zoom, smoothing and speed, back and forth over
// themselves, give valid paths within their smoothing distance.
seed = 5;
for (let trial = 0; trial < 150; trial++) {
  const zoom = Math.exp(random() * 9), smoothing = 1 + (random() + 0.5) * 12, closed = trial % 3 === 0;
  const step = Math.exp(random() * 6), stroke = [[0, 0]];
  let heading = 0;
  for (let i = 0, count = 2 + Math.floor((random() + 0.5) * 80); i < count; i++) {
    heading += random() * (trial % 2 ? 6 : 1);
    stroke.push([stroke.at(-1)[0] + Math.cos(heading) * step / zoom, stroke.at(-1)[1] + Math.sin(heading) * step / zoom]);
  }
  const fit = fitStroke(stroke, {zoom, smoothing, closed}), d = strokePath(fit, zoom);
  if (!d) { assert.ok(!fit || fit.curves.length, `trial ${trial}`); continue; }
  parsed(d);
  assert.equal(d.endsWith('Z'), closed);
  assert.ok(stray(stroke, d, zoom) <= smoothing + 1, `trial ${trial}: strays ${stray(stroke, d, zoom)} px at ${smoothing}`);
}

// The smoothing distance is kept to its range; settings fall back when broken.
assert.equal(smoothingDistance(0), SMOOTHING.least);
assert.equal(smoothingDistance(99), SMOOTHING.most);
assert.equal(smoothingDistance('x'), SMOOTHING.initial);
assert.deepEqual(pencilSettings({smoothing: 'x', closed: 'yes'}), {smoothing: SMOOTHING.initial, closed: false});
assert.deepEqual(pencilSettings({smoothing: 50, closed: true}), {smoothing: SMOOTHING.most, closed: true});
assert.equal(strokeStatus('M0 0 C1 1 2 2 3 3'), 'Pencil · 2 points');
assert.equal(strokeStatus('M0 0 C1 1 2 2 3 3 C4 4 5 5 0 0 Z'), 'Pencil · 2 points');
assert.equal(TOOL_LEVEL.pencil, 'create');
assert.equal(names.pencil, 'Pencil');
assert.match(hints.pencil, /screen pixels/);

// The controller: preview, completion, paint and container.
class Field {
  constructor() { this.value = ''; this.attributes = {}; }
  setAttribute(name, value) { this.attributes[name] = value; }
  blur() {}
}
const fields = new Map();
const $ = id => { if (!fields.has(id)) fields.set(id, new Field()); return fields.get(id); };
const store = new Map(), localStorage = {getItem: key => store.get(key) ?? null, setItem: (key, value) => store.set(key, value)};
const actions = [], toasts = [], appended = [];
let overlays = 0;
const xmlElement = (name, attributes) => ({name, attributes, children: [], append(...children) { this.children.push(...children); }});
const editor = Object.assign(createEditorState(), {tool: 'pencil', zoom: 2, state: {selection: {objects: []}}, overlayFrame: {content: {append: (...items) => appended.push(...items)}}});
const paint = drawingSettings();
const controller = () => createPencilController(editor, {
  $, localStorage, xmlElement, toast: (message, error) => toasts.push([message, Boolean(error)]),
  drawOverlay: () => { overlays++; },
  drawingPaint: closed => drawingPaint(paint, closed),
  drawingContainer: () => editor.scope ? {parent: editor.scope} : {},
  action: async (command, body, label) => { actions.push([command, body, label]); return true; },
}).wire();
const pencil = controller();
assert.equal($('pencil-smoothing').value, '2');
assert.equal($('pencil-close').attributes['aria-pressed'], 'false');
const gestures = canvasGestures({
  point: event => ({x: event.clientX / editor.zoom, y: event.clientY / editor.zoom}), drawOverlay: () => { overlays++; },
  view: {zoom: () => editor.zoom},
  pencil: {preview: pencil.preview, finish: pencil.finish},
});
const event = (x, y, extra = {}) => ({clientX: x, clientY: y, ...extra});
const press = (x, y) => {
  const gesture = {kind: 'pencil', x, y, moved: false};
  gestures.pencil.press(gesture, event(x, y));
  editor.drag = gesture;
  return gesture;
};
// Moves a stroke along screen points, as the canvas does: moved past three
// pixels from the press.
const drag = (gesture, points) => {
  for (const [x, y] of points) {
    gesture.moved ||= Math.hypot(x - gesture.x, y - gesture.y) > 3;
    gestures.pencil.move(gesture, event(x, y));
  }
};
const curve = [];
for (let i = 1; i <= 60; i++) curve.push([100 + 4 * i, 100 + 60 * Math.sin(i / 10)]);

// Below the drag threshold nothing previews; then the fit does, painted with
// the drawing paint, an outline and its points.
const stroke = press(100, 100);
drag(stroke, [[101, 100], [102, 101]]);
assert.equal(stroke.d, undefined); assert.equal(overlays, 0);
drag(stroke, curve);
assert.ok(stroke.d.startsWith('M50 50 C'));
pencil.drawPencil();
const [preview] = appended;
assert.deepEqual(preview.children[0].attributes, {d: stroke.d, fill: 'none', stroke: DRAWING_DEFAULTS.fill, 'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round'});
assert.equal(preview.children[1].attributes['stroke-width'], 0.5, 'the outline is a screen pixel thick');
assert.equal(preview.children.length - 2, strokeNodes(stroke.d), 'each point of the path shows');
// Samples closer than a screen pixel to the last are left out.
const before = stroke.points.length;
gestures.pencil.move(stroke, event(340.5, 100 + 60 * Math.sin(6)));
assert.equal(stroke.points.length, before);
// The path added is the fit previewed, in one edit, with the open path's paint.
const shownD = stroke.d;
editor.drag = null;
await gestures.pencil.release(stroke);
assert.deepEqual(actions.pop(), ['add_path', {d: shownD, fill: 'none', stroke: DRAWING_DEFAULTS.fill, stroke_width: 2}, 'Drawing freehand path…']);

// The samples the browser coalesced between events are each taken.
const quick = press(0, 0);
quick.moved = true;
gestures.pencil.move(quick, {clientX: 40, clientY: 0, getCoalescedEvents: () => [event(10, 5), event(20, 8), event(30, 5), event(40, 0)]});
assert.deepEqual(quick.points, [[0, 0], [5, 2.5], [10, 4], [15, 2.5], [20, 0]]);
editor.drag = null;

// Close shape, pressed mid-stroke, closes the stroke previewed and drawn,
// filled with the drawing paint, into the entered group; it is remembered.
editor.scope = 'group';
const closing = press(100, 100);
drag(closing, curve);
assert.equal(closing.d.endsWith('Z'), false);
const drawn = overlays;
$('pencil-close').onclick();
assert.equal(overlays, drawn + 1, 'the preview is redrawn at once');
assert.equal($('pencil-close').attributes['aria-pressed'], 'true');
pencil.drawPencil();
assert.equal(appended.at(-1).children[0].attributes.fill, DRAWING_DEFAULTS.fill);
assert.ok(closing.d.endsWith(' Z'));
editor.drag = null;
await gestures.pencil.release(closing);
const [command, body] = actions.pop();
assert.equal(command, 'add_path');
assert.deepEqual(body, {d: closing.d, fill: DRAWING_DEFAULTS.fill, stroke: 'none', stroke_width: 2, parent: 'group'});
editor.scope = null;

// Smoothing: out of range is kept to it, not a number puts the field back;
// both settings are remembered for a new window.
$('pencil-smoothing').value = '50'; $('pencil-smoothing').onchange();
assert.equal($('pencil-smoothing').value, '20');
$('pencil-smoothing').value = 'abc'; $('pencil-smoothing').onchange();
assert.equal($('pencil-smoothing').value, '20');
$('pencil-smoothing').value = '6'; $('pencil-smoothing').onchange();
assert.deepEqual(JSON.parse(store.get('vectrify-pencil')), {smoothing: 6, closed: true});
assert.deepEqual(controller().settings(), {smoothing: 6, closed: true});
pencil.wire();
$('pencil-close').onclick();
// More smoothing on the same stroke gives no more points.
const smoother = press(100, 100);
drag(smoother, curve);
assert.ok(strokeNodes(smoother.d) <= strokeNodes(shownD));
// Zooming mid-stroke fits again at the new zoom, preview and path alike.
editor.zoom = 4;
pencil.drawPencil();
assert.equal(appended.at(-1).children[0].attributes.d, smoother.d);
assert.equal(smoother.d, strokePath(fitStroke(smoother.points, {zoom: 4, smoothing: 6}), 4));
editor.drag = null;
await gestures.pencil.release(smoother);
assert.equal(actions.pop()[1].d, smoother.d);
editor.zoom = 2;

// A click, or a stroke too short to fit, draws nothing and says so.
const click = press(100, 100);
editor.drag = null;
await gestures.pencil.release(click);
assert.deepEqual(toasts.pop(), ['Drag to draw a freehand path.', false]);
const short = press(100, 100);
short.moved = true; short.points.push([50.5, 50]);
editor.drag = null;
await gestures.pencil.release(short);
assert.match(toasts.pop()[0], /too short/);
assert.equal(actions.length, 0);

// Escape cancels a stroke under way: nothing is drawn. P picks the tool, and
// in Pencil the 1 key is Close shape.
const keys = createKeysController(editor, {
  $, document: {activeElement: {tagName: 'BODY'}, querySelector: () => null, querySelectorAll: () => []}, gestures, input: {waiting: 0},
  drawOverlay: () => {}, renderDrawing: () => {}, renderInspector: () => {}, later: fn => fn(),
  stage: {classList: {remove() {}}}, stepUp: () => assert.fail('Escape cancels before stepping up'),
});
const escaped = press(100, 100);
drag(escaped, curve);
keys.keyDown({key: 'Escape', code: 'Escape', preventDefault() {}});
assert.equal(editor.drag, null);
assert.equal(actions.length, 0);
assert.equal(keys.TOOL_KEYS.p, 'pencil');

// A stroke pressed while an edit runs is held, then drawn once, as drawn.
const held = new HeldGesture({...event(100, 100), type: 'pointerdown', pointerId: 1, button: 0});
for (const [x, y] of curve) held.record({...event(x, y), type: 'pointermove', pointerId: 1});
held.record({...event(340, 100), type: 'pointerup', pointerId: 1});
let replayed, done;
held.replay({
  down: e => { replayed = press(e.clientX, e.clientY); },
  move: e => drag(replayed, [[e.clientX, e.clientY]]),
  up: () => { editor.drag = null; done = gestures.pencil.release(replayed); },
});
await done;
assert.equal(actions.pop()[1].d, replayed.d);
assert.equal(actions.length, 0);

// The canvas starts a stroke wherever it is pressed, even off the artboard;
// Space or the middle button pans instead.
const started = [];
const canvas = createCanvasController(editor, {
  $: () => ({getBoundingClientRect: () => ({left: 0, top: 0, right: 100, bottom: 100})}),
  stage: {focus() {}, setPointerCapture() {}},
  startGesture: (kind, _event, common) => started.push({kind, deselectOutside: common.deselectOutside}),
});
editor.tool = 'pencil'; editor.drag = null;
canvas.pressStage(event(500, 500, {button: 0}));
assert.deepEqual(started.pop(), {kind: 'pencil', deselectOutside: false});
canvas.pressStage(event(50, 50, {button: 1}));
assert.equal(started.pop().kind, 'pan');
editor.space = true;
canvas.pressStage(event(50, 50, {button: 0}));
assert.equal(started.pop().kind, 'pan');
editor.space = false;
