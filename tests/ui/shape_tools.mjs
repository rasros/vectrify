// Shape tools: geometry, modifiers, preview, drawing paint and completion,
// without a browser DOM.
import assert from 'node:assert/strict';

import {drawable, lineEnds, roundPoint, shapeBox, shapeFrom, shapeSize, shapeStatus, sizedEnd} from '../../src/vectrify/ui/static/shapes.js';
import {canvasGestures} from '../../src/vectrify/ui/static/gestures.js';
import {HeldGesture} from '../../src/vectrify/ui/static/input.js';
import {TOOL_LEVEL} from '../../src/vectrify/ui/static/selection.js';
import {DRAWING_DEFAULTS, createShapesController, drawingPaint, drawingSettings} from '../../src/vectrify/ui/static/app_shapes.js';
import {createEditorState} from '../../src/vectrify/ui/static/app_state.js';
import {createKeysController} from '../../src/vectrify/ui/static/app_keys.js';
import {createCanvasController} from '../../src/vectrify/ui/static/app_canvas.js';

const close = (actual, expected) => {
  for (const key of Object.keys(expected)) assert.ok(Math.abs(actual[key] - expected[key]) < 1e-9, `${key}: ${actual[key]} ≠ ${expected[key]}`);
};

// Boxes: Shift squares on the longer side, Alt centres on the press, both together.
const start = {x: 0, y: 0}, end = {x: 30, y: -10};
assert.deepEqual(shapeBox(start, end), {x: 0, y: -10, width: 30, height: 10});
assert.deepEqual(shapeBox(start, end, {square: true}), {x: 0, y: -30, width: 30, height: 30});
assert.deepEqual(shapeBox(start, end, {centre: true}), {x: -30, y: -10, width: 60, height: 20});
assert.deepEqual(shapeBox(start, end, {square: true, centre: true}), {x: -30, y: -30, width: 60, height: 60});

// Lines: Shift turns to 45° steps keeping the length, level lines exactly so;
// Alt centres on the press.
assert.deepEqual(lineEnds(start, {x: 10, y: 1}, {snap: true}), {x1: 0, y1: 0, x2: Math.hypot(10, 1), y2: 0});
close(lineEnds(start, {x: 10, y: 9}, {snap: true}), {x2: Math.hypot(10, 9) / Math.SQRT2, y2: Math.hypot(10, 9) / Math.SQRT2});
assert.deepEqual(lineEnds({x: 5, y: 5}, {x: 8, y: 9}, {centre: true}), {x1: 2, y1: 1, x2: 8, y2: 9});

// Ellipses come out as circles when their radii are equal; rectangles round
// at most to half their shorter side.
assert.equal(shapeFrom('ellipse', start, {x: 20, y: 10}).tag, 'ellipse');
assert.deepEqual(shapeFrom('ellipse', start, {x: 20, y: 10}, {square: true}), {tag: 'circle', attributes: {cx: 10, cy: 10, r: 10}});
assert.equal(shapeFrom('rect', start, end, {radius: 50}).attributes.rx, 5);
assert.equal('rx' in shapeFrom('rect', start, end).attributes, false);
assert.equal(drawable(shapeFrom('rect', start, {x: 30, y: 0})), false);
assert.equal(drawable(shapeFrom('line', start, start)), false);
assert.equal(drawable(shapeFrom('line', start, {x: 0, y: 1})), true);

// Sizes and the click size: a line's angle counts counterclockwise.
assert.deepEqual(shapeSize(shapeFrom('line', start, {x: 0, y: -10})), {length: 10, angle: 90});
assert.deepEqual(shapeFrom('rect', {x: 100, y: 100}, sizedEnd('rect', {x: 100, y: 100}, {width: 40, height: 20}, true), {centre: true}).attributes,
  {x: 80, y: 90, width: 40, height: 20});
close(shapeFrom('line', start, sizedEnd('line', start, {length: 10, angle: 90})).attributes, {x1: 0, y1: 0, x2: 0, y2: -10});
assert.equal(shapeStatus(shapeFrom('rect', start, {x: 12.345, y: 6})), 'Rectangle · 12.35 × 6');
assert.equal(shapeStatus(shapeFrom('line', start, {x: 10, y: 0})), 'Line · 10 at 0°');
for (const tool of ['rect', 'ellipse', 'line']) assert.equal(TOOL_LEVEL[tool], 'create');
// Pointer positions round to a tenth of a screen pixel, in decimal places.
assert.deepEqual(roundPoint({x: 50.000001118, y: 49.96}, 1.92), {x: 50, y: 49.96});
assert.deepEqual(roundPoint({x: 50.04, y: 7.3}, 0.5), {x: 50, y: 7.3});
assert.deepEqual(roundPoint({x: 50.4, y: 7.6}, 0.05), {x: 50, y: 8});
assert.deepEqual(roundPoint({x: 1.23456, y: 0}, 256), {x: 1.2346, y: 0});

// Remembered settings: junk and missing values fall back, and something
// always paints.
assert.deepEqual(drawingSettings({fill: 'red', width: -1, size: {width: 'x'}}), drawingSettings());
assert.equal(drawingSettings({fillOn: false, strokeOn: false}).fillOn, true);
assert.deepEqual(drawingPaint(drawingSettings(), true), {fill: DRAWING_DEFAULTS.fill, stroke: 'none', stroke_width: 2});
assert.deepEqual(drawingPaint(drawingSettings(), false), {fill: 'none', stroke: DRAWING_DEFAULTS.fill, stroke_width: 2});

class Field { constructor() { this.value = ''; this.checked = false; } blur() {} }
const fields = new Map();
const $ = id => { if (!fields.has(id)) fields.set(id, new Field()); return fields.get(id); };
const store = new Map(), localStorage = {getItem: key => store.get(key) ?? null, setItem: (key, value) => store.set(key, value)};
const actions = [], toasts = [], appended = [];
let overlays = 0;
const xmlElement = (name, attributes) => ({name, attributes, children: [], append(...children) { this.children.push(...children); }});
const editor = Object.assign(createEditorState(), {zoom: 2, state: {selection: {objects: []}}, overlayFrame: {content: {append: (...items) => appended.push(...items)}}});
const document = {activeElement: null, querySelector: () => null, querySelectorAll: () => []};
const controller = deps => createShapesController(editor, {
  $, document, localStorage, xmlElement, toast: message => toasts.push(message),
  action: async (command, body, label) => { actions.push([command, body, label]); return true; }, ...deps,
}).wire();
const shapes = controller();
const gestures = canvasGestures({
  point: event => ({x: event.clientX, y: event.clientY}), drawOverlay: () => { overlays++; },
  shapes: {preview: shapes.preview, finish: shapes.finish, cancel: shapes.cancel},
});
const event = (x, y, extra = {}) => ({clientX: x, clientY: y, shiftKey: false, altKey: false, ...extra});
const press = (tool, x, y, extra) => {
  const gesture = {kind: 'shape', tool, x, y, moved: false};
  gestures.shape.press(gesture, event(x, y, extra));
  editor.drag = gesture;
  return gesture;
};
assert.equal($('draw-fill').value, DRAWING_DEFAULTS.fill);
assert.equal($('draw-fill-on').checked, true);
assert.equal($('shape-width').value, '100');

// Below the drag threshold nothing previews. A drag previews the shape
// painted as it will be, and the strip shows its size as it changes.
const rect = press('rect', 10, 10);
gestures.shape.move(rect, event(50, 30));
assert.equal(rect.shape, undefined); assert.equal(overlays, 0);
rect.moved = true;
gestures.shape.move(rect, event(50, 30, {shiftKey: true}));
assert.deepEqual(rect.shape, {tag: 'rect', attributes: {x: 10, y: 10, width: 40, height: 40}});
assert.equal($('shape-width').value, '40');
shapes.drawShape();
const [preview] = appended;
assert.equal(preview.children[0].name, 'rect');
assert.deepEqual(preview.children[0].attributes, {x: 10, y: 10, width: 40, height: 40, fill: DRAWING_DEFAULTS.fill, stroke: 'none', 'stroke-width': 2});
assert.equal(preview.children[1].attributes['stroke-width'], 0.5, 'the outline is a screen pixel thick');
// Pressing Alt without moving centres it at once.
gestures.shape.modifiers(rect, {shiftKey: false, altKey: true});
assert.deepEqual(rect.shape.attributes, {x: -30, y: -10, width: 80, height: 40});
editor.drag = null;
await gestures.shape.release(rect);
assert.deepEqual(actions.pop(), ['add_shape', {shape: 'rect', x: -30, y: -10, width: 80, height: 40, fill: DRAWING_DEFAULTS.fill, stroke: 'none', stroke_width: 2}, 'Drawing rectangle…']);
assert.deepEqual(JSON.parse(store.get('vectrify-drawing')).size, {width: 80, height: 40, length: 100, angle: 0}, 'the size drawn is remembered');

// In an entered group the shape goes there. A click places the remembered
// size, here around the click.
editor.scope = 'group';
const click = press('ellipse', 100, 100, {altKey: true});
await gestures.shape.release(click);
assert.deepEqual(actions.pop()[1], {shape: 'ellipse', cx: 100, cy: 100, rx: 40, ry: 20, fill: DRAWING_DEFAULTS.fill, stroke: 'none', stroke_width: 2, parent: 'group'});
editor.scope = null;

// A line takes the stroke, or the fill colour without one, and Shift turns it
// to 45° steps.
const line = press('line', 0, 0);
line.moved = true;
gestures.shape.move(line, event(10, 1, {shiftKey: true}));
await gestures.shape.release(line);
const [, lineBody, lineLabel] = actions.pop();
assert.equal(lineLabel, 'Drawing line…');
assert.deepEqual({...lineBody, x2: Math.round(lineBody.x2 * 1000) / 1000}, {shape: 'line', x1: 0, y1: 0, x2: 10.05, y2: 0, fill: 'none', stroke: DRAWING_DEFAULTS.fill, stroke_width: 2});

// A drag without a width and a height draws nothing and says why.
const flat = press('rect', 0, 0);
flat.moved = true;
gestures.shape.move(flat, event(40, 0));
await gestures.shape.release(flat);
assert.equal(actions.length, 0);
assert.match(toasts.pop(), /width and a height/);
assert.equal($('shape-width').value, '80', 'the strip shows the remembered size again');

// Escape cancels a drag under way: nothing is drawn and the strip shows the
// remembered size again.
const keys = createKeysController(editor, {
  $, document: Object.assign(document, {activeElement: {tagName: 'BODY'}}), gestures, input: {waiting: 0},
  drawOverlay: () => {}, renderDrawing: () => {}, renderInspector: () => {}, later: fn => fn(),
  stage: {classList: {remove() {}}}, stepUp: () => assert.fail('Escape cancels before stepping up'),
});
const escaped = press('rect', 0, 0);
escaped.moved = true;
gestures.shape.move(escaped, event(70, 70));
assert.equal($('shape-width').value, '70');
keys.keyDown({key: 'Shift', code: 'ShiftLeft', shiftKey: true, altKey: false, preventDefault() {}});
assert.equal(escaped.square, true, 'Shift changes the drag as it is pressed');
keys.keyDown({key: 'Escape', code: 'Escape', preventDefault() {}});
assert.equal(editor.drag, null);
assert.equal($('shape-width').value, '80');
assert.equal(actions.length, 0);
assert.equal(keys.TOOL_KEYS.m, 'rect'); assert.equal(keys.TOOL_KEYS.e, 'ellipse'); assert.equal(keys.TOOL_KEYS.l, 'line');

// A shape pressed while an edit runs is held, then drawn once, as dragged.
const held = new HeldGesture({...event(0, 0), type: 'pointerdown', pointerId: 1, button: 0});
held.record({...event(20, 30), type: 'pointermove', pointerId: 1});
held.record({...event(20, 30), type: 'pointerup', pointerId: 1});
let replayed, done;
held.replay({
  down: e => { replayed = press('ellipse', e.clientX, e.clientY); },
  move: e => { replayed.moved = true; gestures.shape.move(replayed, e); },
  up: () => { editor.drag = null; done = gestures.shape.release(replayed); },
});
await done;
assert.deepEqual(actions.pop()[1].rx, 10);
assert.equal(actions.length, 0);

// The drawing paint: turning the fill off with the stroke off turns the
// stroke on; colours turn their paint on; widths must be positive. Draw path
// and the shapes take it alike, and it is remembered.
$('draw-fill-on').checked = false; $('draw-fill-on').onchange();
assert.deepEqual(shapes.paint(true), {fill: 'none', stroke: DRAWING_DEFAULTS.stroke, stroke_width: 2});
$('draw-fill').value = '#112233'; $('draw-fill').oninput();
$('draw-stroke').value = '#445566'; $('draw-stroke').oninput();
$('draw-width').value = '-3'; $('draw-width').onchange();
assert.equal($('draw-width').value, '2');
$('draw-width').value = '0.5'; $('draw-width').onchange();
$('shape-radius').value = '6'; $('shape-radius').onchange();
assert.deepEqual(shapes.paint(true), {fill: '#112233', stroke: '#445566', stroke_width: 0.5});
assert.deepEqual(shapes.paint(false), {fill: 'none', stroke: '#445566', stroke_width: 0.5});
const rounded = press('rect', 0, 0);
rounded.moved = true;
gestures.shape.move(rounded, event(30, 20));
assert.equal(rounded.shape.attributes.rx, 6);
editor.drag = null;
const again = controller();
assert.deepEqual(again.settings(), shapes.settings(), 'a new window remembers the drawing paint and sizes');
assert.equal($('shape-radius').value, '6');

// The canvas starts a shape wherever it is pressed, even off the artboard,
// in each shape tool.
const started = [];
const canvas = createCanvasController(editor, {
  $: () => ({getBoundingClientRect: () => ({left: 0, top: 0, right: 100, bottom: 100})}),
  stage: {focus() {}, setPointerCapture() {}},
  startGesture: (kind, _event, common, details) => started.push({kind, deselectOutside: common.deselectOutside, ...details}),
});
for (const tool of ['rect', 'ellipse', 'line']) {
  editor.tool = tool; editor.drag = null;
  canvas.pressStage(event(500, 500, {button: 0}));
  assert.deepEqual(started.pop(), {kind: 'shape', deselectOutside: false, tool});
}
