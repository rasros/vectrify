// The Eyedropper: its choices and the target shown before a click, refusals
// for a missing reference and locked paint, sampling at document points at
// any pan and zoom, one paint edit per sample and cancellation, without a
// browser DOM.
import assert from 'node:assert/strict';

import {MIN_SAMPLE_RADIUS, SAMPLE_PIXELS, createEyedropperController, sampleRadius} from '../../src/vectrify/ui/static/app_eyedropper.js';
import {canvasGestures} from '../../src/vectrify/ui/static/gestures.js';
import {TOOL_LEVEL} from '../../src/vectrify/ui/static/selection.js';
import {createEditorState} from '../../src/vectrify/ui/static/app_state.js';
import {createKeysController} from '../../src/vectrify/ui/static/app_keys.js';
import {createCanvasController} from '../../src/vectrify/ui/static/app_canvas.js';
import {createShapesController} from '../../src/vectrify/ui/static/app_shapes.js';
import {names, hints} from '../../src/vectrify/ui/static/app_config.js';

class Field {
  constructor() { this.value = ''; this.textContent = ''; this.attributes = {}; this.classes = new Set(); this.classList = {toggle: (name, on) => on ? this.classes.add(name) : this.classes.delete(name)}; }
  setAttribute(name, value) { this.attributes[name] = value; }
  blur() {}
}
const fields = new Map();
const $ = id => { if (!fields.has(id)) fields.set(id, new Field()); return fields.get(id); };
const xmlElement = (name, attributes) => ({name, attributes, textContent: '', children: [], append(...children) { this.children.push(...children); }});

const objects = {
  plain: {id: 'plain', tag: 'rect', label: 'Rectangle 1', parent: 'root', inherited_locks: []},
  other: {id: 'other', tag: 'path', label: 'Leaf', parent: 'root', inherited_locks: []},
  locked: {id: 'locked', tag: 'path', label: 'Sky', parent: 'root', inherited_locks: ['paint']},
  stroke: {id: 'stroke', tag: 'path', label: 'Outline', parent: 'root', inherited_locks: ['stroke']},
  clip: {id: 'clip', tag: 'clipPath', label: 'Clip 1', parent: 'root', inherited_locks: []},
  contour: {id: 'contour', tag: 'path', label: 'Clip contour', parent: 'clip', inherited_locks: []},
};
const editor = Object.assign(createEditorState(), {
  tool: 'eyedropper', zoom: 4, overlayFrame: null,
  state: {epoch: 'e', revision: 3, reference: null, selection: {objects: []}},
});
const appended = [], toasts = [], actions = [], requests = [], drawn = [];
let answer = {colour: '#12ab34'}, overlays = 0;
const stageEvents = {};
const eyedropper = createEyedropperController(editor, {
  $, xmlElement, object: id => objects[id],
  plural: (n, one) => `${n} ${one}${n === 1 ? '' : 's'}`,
  drawOverlay: () => { overlays++; },
  request: async (path, body) => { requests.push({path, body}); if (answer instanceof Error) throw answer; return {...answer}; },
  action: async (command, body, label) => { actions.push([command, body, label]); return true; },
  setDrawingColour: (kind, colour) => drawn.push([kind, colour]),
  setTool: async tool => { editor.tool = tool; },
  stage: {addEventListener: (name, listener) => { stageEvents[name] = listener; }},
  toast: (message, error) => toasts.push({message, error: !!error}),
}).wire();
const select = (...ids) => { editor.state.selection.objects = ids; };
const draw = () => {
  appended.length = 0;
  editor.overlayFrame = {content: {append: (...items) => appended.push(...items)}};
  eyedropper.drawEyedropper();
  editor.overlayFrame = null;
  return appended[0];
};
const label = mark => mark.children.find(child => child.name === 'text').textContent;
const swatch = mark => mark.children.find(child => child.name === 'rect');

// The tool: I picks it up, it keeps the selection shown, and it says what a
// click does.
const keys = createKeysController(editor, {$, document: {activeElement: {tagName: 'BODY'}, querySelector: () => null, querySelectorAll: () => []}, gestures: {}});
assert.equal(keys.TOOL_KEYS.i, 'eyedropper');
assert.equal(TOOL_LEVEL.eyedropper, 'objects');
assert.equal(names.eyedropper, 'Eyedropper');
assert.match(hints.eyedropper, /Escape/);

// The radius sampled is a few screen pixels, never under the session's floor.
assert.equal(sampleRadius(1), SAMPLE_PIXELS);
assert.equal(sampleRadius(0.25), SAMPLE_PIXELS * 4);
assert.equal(sampleRadius(100), MIN_SAMPLE_RADIUS);

// Before a click the strip and the mark at the pointer show the source and
// the target: the selection's fill, or with nothing selected the drawing
// paint's.
assert.equal(draw(), undefined, 'no mark until the pointer is over the canvas');
assert.equal($('eyedropper-status').textContent, 'Artwork → drawing fill');
assert.equal($('eyedropper-artwork').attributes['aria-pressed'], 'true');
assert.equal($('eyedropper-fill').attributes['aria-pressed'], 'true');
editor.eyedropperHover = {x: 10, y: 20};
select('plain', 'other');
let mark = draw();
assert.equal($('eyedropper-status').textContent, 'Artwork → fill of 2 objects');
assert.equal(label(mark), 'Artwork → Fill');
assert.equal(swatch(mark).attributes.class, 'eyedropper-swatch fill');
assert.deepEqual(mark.children.slice(0, 2).map(ring => [ring.attributes.cx, ring.attributes.cy, ring.attributes.r]), [[10, 20, 0.75], [10, 20, 0.75]], 'the ring shows the disc sampled, at least 3 screen pixels');
await eyedropper.choose({target: 'stroke'});
mark = draw();
assert.equal(label(mark), 'Artwork → Stroke');
assert.equal(swatch(mark).attributes.class, 'eyedropper-swatch stroke');
assert.equal($('eyedropper-stroke').attributes['aria-pressed'], 'true');
assert.equal($('eyedropper-fill').attributes['aria-pressed'], 'false');
// Leaving the canvas takes the mark away.
stageEvents.pointerleave();
assert.equal(editor.eyedropperHover, null);
assert.equal(draw(), undefined);
editor.eyedropperHover = {x: 10, y: 20};

// Without a reference, sampling it is refused with the reason, before any
// request; the reason shows in the strip and at the pointer.
assert.equal(eyedropper.sourceRefusal('reference'), 'No reference is loaded: load an image under Reference to sample it');
assert.equal(eyedropper.sourceRefusal('artwork'), '');
await eyedropper.choose({source: 'reference'});
mark = draw();
assert.equal(label(mark), 'No reference');
assert.match(mark.attributes.class, /refused/);
assert.equal($('eyedropper-status').textContent, 'No reference is loaded: load an image under Reference to sample it');
assert.ok($('eyedropper-status').classes.has('refused'));
assert.equal(await eyedropper.sample({x: 1, y: 2}), false);
assert.deepEqual(toasts.pop(), {message: 'No reference is loaded: load an image under Reference to sample it', error: true});
assert.equal(requests.length + actions.length, 0);
editor.state.reference = {name: 'r.png', opacity: 0.5};
assert.equal(label(draw()), 'Reference → Stroke');

// Locked paint, or a lock on the target, is refused naming the object;
// clipping contours have no paint.
select('plain', 'locked');
assert.equal(eyedropper.targetRefusal('fill'), 'Sky: paint is locked; unlock it to sample a colour into its fill');
select('stroke');
assert.equal(eyedropper.targetRefusal('fill'), '');
assert.equal(eyedropper.targetRefusal('stroke'), 'Outline: paint is locked; unlock it to sample a colour into its stroke');
assert.equal(label(draw()), 'Paint locked');
assert.equal(await eyedropper.sample({x: 1, y: 2}), false);
assert.equal(toasts.pop().message, 'Outline: paint is locked; unlock it to sample a colour into its stroke');
select('contour');
assert.equal(eyedropper.targetRefusal('fill'), 'Clip contour: definitions and clipping contours are not painted');
assert.equal(requests.length + actions.length, 0);

// A click on the canvas presses the eyedropper wherever it is, the
// selection staying; the sample is taken at the document point under the
// pointer, here at a pan of (30, 40) and a zoom of 4.
const started = [];
const canvas = createCanvasController(editor, {
  $: () => ({getBoundingClientRect: () => ({left: 0, top: 0, right: 100, bottom: 100})}),
  stage: {focus() {}, setPointerCapture() {}},
  startGesture: (kind, _event, common) => started.push({kind, deselectOutside: common.deselectOutside}),
});
canvas.pressStage({clientX: 500, clientY: 500, button: 0});
assert.deepEqual(started.pop(), {kind: 'eyedropper', deselectOutside: false});
const view = {pan: {x: 30, y: 40}, zoom: 4};
const gestures = canvasGestures({
  point: event => ({x: (event.clientX - view.pan.x) / view.zoom, y: (event.clientY - view.pan.y) / view.zoom}),
  drawOverlay: () => { overlays++; },
  eyedropper: {sample: eyedropper.sample},
});
select('plain', 'other');
await eyedropper.choose({source: 'reference', target: 'fill'});
const press = (x, y) => { const gesture = {kind: 'eyedropper', moved: false}; gestures.eyedropper.press(gesture, {clientX: x, clientY: y}); editor.drag = gesture; return gesture; };
let gesture = press(70, 80);
assert.deepEqual(gesture.at, {x: 10, y: 10});
gestures.eyedropper.move(gesture, {clientX: 110, clientY: 120});
assert.equal(draw().children[0].attributes.cx, 20, 'the mark follows the press');
editor.drag = null;
await gestures.eyedropper.release(gesture);
assert.deepEqual(requests.pop(), {path: '/api/sample', body: {epoch: 'e', revision: 3, source: 'reference', at: [20, 20], radius: 0.5}});
// One paint edit, through the paint command the inspector's colours use.
assert.deepEqual(actions.pop(), ['paint', {changes: {fill: '#12ab34'}}, 'Applying the sampled colour…']);
assert.equal(toasts.pop().message, 'Set the fill of 2 objects to #12ab34, sampled from the reference. Undo puts it back.');
// Zoomed out, the same screen point is another document point, sampled wider.
view.zoom = 0.5; editor.zoom = 0.5;
gesture = press(70, 80);
editor.drag = null;
await gestures.eyedropper.release(gesture);
assert.deepEqual(requests.pop().body.at, [80, 80]);
assert.equal(actions.pop()[0], 'paint');
toasts.pop();
assert.equal(requests.at(-1), undefined);

// Escape while pressing cancels: nothing is sampled or changed.
const escapeKeys = createKeysController(editor, {
  $, document: {activeElement: {tagName: 'BODY'}, querySelector: () => null, querySelectorAll: () => []}, gestures, input: {waiting: 0},
  drawOverlay: () => {}, renderDrawing: () => {}, renderInspector: () => {}, later: fn => fn(),
  stage: {classList: {remove() {}}}, stepUp: () => assert.fail('Escape cancels before stepping up'),
});
gesture = press(70, 80);
escapeKeys.keyDown({key: 'Escape', code: 'Escape', preventDefault() {}});
assert.equal(editor.drag, null);
assert.equal(requests.length + actions.length, 0);

// A sample that fails says why and changes nothing.
answer = new Error('Another drawing was opened. This edit belongs to the previous drawing.');
assert.equal(await eyedropper.sample({x: 1, y: 1}), false);
assert.deepEqual(toasts.pop(), {message: 'Another drawing was opened. This edit belongs to the previous drawing.', error: true});
assert.equal(actions.length, 0);
answer = {colour: '#445566'};

// With nothing selected the sample becomes the drawing paint new shapes
// take, without an edit.
select();
await eyedropper.choose({source: 'artwork', target: 'stroke'});
assert.equal(await eyedropper.sample({x: 5, y: 5}), true);
assert.deepEqual(drawn.pop(), ['stroke', '#445566']);
assert.equal(actions.length, 0);
assert.equal(toasts.pop().message, 'Sampled #445566 from the artwork: new shapes and paths take it as their stroke.');

// The drawing paint takes a sampled colour as its picker sets one: turned
// on and remembered.
const store = new Map();
const shapes = createShapesController(createEditorState(), {
  $, document: {activeElement: null}, xmlElement, toast() {}, action: async () => true,
  localStorage: {getItem: key => store.get(key) ?? null, setItem: (key, value) => store.set(key, value)},
}).wire();
shapes.setColour('stroke', '#445566');
assert.equal(shapes.settings().stroke, '#445566');
assert.equal(shapes.settings().strokeOn, true);
assert.equal($('draw-stroke').value, '#445566');
assert.equal(JSON.parse(store.get('vectrify-drawing')).stroke, '#445566');
shapes.setColour('fill', 'not a colour');
assert.notEqual(shapes.settings().fill, 'not a colour');
assert.ok(overlays > 0);
