// Exercise canvas presses and releases through the real picking and resize code.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {canvasGestures} from '../../src/vectrify/ui/static/gestures.js';
import {HeldGesture} from '../../src/vectrify/ui/static/input.js';
import {frameHandle, resizeScale} from '../../src/vectrify/ui/static/resize.js';

const source = readFileSync(new URL('../../src/vectrify/ui/static/app.js', import.meta.url), 'utf8');
const section = (start, end) => source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start)));
const matrix = {inverse() { return this; }};
class Point {
  constructor(x, y) { this.x = x; this.y = y; }
  matrixTransform() { return this; }
}
const bounds = {left: 0, top: 0, right: 400, bottom: 400};
const frame = {left: 100, top: 100, right: 300, bottom: 200};
const edits = [], picked = [], warnings = [];
let hits = ['front', 'middle', 'back'], refusal = '';
const element = {
  parentElement: {getScreenCTM: () => matrix},
  setAttribute() {}, removeAttribute() {},
};
const context = vm.createContext({
  frameHandle, DOMPoint: Point,
  stage: {focus() {}, setPointerCapture() {}, hasPointerCapture: () => false, classList: {remove() {}}},
  overlay: {getScreenCTM: () => matrix},
  $: () => ({getBoundingClientRect: () => bounds}),
  svgElement: () => element,
  object: id => ({id, tag: 'path', attributes: {}}),
  level: () => 'objects', clickTargets: ids => ids,
  hitStack: () => [...hits], topSelection: () => vm.runInContext('state.selection.objects', context),
  point: event => new Point(event.clientX, event.clientY), localToOverlay: () => matrix,
  resizeRefusal: () => refusal, toast: message => warnings.push(message), drawOverlay() {}, revealObject() {},
  selectedPoints: () => [],
  action: async (command, data) => {
    if (command === 'select') {
      context.selection = data.objects;
      vm.runInContext('state.selection.objects = selection', context);
      picked.push([...data.objects]);
    } else edits.push([command, data]);
    return true;
  },
});
vm.runInContext(`
  let state = {selection: {objects: []}}, clickCycle = null, lastPick = null;
  let drag = null, tool = 'select', space = false, scope = null, focusPoint = null;
  let selectionBox = null;
  const DOUBLE_CLICK = 400, FRAME_REACH = 6;
  ${section('async function selectObject(', 'function focusSelection(')}
  ${section('function sameClickSpot(', '// What clicks at a spot pick')}
  ${section('async function selectAtPoint(', '// The points a point command')}
  ${section('function frameAt(', '// The cursor over the frame')}
  ${section('function pressFrame(', '// Other objects\' bounds')}
  ${section('function startGesture(', '// A press on the canvas')}
  ${section('function pressStage(', '// In Nodes the unselected path')}
  ${section('async function releaseStage(', '// Double-click a group')}
`, context);
context.gestures = canvasGestures({
  point: context.point, drawOverlay() {},
  selection: {
    pick: vm.runInContext('selectAtPoint', context),
    renderDrawing() {},
    move: offsets => edits.push(['move', offsets]),
  },
  resize: {
    result: event => resizeScale(frame, vm.runInContext('drag.handle', context), {x: event.clientX, y: event.clientY}),
    preview() {}, finish: result => edits.push(['resize', result]),
    refuse: context.toast,
  },
});
const press = vm.runInContext('pressStage', context), release = vm.runInContext('releaseStage', context);
const reset = () => {
  picked.length = 0; edits.length = 0; warnings.length = 0; hits = ['front', 'middle', 'back']; refusal = '';
  vm.runInContext('state.selection.objects = []; clickCycle = lastPick = selectionBox = null', context);
};
const click = async (x, y, timeStamp, extra = {}) => {
  const event = {clientX: x, clientY: y, timeStamp, pointerId: 1, button: 0, ...extra};
  press(event); await release(event);
};

// Once a frame appears under the click, its edges and corners still cycle.
for (const [x, y] of [[100, 150], [300, 150], [200, 100], [200, 200], [100, 100], [300, 100], [300, 200], [100, 200]]) {
  reset();
  await click(x, y, 0);
  context.frame = frame;
  vm.runInContext('selectionBox = frame', context);
  await click(x, y, 600);
  await click(x, y, 1200);
  await click(x, y, 1800);
  assert.deepEqual(picked, [['front'], ['middle'], ['back'], ['front']], `cycle at ${x}, ${y}`);
  assert.equal(edits.length, 0, 'clicking never resizes');
}

// A thin shape can have its entire body inside the resize reach.
reset(); context.frame = {left: 100, top: 100, right: 104, bottom: 200};
await click(102, 150, 0); vm.runInContext('selectionBox = frame', context);
await click(102, 150, 600);
assert.deepEqual(picked, [['front'], ['middle']]);

// Shift-click toggles even at the frame; the second click of a double-click stays put.
reset(); context.frame = frame;
await click(300, 150, 0); vm.runInContext('selectionBox = frame', context);
await click(300, 150, 100);
assert.deepEqual(picked, [['front'], ['front']]);
await click(300, 150, 700, {shiftKey: true});
assert.deepEqual(picked.at(-1), []);

// A held click replayed after an edit uses the current hit stack.
reset(); await click(300, 150, 0); vm.runInContext('selectionBox = frame', context);
const held = new HeldGesture({type: 'pointerdown', clientX: 300, clientY: 150, timeStamp: 600, pointerId: 1, button: 0});
held.record({type: 'pointerup', clientX: 300, clientY: 150, timeStamp: 650, pointerId: 1, button: 0});
hits = ['back', 'middle'];
let completion;
held.replay({down: press, up: event => { completion = release(event); }});
await completion;
assert.deepEqual(picked.at(-1), ['back']);

// Actual drags on an edge still resize, with no click selection on release.
reset(); await click(300, 150, 0); vm.runInContext('selectionBox = frame', context);
press({clientX: 300, clientY: 150, timeStamp: 600, pointerId: 1, button: 0});
vm.runInContext('drag.moved = true; gestures.resize.move(drag, {clientX: 350, clientY: 150})', context);
await release({pointerId: 1});
assert.equal(edits[0][0], 'resize');
assert.equal(edits[0][1].sx, 1.25);
assert.deepEqual(picked, [['front']]);

// Resize locks block dragging, while ordinary frame clicks still cycle.
reset(); await click(300, 150, 0); vm.runInContext('selectionBox = frame', context);
refusal = 'Geometry is locked';
await click(300, 150, 600);
assert.deepEqual(picked, [['front'], ['middle']]);
assert.equal(warnings.length, 0);
press({clientX: 300, clientY: 150, timeStamp: 1200, pointerId: 1, button: 0});
vm.runInContext(`
  drag.moved = true;
  gestures.resize.move(drag, {clientX: 350, clientY: 150});
  gestures.resize.move(drag, {clientX: 360, clientY: 150});
`, context);
await release({pointerId: 1});
assert.deepEqual(warnings, ['Geometry is locked']);
assert.equal(edits.length, 0);
assert.deepEqual(picked, [['front'], ['middle']]);
