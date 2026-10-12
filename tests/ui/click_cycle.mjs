// Exercise canvas presses and releases through the real picking and resize code.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {canvasGestures} from '../../src/vectrify/ui/static/gestures.js';
import {HeldGesture} from '../../src/vectrify/ui/static/input.js';
import {frameHandle, resizeScale} from '../../src/vectrify/ui/static/resize.js';
import {pickTarget, selectionContains} from '../../src/vectrify/ui/static/selection.js';

const source = readFileSync(new URL('../../src/vectrify/ui/static/app.js', import.meta.url), 'utf8');
const section = (start, end) => source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start)));
const matrix = {inverse() { return this; }};
class Point {
  constructor(x, y) { this.x = x; this.y = y; }
  matrixTransform() { return this; }
}
globalThis.DOMPoint = Point;
const bounds = {left: 0, top: 0, right: 400, bottom: 400};
const frame = {left: 100, top: 100, right: 300, bottom: 200};
const edits = [], picked = [], warnings = [];
let hits = ['front', 'middle', 'back'], refusal = '';
let parents = new Map();
const element = {
  parentElement: {getScreenCTM: () => matrix},
  setAttribute() {}, removeAttribute() {}, closest: () => null,
};
const context = vm.createContext({
  frameHandle, DOMPoint: Point, pickTarget, selectionContains,
  stage: {focus() {}, setPointerCapture() {}, hasPointerCapture: () => false, classList: {remove() {}}},
  overlay: {getScreenCTM: () => matrix},
  $: () => ({getBoundingClientRect: () => bounds}),
  svgElement: () => element,
  screenMatrix: element => element.getScreenCTM(),
  object: id => ({id, parent: parents.get(id) || 'root', tag: 'path', attributes: {}}),
  level: () => 'objects', parents: () => parents,
  hitStack: () => [...hits], topSelection: () => vm.runInContext('state.selection.objects', context),
  point: event => new Point(event.clientX, event.clientY), localToOverlay: () => matrix,
  resizeRefusal: () => refusal, toast: message => warnings.push(message), drawOverlay() {}, revealObject() {}, focusSelection() {},
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
  let state = {root: 'root', selection: {objects: []}}, clickCycle = null, lastPick = null;
  let drag = null, tool = 'select', space = false, scope = null, focusPoint = null;
  let selectionBox = null;
  const DOUBLE_CLICK = 400, FRAME_REACH = 6;
  ${section('async function selectObject(', 'function focusSelection(')}
  ${section('function sameClickSpot(', '// What clicks at a spot pick')}
  ${section('function clickTargets(', 'async function selectAtPoint(')}
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
    box: () => picked.push(['boxed']),
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
  picked.length = 0; edits.length = 0; warnings.length = 0; hits = ['front', 'middle', 'back']; refusal = ''; parents = new Map();
  vm.runInContext('state.selection.objects = []; clickCycle = lastPick = selectionBox = null', context);
};
const click = async (x, y, timeStamp, extra = {}) => {
  const event = {clientX: x, clientY: y, timeStamp, pointerId: 1, button: 0, ...extra};
  press(event); await release(event);
};

// Ordinary clicks toggle a lone object in either object tool, even at its frame.
for (const tool of ['select', 'move']) {
  for (const [x, y] of [[200, 150], [300, 150], [300, 200]]) {
    reset(); hits = ['front']; context.tool = tool; context.frame = frame;
    vm.runInContext('tool = globalThis.tool', context);
    await click(x, y, 0);
    vm.runInContext('selectionBox = frame', context);
    await click(x, y, 600);
    await click(x, y, 1200);
    assert.deepEqual(picked, [['front'], [], ['front']]);
    assert.equal(edits.length, 0);
  }
}
vm.runInContext("tool = 'select'", context);

// Clicking a selected member removes only that member; a new object replaces.
reset(); hits = ['front'];
vm.runInContext("state.selection.objects = ['front', 'middle']", context);
await click(200, 150, 0);
assert.deepEqual(picked, [['middle']]);
await click(200, 150, 600);
assert.deepEqual(picked.at(-1), ['front']);

// Tree clicks use the same toggle without changing explicit selection requests.
reset();
const selectObject = vm.runInContext('selectObject', context);
await selectObject('front', false, true, true);
await selectObject('front', false, true, true);
assert.deepEqual(picked, [['front'], []]);
await selectObject('front');
await selectObject('front');
assert.deepEqual(picked.slice(-2), [['front'], ['front']]);

// Canvas picking reaches the leaf through nested groups, including an entered one.
for (const scope of [null, 'outer', 'inner']) {
  reset(); hits = ['front']; context.scope = scope;
  parents = new Map([['front', 'inner'], ['inner', 'outer'], ['outer', 'root']]);
  vm.runInContext('scope = globalThis.scope', context);
  await click(200, 150, 0);
  await click(200, 150, 600);
  assert.deepEqual(picked, [['front'], []]);
}
vm.runInContext('scope = null', context);

// Select box-selects from a nested leaf; Move drags its selected group.
// Clicking picks the leaf in both tools.
for (const tool of ['select', 'move']) {
  reset(); hits = ['front']; context.tool = tool;
  parents = new Map([['front', 'inner'], ['inner', 'outer'], ['outer', 'root']]);
  vm.runInContext("tool = globalThis.tool; state.selection.objects = ['outer']", context);
  press({clientX: 200, clientY: 150, timeStamp: 0, pointerId: 1, button: 0});
  assert.equal(vm.runInContext('drag.kind', context), tool === 'select' ? 'box' : 'move');
  vm.runInContext('drag.moved = true; gestures[drag.kind].move(drag, {clientX: 230, clientY: 130})', context);
  await release({pointerId: 1});
  assert.deepEqual(edits, tool === 'select' ? [] : [['move', {outer: [30, -20]}]]);
  assert.deepEqual(picked, tool === 'select' ? [['boxed']] : []);
  picked.length = 0;
  await click(200, 150, 600);
  assert.deepEqual(picked, [['front']]);
}
vm.runInContext("tool = 'select'", context);

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

// Move drags thin line frames and the interior of ordinary frames.
for (const [box, x, y] of [
  [{left: 100, top: 100, right: 300, bottom: 100}, 200, 100],
  [{left: 100, top: 100, right: 100, bottom: 200}, 100, 150],
  [{left: 100, top: 100, right: 104, bottom: 200}, 102, 150],
  [frame, 200, 150],
]) {
  reset(); context.frame = box;
  vm.runInContext("tool = 'move'; state.selection.objects = ['front', 'middle']; selectionBox = frame", context);
  const event = {clientX: x, clientY: y, timeStamp: 0, pointerId: 1, button: 0};
  press(event);
  assert.equal(vm.runInContext('drag.kind', context), 'move');
  context.end = {clientX: x + 30, clientY: y - 20};
  vm.runInContext('drag.moved = true; gestures.move.move(drag, end)', context);
  await release({pointerId: 1});
  assert.deepEqual(edits, [['move', {front: [30, -20], middle: [30, -20]}]]);
  assert.deepEqual(picked, [], 'moving preserves the whole selection');
  vm.runInContext("tool = 'select'", context);
}

// Move still supports picking, Shift toggles, and empty-canvas box selection.
reset();
vm.runInContext("tool = 'move'", context);
await click(200, 100, 0);
assert.deepEqual(picked, [['front']]);
context.frame = {left: 100, top: 100, right: 300, bottom: 100};
vm.runInContext('selectionBox = frame', context);
await click(200, 100, 600, {shiftKey: true});
assert.deepEqual(picked.at(-1), []);
hits = [];
press({clientX: 350, clientY: 300, timeStamp: 1200, pointerId: 1, button: 0});
assert.equal(vm.runInContext('drag.kind', context), 'box');
await release({pointerId: 1});
vm.runInContext("tool = 'select'", context);

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

// In Move, edge drags resize without changing the click selection.
vm.runInContext("tool = 'move'", context);
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
