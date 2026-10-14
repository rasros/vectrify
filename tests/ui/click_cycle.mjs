import {createEditorState} from '../../src/vectrify/ui/static/app_state.js';
import {createSelectionController} from '../../src/vectrify/ui/static/app_selection.js';
import {createResizingController} from '../../src/vectrify/ui/static/app_resizing.js';
import {createCanvasEventsController} from '../../src/vectrify/ui/static/app_canvas_events.js';
import {createCanvasController} from '../../src/vectrify/ui/static/app_canvas.js';
// Exercise canvas presses and releases through the real picking and resize code.
import assert from 'node:assert/strict';

import {canvasGestures} from '../../src/vectrify/ui/static/gestures.js';
import {HeldGesture} from '../../src/vectrify/ui/static/input.js';
import {frameHandle, resizeScale} from '../../src/vectrify/ui/static/resize.js';
import {pickTarget, selectionContains} from '../../src/vectrify/ui/static/selection.js';

const matrix = {a: 1, b: 0, c: 0, d: 1, inverse() { return this; }};
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
const context = Object.assign(createEditorState(), {
  frameHandle, DOMPoint: Point, pickTarget, selectionContains,
  stage: {focus() {}, setPointerCapture() {}, hasPointerCapture: () => false, classList: {remove() {}}},
  overlay: {getScreenCTM: () => matrix},
  $: () => ({getBoundingClientRect: () => bounds}),
  svgElement: () => element,
  screenMatrix: element => element.getScreenCTM(),
  object: id => ({id, parent: id === 'root' ? null : parents.get(id) || 'root', tag: 'path', attributes: {}, inherited_locks:[]}),
  level: () => 'objects', parents: () => parents,
  hitStack: () => [...hits], topSelection: () => (context.state.selection.objects),
  point: event => new Point(event.clientX, event.clientY), localToOverlay: () => matrix,
  resizeRefusal: () => refusal, toast: message => warnings.push(message), drawOverlay() {}, revealObject() {}, focusSelection() {},
  selectedPoints: () => [],
  action: async (command, data) => {
    if (command === 'select') {
      context.selection = data.objects;
      (context.state.selection.objects = context.selection);
      picked.push([...data.objects]);
    } else edits.push([command, data]);
    return true;
  },
});
context.state = {root: 'root', selection: {objects: []}, objects: [], bounds: [0, 0, 400, 400]}, context.clickCycle = null, context.lastPick = null;
   context.drag = null, context.tool = 'select', context.space = false, context.scope = null, context.focusPoint = null;
   context.selectionBox = null;
   context.DOUBLE_CLICK = 400, context.FRAME_REACH = 6;
const context_selection = createSelectionController(context, {
  $: (...args) => context.$(...args),
  action: (...args) => context.action(...args),
  contourAt: (...args) => context.contourAt(...args),
  document: context.document,
  focusSelection: (...args) => context.focusSelection(...args),
  geometryNodes: (...args) => context.geometryNodes(...args),
  later: (...args) => context.later(...args),
  level: (...args) => context.level(...args),
  nodeAt: (...args) => context.nodeAt(...args),
  object: (...args) => context.object(...args),
  overlay: context.overlay,
  parents: (...args) => context.parents(...args),
  pointPaths: (...args) => context.pointPaths(...args),
  pointTargetsNow: (...args) => context.pointTargetsNow(...args),
  renderCommands: (...args) => context.renderCommands(...args),
  request: (...args) => context.request(...args),
  revealObject: (...args) => context.revealObject(...args),
  runCommand: (...args) => context.runCommand(...args),
  scheduleStrip: (...args) => context.scheduleStrip(...args),
  selectPoints: (...args) => context.selectPoints(...args),
  selectedPoints: (...args) => context.selectedPoints(...args),
  sharingPaths: (...args) => context.sharingPaths(...args),
  svgElement: (...args) => context.svgElement(...args),
  toast: (...args) => context.toast(...args),
});
context.selectObject = context_selection.selectObject;
context.sameClickSpot = context_selection.sameClickSpot;
context.clickTargets = context_selection.clickTargets;
context.selectAtPoint = context_selection.selectAtPoint;
context.clickEmpty = context_selection.clickEmpty;
const context_resizing = createResizingController(context, {
  resizeRefusal: (...args) => context.resizeRefusal(...args),
  DOMMatrix: context.DOMMatrix,
  DOMPoint: context.DOMPoint,
  hitStack: (...args) => context.hitStack(...args),
  localToOverlay: (...args) => context.localToOverlay(...args),
  object: (...args) => context.object(...args),
  objectBounds: () => null,
  overlay: context.overlay,
  parents: (...args) => context.parents(...args),
  point: (...args) => context.point(...args),
  requestAnimationFrame: context.requestAnimationFrame,
  stage: context.stage,
  startGesture: (...args) => context.startGesture(...args),
  svgElement: (...args) => context.svgElement(...args),
  topSelection: (...args) => context.topSelection(...args),
});
context.frameAt = context_resizing.frameAt;
context.pressFrame = context_resizing.pressFrame;
context.moveDrag = context_resizing.moveDrag;
const context_canvas_events = createCanvasEventsController(context, {
  $: (...args) => context.$(...args),
  ResizeObserver: context.ResizeObserver,
  action: (...args) => context.action(...args),
  cancelStage: (...args) => context.cancelStage(...args),
  cutWithKnife: (...args) => context.cutWithKnife(...args),
  document: context.document,
  drawOverlay: (...args) => context.drawOverlay(...args),
  enterObject: (...args) => context.enterObject(...args),
  finishBox: (...args) => context.finishBox(...args),
  finishPath: (...args) => context.finishPath(...args),
  finishPointDrag: (...args) => context.finishPointDrag(...args),
  fit: (...args) => context.fit(...args),
  hitStack: (...args) => context.hitStack(...args),
  input: context.input,
  knifeEnd: (...args) => context.knifeEnd(...args),
  later: (...args) => context.later(...args),
  level: (...args) => context.level(...args),
  moveStage: (...args) => context.moveStage(...args),
  object: (...args) => context.object(...args),
  point: (...args) => context.point(...args),
  pressStage: (...args) => context.pressStage(...args),
  previewPointDrag: (...args) => context.previewPointDrag(...args),
  previewResize: (...args) => context.previewResize(...args),
  redrawOutline: (...args) => context.redrawOutline(...args),
  releaseStage: (...args) => context.releaseStage(...args),
  renderDrawing: (...args) => context.renderDrawing(...args),
  renderInspector: (...args) => context.renderInspector(...args),
  renderNodeInspector: (...args) => context.renderNodeInspector(...args),
  resizeDrag: (...args) => context.resizeDrag(...args),
  restoreValues: (...args) => context.restoreValues(...args),
  selectAtPoint: (...args) => context.selectAtPoint(...args),
  selectPoints: (...args) => context.selectPoints(...args),
  setTool: (...args) => context.setTool(...args),
  snappedDrag: (...args) => context.snappedDrag(...args),
  stage: context.stage,
  toast: (...args) => context.toast(...args),
  updateView: (...args) => context.updateView(...args),
  zoomAt: (...args) => context.zoomAt(...args),
});
context.startGesture = context_canvas_events.startGesture;
context.gestures = {};
const context_canvas = createCanvasController(context, {
  $: (...args) => context.$(...args),
  DOMPoint: context.DOMPoint,
  action: (...args) => context.action(...args),
  clickEmpty: (...args) => context.clickEmpty(...args),
  clickTargets: (...args) => context.clickTargets(...args),
  drawOverlay: (...args) => context.drawOverlay(...args),
  drawing: context.drawing,
  finishPath: (...args) => context.finishPath(...args),
  frameAt: (...args) => context.frameAt(...args),
  geometryNodes: (...args) => context.geometryNodes(...args),
  gestures: context.gestures,
  hitStack: (...args) => context.hitStack(...args),
  hoverResize: (...args) => context.hoverResize(...args),
  later: (...args) => context.later(...args),
  level: (...args) => context.level(...args),
  loadGeometries: (...args) => context.loadGeometries(...args),
  localToOverlay: (...args) => context.localToOverlay(...args),
  nearKey: (...args) => context.nearKey(...args),
  nearestPoint: (...args) => context.nearestPoint(...args),
  nodeAt: (...args) => context.nodeAt(...args),
  nodeHandles: (...args) => context.nodeHandles(...args),
  object: (...args) => context.object(...args),
  overlay: context.overlay,
  parents: (...args) => context.parents(...args),
  point: (...args) => context.point(...args),
  pointPaths: (...args) => context.pointPaths(...args),
  pressFrame: (...args) => context.pressFrame(...args),
  redrawLines: (...args) => context.redrawLines(...args),
  renderDrawing: (...args) => context.renderDrawing(...args),
  renderInspector: (...args) => context.renderInspector(...args),
  renderNodeInspector: (...args) => context.renderNodeInspector(...args),
  renderStatus: (...args) => context.renderStatus(...args),
  requestAnimationFrame: context.requestAnimationFrame,
  sameClickSpot: (...args) => context.sameClickSpot(...args),
  screenMatrix: (...args) => context.screenMatrix(...args),
  selectPoints: (...args) => context.selectPoints(...args),
  selectedHandle: (...args) => context.selectedHandle(...args),
  selectedPoints: (...args) => context.selectedPoints(...args),
  setTool: (...args) => context.setTool(...args),
  sharingPaths: (...args) => context.sharingPaths(...args),
  stage: context.stage,
  startGesture: (...args) => context.startGesture(...args),
  svgElement: (...args) => context.svgElement(...args),
  toast: (...args) => context.toast(...args),
});
context.pressStage = context_canvas.pressStage;
context.releaseStage = context_canvas.releaseStage;
Object.assign(context.gestures, canvasGestures({
  point: context.point, drawOverlay() {},
  selection: {
    pick: (context.selectAtPoint),
    renderDrawing() {},
    box: () => picked.push(['boxed']),
    move: offsets => edits.push(['move', offsets]),
  },
  moving: {drag: event => context.moveDrag(event)},
  resize: {
    result: event => resizeScale(frame, (context.drag.handle), {x: event.clientX, y: event.clientY}),
    preview() {}, finish: result => edits.push(['resize', result]),
    refuse: context.toast,
  },
}));
const press = (context.pressStage), release = (context.releaseStage);
const reset = () => {
  picked.length = 0; edits.length = 0; warnings.length = 0; hits = ['front', 'middle', 'back']; refusal = ''; parents = new Map();
  (()=>{context.state.selection.objects = []; context.clickCycle = context.lastPick = context.selectionBox = null})();
};
const click = async (x, y, timeStamp, extra = {}) => {
  const event = {clientX: x, clientY: y, timeStamp, pointerId: 1, button: 0, ...extra};
  press(event); await release(event);
};

// Ordinary clicks toggle a lone object in either object tool, even at its frame.
for (const tool of ['select', 'move']) {
  for (const [x, y] of [[200, 150], [300, 150], [300, 200]]) {
    reset(); hits = ['front']; context.tool = tool; context.frame = frame;
    (context.tool = context.tool);
    await click(x, y, 0);
    (context.selectionBox = context.frame);
    await click(x, y, 600);
    await click(x, y, 1200);
    assert.deepEqual(picked, [['front'], [], ['front']]);
    assert.equal(edits.length, 0);
  }
}
(context.tool = 'select');

// Clicking a selected member removes only that member; a new object replaces.
reset(); hits = ['front'];
(context.state.selection.objects = ['front', 'middle']);
await click(200, 150, 0);
assert.deepEqual(picked, [['middle']]);
await click(200, 150, 600);
assert.deepEqual(picked.at(-1), ['front']);

// Tree clicks use the same toggle without changing explicit selection requests.
reset();
const selectObject = (context.selectObject);
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
  (context.scope = context.scope);
  await click(200, 150, 0);
  await click(200, 150, 600);
  assert.deepEqual(picked, [['front'], []]);
}
(context.scope = null);

// Select box-selects from a nested leaf; Move drags its selected group.
// Clicking picks the leaf in both tools.
for (const tool of ['select', 'move']) {
  reset(); hits = ['front']; context.tool = tool;
  parents = new Map([['front', 'inner'], ['inner', 'outer'], ['outer', 'root']]);
  (()=>{context.tool = context.tool; context.state.selection.objects = ['outer']})();
  press({clientX: 200, clientY: 150, timeStamp: 0, pointerId: 1, button: 0});
  assert.equal((context.drag.kind), tool === 'select' ? 'box' : 'move');
  (()=>{context.drag.moved = true; context.gestures[context.drag.kind].move(context.drag, {clientX: 230, clientY: 130})})();
  await release({pointerId: 1});
  assert.deepEqual(edits, tool === 'select' ? [] : [['move', {outer: [30, -20]}]]);
  assert.deepEqual(picked, tool === 'select' ? [['boxed']] : []);
  picked.length = 0;
  await click(200, 150, 600);
  assert.deepEqual(picked, [['front']]);
}
(context.tool = 'select');

// Once a frame appears under the click, its edges and corners still cycle.
for (const [x, y] of [[100, 150], [300, 150], [200, 100], [200, 200], [100, 100], [300, 100], [300, 200], [100, 200]]) {
  reset();
  await click(x, y, 0);
  context.frame = frame;
  (context.selectionBox = context.frame);
  await click(x, y, 600);
  await click(x, y, 1200);
  await click(x, y, 1800);
  assert.deepEqual(picked, [['front'], ['middle'], ['back'], ['front']], `cycle at ${x}, ${y}`);
  assert.equal(edits.length, 0, 'clicking never resizes');
}

// A thin shape can have its entire body inside the resize reach.
reset(); context.frame = {left: 100, top: 100, right: 104, bottom: 200};
await click(102, 150, 0); (context.selectionBox = context.frame);
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
  (()=>{context.tool = 'move'; context.state.selection.objects = ['front', 'middle']; context.selectionBox = context.frame})();
  const event = {clientX: x, clientY: y, timeStamp: 0, pointerId: 1, button: 0};
  press(event);
  assert.equal((context.drag.kind), 'move');
  context.end = {clientX: x + 30, clientY: y - 20};
  (()=>{context.drag.moved = true; context.gestures.move.move(context.drag, context.end)})();
  await release({pointerId: 1});
  assert.deepEqual(edits, [['move', {front: [30, -20], middle: [30, -20]}]]);
  assert.deepEqual(picked, [], 'moving preserves the whole selection');
  (context.tool = 'select');
}

// Move still supports picking, Shift toggles, and empty-canvas box selection.
reset();
(context.tool = 'move');
await click(200, 100, 0);
assert.deepEqual(picked, [['front']]);
context.frame = {left: 100, top: 100, right: 300, bottom: 100};
(context.selectionBox = context.frame);
await click(200, 100, 600, {shiftKey: true});
assert.deepEqual(picked.at(-1), []);
hits = [];
press({clientX: 350, clientY: 300, timeStamp: 1200, pointerId: 1, button: 0});
assert.equal((context.drag.kind), 'box');
await release({pointerId: 1});
(context.tool = 'select');

// Shift-click toggles even at the frame; the second click of a double-click stays put.
reset(); context.frame = frame;
await click(300, 150, 0); (context.selectionBox = context.frame);
await click(300, 150, 100);
assert.deepEqual(picked, [['front'], ['front']]);
await click(300, 150, 700, {shiftKey: true});
assert.deepEqual(picked.at(-1), []);

// A held click replayed after an edit uses the current hit stack.
reset(); await click(300, 150, 0); (context.selectionBox = context.frame);
const held = new HeldGesture({type: 'pointerdown', clientX: 300, clientY: 150, timeStamp: 600, pointerId: 1, button: 0});
held.record({type: 'pointerup', clientX: 300, clientY: 150, timeStamp: 650, pointerId: 1, button: 0});
hits = ['back', 'middle'];
let completion;
held.replay({down: press, up: event => { completion = release(event); }});
await completion;
assert.deepEqual(picked.at(-1), ['back']);

// In Move, edge drags resize without changing the click selection.
(context.tool = 'move');
reset(); await click(300, 150, 0); (context.selectionBox = context.frame);
press({clientX: 300, clientY: 150, timeStamp: 600, pointerId: 1, button: 0});
(()=>{context.drag.moved = true; context.gestures.resize.move(context.drag, {clientX: 350, clientY: 150})})();
await release({pointerId: 1});
assert.equal(edits[0][0], 'resize');
assert.equal(edits[0][1].sx, 1.25);
assert.deepEqual(picked, [['front']]);

// Resize locks block dragging, while ordinary frame clicks still cycle.
reset(); await click(300, 150, 0); (context.selectionBox = context.frame);
refusal = 'Geometry is locked';
await click(300, 150, 600);
assert.deepEqual(picked, [['front'], ['middle']]);
assert.equal(warnings.length, 0);
press({clientX: 300, clientY: 150, timeStamp: 1200, pointerId: 1, button: 0});
context.drag.moved = true;
  context.gestures.resize.move(context.drag, {clientX: 350, clientY: 150});
  context.gestures.resize.move(context.drag, {clientX: 360, clientY: 150});
await release({pointerId: 1});
assert.deepEqual(warnings, ['Geometry is locked']);
assert.equal(edits.length, 0);
assert.deepEqual(picked, [['front'], ['middle']]);
