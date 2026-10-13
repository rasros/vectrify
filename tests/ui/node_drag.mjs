import {createEditorState} from '../../src/vectrify/ui/static/app_state.js';
import {createToolsController} from '../../src/vectrify/ui/static/app_tools.js';
import {createCanvasController} from '../../src/vectrify/ui/static/app_canvas.js';
import {createSelectionController} from '../../src/vectrify/ui/static/app_selection.js';
import {createCanvasEventsController} from '../../src/vectrify/ui/static/app_canvas_events.js';
// Exercise Shift constraints and selection through the editor's node drag code.
import assert from 'node:assert/strict';

import {snapIndex, snapPoint} from '../../src/vectrify/ui/static/snap.js';
import {clickPoint, clickPointPath, pointKey, splitKey} from '../../src/vectrify/ui/static/selection.js';
import {canvasGestures} from '../../src/vectrify/ui/static/gestures.js';

class Point {
  constructor(x, y) { this.x = x; this.y = y; }
  matrixTransform() { return new Point(this.x, this.y); }
}
const nodes = [{id: 'a', command: 'M', values: [20, 30]}, {id: 'b', command: 'L', values: [40, 50]}];
const geometry = {id: 'g', subpaths: [{nodes}]};
let selected = ['p a', 'p b'], targets = [];
const committed = [];
const context = Object.assign(createEditorState(), {
  DOMPoint: Point, snapPoint, snapIndex, clickPoint, clickPointPath, pointKey, splitKey,
  state: {bounds: [0, 0, 100, 100], selection: {objects: ['p']}}, zoom: 1, SNAP_RADIUS: 8,
  geometries: new Map([['p', geometry]]), geometryNodes: () => nodes,
  nodeAt: key => nodes.find(n => n.id === splitKey(key)[1]),
  point: event => new Point(event.clientX, event.clientY),
  snapTargets: () => snapIndex(targets, 8), selectedPoints: () => selected, pointPaths: () => ['p'],
  selectPoints: (objects, points) => { selected = points; },
  svgElement: () => ({hasAttribute() { return false; }, setAttribute() {}}), localToOverlay: () => ({inverse() { return this; }}),
  sharingPaths: () => [],
  startGesture: (kind, event, common, details) => { context.drag = {kind, ...common, ...details}; },
});
context.focusPoint = null, context.activeHandle = null;
const context_tools = createToolsController(context, {
  $: (...args) => context.$(...args),
  DOMPoint: context.DOMPoint,
  action: (...args) => context.action(...args),
  drawOverlay: (...args) => context.drawOverlay(...args),
  drawing: context.drawing,
  filledPath: (...args) => context.filledPath(...args),
  geometryNodes: (...args) => context.geometryNodes(...args),
  localToOverlay: (...args) => context.localToOverlay(...args),
  point: (...args) => context.point(...args),
  pointPaths: (...args) => context.pointPaths(...args),
  setTool: (...args) => context.setTool(...args),
  svgElement: (...args) => context.svgElement(...args),
  toast: (...args) => context.toast(...args),
  xmlElement: (...args) => context.xmlElement(...args),
});
context.snappedDrag = context_tools.snappedDrag;
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
context.pathData = context_canvas.pathData;
context.valuesById = context_canvas.valuesById;
context.restoreValues = context_canvas.restoreValues;
context.carryHandles = context_canvas.carryHandles;
context.previewPointDrag = context_canvas.previewPointDrag;
context.pressPoint = context_canvas.pressPoint;
const snappedDrag = (context.snappedDrag);
const pressPoint = (context.pressPoint);
const previewPointDrag = (context.previewPointDrag);
const gestures = canvasGestures({
  nodes: {
    snap: snappedDrag, preview: previewPointDrag, renderInspector() {}, renderNodeInspector() {},
    select: (objects, points) => { selected = points; },
    finish: () => committed.push(nodes.map(n => [...n.values])),
  },
  drawOverlay() {},
});
const event = (x, y, extra = {}) => ({clientX: x, clientY: y, shiftKey: true, ...extra});
const position = p => [p.x, p.y];
const press = (shift = true) => pressPoint(
  {target: {dataset: {object: 'p', node: 'a', part: 'endpoint'}}}, {shift, moved: false},
);

// Shift can be held before dragging an already selected node: both nodes move.
press();
assert.equal(context.drag.kind, 'node');
assert.deepEqual(selected, ['p a', 'p b']);
context.drag.moved = true;
gestures.node.move(context.drag, event(65, 36));
assert.deepEqual(nodes.map(n => [...n.values]), [[65, 30], [85, 50]]);
await gestures.node.release(context.drag);
assert.deepEqual(committed, [[[65, 30], [85, 50]]]);

// A linked inset's rendered perimeter must not be overwritten by its base
// geometry during the optimistic drag. The server derives it on release.
const writes = [], element = {hasAttribute: () => false, setAttribute: (...args) => writes.push(['body', ...args])};
const insetElement = {hasAttribute: name => name === 'data-vectrify-clip-inset', setAttribute: (...args) => writes.push(['clip', ...args])};
context.svgElement = id => id === 'clip' ? insetElement : element;
context.sharingPaths = () => ['clip'];
gestures.node.move(context.drag, event(70, 36));
assert.ok(writes.some(write => write[0] === 'body'));
assert.ok(!writes.some(write => write[0] === 'clip'));
context.svgElement = () => element;
context.sharingPaths = () => [];

// A Shift-click still removes the selected node, including the last one.
press();
await gestures.node.release(context.drag);
assert.deepEqual(selected, ['p b']);
pressPoint({target: {dataset: {object: 'p', node: 'b', part: 'endpoint'}}}, {shift: true, moved: false});
await gestures.node.release(context.drag);
assert.deepEqual(selected, []);

// Pinned points still toggle on click, and cannot start a drag.
selected = ['p a']; nodes[0].pinned = true;
press();
assert.equal(context.drag.kind, 'point-click');
assert.deepEqual(selected, []);
nodes[0].pinned = false;

// Constrain against the original position, switching axes and releasing Shift.
context.drag = {start: new Point(20, 30), part: 'endpoint', moving: []};
assert.deepEqual(position(snappedDrag(event(65, 36))), [65, 30]);
assert.deepEqual(position(snappedDrag(event(24, 70))), [20, 70]);
assert.deepEqual(position(snappedDrag(event(-10, 28))), [-10, 30]);
assert.deepEqual(position(snappedDrag(event(24, 70, {shiftKey: false}))), [24, 70]);
for (const modifier of ['altKey', 'ctrlKey', 'metaKey']) {
  assert.deepEqual(position(snappedDrag(event(24, 70, {[modifier]: true}))), [20, 70]);
}

// Compatible point targets still snap; closer off-axis points cannot win.
targets = [[63, 31], [67, 30]]; context.drag.snaps = null;
assert.deepEqual(position(snappedDrag(event(64, 35))), [67, 30]);
targets = [[20, 74], [21, 71]]; context.drag.snaps = null;
assert.deepEqual(position(snappedDrag(event(24, 70))), [20, 74]);
targets = [[63, 31]]; context.drag.snaps = null;
assert.deepEqual(position(snappedDrag(event(64, 35))), [64, 30]);
assert.equal(context.drag.snap, null);

// Edges and corners cannot pull a constrained drag away from its axis.
targets = []; context.drag.snaps = null; context.drag.start = new Point(20, 3);
assert.deepEqual(position(snappedDrag(event(96, 5))), [100, 3]);
assert.equal(context.drag.snap.target.y, undefined);
context.drag.start = new Point(3, 30);
assert.deepEqual(position(snappedDrag(event(5, 96))), [3, 100]);
assert.equal(context.drag.snap.target.x, undefined);
context.drag.start = new Point(20, 0);
assert.deepEqual(position(snappedDrag(event(96, 2))), [100, 0]);
assert.equal(context.drag.snap.target.kind, 'point');

// Unconstrained drags retain ordinary snapping in both coordinates.
targets = [[63, 31]]; context.drag.snaps = null;
assert.deepEqual(position(snappedDrag(event(64, 35, {shiftKey: false}))), [63, 31]);

// Points and handles can extend beyond the artboard (including clipped paths).
// Run their clicks through both canvas handlers: release must keep what press
// selected, while an empty click outside must still clear the selection.
let circles = [];
Object.assign(context, {
  tool: 'nodes', space: false, POINT_REACH: 10, gestures, clickCycle: null, drag: null,
  stage: {focus() {}, setPointerCapture() {}, hasPointerCapture: () => false, classList: {remove() {}}},
  $: () => ({getBoundingClientRect: () => ({left: 0, right: 100, top: 0, bottom: 100})}),
  overlay: {querySelectorAll: () => circles}, hitStack: () => [],
  renderNodeInspector() {}, drawOverlay() {},
  selectPoints: (objects, points) => { context.state.selection.objects = objects; selected = points; },
  selectObject: () => { context.state.selection.objects = []; },
});
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
context.nearestPoint = context_selection.nearestPoint;
context.clickEmpty = context_selection.clickEmpty;
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
context.pressStage = context_canvas.pressStage;
context.releaseStage = context_canvas.releaseStage;
const pressStage = (context.pressStage);
const releaseStage = (context.releaseStage);
const canvasClick = async (x, y, extra = {}) => {
  const e = event(x, y, {button: 0, pointerId: 1, timeStamp: 1, shiftKey: false, ...extra});
  pressStage(e);
  await releaseStage(e);
};
const circle = (x, y, node, part = 'endpoint', anchor) => ({
  dataset: {object: 'p', node, part, anchor},
  getBoundingClientRect: () => ({x: x - 2, y: y - 2, width: 4, height: 4}),
});

nodes[0].values = [101, 50]; nodes[1].values = [40, 50];
selected = [];
circles = [circle(101, 50, 'a')];
await canvasClick(101, 50);
assert.deepEqual([...selected], ['p a']);
assert.deepEqual([...context.state.selection.objects], ['p']);

// The reach around an inside point also works from outside the artboard.
selected = []; nodes[0].values = [99, 50];
circles = [circle(99, 50, 'a')];
await canvasClick(104, 50);
assert.deepEqual([...selected], ['p a']);

// Shift-click toggles only the clicked point, preserving its selected peers.
selected = ['p a', 'p b'];
await canvasClick(104, 50, {shiftKey: true});
assert.deepEqual([...selected], ['p b']);

// Pinned points still select on click; off-artboard handles keep their anchor.
selected = []; nodes[0].pinned = true;
await canvasClick(104, 50);
assert.deepEqual([...selected], ['p a']);
nodes[0].pinned = false;
selected = []; nodes[1].values = [105, 50, 90, 50, 80, 50];
circles = [circle(105, 50, 'b', '0', 'p b')];
await canvasClick(105, 50);
assert.deepEqual([...selected], ['p b']);
assert.equal((context.activeHandle?.anchor), 'p b');

circles = [];
await canvasClick(150, 50);
assert.deepEqual([...selected], []);
assert.deepEqual([...context.state.selection.objects], ['p']);
await canvasClick(150, 50);
assert.deepEqual([...context.state.selection.objects], []);
