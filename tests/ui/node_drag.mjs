// Exercise Shift constraints and selection through the editor's node drag code.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {snapIndex, snapPoint} from '../../src/vectrify/ui/static/snap.js';
import {clickPoint, clickPointPath, pointKey, splitKey} from '../../src/vectrify/ui/static/selection.js';
import {canvasGestures} from '../../src/vectrify/ui/static/gestures.js';

const source = readFileSync(new URL('../../src/vectrify/ui/static/app.js', import.meta.url), 'utf8');
const section = (start, end) => source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start)));
class Point {
  constructor(x, y) { this.x = x; this.y = y; }
  matrixTransform() { return new Point(this.x, this.y); }
}
const nodes = [{id: 'a', command: 'M', values: [20, 30]}, {id: 'b', command: 'L', values: [40, 50]}];
const geometry = {id: 'g', subpaths: [{nodes}]};
let selected = ['p a', 'p b'], targets = [];
const committed = [];
const context = vm.createContext({
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
vm.runInContext(`
  let focusPoint = null, activeHandle = null;
  ${section('function snappedDrag(', 'function drawSnap(')}
  ${section('function pathData(', '// Press on a point:')}
  ${section('function pressPoint(', '// Box select:')}
`, context);
const snappedDrag = vm.runInContext('snappedDrag', context);
const pressPoint = vm.runInContext('pressPoint', context);
const previewPointDrag = vm.runInContext('previewPointDrag', context);
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
vm.runInContext(`
  ${section('function nearestPoint(', '// The selected points,')}
  ${section('function clickEmpty(', '// The points a point command')}
  ${section('function startGesture(', '// A press on the canvas')}
  ${section('function pressStage(', '// In Nodes the unselected')}
  ${section('async function releaseStage(', '// Double-click a group')}
`, context);
const pressStage = vm.runInContext('pressStage', context);
const releaseStage = vm.runInContext('releaseStage', context);
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
assert.equal(vm.runInContext('activeHandle?.anchor', context), 'p b');

circles = [];
await canvasClick(150, 50);
assert.deepEqual([...selected], []);
assert.deepEqual([...context.state.selection.objects], ['p']);
await canvasClick(150, 50);
assert.deepEqual([...context.state.selection.objects], []);
