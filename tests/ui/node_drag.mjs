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
  svgElement: () => ({setAttribute() {}}), localToOverlay: () => ({inverse() { return this; }}),
  sharingPaths: () => [],
  startGesture: (kind, event, common, details) => { context.drag = {kind, ...common, ...details}; },
});
vm.runInContext(`
  let focusPoint = null;
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
