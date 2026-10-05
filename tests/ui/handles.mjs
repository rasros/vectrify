// Exercise the editor's handle selection, rendering and command routing.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {clickPoint, clickPointPath, pointKey, splitKey} from '../../src/vectrify/ui/static/selection.js';

const source = readFileSync(new URL('../../src/vectrify/ui/static/app.js', import.meta.url), 'utf8');
const section = (start, end) => source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start)));
class Point {
  constructor(x, y) { this.x = x; this.y = y; }
  matrixTransform() { return this; }
}
let nodes = [
  {id: 'a', command: 'M', values: [0, 0]},
  {id: 'b', command: 'C', values: [3, 1, 8, 4, 10, 0]},
  {id: 'c', command: 'C', values: [14, 2, 17, 9, 20, 10]},
];
let selected = ['p b'], closed = false;
const sent = [], children = [];
const context = vm.createContext({
  DOMPoint: Point, clickPoint, clickPointPath, pointKey, splitKey,
  state: {selection: {objects: ['p']}}, tool: 'nodes', zoom: 1, HANDLE_SPREAD: 16, nearPoint: null,
  geometries: new Map([['p', {id: 'g', get subpaths() { return [{nodes, closed}]; }}]]),
  nodeAt: key => nodes.find(n => n.id === splitKey(key)[1]),
  contourAt: () => ({nodes, closed}), selectedPoints: () => selected, pointPaths: () => ['p'],
  selectPoints: (_objects, points) => { selected = points; vm.runInContext('activeHandle = null', context); },
  geometryNodes: () => nodes,
  svgElement: () => ({setAttribute() {}}), localToOverlay: () => ({inverse() { return this; }}), sharingPaths: () => [], renderDrawing() {},
  startGesture: (_kind, _event, _common, details) => { context.drag = details; },
  renderNodeInspector() {}, renderInspector() {}, drawOverlay() {},
  action: (command, data) => { sent.push({command, data}); },
  level: () => context.tool === 'nodes' ? 'points' : 'objects',
  pointPairs: () => selected.map(splitKey),
  overlayFrame: {content: {append: item => children.push(item)}},
  xmlElement: (name, attrs) => ({name, ...attrs, dataset: {}}),
});
vm.runInContext(`let activeHandle = null, focusPoint = null;
  ${section('function pathData(', '// Press on a point:')}
  ${section('async function finishPointDrag(', "stage.addEventListener('pointerup'")}
  ${section('function nodeHandles(', '// The point section')}
  ${section('function pressPoint(', '// Box select:')}
  ${section('function drawHandles(', '// The rubber band')}
  ${section('function deleteSelection(', 'const segmentPicked')}
  ${section('async function stepUp()', "stage.addEventListener('pointercancel'")}
`, context);
const call = name => vm.runInContext(name, context);
const handle = () => call('selectedHandle')();
const press = (node, part, anchor) => call('pressPoint')({target: {dataset: {object: 'p', node, part, anchor}}}, {shift: false});
const plain = value => JSON.parse(JSON.stringify(value));

// The outgoing control lives on the next segment; deletion targets that slot,
// while straightening acts on its owning anchor and preserves the chosen side.
press('c', '0', 'p b');
assert.deepEqual(plain(handle()), {object: 'p', node: 'c', offset: 0, anchor: 'p b'});
call('drawHandles')('p b');
const circles = children.filter(n => n.name === 'circle');
assert.equal(circles[0].class, 'handle');
assert.equal(circles[1].class, 'handle selected');
assert.equal(circles[1].dataset.anchor, 'p b');
call('straightenHandles')();
assert.deepEqual(plain(sent.pop()), {command: 'straighten_handles', data: {points: [['p', 'b']], aligned: true, side: 0}});
call('deleteSelection')();
assert.deepEqual(plain(sent.pop()), {command: 'delete_handle', data: {object: 'p', node: 'c', offset: 0}});

// Escape clears the handle first, leaving the anchor selected.
await call('stepUp')();
assert.equal(handle(), null);
assert.deepEqual(plain(selected), ['p b']);
call('deleteSelection')();
assert.equal(sent.pop().command, 'delete_node');

// Clicking a hover handle selects its anchor, even when its segment endpoint
// is a different node. Multiple selected anchors do not change its target.
selected = [];
press('c', '0', 'p b');
assert.deepEqual(plain(selected), ['p b']);
assert.ok(handle());
selected = ['p a', 'p b'];
press('b', '2', 'p b');
call('straightenHandles')();
assert.deepEqual(plain(sent.pop()).data, {points: [['p', 'b']], aligned: true, side: 2});
press('b', 'endpoint');
assert.equal(handle(), null);
call('straightenHandles')();
assert.deepEqual(plain(sent.pop()).data, {points: [['p', 'a'], ['p', 'b']], aligned: true});

// A retracted control, removed segment, changed selection or tool cannot
// leave a stale handle selection that intercepts later Delete presses.
selected = ['p b'];
press('c', '0', 'p b');
nodes[2].values.splice(0, 2, 10, 0);
assert.equal(handle(), null);
nodes[2].values.splice(0, 2, 14, 2);
press('c', '0', 'p b');
nodes[2].command = 'L';
assert.equal(handle(), null);
nodes[2].command = 'C';
press('c', '0', 'p b');
selected = ['p a'];
assert.equal(handle(), null);
selected = ['p b']; context.tool = 'select';
assert.equal(handle(), null);
call('deleteSelection')();
assert.equal(sent.pop().command, 'delete');
context.tool = 'nodes';

// Either representation of a closed seam owns its incoming handle.
closed = true;
nodes = [
  {id: 'a', command: 'M', values: [0, 0]},
  {id: 'b', command: 'C', values: [3, 1, 8, 4, 10, 0]},
  {id: 'c', command: 'C', values: [14, 2, -2, -4, 0, 0]},
];
for (const anchor of ['p a', 'p c']) {
  selected = [anchor];
  press('c', '2', anchor);
  assert.equal(handle().anchor, anchor);
  call('deleteSelection')();
  assert.deepEqual(plain(sent.pop()).data, {object: 'p', node: 'c', offset: 2});
}

// Toggle state determines the next press; disabling leaves positions intact.
closed = false;
nodes = [
  {id: 'a', command: 'M', values: [0, 0]},
  {id: 'b', command: 'C', values: [3, 1, 8, -1, 10, 0], handles_aligned: true},
  {id: 'c', command: 'C', values: [14, 2, 17, 9, 20, 10]},
];
selected = ['p b'];
press('c', '0', 'p b');
call('straightenHandles')();
assert.deepEqual(plain(sent.pop()).data, {points: [['p', 'b']], aligned: false, side: 0});

// During an aligned drag, the opposite follows in the preview and keeps its
// own length; committing includes the slot so the backend enforces it too.
call('previewPointDrag')(new Point(10, 8));
assert.equal(nodes[1].values[2], 10);
assert.ok(Math.abs(nodes[1].values[3] + Math.sqrt(5)) < 1e-9);
assert.deepEqual(plain(nodes[2].values.slice(0, 2)), [10, 8]);
await call('finishPointDrag')(context.drag);
assert.equal(sent.at(-1).command, 'node');
assert.equal(sent.at(-1).data.handle_offset, 0);
assert.deepEqual(plain(nodes[1].values.slice(2, 4)), [8, -1]);

nodes[1].handles_aligned = false;
press('c', '0', 'p b');
call('previewPointDrag')(new Point(10, 8));
assert.deepEqual(plain(nodes[1].values.slice(2, 4)), [8, -1]);
