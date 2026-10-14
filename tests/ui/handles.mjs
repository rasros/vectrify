import {createEditorState} from '../../src/vectrify/ui/static/app_state.js';
import {createCanvasController} from '../../src/vectrify/ui/static/app_canvas.js';
import {createSelectionController} from '../../src/vectrify/ui/static/app_selection.js';
import {createOverlayController} from '../../src/vectrify/ui/static/app_overlay.js';
import {createCommandsController} from '../../src/vectrify/ui/static/app_commands.js';
// Exercise the editor's handle selection, rendering and command routing.
import assert from 'node:assert/strict';

import {clickPoint, clickPointPath, pointKey, splitKey} from '../../src/vectrify/ui/static/selection.js';

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
const context = Object.assign(createEditorState(), {
  DOMPoint: Point, clickPoint, clickPointPath, pointKey, splitKey,
  state: {selection: {objects: ['p']}}, tool: 'nodes', zoom: 1, HANDLE_SPREAD: 16, nearPoint: null,
  geometries: new Map([['p', {id: 'g', get subpaths() { return [{nodes, closed}]; }}]]),
  nodeAt: key => nodes.find(n => n.id === splitKey(key)[1]),
  contourAt: () => ({nodes, closed}), selectedPoints: () => selected, pointPaths: () => ['p'],
  selectPoints: (_objects, points) => { selected = points; (context.activeHandle = null); },
  geometryNodes: () => nodes,
  svgElement: () => ({hasAttribute() { return false; }, setAttribute() {}}), localToOverlay: () => ({inverse() { return this; }}), sharingPaths: () => [], renderDrawing() {},
  startGesture: (_kind, _event, _common, details) => { context.drag = details; },
  renderNodeInspector() {}, renderInspector() {}, drawOverlay() {},
  action: (command, data) => { sent.push({command, data}); },
  level: () => context.tool === 'nodes' ? 'points' : 'objects',
  pointPairs: () => selected.map(splitKey),
  overlayFrame: {content: {append: item => children.push(item)}},
  xmlElement: (name, attrs) => ({name, ...attrs, dataset: {}}),
});
context.activeHandle = null, context.focusPoint = null;
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
context.finishPointDrag = context_canvas.finishPointDrag;
context.pressPoint = context_canvas.pressPoint;
context.stepUp = context_canvas.stepUp;
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
context.nodeHandles = context_selection.nodeHandles;
context.handleCount = context_selection.handleCount;
context.selectedHandle = context_selection.selectedHandle;
context.straightenHandles = context_selection.straightenHandles;
const context_overlay = createOverlayController(context, {
  $: (...args) => context.$(...args),
  CSS: context.CSS,
  DOMPoint: context.DOMPoint,
  drawFlash: (...args) => context.drawFlash(...args),
  drawKnife: (...args) => context.drawKnife(...args),
  drawOverlay: (...args) => context.drawOverlay(...args),
  drawPathDraft: (...args) => context.drawPathDraft(...args),
  drawPoints: (...args) => context.drawPoints(...args),
  drawRedraw: (...args) => context.drawRedraw(...args),
  drawSnap: (...args) => context.drawSnap(...args),
  drawing: context.drawing,
  getComputedStyle: context.getComputedStyle,
  later: (...args) => context.later(...args),
  level: (...args) => context.level(...args),
  localToOverlay: (...args) => context.localToOverlay(...args),
  nodeAt: (...args) => context.nodeAt(...args),
  nodeHandles: (...args) => context.nodeHandles(...args),
  object: (...args) => context.object(...args),
  overlay: context.overlay,
  renderStatus: (...args) => context.renderStatus(...args),
  resizeRefusal: (...args) => context.resizeRefusal(...args),
  svgElement: (...args) => context.svgElement(...args),
  topSelection: (...args) => context.topSelection(...args),
  xmlElement: (...args) => context.xmlElement(...args),
});
context.drawHandles = context_overlay.drawHandles;
const context_commands = createCommandsController(context, {
  $: (...args) => context.$(...args),
  COMMANDS: context.COMMANDS,
  action: (...args) => context.action(...args),
  clickTargets: (...args) => context.clickTargets(...args),
  colorHex: (...args) => context.colorHex(...args),
  commandById: context.commandById,
  commandName: (...args) => context.commandName(...args),
  commandRefusal: (...args) => context.commandRefusal(...args),
  document: context.document,
  drawOverlay: (...args) => context.drawOverlay(...args),
  enable: (...args) => context.enable(...args),
  hitStack: (...args) => context.hitStack(...args),
  joinEnds: (...args) => context.joinEnds(...args),
  later: (...args) => context.later(...args),
  level: (...args) => context.level(...args),
  object: (...args) => context.object(...args),
  paintSwatch: (...args) => context.paintSwatch(...args),
  pointContours: (...args) => context.pointContours(...args),
  pointPairs: (...args) => context.pointPairs(...args),
  pointPaths: (...args) => context.pointPaths(...args),
  renderInspector: (...args) => context.renderInspector(...args),
  resolvedPaint: (...args) => context.resolvedPaint(...args),
  selectObject: (...args) => context.selectObject(...args),
  selectPoints: (...args) => context.selectPoints(...args),
  selectedHandle: (...args) => context.selectedHandle(...args),
  selectedPoints: (...args) => context.selectedPoints(...args),
  stage: context.stage,
  toast: (...args) => context.toast(...args),
  window: context.window,
});
context.deleteSelection = context_commands.deleteSelection;
const call = name => context[name];
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
selected = [];
call('deleteSelection')();
assert.equal(sent.pop().command, 'delete', 'Nodes with no points deletes the selected object');
context.state.selection.nodes = ['pending'];
call('deleteSelection')();
assert.equal(sent.length, 0, 'pending point geometry must not delete the selected object');
context.state.selection.nodes = [];

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

// The inspector explains the open-end limit without waiting for a failed
// Two-handles click; it disappears for interior points and closed contours.
const inspectorFields = new Map();
const inspector = createSelectionController(context, {
  $: id => { if (!inspectorFields.has(id)) inspectorFields.set(id, {}); return inspectorFields.get(id); },
  document: {querySelectorAll: () => []},
  contourAt: () => ({nodes, closed}), nodeAt: context.nodeAt,
  geometryNodes: () => nodes, pointPaths: () => ['p'], selectedPoints: () => selected,
  level: () => 'points', pointTargetsNow: () => ({instances: []}), sharingPaths: () => [],
  renderCommands() {}, scheduleStrip() {},
});
context.activeHandle = null; closed = false;
for (const id of ['a', 'c']) {
  selected = [`p ${id}`]; inspector.renderNodeInspector();
  assert.match(inspectorFields.get('node-hint').textContent, /open.*only have one handle.*Join points/);
  assert.equal(inspectorFields.get('node-hint').hidden, false);
}
selected = ['p b']; inspector.renderNodeInspector();
assert.equal(inspectorFields.get('node-hint').hidden, true);
closed = true; selected = ['p a']; inspector.renderNodeInspector();
assert.equal(inspectorFields.get('node-hint').hidden, true);
