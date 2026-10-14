// Select all and arrow-key nudging through the editor's selection and keys
// controllers. Run by test_select_nudge.py.
import assert from 'node:assert/strict';
import {createEditorState} from '../../src/vectrify/ui/static/app_state.js';
import {createSelectionController} from '../../src/vectrify/ui/static/app_selection.js';
import {createCanvasController} from '../../src/vectrify/ui/static/app_canvas.js';
import {createKeysController} from '../../src/vectrify/ui/static/app_keys.js';
import {inputQueue} from '../../src/vectrify/ui/static/input.js';
import {NUDGE_RUN} from '../../src/vectrify/ui/static/app_config.js';

// A 2D affine map as DOMMatrix has it, with its inverse.
class Matrix {
  constructor(a, b, c, d, e = 0, f = 0) { Object.assign(this, {a, b, c, d, e, f}); }
  inverse() {
    const {a, b, c, d, e, f} = this, det = a * d - b * c;
    return new Matrix(d / det, -b / det, -c / det, a / det, (c * f - d * e) / det, (b * e - a * f) / det);
  }
  apply([x, y]) { return [this.a * x + this.c * y + this.e, this.b * x + this.d * y + this.f]; }
}

// The drawing: a layer group with a path and a rectangle, a path at the top,
// definitions holding a gradient and a clipping path with its boundary.
const objects = [
  {id: 'layer', tag: 'g', label: 'Layer', parent: 'root', resource: false},
  {id: 'inner', tag: 'path', label: 'Inner', parent: 'layer', resource: false},
  {id: 'box', tag: 'rect', label: 'Box', parent: 'layer', resource: false},
  {id: 'free', tag: 'path', label: 'Free', parent: 'root', resource: false},
  {id: 'defs', tag: 'defs', label: 'Definitions', parent: 'root', resource: true},
  {id: 'grad', tag: 'linearGradient', label: 'Gradient', parent: 'defs', resource: true},
  {id: 'clip', tag: 'clipPath', label: 'Clip', parent: 'defs', resource: true},
  {id: 'edge', tag: 'path', label: 'Edge', parent: 'clip', resource: true},
  {id: 'loose', tag: 'linearGradient', label: 'Loose gradient', parent: 'root', resource: false},
];
for (const item of objects) Object.assign(item, {locks: [], inherited_locks: []});
const object = id => objects.find(item => item.id === id);

// Inner's frame is turned 90° and scaled 2 by its group: one document unit
// right is half a unit down its own -y axis. Its second point has handles.
const geometries = () => new Map([
  ['inner', {id: 'gi', subpaths: [{closed: false, nodes: [
    {id: 'a', command: 'M', values: [0, 0]},
    {id: 'b', command: 'C', values: [1, 1, 2, 2, 3, 3]},
    {id: 'c', command: 'L', values: [5, 5]},
  ]}]}],
  ['free', {id: 'gf', subpaths: [{closed: false, nodes: [{id: 'p', command: 'M', values: [10, 10]}, {id: 'q', command: 'L', values: [20, 10]}]}]}],
]);
const frames = {inner: new Matrix(0, 2, -2, 0, 50, 50), free: new Matrix(1, 0, 0, 1)};

const editor = Object.assign(createEditorState(), {
  state: {epoch: 'e', revision: 0, root: 'root', selection: {objects: [], nodes: []}, objects},
  geometries: geometries(),
});
let pointLevel = false, points = [], clock = 0, busy = false, waiters = [];
const actions = [], toasts = [], selected = [];
const queue = inputQueue(() => busy, () => busy ? new Promise(resolve => waiters.push(resolve)) : Promise.resolve());
const finish = () => { busy = false; const ready = waiters; waiters = []; ready.forEach(resolve => resolve()); };
const settle = async () => { for (let i = 0; i < 5; i++) await new Promise(resolve => setTimeout(resolve)); };

const nodes = id => editor.geometries.get(id)?.subpaths.flatMap(s => s.nodes) || [];
const nodeAt = key => { const [id, node] = key.split(' '); return nodes(id).find(n => n.id === node); };
const canvas = createCanvasController(editor, {});
const selection = createSelectionController(editor, {
  action: async (...args) => { actions.push(args); return true; },
  carryHandles: canvas.carryHandles, restoreValues: canvas.restoreValues, valuesById: canvas.valuesById,
  geometryNodes: nodes, nodeAt, object, level: () => pointLevel ? 'points' : 'objects',
  later: fn => queue.run(fn), laterMerged: (...args) => queue.coalesce(...args),
  localToOverlay: element => frames[element.id], svgElement: id => ({id}),
  moveLockRefusal: ids => {
    const item = ids.map(object).find(one => one.inherited_locks.includes('transform'));
    return item ? `${item.label}: position is locked; unlock it to move` : '';
  },
  now: () => clock,
  pointPaths: () => editor.state.selection.objects.filter(id => object(id)?.tag === 'path'),
  selectPoints: (objectIds, keys) => { selected.push([objectIds, keys]); return true; },
  selectedPoints: () => points,
  toast: (message, error) => toasts.push({message, error: !!error}),
  topSelection: () => editor.state.selection.objects,
});

// Select all, at the top: the drawn objects, without definitions, gradients
// or clipping boundaries.
assert.equal(selection.selectAllRefusal(), '');
await selection.selectAll();
assert.deepEqual(actions.pop(), ['select', {objects: ['layer', 'free']}, 'Selecting…']);
// Inside an entered group, its children.
editor.scope = 'layer';
await selection.selectAll();
assert.deepEqual(actions.pop(), ['select', {objects: ['inner', 'box']}, 'Selecting…']);
editor.scope = null;
// A drawing of definitions only has nothing to select.
const kept = objects.splice(0, 4);
assert.equal(selection.selectAllRefusal(), 'The drawing has nothing to select');
objects.unshift(...kept);
// In a point tool, every point of the selected paths, the selection kept.
pointLevel = true;
assert.equal(selection.selectAllRefusal(), 'Select paths first: Select all then picks all their points');
editor.state.selection.objects = ['inner', 'free', 'box'];
assert.equal(selection.selectAllRefusal(), '');
await selection.selectAll();
assert.deepEqual(selected.pop(), [['inner', 'free', 'box'], ['inner a', 'inner b', 'inner c', 'free p', 'free q']]);

// A point nudge moves the points in each path's own frame, through nested
// transforms, with the handles they take along, as a drag does.
points = ['inner b', 'free q'];
const before = JSON.stringify([...editor.geometries]);
const {changes} = selection.pointNudge(1, 0);
assert.deepEqual(Object.keys(changes), ['inner', 'free']);
assert.deepEqual(changes.free, {q: [21, 10]});
// Inner's point moves half a unit along -y in its frame: one unit right on the page.
assert.deepEqual(changes.inner.b, [1, 1, 2, 1.5, 3, 2.5]);
const [x0, y0] = frames.inner.apply([3, 3]), [x1, y1] = frames.inner.apply(changes.inner.b.slice(-2));
assert.deepEqual([x1 - x0, y1 - y0], [1, 0]);
// Its incoming handle rides along; the line after it has none to carry.
assert.deepEqual(changes.inner.c, undefined);
assert.equal(JSON.stringify([...editor.geometries]), before, 'the shown geometry is left as it was');
// Pinned points and locked geometry refuse, saying why.
nodeAt('free q').pinned = true;
assert.equal(selection.pointNudge(1, 0).refusal, 'Pinned points stay put: unpin them to move them');
delete nodeAt('free q').pinned;
object('free').inherited_locks = ['geometry'];
assert.equal(selection.pointNudge(1, 0).refusal, 'Free: geometry is locked; unlock it to move its points');
object('free').inherited_locks = [];

// Nudges pressed while an edit runs add up into one edit, sent when it ends.
editor.state.selection.nodes = ['b', 'q'];
selection.nudge(1, 0);
assert.equal(actions.at(-1)[0], 'move_nodes', 'with nothing running a nudge is sent at once');
const firstRun = actions.at(-1)[1].nudge;
busy = true;
for (let i = 0; i < 10; i++) { clock += 30; selection.nudge(1, 0); }
assert.equal(queue.waiting, 1, 'held repeats wait as one nudge, not ten');
finish(); await settle();
const sent = actions.at(-1);
assert.deepEqual(sent[1].changes.free, {q: [30, 10]}, 'ten repeats move ten units in one edit');
assert.equal(sent[1].nudge, firstRun, 'presses close together are one run: one undo step');
// A pause longer than a run starts a new one, as does another selection.
clock += NUDGE_RUN + 1;
selection.nudge(0, 1);
const secondRun = actions.at(-1)[1].nudge;
assert.notEqual(secondRun, firstRun);
clock += 10;
editor.state.selection.nodes = ['q'];
points = ['free q'];
selection.nudge(0, 1);
assert.notEqual(actions.at(-1)[1].nudge, secondRun);
assert.deepEqual(actions.at(-1)[1].changes, {free: {q: [20, 11]}});

// Objects move through the move command, in document units.
pointLevel = false; points = [];
editor.state.selection = {objects: ['inner', 'free'], nodes: []};
actions.length = 0;
selection.nudge(-10, 0);
const [command, data, label] = actions.pop();
assert.deepEqual([command, data.dx, data.dy, label], ['move', -10, 0, 'Moving selection…']);
assert.equal(typeof data.nudge, 'string');
// Nodes with an object but no points selected moves that object too.
pointLevel = true;
selection.nudge(0, 1);
assert.equal(actions.pop()[0], 'move');
// An unresolved server point selection must not move the entire object.
editor.state.selection.nodes = ['missing'];
selection.nudge(1, 0); await settle();
assert.equal(actions.length, 0);
editor.state.selection.nodes = [];
// A locked position and definitions refuse in Nodes too, and nothing is sent.
object('free').inherited_locks = ['transform'];
selection.nudge(1, 0); await settle();
assert.deepEqual(toasts.pop(), {message: 'Free: position is locked; unlock it to move', error: true});
object('free').inherited_locks = [];
editor.state.selection.objects = ['defs'];
selection.nudge(1, 0); await settle();
assert.equal(toasts.pop().message, 'Definitions keep their place: select drawing objects to move');
assert.equal(actions.length, 0);

// The keys: arrows nudge one document unit, ten with Shift, on the canvas;
// in text fields, dialogs, the object tree and menus they keep their meaning.
const nudges = [], commands = [];
let active = {tagName: 'BODY', closest: () => null}, dialog = null;
const keys = createKeysController(editor, {
  document: {get activeElement() { return active; }, querySelector: () => dialog},
  input: {waiting: 0}, later: fn => fn(), nudge: (dx, dy) => nudges.push([dx, dy]),
  runCommand: id => commands.push(id),
});
function press(key, modifiers = {}) {
  const event = {key, code: key, repeat: false, shiftKey: false, ctrlKey: false, metaKey: false, altKey: false, ...modifiers, prevented: false};
  event.preventDefault = () => { event.prevented = true; };
  keys.keyDown(event);
  return event;
}
editor.state.selection.objects = ['free'];
assert.equal(press('ArrowRight').prevented, true);
press('ArrowLeft', {shiftKey: true}); press('ArrowUp'); press('ArrowDown', {shiftKey: true});
assert.deepEqual(nudges, [[1, 0], [-10, 0], [0, -1], [0, 10]]);
// Held repeats go on to the nudge, which adds them up.
press('ArrowRight', {repeat: true});
assert.deepEqual(nudges.pop(), [1, 0]);
nudges.length = 0;
// Nothing selected: the arrow keeps its default.
editor.state.selection.objects = [];
assert.equal(press('ArrowRight').prevented, false);
editor.state.selection.objects = ['free'];
// Not during a gesture, nor with a modifier.
editor.drag = {kind: 'move'}; press('ArrowRight'); editor.drag = null;
press('ArrowRight', {altKey: true}); press('ArrowRight', {ctrlKey: true});
// The tree, menus and lists navigate with arrows.
active = {tagName: 'DIV', closest: selector => selector.includes('[role="tree"]') ? {} : null};
assert.equal(press('ArrowDown').prevented, false);
// Text fields keep arrows and Ctrl/⌘ A.
active = {tagName: 'INPUT', closest: () => null};
assert.equal(press('ArrowLeft').prevented, false);
assert.equal(press('a', {ctrlKey: true}).prevented, false);
// Dialogs, such as the command palette, keep them too.
active = {tagName: 'BODY', closest: () => null}; dialog = {};
press('ArrowUp'); press('a', {metaKey: true});
dialog = null;
assert.deepEqual(nudges, []);
assert.deepEqual(commands, []);
// On the canvas, Ctrl/⌘ A runs Select all.
assert.equal(press('a', {ctrlKey: true}).prevented, true);
press('a', {metaKey: true});
assert.deepEqual(commands, ['select-all', 'select-all']);
