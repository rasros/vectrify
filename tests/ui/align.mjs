// Align and Distribute in Arrange: the moves they make of the selection's
// bounds, the reference and key object, and the refusals they explain. Run by
// test_align.py.
import assert from 'node:assert/strict';
import {alignMoves, distributeMoves, distributeName, distributeOrder, unionBox} from '../../src/vectrify/ui/static/align.js';
import {createAlignController} from '../../src/vectrify/ui/static/app_align.js';

const near = (actual, expected) => {
  assert.deepEqual(Object.keys(actual).sort(), Object.keys(expected).sort());
  for (const [id, [dx, dy]] of Object.entries(expected)) {
    assert.ok(Math.abs(actual[id][0] - dx) < 1e-9 && Math.abs(actual[id][1] - dy) < 1e-9, `${id}: ${actual[id]} != ${[dx, dy]}`);
  }
};
const moved = (items, moves) => items.map(({id, box}) => {
  const [dx, dy] = moves[id] || [0, 0];
  return {id, box: {left: box.left + dx, top: box.top + dy, right: box.right + dx, bottom: box.bottom + dy}};
});
const size = ({box}) => [box.right - box.left, box.bottom - box.top];

const items = [
  {id: 'a', box: {left: 0, top: 0, right: 10, bottom: 10}},
  {id: 'b', box: {left: 30, top: 5, right: 50, bottom: 25}},
  {id: 'c', box: {left: 70, top: 40, right: 75, bottom: 100}},
];
const frame = unionBox(items.map(item => item.box));
assert.deepEqual(frame, {left: 0, top: 0, right: 75, bottom: 100});

// Alignment lines each box up with the target; what is already there stays.
near(alignMoves(items, 'left', frame), {b: [-30, 0], c: [-70, 0]});
near(alignMoves(items, 'right', frame), {a: [65, 0], b: [25, 0]});
near(alignMoves(items, 'centre', frame), {a: [32.5, 0], b: [-2.5, 0], c: [-35, 0]});
near(alignMoves(items, 'top', frame), {b: [0, -5], c: [0, -40]});
near(alignMoves(items, 'middle', frame), {a: [0, 45], b: [0, 35], c: [0, -20]});
near(alignMoves(items, 'bottom', frame), {a: [0, 90], b: [0, 75]});
// A key object is the target and never moves, even when it is not at the edge.
near(alignMoves(items, 'left', items[1].box, {fixed: 'b'}), {a: [30, 0], c: [-40, 0]});
near(alignMoves(items, 'bottom', items[1].box, {fixed: 'b'}), {a: [0, 15], c: [0, -75]});
// The artboard is a target like any other box.
near(alignMoves(items, 'right', {left: 0, top: 0, right: 200, bottom: 100}), {a: [190, 0], b: [150, 0], c: [125, 0]});

// Equal gaps: the outer two stay, and the space between neighbours' edges is
// the same; sizes never change.
const row = [
  {id: 'p', box: {left: 0, top: 0, right: 10, bottom: 10}},
  {id: 'r', box: {left: 90, top: 0, right: 100, bottom: 10}},
  {id: 'q', box: {left: 15, top: 0, right: 45, bottom: 30}},
  {id: 's', box: {left: 60, top: 0, right: 65, bottom: 5}},
];
assert.deepEqual(distributeOrder(row, 'x').map(item => item.id), ['p', 'q', 's', 'r']);
let moves = distributeMoves(row, 'x', 'gaps');
assert.ok(!moves.p && !moves.r, 'the outer two stay');
let after = distributeOrder(moved(row, moves), 'x');
const gaps = after.slice(1).map((item, i) => item.box.left - after[i].box.right);
gaps.forEach(gap => assert.ok(Math.abs(gap - gaps[0]) < 1e-9, `gaps ${gaps}`));
assert.ok(Math.abs(gaps[0] - 15) < 1e-9);
assert.deepEqual(moved(row, moves).map(size), row.map(size));
assert.ok(Object.values(moves).every(([, dy]) => dy === 0), 'only the chosen axis moves');
// Equal centres: the centres are the same distance apart.
moves = distributeMoves(row, 'x', 'centres');
after = distributeOrder(moved(row, moves), 'x');
const centres = after.map(({box}) => (box.left + box.right) / 2);
centres.slice(1).forEach((centre, i) => assert.ok(Math.abs(centre - centres[i] - 30) < 1e-9, `centres ${centres}`));
assert.deepEqual(moved(row, moves).map(size), row.map(size));
// Vertically too, and objects already evenly spaced stay.
const column = row.map(({id, box}) => ({id, box: {left: box.top, top: box.left, right: box.bottom, bottom: box.right}}));
near(distributeMoves(column, 'y', 'gaps'), Object.fromEntries(Object.entries(distributeMoves(row, 'x', 'gaps')).map(([id, [dx]]) => [id, [0, dx]])));
assert.deepEqual(distributeMoves(moved(row, distributeMoves(row, 'x', 'gaps')), 'x', 'gaps'), {});
assert.deepEqual(distributeMoves(row.slice(0, 2), 'x', 'gaps'), {}, 'two objects have nothing between them');
assert.equal(distributeName('y', 'centres'), 'Distribute vertically: equal centres');

// The controller: just enough DOM for the reference and the key picker.
class Control {
  constructor(id) { this.id = id; this.value = ''; this.hidden = false; this.title = ''; this.children = []; this.listeners = new Map(); this.textContent = ''; }
  addEventListener(name, handler) { this.listeners.set(name, handler); }
  replaceChildren(...children) { this.children = children; }
  fire(name, value) { this.value = value; this.listeners.get(name)({target: this}); }
}
const controls = new Map();
const $ = id => { if (!controls.has(id)) controls.set(id, new Control(id)); return controls.get(id); };
const document = {createElement: tag => new Control(tag)};
const objects = {
  a: {id: 'a', label: 'Square', locks: [], inherited_locks: [], parent: 'root', tag: 'rect'},
  b: {id: 'b', label: 'Panel', locks: [], inherited_locks: [], parent: 'root', tag: 'g'},
  c: {id: 'c', label: 'Bar', locks: [], inherited_locks: [], parent: 'root', tag: 'rect'},
  inside: {id: 'inside', label: 'Dot', locks: [], inherited_locks: [], parent: 'b', tag: 'circle'},
  defs: {id: 'defs', label: 'Definitions', locks: [], inherited_locks: [], parent: 'root', tag: 'defs', resource: true},
};
const bounds = Object.fromEntries(items.map(item => [item.id, item.box]));
const sent = [], toasts = [], drawn = [];
const editor = {align: {reference: 'selection', key: null, seen: []}, overlayFrame: {content: {append: shape => drawn.push(shape)}}};
const select = (...ids) => {
  editor.state = {selection: {objects: ids, nodes: []}, objects: Object.values(objects), bounds: [0, 0, 200, 100]};
  align.renderAlign();
};
const align = createAlignController(editor, {
  $, document,
  action: async (command, data) => { sent.push({command, data}); return true; },
  drawOverlay() {}, later: fn => fn(), level: () => 'objects',
  object: id => objects[id], objectBounds: id => bounds[id] ? {...bounds[id]} : null,
  renderCommands() {}, toast: (message, error) => toasts.push({message, error: !!error}),
  topSelection: () => editor.state.selection.objects,
  xmlElement: (tag, attributes) => ({tag, attributes}),
}).wire();

// Selection: the selection frame is the target; one object alone has none.
select('a');
assert.match(align.alignRefusal('left'), /Select two or more objects to align them to each other, or align to the artboard/);
select('a', 'b', 'c');
assert.equal(align.alignRefusal('left'), '');
await align.align('left');
assert.deepEqual(sent.pop(), {command: 'align', data: {moves: {b: [-30, 0], c: [-70, 0]}}});
assert.match(align.alignDescription('left'), /left edges with the selection frame's/);
// Already lined up: nothing is sent, and the toast says so.
bounds.b = {left: 0, top: 5, right: 20, bottom: 25}; bounds.c = {left: 0, top: 40, right: 5, bottom: 100};
select('a', 'b', 'c');
await align.align('left');
assert.equal(sent.length, 0);
assert.match(toasts.pop().message, /Already lined up/);
Object.assign(bounds, Object.fromEntries(items.map(item => [item.id, item.box])));

// Artboard: one object aligns too.
$('align-reference').fire('change', 'artboard');
assert.equal(editor.align.reference, 'artboard');
assert.equal($('align-key-field').hidden, true);
select('c');
await align.align('right');
assert.deepEqual(sent.pop(), {command: 'align', data: {moves: {c: [125, 0]}}});
assert.match(align.alignDescription('right'), /the artboard's/);

// Key object: the last object added alone is the key; it stays and shows its
// own frame; the picker lists the selection with the key chosen.
$('align-reference').fire('change', 'key');
select('a');
select('a', 'c');
select('a', 'c', 'b');
assert.equal(align.keyObject(), 'b');
assert.equal($('align-key-field').hidden, false);
assert.deepEqual($('align-key').children.map(option => option.value), ['a', 'c', 'b']);
assert.equal($('align-key').value, 'b');
await align.align('left');
assert.deepEqual(sent.pop(), {command: 'align', data: {moves: {a: [30, 0], c: [-40, 0]}}});
drawn.length = 0; align.drawKeyObject();
assert.deepEqual(drawn.map(shape => [shape.tag, shape.attributes.class, shape.attributes.x]), [['rect', 'key-object', 30]]);
// The picker chooses another key; deselecting the key forgets it, and a box
// select of several adds no key, so Align asks for one.
$('align-key').fire('change', 'c');
assert.equal(align.keyObject(), 'c');
await align.align('top');
assert.deepEqual(sent.pop(), {command: 'align', data: {moves: {a: [0, 40], b: [0, 35]}}});
select('a', 'b');
assert.equal(align.keyObject(), null);
select();
select('a', 'b', 'c');
assert.equal(align.keyObject(), null);
assert.match(align.alignRefusal('left'), /Choose the key object/);
assert.equal($('align-key').value, '');
assert.equal($('align-key').children[0].disabled, true);
// A locked key object stays, so it does not refuse; a locked object that
// would move does, by name, as does one locked inside a moving group.
$('align-key').fire('change', 'a');
objects.a.inherited_locks = ['transform'];
assert.equal(align.alignRefusal('left'), '');
objects.inside.locks = ['transform'];
select('a', 'b', 'c');
assert.equal(align.alignRefusal('left'), 'Dot in Panel: position is locked; unlock it to move');
objects.inside.locks = [];
objects.c.inherited_locks = ['transform'];
select('a', 'b', 'c');
assert.equal(align.alignRefusal('left'), 'Bar: position is locked; unlock it to move');
await align.align('left');
assert.equal(sent.length, 0);
assert.deepEqual(toasts.pop(), {message: 'Bar: position is locked; unlock it to move', error: true});
objects.a.inherited_locks = []; objects.c.inherited_locks = [];

// Distribute: three or more objects, the outer two stay; with a key object,
// the key must be one of them.
$('align-reference').fire('change', 'selection');
select('a', 'b');
assert.equal(align.distributeRefusal('x', 'gaps'), 'Select three or more objects to distribute');
select('a', 'b', 'c');
await align.distribute('x', 'gaps');
assert.equal(sent.length, 0, 'the gaps are already 20 each');
assert.equal(toasts.pop().message, 'Already spaced with equal gaps');
await align.distribute('x', 'centres');
assert.deepEqual(sent.pop(), {command: 'distribute', data: {moves: {b: [-1.25, 0]}}});
assert.match(align.distributeDescription('x', 'gaps'), /^Distribute horizontally: equal gaps: .*without resizing/);
$('align-reference').fire('change', 'key');
$('align-key').fire('change', 'b');
assert.match(align.distributeRefusal('x', 'gaps'), /The key object, Panel, lies between the others/);
$('align-key').fire('change', 'c');
assert.equal(align.distributeRefusal('x', 'centres'), '');

// Definitions have no place to align.
select('a', 'defs');
assert.match(align.alignRefusal('left'), /Definitions and clipping boundaries/);
select();
assert.equal(align.alignRefusal('left'), 'Select objects to align');
