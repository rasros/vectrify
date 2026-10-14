// The Transform section's X, Y, W and H: the edits they make of the selection
// frame and how the fields show it. Run by test_transform_fields.py.
import assert from 'node:assert/strict';
import {MIN_SIZE, boxFields, fieldEdit, formatLength, sizeRefusal} from '../../src/vectrify/ui/static/transform.js';
import {createTransformController} from '../../src/vectrify/ui/static/app_transform.js';

const near = (actual, expected) => {
  if (Array.isArray(expected)) { expected.forEach((value, i) => near(actual[i], value)); return; }
  if (typeof expected === 'object') { for (const [key, value] of Object.entries(expected)) near(actual[key], value); return; }
  assert.ok(Math.abs(actual - expected) < 1e-9, `${actual} != ${expected}`);
};
// Where a resize puts the box: what the server's scale about the anchor does.
const scaled = (box, {anchor: [ax, ay], scale: [sx, sy]}) => ({left: ax + (box.left - ax) * sx, top: ay + (box.top - ay) * sy, right: ax + (box.right - ax) * sx, bottom: ay + (box.bottom - ay) * sy});

const box = {left: 10, top: 20, right: 110, bottom: 70};
assert.deepEqual(boxFields(box), {x: 10, y: 20, w: 100, h: 50});
assert.equal(formatLength(12.34567), '12.346');
assert.equal(formatLength(-0.0001), '0');
assert.equal(formatLength(40), '40');

// X and Y move by the difference, in document units.
assert.deepEqual(fieldEdit(box, 'x', 25), {command: 'move', data: {dx: 15, dy: 0}});
assert.deepEqual(fieldEdit(box, 'y', -5), {command: 'move', data: {dx: 0, dy: -25}});
assert.equal(fieldEdit(box, 'x', 10), null);
// W and H scale about the top-left corner, which stays.
let edit = fieldEdit(box, 'w', 150);
near(edit.data, {anchor: [10, 20], scale: [1.5, 1]});
assert.equal(edit.command, 'resize');
near(scaled(box, edit.data), {left: 10, top: 20, right: 160, bottom: 70});
edit = fieldEdit(box, 'h', 25);
near(scaled(box, edit.data), {left: 10, top: 20, right: 110, bottom: 45});
assert.equal(fieldEdit(box, 'h', 50), null);
// The proportion lock scales both sides alike.
edit = fieldEdit(box, 'w', 50, {keepRatio: true});
near(edit.data.scale, [.5, .5]);
near(scaled(box, edit.data), {right: 60, bottom: 45});
edit = fieldEdit(box, 'h', 100, {keepRatio: true});
near(scaled(box, edit.data), {right: 210, bottom: 120});

// No edit makes a dimension zero, negative, tiny or invalid.
for (const value of [0, -4, MIN_SIZE / 2, NaN, Infinity]) assert.ok(fieldEdit(box, 'w', value).refusal, `W ${value}`);
assert.match(fieldEdit(box, 'x', NaN).refusal, /X needs a number/);
assert.match(fieldEdit(box, 'h', 0).refusal, /at least 0\.001/);
assert.match(fieldEdit({left: 0, top: 0, right: 100, bottom: 1}, 'w', .05, {keepRatio: true}).refusal, /Height would be less/);
// A horizontal line has no height: it shows 0, cannot be given one, and
// keeps none when its width scales with the proportion lock.
const line = {left: 0, top: 5, right: 80, bottom: 5 + 1e-9};
assert.deepEqual(boxFields(line), {x: 0, y: 5, w: 80, h: 0});
assert.match(fieldEdit(line, 'h', 10).refusal, /no height to scale/);
assert.match(sizeRefusal(boxFields({left: 3, top: 0, right: 3, bottom: 9}), 'w'), /no width/);
near(fieldEdit(line, 'w', 160, {keepRatio: true}).data.scale, [2, 1]);
assert.equal(sizeRefusal(boxFields(line), 'w'), '');

// The controller: just enough DOM for its fields.
class Field {
  constructor(id) { this.id = id; this.value = ''; this.dataset = {}; this.disabled = false; this.title = ''; this.parentElement = {title: ''}; this.listeners = new Map(); this.attributes = new Map(); }
  addEventListener(name, handler) { this.listeners.set(name, handler); }
  fire(name, event = {}) { const full = {key: '', preventDefault() {}, stopPropagation() { full.stopped = true; }, ...event}; this.listeners.get(name)?.(full); return full; }
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  blur() { this.blurred = true; }
  select() {}
}
const fields = new Map();
const $ = id => { if (!fields.has(id)) fields.set(id, new Field(id)); return fields.get(id); };
const sent = [], toasts = [];
let frame = {...box}, resizeLock = '';
const items = {a: {id: 'a', label: 'Shape', inherited_locks: []}, b: {id: 'b', label: 'Line', inherited_locks: []}};
const editor = {state: {selection: {objects: ['a', 'b'], nodes: []}}, keepRatio: false};
const transform = createTransformController(editor, {
  $,
  action: async (command, data) => { sent.push({command, data}); return true; },
  later: fn => fn(),
  object: id => items[id],
  resizeRefusal: () => resizeLock,
  selectionBounds: () => frame && {...frame},
  toast: (message, error) => toasts.push({message, error}),
  topSelection: () => editor.state.selection.objects,
}).wire();

transform.renderTransform();
assert.deepEqual(['x', 'y', 'w', 'h'].map(f => $(`transform-${f}`).value), ['10', '20', '100', '50']);
assert.ok(['x', 'y', 'w', 'h'].every(f => !$(`transform-${f}`).disabled));
assert.equal($('transform-ratio').attributes.get('aria-pressed'), 'false');

// Typing then committing a field sends one move or resize for the selection.
const type = async (field, value) => { const input = $(`transform-${field}`); input.value = value; input.fire('input'); await input.listeners.get('change')(); };
await type('x', '35');
assert.deepEqual(sent.pop(), {command: 'move', data: {dx: 25, dy: 0}});
await type('w', '200');
near(sent.pop().data, {anchor: [10, 20], scale: [2, 1]});
$('transform-ratio').onclick();
assert.equal(editor.keepRatio, true);
assert.equal($('transform-ratio').attributes.get('aria-pressed'), 'true');
await type('h', '25');
near(sent.pop().data, {anchor: [10, 20], scale: [.5, .5]});
// Committing what is shown changes nothing, and a refused value is put back.
await type('y', '20');
await type('w', '100.0');
assert.equal(sent.length, 0);
await type('w', '0');
assert.equal(sent.length, 0);
assert.deepEqual(toasts.pop(), {message: 'Width must be at least 0.001', error: true});
assert.equal($('transform-w').value, '100');
// A field being typed in keeps its text when the inspector redraws; Escape
// puts the frame's value back.
$('transform-x').value = '7'; $('transform-x').fire('input');
transform.renderTransform();
assert.equal($('transform-x').value, '7');
assert.equal($('transform-x').fire('keydown', {key: 'Escape'}).stopped, true);
assert.equal($('transform-x').value, '10');
assert.equal($('transform-x').blurred, true);
// Leaving a field whose text came back unchanged, so without a change
// event, shows the frame again too.
$('transform-y').value = '20'; $('transform-y').fire('input');
frame = {...box, top: 30, bottom: 80};
$('transform-y').fire('blur');
assert.equal($('transform-y').value, '30');
frame = {...box};

// Locks keep their policy: a position lock disables all four fields, a
// geometry lock only the size; each says why.
items.b.inherited_locks = ['transform']; resizeLock = 'Line: position is locked; unlock it to resize';
transform.renderTransform();
assert.ok(['x', 'y', 'w', 'h'].every(f => $(`transform-${f}`).disabled));
assert.equal($('transform-x').parentElement.title, 'X: Line: position is locked; unlock it to move');
assert.equal(transform.fieldRefusal('w'), resizeLock);
assert.equal($('transform-ratio').disabled, true);
items.b.inherited_locks = ['geometry']; resizeLock = 'Line: geometry is locked; unlock it to resize';
transform.renderTransform();
assert.equal($('transform-x').disabled, false);
assert.equal($('transform-w').disabled, true);
resizeLock = '';

// A thin line's missing side is shown as 0 and cannot be edited.
frame = {...line};
transform.renderTransform();
assert.equal($('transform-h').value, '0');
assert.equal($('transform-h').disabled, true);
assert.match($('transform-h').parentElement.title, /no height to scale/);
assert.equal($('transform-w').disabled, false);
// Without bounds, nothing can be typed.
frame = null;
transform.renderTransform();
assert.ok(['x', 'y', 'w', 'h'].every(f => $(`transform-${f}`).disabled && $(`transform-${f}`).value === ''));
assert.match(transform.fieldRefusal('x'), /no bounds/);
editor.state.selection.objects = [];
assert.equal(transform.fieldRefusal('x'), 'Select objects first');

// Move by offsets in document units through the same move command.
editor.state.selection.objects = ['a'];
$('move-x').value = '4'; $('move-y').value = '-2';
await $('move-apply').onclick();
assert.deepEqual(sent.pop(), {command: 'move', data: {dx: 4, dy: -2}});
assert.equal($('move-x').value, '0');
