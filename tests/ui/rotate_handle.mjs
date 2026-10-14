// The rotation handle, the pivot and the flips in Move: where they are hit,
// how a drag previews and commits, and Escape. Run by test_rotate_handle.py,
// which checks the printed preview against what the server commits.
import assert from 'node:assert/strict';
import {createEditorState} from '../../src/vectrify/ui/static/app_state.js';
import {createRotatingController} from '../../src/vectrify/ui/static/app_rotating.js';
import {canvasGestures} from '../../src/vectrify/ui/static/gestures.js';

class Matrix {
  constructor(a = 1, b = 0, c = 0, d = 1, e = 0, f = 0) { Object.assign(this, {a, b, c, d, e, f}); }
  multiply(m) {
    return new Matrix(this.a*m.a + this.c*m.b, this.b*m.a + this.d*m.b,
      this.a*m.c + this.c*m.d, this.b*m.c + this.d*m.d,
      this.a*m.e + this.c*m.f + this.e, this.b*m.e + this.d*m.f + this.f);
  }
  inverse() {
    const det = this.a*this.d - this.b*this.c;
    return new Matrix(this.d/det, -this.b/det, -this.c/det, this.a/det,
      (this.c*this.f - this.d*this.e)/det, (this.b*this.e - this.a*this.f)/det);
  }
  translate(x, y) { return this.multiply(new Matrix(1, 0, 0, 1, x, y)); }
  scale(x, y = x) { return this.multiply(new Matrix(x, 0, 0, y)); }
  rotate(degrees) { const r = degrees * Math.PI / 180; return this.multiply(new Matrix(Math.cos(r), Math.sin(r), -Math.sin(r), Math.cos(r))); }
  toString() { return `matrix(${[this.a, this.b, this.c, this.d, this.e, this.f].join(', ')})`; }
}
class Point {
  constructor(x, y) { this.x = x; this.y = y; }
  matrixTransform(m) { return {x: m.a * this.x + m.c * this.y + m.e, y: m.b * this.x + m.d * this.y + m.f}; }
}
const near = (actual, expected, what = '') => assert.ok(Math.abs(actual - expected) < 1e-9, `${what} ${actual} != ${expected}`);

// Two selected objects: one inside a turned, scaled group with a transform of
// its own, one at the top level. The overlay is drawn at twice the size,
// 10 pixels in from the screen's corner.
const group = new Matrix().translate(100, 20).rotate(90).scale(2);
const attributes = (element, values) => Object.assign(element, {
  values, setAttribute(name, value) { values[name] = value; }, removeAttribute(name) { delete values[name]; }, getAttribute: name => values[name] ?? null,
});
const elements = {
  inner: attributes({parentElement: {frame: group}}, {transform: 'translate(3 4)'}),
  plain: attributes({parentElement: {frame: new Matrix()}}, {}),
};
const objects = {
  inner: {id: 'inner', label: 'Inner', inherited_locks: [], attributes: {transform: 'translate(3 4)'}},
  plain: {id: 'plain', label: 'Plain', inherited_locks: [], attributes: {}},
};
const screen = new Matrix().translate(10, 10).scale(2);
const editor = Object.assign(createEditorState(), {
  state: {revision: 4, selection: {objects: ['inner', 'plain'], nodes: []}},
  tool: 'move', zoom: 2, selectionBox: {left: 0, top: 0, right: 100, bottom: 50},
});
const sent = [], drawn = [];
let started = null;
const rotating = createRotatingController(editor, {
  DOMMatrix: Matrix, DOMPoint: Point,
  action: async (command, data) => { sent.push({command, data}); editor.state = {...editor.state, revision: editor.state.revision + 1}; return true; },
  drawOverlay() {},
  localToOverlay: element => element.frame ?? null,
  object: id => objects[id],
  overlay: {getScreenCTM: () => screen},
  point: event => ({x: (event.clientX - 10) / 2, y: (event.clientY - 10) / 2}),
  selectionBounds: () => editor.selectionBox && {...editor.selectionBox},
  startGesture: (kind, _event, common, details) => { started = kind; editor.drag = {...common, kind, ...details}; },
  svgElement: id => elements[id],
  topSelection: () => editor.state.selection.objects,
  xmlElement: (tag, attrs) => ({tag, attrs}),
}).wire();
editor.overlayFrame = {content: {append: (...items) => drawn.push(...items)}};
const onScreen = (x, y) => [10 + 2 * x, 10 + 2 * y];

// The pivot starts at the frame's centre; the handle is 24 screen pixels
// above the middle of its top edge.
assert.deepEqual(rotating.pivot(), {x: 50, y: 25});
assert.equal(rotating.rotationAt(...onScreen(50, -12)), 'rotate');
assert.equal(rotating.rotationAt(...onScreen(50, 25)), 'pivot');
assert.equal(rotating.rotationAt(...onScreen(20, 25)), null);
assert.equal(rotating.rotationCursor(...onScreen(50, -12)), 'grab');
assert.equal(rotating.rotationCursor(...onScreen(50, 25)), 'crosshair');
rotating.drawRotation();
assert.deepEqual(drawn.map(item => item.attrs.class), ['rotate-stem', 'rotate-handle', 'rotate-pivot-halo', 'rotate-pivot-halo', 'rotate-pivot', 'rotate-pivot']);
// Neither shows outside Move, nor while position or geometry is locked; a
// frame too small to grab the pivot apart from moves wherever pressed.
editor.tool = 'select';
assert.equal(rotating.rotationAt(...onScreen(50, -12)), null);
editor.tool = 'move';
objects.plain.inherited_locks = ['geometry'];
assert.equal(rotating.rotationAt(...onScreen(50, -12)), null);
assert.equal(rotating.turnRefusal('flip'), 'Plain: geometry is locked; unlock it to flip');
drawn.length = 0; rotating.drawRotation();
assert.equal(drawn.length, 0);
objects.plain.inherited_locks = [];
editor.selectionBox = {left: 48, top: 24, right: 52, bottom: 26};
assert.equal(rotating.rotationAt(...onScreen(50, 25)), null);
editor.selectionBox = {left: 0, top: 0, right: 100, bottom: 50};

// A drag on the handle turns about the pivot; Shift snaps to 15° steps. The
// preview composes the turn in each object's parent frame.
const gestures = canvasGestures({
  point: () => ({}), drawOverlay() {},
  selection: {pick: () => sent.push({command: 'pick'}), renderDrawing: () => sent.push({command: 'render'})},
  rotate: {result: rotating.rotateDrag, preview: rotating.previewRotate, finish: rotating.finishRotate, placePivot: rotating.placePivot, restorePivot: rotating.restorePivot},
});
const press = (part, x, y) => {
  const [sx, sy] = onScreen(x, y), event = {clientX: sx, clientY: sy};
  rotating.pressRotation(event, {x: sx, y: sy, moved: false}, part);
  return editor.drag;
};
const drag = (gesture, x, y, extra = {}) => {
  const [sx, sy] = onScreen(x, y);
  gesture.moved = true;
  gestures[gesture.kind].move(gesture, {clientX: sx, clientY: sy, shiftKey: false, altKey: false, ctrlKey: false, metaKey: false, ...extra});
};
let gesture = press('rotate', 50, -12);
assert.equal(started, 'rotate');
assert.deepEqual(gesture.pivot, {x: 50, y: 25});
// From straight above the pivot to its right is a quarter turn clockwise.
drag(gesture, 87, 25);
near(gesture.result.angle, 90, 'quarter turn');
drag(gesture, 87, 27, {shiftKey: true});
assert.equal(gesture.result.angle, 90);
drag(gesture, 87, 40);
const angle = gesture.result.angle;
const page = new Matrix().translate(50, 25).rotate(angle).translate(-50, -25);
assert.equal(elements.plain.values.transform, String(page));
assert.ok(elements.inner.values.transform.endsWith(' translate(3 4)'));
// The turned frame is drawn instead of the bounds, with the handle turned too.
drawn.length = 0; rotating.drawRotation();
assert.deepEqual(drawn.map(item => item.attrs.class), ['resize-frame', 'rotate-stem', 'rotate-handle', 'rotate-pivot-halo', 'rotate-pivot-halo', 'rotate-pivot', 'rotate-pivot']);
const preview = {angle, pivot: [50, 25], inner: elements.inner.values.transform, plain: elements.plain.values.transform};
// Escape restores what the objects had.
gestures.rotate.cancel(gesture);
assert.equal(elements.inner.values.transform, 'translate(3 4)');
assert.equal(elements.plain.values.transform, undefined);
// A click on the handle turns nothing; a drag commits one rotate edit, and
// the pivot it turned about stays for the next turn.
gesture = press('rotate', 50, -12);
await gestures.rotate.release(gesture);
assert.deepEqual(sent.splice(0), [{command: 'render'}]);
gesture = press('rotate', 50, -12);
drag(gesture, 87, 40);
await gestures.rotate.release(gesture);
assert.deepEqual(sent.splice(0), [{command: 'rotate', data: {angle, pivot: [50, 25]}}]);
editor.selectionBox = {left: 10, top: 0, right: 80, bottom: 70};
assert.deepEqual(rotating.pivot(), {x: 50, y: 25});
// Any other change, or another selection, puts it back at the centre.
editor.state = {...editor.state, revision: editor.state.revision + 1};
assert.deepEqual(rotating.pivot(), {x: 45, y: 35});
editor.selectionBox = {left: 0, top: 0, right: 100, bottom: 50};

// The pivot drags, snapping to the frame's corners, edge middles and centre
// unless Alt is held; Escape puts it back.
gesture = press('pivot', 50, 25);
assert.equal(started, 'pivot');
// A click on the pivot picks beneath it instead.
await gestures.pivot.release(gesture);
assert.deepEqual(sent.splice(0), [{command: 'pick'}]);
gesture = press('pivot', 50, 25);
drag(gesture, 98, 1);
assert.deepEqual(rotating.pivot(), {x: 100, y: 0});
drag(gesture, 98, 1, {altKey: true});
assert.deepEqual(rotating.pivot(), {x: 98, y: 1});
await gestures.pivot.release(gesture);
editor.state = {...editor.state, selection: {objects: ['plain', 'inner'], nodes: []}};
assert.deepEqual(rotating.pivot(), {x: 98, y: 1}, 'the same objects keep the pivot');
gesture = press('pivot', 98, 1);
drag(gesture, 30, 30);
gestures.pivot.cancel(gesture);
assert.deepEqual(rotating.pivot(), {x: 98, y: 1});
editor.state = {...editor.state, selection: {objects: ['plain'], nodes: []}};
assert.deepEqual(rotating.pivot(), {x: 50, y: 25});
// Rotation by the field turns about the pivot too.
await rotating.rotateBy(30);
assert.deepEqual(sent.splice(0), [{command: 'rotate', data: {angle: 30, pivot: [50, 25]}}]);

// Flips mirror about the frame's centre, whatever the pivot.
editor.pivot = {x: 1, y: 2, key: 'plain', revision: editor.state.revision};
await rotating.flip('horizontal');
await rotating.flip('vertical');
assert.deepEqual(sent.splice(0), [{command: 'flip_horizontal', data: {pivot: [50, 25]}}, {command: 'flip_vertical', data: {pivot: [50, 25]}}]);
assert.equal(rotating.rotationStatus({angle: -372.25}), 'Rotate · -12.3°');

console.log(JSON.stringify(preview));
