// The Gradient tool's canvas handles: where they are drawn and hit, how a
// drag previews, snaps and commits, Escape, locks and the shared/private
// status. Run by test_gradient_handles.py, which commits the printed edits
// and checks the committed endpoints land where the drags put them.
import assert from 'node:assert/strict';
import {createEditorState} from '../../src/vectrify/ui/static/app_state.js';
import {createGradientsController} from '../../src/vectrify/ui/static/app_gradients.js';
import {canvasGestures} from '../../src/vectrify/ui/static/gestures.js';
import {constrainEndpoint, endpointNumber, endpointText, unitsMatrix} from '../../src/vectrify/ui/static/gradient.js';

class Matrix {
  constructor(values = [1, 0, 0, 1, 0, 0]) { [this.a, this.b, this.c, this.d, this.e, this.f] = values; }
  multiply(m) {
    return new Matrix([this.a*m.a + this.c*m.b, this.b*m.a + this.d*m.b,
      this.a*m.c + this.c*m.d, this.b*m.c + this.d*m.d,
      this.a*m.e + this.c*m.f + this.e, this.b*m.e + this.d*m.f + this.f]);
  }
  inverse() {
    const det = this.a*this.d - this.b*this.c;
    return new Matrix([this.d/det, -this.b/det, -this.c/det, this.a/det,
      (this.c*this.f - this.d*this.e)/det, (this.b*this.e - this.a*this.f)/det]);
  }
  translate(x, y) { return this.multiply(new Matrix([1, 0, 0, 1, x, y])); }
  scale(x, y = x) { return this.multiply(new Matrix([x, 0, 0, y, 0, 0])); }
  rotate(degrees, cx = 0, cy = 0) { const r = degrees * Math.PI / 180; return this.translate(cx, cy).multiply(new Matrix([Math.cos(r), Math.sin(r), -Math.sin(r), Math.cos(r), 0, 0])).translate(-cx, -cy); }
}
class Point {
  constructor(x, y) { this.x = x; this.y = y; }
  matrixTransform(m) { return new Point(m.a * this.x + m.c * this.y + m.e, m.b * this.x + m.d * this.y + m.f); }
}
const near = (actual, expected, what = '', tolerance = 1e-9) => assert.ok(Math.abs(actual - expected) <= tolerance, `${what} ${actual} != ${expected}`);
const nearPoint = (actual, expected, what = '', tolerance = 1e-9) => { near(actual.x, expected.x, `${what} x`, tolerance); near(actual.y, expected.y, `${what} y`, tolerance); };

// The pure helpers: percentages of the bounds or of the viewport, written
// back as they were spelled.
assert.equal(endpointNumber('25%', 'objectBoundingBox', 200), 0.25);
assert.equal(endpointNumber('25%', 'userSpaceOnUse', 200), 50);
assert.ok(Number.isNaN(endpointNumber('', 'userSpaceOnUse', 200)));
assert.equal(endpointText(0.123456789, 'objectBoundingBox', true), '12.346%');
assert.equal(endpointText(0.123456789, 'objectBoundingBox', false), '0.12346');
assert.equal(endpointText(-0.0000001, 'userSpaceOnUse', false), '0');
assert.equal(unitsMatrix('objectBoundingBox', {x: 1, y: 2, width: 0, height: 5}), null);
nearPoint(constrainEndpoint({x: 0, y: 0}, {x: 10, y: 1}), {x: Math.hypot(10, 1), y: 0}, 'constrained');

// test_gradient_handles.py builds the same drawing. Shape a is in a turned,
// scaled group with a transform of its own and shares the gradient "box"
// (objectBoundingBox, turned by its gradientTransform) with b; shape c, turned
// itself, has "user" (userSpaceOnUse, also transformed). The viewport is
// 200 × 100 and the overlay is drawn at twice the size, 10 pixels in.
const element = (attributes, extra = {}) => Object.assign({
  values: attributes,
  getAttribute: name => attributes[name] ?? null,
  setAttribute(name, value) { attributes[name] = value; },
  removeAttribute(name) { delete attributes[name]; },
}, extra);
const gradient = (attributes, transform) => element(attributes, {localName: 'linearGradient', gradientTransform: {baseVal: {consolidate: () => ({matrix: transform})}}});
const frames = {
  a: new Matrix().translate(100, 20).rotate(90).scale(2).translate(3, 4),
  b: new Matrix(),
  c: new Matrix().rotate(10, 150, 30),
};
const boxes = {a: {x: 10, y: 20, width: 40, height: 20}, b: {x: 0, y: 60, width: 50, height: 30}, c: {x: 120, y: 10, width: 60, height: 40}};
const transforms = {box: new Matrix().rotate(30, 0.5, 0.5), user: new Matrix().translate(5, 0).scale(1.5, 1)};
const elements = {
  a: element({}, {frame: frames.a, getBBox: () => boxes.a}),
  b: element({}, {frame: frames.b, getBBox: () => boxes.b}),
  c: element({}, {frame: frames.c, getBBox: () => boxes.c}),
  g: element({}, {frame: new Matrix(), getBBox: () => boxes.b}),
  box: gradient({x1: '10%', y1: '0%', x2: '90%', gradientTransform: 'rotate(30 0.5 0.5)'}, transforms.box),
  user: gradient({gradientUnits: 'userSpaceOnUse', x1: '130', y1: '20', x2: '25%', y2: '50%', gradientTransform: 'translate(5 0) scale(1.5 1)'}, transforms.user),
};
const fill = (id, users) => ({id, private: users === 1, users, attributes: {...elements[id].values}, stops: []});
const objects = [
  {id: 'a', tag: 'rect', label: 'Rect 1', inherited_locks: [], fill_gradient: fill('box', 2)},
  {id: 'b', tag: 'rect', label: 'Rect 2', inherited_locks: [], fill_gradient: fill('box', 2)},
  {id: 'c', tag: 'rect', label: 'Rect 3', inherited_locks: [], fill_gradient: fill('user', 1)},
  {id: 'g', tag: 'g', label: 'Group 1', inherited_locks: [], fill_gradient: fill('box', 2)},
  {id: 'box', tag: 'linearGradient', label: 'Linear gradient 1', inherited_locks: [], fill_gradient: fill('box', 2)},
];
const object = id => objects.find(o => o.id === id);
const screen = new Matrix().translate(10, 10).scale(2);
const onScreen = p => [10 + 2 * p.x, 10 + 2 * p.y];
const editor = Object.assign(createEditorState(), {
  state: {revision: 1, bounds: [0, 0, 200, 100], objects, selection: {objects: ['a'], nodes: []}},
  tool: 'gradient', zoom: 2,
});
const fields = new Map(), field = id => {
  if (!fields.has(id)) fields.set(id, {value: '', style: {}, querySelector: () => ({}), replaceChildren() {}, append() {}});
  return fields.get(id);
};
const control = () => ({setAttribute() {}, append() {}});
const sent = [], toasts = [], drawn = [];
let started = null;
const gradients = createGradientsController(editor, {
  $: field, DOMMatrix: Matrix, DOMPoint: Point,
  action: async (command, data) => { sent.push({command, data}); return true; },
  colorHex: () => '#000000', cssColour: colour => colour,
  document: {createElement: control},
  drawOverlay() {},
  gradientCss: () => null,
  localToOverlay: target => target.frame ?? null,
  oneObject: () => editor.state.selection.objects.length === 1 ? object(editor.state.selection.objects[0]) : null,
  overlay: {getScreenCTM: () => screen},
  paintGradient: () => elements.box, paintValue: () => 'url(#art-box)',
  point: event => ({x: (event.clientX - 10) / 2, y: (event.clientY - 10) / 2}),
  startGesture: (kind, _event, common, details) => { started = kind; editor.drag = {...common, kind, ...details}; },
  svgElement: id => elements[id],
  toast: (message, error) => toasts.push({message, error}),
  xmlElement: (tag, attrs) => ({tag, attrs}),
});
editor.overlayFrame = {content: {append: (...items) => drawn.push(...items)}};

// Where a point of *gradient* painting shape *id* lands on the page, mapped
// as SVG paints it.
const onPage = (id, gradient, units, local) => {
  const bounds = unitsMatrix(units, boxes[id]);
  const matrix = frames[id].multiply(new Matrix([bounds.a, bounds.b, bounds.c, bounds.d, bounds.e, bounds.f])).multiply(transforms[gradient]);
  return new Point(local.x, local.y).matrixTransform(matrix);
};

// The handles sit at the endpoints through the group, the shape's own
// transform, its bounds and the gradientTransform; a missing y2 is 0%.
let shown = gradients.placement();
nearPoint(shown.start, onPage('a', 'box', 'objectBoundingBox', {x: 0.1, y: 0}), 'a start');
nearPoint(shown.end, onPage('a', 'box', 'objectBoundingBox', {x: 0.9, y: 0}), 'a end');
assert.equal(gradients.handleAt(...onScreen(shown.end)), 'end');
assert.equal(gradients.handleAt(...onScreen(shown.start)), 'start');
assert.equal(gradients.handleAt(...onScreen({x: shown.start.x + 10, y: shown.start.y + 10})), null);
assert.equal(gradients.handleCursor(...onScreen(shown.end)), 'move');
gradients.drawHandles();
assert.deepEqual(drawn.map(item => [item.tag, item.attrs.class]), [
  ['line', 'gradient-line-halo'], ['line', 'gradient-line shared'], ['rect', 'gradient-handle gradient-start shared'], ['circle', 'gradient-handle gradient-end shared'],
]);
assert.equal(gradients.handlesStatus(), 'Shared gradient: changes all 2 shapes using it');
// Other tools, groups, the gradient itself and unpainted shapes show none.
for (const [tool, selected, refusal] of [
  ['move', 'a', ''],
  ['gradient', 'g', 'Gradient handles place the fill of one path or basic shape'],
  ['gradient', 'box', 'Select a shape using this gradient to place it on the canvas'],
  ['gradient', null, 'Select one shape with a linear gradient fill'],
]) {
  editor.tool = tool; editor.state.selection.objects = selected ? [selected] : [];
  assert.equal(gradients.placement(), null, `${tool} ${selected}`);
  assert.equal(gradients.handlesRefusal(), refusal);
  drawn.length = 0; gradients.drawHandles(); assert.equal(drawn.length, 0);
}
editor.tool = 'gradient'; editor.state.selection.objects = ['a'];
shown = gradients.placement();

const gestures = canvasGestures({
  point: () => ({}), drawOverlay() {},
  selection: {pick: () => sent.push({command: 'pick'})},
  gradient: {result: gradients.handleDrag, preview: gradients.previewHandle, finish: gradients.finishHandle, restore: gradients.restoreHandle, refuse: message => toasts.push({message, error: true})},
});
const press = (name, at) => {
  const [x, y] = onScreen(at), event = {clientX: x, clientY: y};
  gradients.pressHandle(event, {x, y, moved: false}, name);
  return editor.drag;
};
// As the canvas does: the gesture ends before it releases or cancels.
const release = gesture => { editor.drag = null; return gestures.gradient.release(gesture); };
const cancel = gesture => { gestures.gradient.cancel(gesture); editor.drag = null; };
const drag = (gesture, at, extra = {}) => {
  const [x, y] = onScreen(at);
  gesture.moved = true;
  gestures.gradient.move(gesture, {clientX: x, clientY: y, shiftKey: false, altKey: false, ctrlKey: false, metaKey: false, ...extra});
};

// A click on a handle picks what is under it and edits nothing.
let gesture = press('end', shown.end);
assert.equal(started, 'gradient');
await release(gesture);
assert.deepEqual(sent.splice(0), [{command: 'pick'}]);

// A drag previews on the drawn gradient, so b changes with a, and the
// Endpoints fields follow. The handle stays under the pointer: a press off
// its centre keeps that offset.
const grabbed = {x: shown.end.x + 0.5, y: shown.end.y - 0.25};
gesture = press('end', grabbed);
const target = {x: 60, y: 70};
drag(gesture, {x: target.x + 0.5, y: target.y - 0.25}, {altKey: true});
assert.deepEqual(Object.keys(gesture.result.attributes), ['x2', 'y2']);
assert.ok(gesture.result.attributes.x2.endsWith('%') && gesture.result.attributes.y2.endsWith('%'), 'percentages stay percentages');
assert.equal(elements.box.values.x2, gesture.result.attributes.x2);
assert.equal(field('gradient-x2').value, gesture.result.attributes.x2);
assert.equal(field('gradient-y2').value, gesture.result.attributes.y2);
nearPoint(gradients.placement().end, target, 'previewed end', 1e-3);
nearPoint(gradients.placement().start, shown.start, 'fixed start');
assert.equal(gradients.handlesStatus(), 'Gradient end · Shared gradient: changes all 2 shapes using it');
drawn.length = 0; gradients.drawHandles();
assert.ok(!drawn.some(item => item.attrs.class === 'snap-target'), 'Alt does not snap');

// Shift turns it about the start in 15° steps.
drag(gesture, {x: target.x + 0.5, y: target.y - 0.25}, {shiftKey: true});
const turned = gradients.placement().end, step = Math.atan2(turned.y - shown.start.y, turned.x - shown.start.x) * 180 / Math.PI / 15;
near(step, Math.round(step), '15° steps', 1e-3);
near(Math.hypot(turned.x - shown.start.x, turned.y - shown.start.y), Math.hypot(target.x - shown.start.x, target.y - shown.start.y), 'length kept', 1e-3);
// Near the shape's corner it snaps there, shown by the snap marker.
const corner = new Point(boxes.a.x + boxes.a.width, boxes.a.y).matrixTransform(frames.a);
drag(gesture, {x: corner.x + 0.5 + 1, y: corner.y - 0.25 - 1});
nearPoint(gradients.placement().end, corner, 'snapped end', 1e-3);
drawn.length = 0; gradients.drawHandles();
assert.equal(drawn.at(-1).attrs.class, 'snap-target');

// Escape restores what the gradient had, a missing y2 removed again, and
// the fields show it.
cancel(gesture);
assert.deepEqual(elements.box.values, {x1: '10%', y1: '0%', x2: '90%', gradientTransform: 'rotate(30 0.5 0.5)'});
assert.equal(field('gradient-x2').value, '90%');
assert.equal(field('gradient-y2').value, '0%');

// The release commits one edit of just the moved end, as previewed.
gesture = press('end', grabbed);
drag(gesture, {x: target.x + 0.5, y: target.y - 0.25}, {altKey: true});
const shared = {gradient: 'box', object: 'a', end: 'end', attributes: {...gesture.result.attributes}, target};
await release(gesture);
assert.deepEqual(sent.splice(0), [{command: 'paint', data: {gradient: {id: 'box', attributes: shared.attributes}}}]);
// A drag back to where the end was edits nothing.
Object.assign(elements.box.values, {x2: '90%'}); delete elements.box.values.y2;
gesture = press('end', shown.end);
drag(gesture, shown.end, {altKey: true});
await release(gesture);
assert.deepEqual(sent.splice(0), []);
assert.deepEqual(elements.box.values, {x1: '10%', y1: '0%', x2: '90%', gradientTransform: 'rotate(30 0.5 0.5)'});

// Drawing coordinates: c's private gradient, through c's turn and the
// gradientTransform; x2 at 25% of the viewport's width is 50.
editor.state.selection.objects = ['c'];
shown = gradients.placement();
nearPoint(shown.start, onPage('c', 'user', 'userSpaceOnUse', {x: 130, y: 20}), 'c start');
nearPoint(shown.end, onPage('c', 'user', 'userSpaceOnUse', {x: 50, y: 50}), 'c end');
assert.equal(gradients.handlesStatus(), 'Private gradient: changes this shape');
drawn.length = 0; gradients.drawHandles();
assert.equal(drawn[1].attrs.class, 'gradient-line');
gesture = press('start', shown.start);
const userTarget = {x: 150.25, y: 45.5};
drag(gesture, userTarget, {altKey: true});
assert.ok(!gesture.result.attributes.x1.endsWith('%'));
nearPoint(gradients.placement().start, userTarget, 'previewed start', 1e-2);
const own = {gradient: 'user', object: 'c', end: 'start', attributes: {...gesture.result.attributes}, target: userTarget};
await release(gesture);
assert.deepEqual(sent.splice(0), [{command: 'paint', data: {gradient: {id: 'user', attributes: own.attributes}}}]);

// A paint lock on any shape using the gradient refuses a drag once, before
// anything is previewed; the handles show it.
editor.state.selection.objects = ['a'];
object('b').inherited_locks = ['paint'];
shown = gradients.placement();
assert.equal(shown.refusal, 'This shared gradient cannot move: Rect 2’s paint is locked');
assert.equal(gradients.handleCursor(...onScreen(shown.end)), 'not-allowed');
drawn.length = 0; gradients.drawHandles();
assert.equal(drawn[1].attrs.class, 'gradient-line shared locked');
gesture = press('end', shown.end);
drag(gesture, target); drag(gesture, {x: 61, y: 71});
assert.deepEqual(toasts.splice(0), [{message: shown.refusal, error: true}]);
assert.equal(elements.box.values.x2, '90%');
await release(gesture);
assert.deepEqual(sent.splice(0), []);
object('b').inherited_locks = [];

console.log(JSON.stringify([shared, own]));
