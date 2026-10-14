// Snapping and axis constraints for moving objects; run by test_move_snap.py.
import assert from 'node:assert/strict';

import {boxLines, snapBox} from '../../src/vectrify/ui/static/snap.js';
import {createResizingController} from '../../src/vectrify/ui/static/app_resizing.js';
import {createToolsController} from '../../src/vectrify/ui/static/app_tools.js';
import {createKeysController} from '../../src/vectrify/ui/static/app_keys.js';
import {canvasGestures} from '../../src/vectrify/ui/static/gestures.js';

// Lines run along each box's edges and centre, spanning the box across.
assert.deepEqual(boxLines([{left: 0, top: 10, right: 20, bottom: 50}]), {
  x: [{at: 0, low: 10, high: 50}, {at: 10, low: 10, high: 50}, {at: 20, low: 10, high: 50}],
  y: [{at: 10, low: 0, high: 20}, {at: 30, low: 0, high: 20}, {at: 50, low: 0, high: 20}],
});

const box = {left: 100, top: 100, right: 140, bottom: 120};
const lines = boxLines([{left: 200, top: 300, right: 260, bottom: 340}]);
// The left edge snaps to another box's left; y has nothing in reach.
let snap = snapBox(box, 97, 0, lines, 4);
assert.equal(snap.dx, 100); assert.equal(snap.dy, 0);
assert.deepEqual(snap.guides, {x: [{at: 200, low: 100, high: 340}], y: []});
// The centre snaps to a centre, the right edge to a left edge.
assert.equal(snapBox(box, 108, 0, lines, 4).dx, 110);
assert.equal(snapBox(box, 62, 0, lines, 4).dx, 60);
// The nearest line wins; nothing beyond the radius snaps.
const pair = boxLines([{left: 199, top: 0, right: 199, bottom: 1}, {left: 202.5, top: 0, right: 202.5, bottom: 1}]);
assert.equal(snapBox(box, 101.5, 0, pair, 4).dx, 102.5);
assert.equal(snapBox(box, 85, 0, lines, 4).dx, 85);
assert.deepEqual(snapBox(box, 85, 0, lines, 4).guides, {x: [], y: []});
// Both axes snap at once; the guides span the moved box and the target.
snap = snapBox(box, 158, 222, lines, 4);
assert.deepEqual([snap.dx, snap.dy], [160, 220]);
assert.deepEqual(snap.guides.x, [{at: 260, low: 300, high: 340}]);
assert.deepEqual(snap.guides.y, [{at: 320, low: 200, high: 300}, {at: 340, low: 200, high: 300}]);
// Every edge or centre that lands on a line gets a guide.
snap = snapBox({left: 0, top: 0, right: 60, bottom: 10}, 199, 0, lines, 4);
assert.equal(snap.dx, 200);
assert.deepEqual(snap.guides.x.map(g => g.at), [200, 230, 260]);
// A locked axis neither moves nor snaps; the other still snaps.
snap = snapBox(box, 97, 199, lines, 4, 'x');
assert.deepEqual([snap.dx, snap.dy], [0, 200]);
assert.deepEqual(snap.guides.x, []);
snap = snapBox(box, 97, 1, lines, 4, 'y');
assert.deepEqual([snap.dx, snap.dy], [100, 0]);

// The resizing controller gathers the targets and reads the modifiers.
class Point {
  constructor(x, y) { this.x = x; this.y = y; }
  matrixTransform() { return this; }
}
const boxes = {
  sel: {left: 100, top: 100, right: 140, bottom: 120},
  child: {left: 105, top: 105, right: 110, bottom: 110},
  group: {left: 0, top: 0, right: 400, bottom: 400},
  other: {left: 200, top: 300, right: 260, bottom: 340},
  hidden: {left: 150, top: 150, right: 151, bottom: 151},
  clip: {left: 153, top: 153, right: 154, bottom: 154},
};
const parentOf = new Map([['sel', 'group'], ['child', 'sel'], ['group', 'root'], ['other', 'root'], ['hidden', 'root'], ['clip', 'clipper'], ['clipper', 'root']]);
const element = id => ({
  closest: selector => id === 'clip' && selector.includes('clipPath') ? {} : null,
  checkVisibility: () => id !== 'hidden',
});
const editor = {
  zoom: 1, tool: 'move',
  state: {
    root: 'root', bounds: [0, 0, 1000, 800], selection: {objects: ['sel']},
    objects: [...parentOf.keys(), 'root'].map(id => ({id, tag: id === 'clipper' ? 'clipPath' : id === 'group' ? 'g' : 'rect'})),
  },
};
const asked = [];
const resizing = createResizingController(editor, {
  DOMPoint: Point,
  localToOverlay: () => null,
  object: id => ({id}),
  objectBounds: id => { asked.push(id); return boxes[id] || null; },
  overlay: {getScreenCTM: () => null},
  parents: () => parentOf,
  // The overlay's frame is the screen divided by the zoom.
  point: e => new Point(e.clientX / editor.zoom, e.clientY / editor.zoom),
  svgElement: element,
  topSelection: () => ['sel'],
});
// The selection never snaps to itself, what it holds or the groups holding
// it; hidden objects and clipping contents are no targets either.
const xs = resizing.moveSnapLines().x.map(line => line.at);
assert.deepEqual(xs, [0, 500, 1000, 200, 230, 260]);
assert.deepEqual(asked, ['other']);

const press = (zoom, at = {x: 0, y: 0}) => {
  editor.zoom = zoom;
  editor.drag = {kind: 'move', start: {x: at.x / zoom, y: at.y / zoom}, box: {...boxes.sel}, moved: true};
};
const drag = (x, y, extra = {}) => resizing.moveDrag({clientX: x, clientY: y, shiftKey: false, altKey: false, ctrlKey: false, metaKey: false, ...extra});
// The reach is SNAP_RADIUS (8) screen pixels at every zoom: the frame's top
// 6 pixels below the artboard's snaps up to it, 10 pixels below does not.
for (const zoom of [0.5, 1, 4]) {
  press(zoom);
  assert.deepEqual(drag(0, (-100 + 6 / zoom) * zoom), {dx: 0, dy: -100}, `snaps 6 px away at zoom ${zoom}`);
  assert.deepEqual(editor.drag.guides.y, [{at: 0, low: 0, high: 1000}]);
  const free = drag(0, (-100 + 10 / zoom) * zoom);
  assert.ok(free.dx === 0 && Math.abs(free.dy - (-100 + 10 / zoom)) < 1e-9, `does not snap 10 px away at zoom ${zoom}`);
  assert.deepEqual(editor.drag.guides, {x: [], y: []});
}
// The targets are gathered once per drag.
press(1); asked.length = 0;
drag(10, 10); drag(20, 20);
assert.deepEqual(asked, ['other']);
// Alt or Ctrl/⌘ drags freely, without guides.
for (const key of ['altKey', 'ctrlKey', 'metaKey']) {
  press(1);
  assert.deepEqual(drag(97, 0, {[key]: true}), {dx: 97, dy: 0});
  assert.equal(editor.drag.guides, null);
}
// Shift keeps the axis the pointer has gone furthest along; the free axis
// still snaps, the held one does not.
press(1);
assert.deepEqual(drag(97, 5, {shiftKey: true}), {dx: 100, dy: 0});
assert.deepEqual(drag(5, 199, {shiftKey: true}), {dx: 0, dy: 200});
assert.deepEqual(editor.drag.guides.x, []);
// Shift and Alt together: constrained, unsnapped.
assert.deepEqual(drag(97, 5, {shiftKey: true, altKey: true}), {dx: 97, dy: 0});
// The axis follows the pointer: past the diagonal it switches.
assert.deepEqual(drag(40, 50, {shiftKey: true, altKey: true}), {dx: 0, dy: 50});
// Without a frame (a clipping contour, say) the move is unsnapped.
press(1); editor.drag.box = null;
assert.deepEqual(drag(97, 0), {dx: 97, dy: 0});

// The guides are drawn while the selection moves.
const drawn = [];
const tools = createToolsController(editor, {
  xmlElement: (name, attributes) => ({name, ...attributes}),
});
editor.overlayFrame = {content: {append: (...shapes) => drawn.push(...shapes)}};
press(1); drag(97, 199);
tools.drawSnap();
assert.deepEqual(drawn, [
  {name: 'line', x1: 200, y1: 300, x2: 200, y2: 340, class: 'snap-edge'},
  {name: 'line', x1: 200, y1: 300, x2: 260, y2: 300, class: 'snap-edge'},
  {name: 'line', x1: 200, y1: 320, x2: 260, y2: 320, class: 'snap-edge'},
]);
drawn.length = 0; editor.drag.moved = false; tools.drawSnap();
assert.equal(drawn.length, 0, 'no guides before the drag starts');

// Shift, Alt and Ctrl/⌘ change the move as they are pressed and released.
const seen = [];
const gestures = {
  move: {modifiers: (gesture, event) => seen.push(['move', event.key])},
  shape: {modifiers: (gesture, event) => seen.push(['shape', event.key])},
};
const keys = createKeysController(editor, {gestures, document: {activeElement: {tagName: 'BODY'}}});
const key = name => ({key: name, code: name, shiftKey: name === 'Shift', altKey: name === 'Alt', ctrlKey: name === 'Control', metaKey: name === 'Meta', preventDefault() {}});
editor.drag = {kind: 'move'};
for (const name of ['Shift', 'Alt', 'Control', 'Meta']) assert.equal(keys.gestureModifiers(key(name)), true);
editor.drag = {kind: 'shape'};
assert.equal(keys.gestureModifiers(key('Control')), false, 'Ctrl does not change a shape');
assert.equal(keys.gestureModifiers(key('Shift')), true);
editor.drag = {kind: 'box'};
assert.equal(keys.gestureModifiers(key('Shift')), false);
assert.deepEqual(seen, [['move', 'Shift'], ['move', 'Alt'], ['move', 'Control'], ['move', 'Meta'], ['shape', 'Shift']]);

// Through the gesture: a move in a scaled group snaps in the document's
// frame and its parent-frame offset lands exactly on the target.
press(1);
const attributes = new Map();
const member = {id: 'sel', before: '', frame: {inverse: () => ({a: 0.5, b: 0, c: 0, d: 0.5})},
  element: {setAttribute: (name, value) => attributes.set(name, value), removeAttribute: name => attributes.delete(name)}};
Object.assign(editor.drag, {members: [member]});
const moves = [];
const live = canvasGestures({
  point: e => new Point(e.clientX, e.clientY), drawOverlay() {},
  selection: {move: offsets => moves.push(offsets), renderDrawing() {}},
  moving: {drag: resizing.moveDrag},
});
live.move.move(editor.drag, {clientX: 97, clientY: 3, shiftKey: false, altKey: false, ctrlKey: false, metaKey: false});
assert.deepEqual(member.offset, [50, 1.5]);
assert.equal(attributes.get('transform'), 'translate(50 1.5)');
await live.move.release(editor.drag);
assert.deepEqual(moves, [{sel: [50, 1.5]}]);
