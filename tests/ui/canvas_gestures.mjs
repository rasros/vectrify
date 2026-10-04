// Canvas previews, completion and cancellation, without a browser DOM.
import assert from 'node:assert/strict';
import {canvasGestures} from '../../src/vectrify/ui/static/gestures.js';
import {HeldGesture} from '../../src/vectrify/ui/static/input.js';

const calls = [], draft = [];
let pan = {x: 10, y: 20}, panning = false, zoom = 2;
const record = name => (...args) => { calls.push([name, ...args]); };
const gestures = canvasGestures({
  point: event => ({x: event.clientX, y: event.clientY}),
  drawOverlay: record('overlay'),
  view: {
    pan: () => pan, setPan: value => { pan = value; }, zoom: () => zoom,
    setPanning: value => { panning = value; }, update: record('view'),
  },
  path: {draft: () => draft, setHover: record('pathHover'), finish: record('finishPath')},
  selection: {pick: record('pick'), box: record('box'), move: record('move'), renderDrawing: record('drawing')},
  nodes: {
    preview: record('nodePreview'), snap: event => ({x: event.clientX + 1, y: event.clientY + 1}),
    finish: record('nodeFinish'), select: record('nodeSelect'), restore: record('nodeRestore'),
    renderInspector: record('inspector'), renderNodeInspector: record('nodeInspector'),
  },
  resize: {result: () => ({sx: 2, sy: 3, anchor: [0, 0]}), preview: record('resizePreview'), finish: record('resizeFinish')},
  knife: {end: event => ({x: event.clientX, y: 100}), finish: record('knife')},
  redraw: {setHover: record('redrawHover'), finish: record('redraw')},
});
const event = (x = 10, y = 20, extra = {}) => ({clientX: x, clientY: y, shiftKey: false, ...extra});
const begin = (kind, details = {}, e = event()) => {
  calls.length = 0;
  const gesture = {kind, x: e.clientX, y: e.clientY, moved: false, ...details};
  gestures[kind].press?.(gesture, e);
  return gesture;
};
const named = name => calls.filter(call => call[0] === name);

// Below the drag threshold, selection gestures pick instead of editing.
for (const kind of ['box', 'move', 'knife', 'redraw']) {
  const gesture = begin(kind, {id: 'a', members: []});
  gestures[kind].move(gesture, event(11, 20));
  await gestures[kind].release(gesture);
  assert.deepEqual(named('pick'), [['pick', gesture]]);
  for (const name of ['box', 'move', 'knife', 'redraw']) assert.equal(named(name).length, 0);
}
// A box updates in screen coordinates and completes with its press modifiers.
const box = begin('box', {shift: true});
box.moved = true;
gestures.box.move(box, event(80, 90));
await gestures.box.release(box);
assert.deepEqual(box.end, {x: 80, y: 90});
assert.equal(named('box')[0][1].shift, true);
assert.equal(named('pick').length, 0);

// Pan moves the view even before the edit-drag threshold and stays on cancel.
const panningGesture = begin('pan');
assert.equal(panning, true);
gestures.pan.move(panningGesture, event(12, 24));
assert.deepEqual(pan, {x: 12, y: 24});
gestures.pan.cancel(panningGesture);
assert.equal(panning, false);
assert.deepEqual(pan, {x: 12, y: 24});
begin('pan'); gestures.pan.release(); assert.equal(panning, false);

// Drawing a curve mirrors its handles. Cancel removes only the current point.
draft.push({x: 0, y: 0});
const curve = begin('drawPath');
assert.equal(draft.length, 2);
assert.equal(draft[1], curve.anchor);
gestures.drawPath.move(curve, event(30, 40));
assert.equal(curve.anchor.out, undefined);
curve.moved = true;
gestures.drawPath.move(curve, event(30, 40));
assert.deepEqual(curve.anchor.out, {x: 30, y: 40});
assert.deepEqual(curve.anchor.in, {x: -10, y: 0});
gestures.drawPath.cancel(curve);
assert.deepEqual(draft, [{x: 0, y: 0}]);
const close = begin('closePath');
await gestures.closePath.release({...close, moved: true});
assert.equal(named('finishPath').length, 0);
await gestures.closePath.release(close);
assert.deepEqual(named('finishPath'), [['finishPath', true]]);

// Knife endpoints come from the snapping calculation; redraw samples use zoom.
const cut = begin('knife', {id: 'a'});
cut.moved = true;
gestures.knife.move(cut, event(30, 40));
assert.deepEqual(cut.end, {x: 30, y: 100});
await gestures.knife.release(cut);
assert.deepEqual(named('knife'), [['knife', cut]]);
const stroke = begin('redraw', {id: 'a'});
gestures.redraw.move(stroke, event(10.5, 20, {shiftKey: true}));
assert.equal(stroke.points.length, 1);
assert.equal(stroke.longWay, true);
stroke.moved = true;
gestures.redraw.move(stroke, event(11, 20));
assert.deepEqual(stroke.points, [[10, 20], [11, 20]]);
assert.equal(stroke.longWay, false);
await gestures.redraw.release(stroke);
assert.deepEqual(named('redraw'), [['redraw', stroke]]);

// A point click collapses multiple points; a drag previews snapped coordinates.
const node = begin('node', {objects: ['a'], key: 'a n', collapse: true, saved: new Map()});
gestures.node.move(node, event(11, 20));
assert.equal(named('nodePreview').length, 0);
await gestures.node.release(node);
assert.deepEqual(named('nodeSelect'), [['nodeSelect', ['a'], ['a n']]]);
node.moved = true;
gestures.node.move(node, event(30, 40));
assert.deepEqual(named('nodePreview'), [['nodePreview', {x: 31, y: 41}]]);
await gestures.node.release(node);
assert.deepEqual(named('nodeFinish'), [['nodeFinish', node]]);
gestures.node.cancel(node);
assert.deepEqual(named('nodeRestore'), [['nodeRestore', node]]);

// No-op resize releases restore the drawing; changed scales commit once.
const resized = begin('resize', {members: []});
await gestures.resize.release(resized);
assert.equal(named('drawing').length, 1);
resized.moved = true;
gestures.resize.move(resized, event(30, 40));
await gestures.resize.release(resized);
assert.deepEqual(named('resizeFinish'), [['resizeFinish', resized.result]]);
resized.result = {sx: 1, sy: 1};
await gestures.resize.release(resized);
assert.equal(named('resizeFinish').length, 1);
assert.equal(named('drawing').length, 2);

// Object movement converts screen offsets into each parent's coordinates.
globalThis.DOMPoint = class {
  constructor(x, y) { this.x = x; this.y = y; }
  matrixTransform(matrix) { return {x: this.x / matrix.scale, y: this.y / matrix.scale}; }
};
const member = (id, before, scale) => {
  const attributes = new Map(before ? [['transform', before]] : []);
  return {id, before, element: {
    parentElement: {getScreenCTM: () => ({inverse: () => ({scale})})},
    setAttribute: (name, value) => attributes.set(name, value),
    removeAttribute: name => attributes.delete(name),
    attributes,
  }};
};
const a = member('a', 'rotate(30)', 2), b = member('b', '', 4);
const moved = begin('move', {members: [a, b]});
moved.moved = true;
gestures.move.move(moved, event(30, 40));
assert.deepEqual(a.offset, [10, 10]);
assert.deepEqual(b.offset, [5, 5]);
assert.equal(a.element.attributes.get('transform'), 'translate(10 10) rotate(30)');
await gestures.move.release(moved);
assert.deepEqual(named('move'), [['move', {a: [10, 10], b: [5, 5]}]]);
// Both transform previews return to their original attributes on cancel.
for (const kind of ['move', 'resize']) {
  a.element.setAttribute('transform', 'preview'); b.element.setAttribute('transform', 'preview');
  gestures[kind].cancel(moved);
  assert.equal(a.element.attributes.get('transform'), 'rotate(30)');
  assert.equal(b.element.attributes.has('transform'), false);
}

// Held gestures still replay the same lifecycle after queued edits complete.
const held = new HeldGesture({...event(), type: 'pointerdown', pointerId: 1});
held.record({...event(30, 40), type: 'pointermove', pointerId: 1});
held.record({...event(30, 40), type: 'pointerup', pointerId: 1});
let live, completion;
assert.equal(held.replay({
  down: e => { live = begin('knife', {id: 'a'}, e); },
  move: e => { live.moved = true; gestures.knife.move(live, e); },
  up: () => { completion = gestures.knife.release(live); },
}), false);
await completion;
assert.equal(named('knife').length, 1);
