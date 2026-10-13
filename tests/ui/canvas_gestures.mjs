import {createEditorState} from '../../src/vectrify/ui/static/app_state.js';
import {createOverlayController} from '../../src/vectrify/ui/static/app_overlay.js';
import {createResizingController} from '../../src/vectrify/ui/static/app_resizing.js';
import {createCanvasController} from '../../src/vectrify/ui/static/app_canvas.js';
// Canvas previews, completion and cancellation, without a browser DOM.
import assert from 'node:assert/strict';

import {canvasGestures} from '../../src/vectrify/ui/static/gestures.js';
import {HeldGesture} from '../../src/vectrify/ui/static/input.js';
import {selectionContains} from '../../src/vectrify/ui/static/selection.js';
import {CURSORS, frameHandle} from '../../src/vectrify/ui/static/resize.js';

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

// Frame clicks pick; no-op drags restore the drawing; changed scales commit once.
const resized = begin('resize', {members: []});
await gestures.resize.release(resized);
assert.deepEqual(named('pick'), [['pick', resized]]);
assert.equal(named('drawing').length, 0);
resized.moved = true;
await gestures.resize.release(resized);
assert.equal(named('drawing').length, 1);
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
// A clipping contour moves in its consumer's frame, even though its
// definition's parent has no rendered screen matrix of its own.
const clipped = member('clip', '', 1);
clipped.element.parentElement = {};
clipped.screenMatrix = () => ({inverse: () => ({scale: 10})});
const clipMove = begin('move', {members: [clipped]}); clipMove.moved = true;
gestures.move.move(clipMove, event(30, 40));
assert.deepEqual(clipped.offset, [2, 2]);
await gestures.move.release(clipMove);
assert.deepEqual(named('move'), [['move', {clip: [2, 2]}]]);
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

// Select must box-select even on selected objects and frame edges. Exercise
// the actual canvas dispatch and frame hit testing, not just the controllers.

const frameShapes = [];
let hits = ['a'];
const dispatch = Object.assign(createEditorState(), {
  tool: 'select', space: false, drag: null, zoom: 1, FRAME_REACH: 6,
  performance: {now: () => 0},
  state: {root: 'root', selection: {objects: ['a']}},
  selectionBox: {left: 10, top: 10, right: 90, bottom: 90},
  DOMPoint: class {
    constructor(x, y) {this.x = x; this.y = y;}
    matrixTransform() {return this;}
  },
  overlay: {getScreenCTM: () => ({})}, frameHandle, CURSORS,
  overlayFrame: {content: {append: shape => frameShapes.push(shape)}},
  xmlElement: (name, attrs) => ({name, ...attrs}), resizeRefusal: () => '',
  stage: {focus() {}, setPointerCapture() {}},
  $: () => ({getBoundingClientRect: () => ({left: 0, top: 0, right: 100, bottom: 100})}),
  hitStack: () => hits, clickTargets: stack => stack, sameClickSpot: () => false,
  topSelection: () => ['a'], object: () => ({attributes: {}}),
  svgElement: () => a.element,
  scope: null, parents: () => new Map(), pickTarget: id => ({id}), selectionContains,
  startGesture: (kind, event, common, details = {}) => {
    dispatch.drag = {...common, kind, ...details};
    gestures[kind].press?.(dispatch.drag, event);
  },
  pressFrame: (_event, common, handle) => {dispatch.drag = {...common, kind: 'resize', handle};},
});
const dispatch_overlay = createOverlayController(dispatch, {
  $: (...args) => dispatch.$(...args),
  CSS: dispatch.CSS,
  DOMPoint: dispatch.DOMPoint,
  drawFlash: (...args) => dispatch.drawFlash(...args),
  drawKnife: (...args) => dispatch.drawKnife(...args),
  drawOverlay: (...args) => dispatch.drawOverlay(...args),
  drawPathDraft: (...args) => dispatch.drawPathDraft(...args),
  drawPoints: (...args) => dispatch.drawPoints(...args),
  drawRedraw: (...args) => dispatch.drawRedraw(...args),
  drawSnap: (...args) => dispatch.drawSnap(...args),
  drawing: dispatch.drawing,
  getComputedStyle: dispatch.getComputedStyle,
  later: (...args) => dispatch.later(...args),
  level: (...args) => dispatch.level(...args),
  localToOverlay: (...args) => dispatch.localToOverlay(...args),
  nodeAt: (...args) => dispatch.nodeAt(...args),
  nodeHandles: (...args) => dispatch.nodeHandles(...args),
  object: (...args) => dispatch.object(...args),
  overlay: dispatch.overlay,
  renderStatus: (...args) => dispatch.renderStatus(...args),
  resizeRefusal: (...args) => dispatch.resizeRefusal(...args),
  svgElement: (...args) => dispatch.svgElement(...args),
  topSelection: (...args) => dispatch.topSelection(...args),
  xmlElement: (...args) => dispatch.xmlElement(...args),
});
dispatch.drawFrame = dispatch_overlay.drawFrame;
const dispatch_resizing = createResizingController(dispatch, {
  resizeRefusal: (...args) => dispatch.resizeRefusal(...args),
  DOMMatrix: dispatch.DOMMatrix,
  DOMPoint: dispatch.DOMPoint,
  hitStack: (...args) => dispatch.hitStack(...args),
  localToOverlay: (...args) => dispatch.localToOverlay(...args),
  object: (...args) => dispatch.object(...args),
  overlay: dispatch.overlay,
  parents: (...args) => dispatch.parents(...args),
  point: (...args) => dispatch.point(...args),
  requestAnimationFrame: dispatch.requestAnimationFrame,
  stage: dispatch.stage,
  startGesture: (...args) => dispatch.startGesture(...args),
  svgElement: (...args) => dispatch.svgElement(...args),
  topSelection: (...args) => dispatch.topSelection(...args),
});
dispatch.frameAt = dispatch_resizing.frameAt;
dispatch.frameCursor = dispatch_resizing.frameCursor;
const dispatch_canvas = createCanvasController(dispatch, {
  $: (...args) => dispatch.$(...args),
  DOMPoint: dispatch.DOMPoint,
  action: (...args) => dispatch.action(...args),
  clickEmpty: (...args) => dispatch.clickEmpty(...args),
  clickTargets: (...args) => dispatch.clickTargets(...args),
  drawOverlay: (...args) => dispatch.drawOverlay(...args),
  drawing: dispatch.drawing,
  finishPath: (...args) => dispatch.finishPath(...args),
  frameAt: (...args) => dispatch.frameAt(...args),
  geometryNodes: (...args) => dispatch.geometryNodes(...args),
  gestures: dispatch.gestures,
  hitStack: (...args) => dispatch.hitStack(...args),
  hoverResize: (...args) => dispatch.hoverResize(...args),
  later: (...args) => dispatch.later(...args),
  level: (...args) => dispatch.level(...args),
  loadGeometries: (...args) => dispatch.loadGeometries(...args),
  localToOverlay: (...args) => dispatch.localToOverlay(...args),
  nearKey: (...args) => dispatch.nearKey(...args),
  nearestPoint: (...args) => dispatch.nearestPoint(...args),
  nodeAt: (...args) => dispatch.nodeAt(...args),
  nodeHandles: (...args) => dispatch.nodeHandles(...args),
  object: (...args) => dispatch.object(...args),
  overlay: dispatch.overlay,
  parents: (...args) => dispatch.parents(...args),
  point: (...args) => dispatch.point(...args),
  pointPaths: (...args) => dispatch.pointPaths(...args),
  pressFrame: (...args) => dispatch.pressFrame(...args),
  redrawLines: (...args) => dispatch.redrawLines(...args),
  renderDrawing: (...args) => dispatch.renderDrawing(...args),
  renderInspector: (...args) => dispatch.renderInspector(...args),
  renderNodeInspector: (...args) => dispatch.renderNodeInspector(...args),
  renderStatus: (...args) => dispatch.renderStatus(...args),
  requestAnimationFrame: dispatch.requestAnimationFrame,
  sameClickSpot: (...args) => dispatch.sameClickSpot(...args),
  screenMatrix: (...args) => dispatch.screenMatrix(...args),
  selectPoints: (...args) => dispatch.selectPoints(...args),
  selectedHandle: (...args) => dispatch.selectedHandle(...args),
  selectedPoints: (...args) => dispatch.selectedPoints(...args),
  setTool: (...args) => dispatch.setTool(...args),
  sharingPaths: (...args) => dispatch.sharingPaths(...args),
  stage: dispatch.stage,
  startGesture: (...args) => dispatch.startGesture(...args),
  svgElement: (...args) => dispatch.svgElement(...args),
  toast: (...args) => dispatch.toast(...args),
});
dispatch.pressStage = dispatch_canvas.pressStage;
const pressStage = (dispatch.pressStage);
const frameCursor = (dispatch.frameCursor);
const drawFrame = (dispatch.drawFrame);
for (const stack of [['a'], ['unselected'], []]) {
  hits = stack;
  for (const [x, y] of [[50, 50], [10, 50], [10, 10]]) {
    calls.length = 0; dispatch.drag = null;
    const before = a.element.attributes.get('transform');
    pressStage(event(x, y, {button: 0}));
    assert.equal(dispatch.drag.kind, 'box');
    dispatch.drag.moved = true;
    gestures.box.move(dispatch.drag, event(95, 95));
    await gestures.box.release(dispatch.drag);
    assert.equal(named('box').length, 1);
    assert.equal(named('move').length, 0);
    assert.equal(named('resizeFinish').length, 0);
    assert.equal(a.element.attributes.get('transform'), before);
    assert.equal(frameCursor(x, y), 'default');
  }
}
drawFrame();
assert.equal(frameShapes.length, 1, 'Select has a passive frame without resize ticks');

// Move retains dragging and owns resizing. Degenerate line frames still move
// at their edges, where there is no frame interior to grab.
dispatch.tool = 'move'; hits = ['a'];
for (const [x, y, kind] of [[50, 50, 'move'], [10, 50, 'resize'], [10, 10, 'resize']]) {
  dispatch.drag = null;
  pressStage(event(x, y, {button: 0}));
  assert.equal(dispatch.drag.kind, kind);
}
frameShapes.length = 0; drawFrame();
assert.equal(frameShapes.length, 5, 'Move shows four resize ticks');
dispatch.selectionBox.bottom = dispatch.selectionBox.top;
dispatch.drag = null;
pressStage(event(10, 10, {button: 0}));
assert.equal(dispatch.drag.kind, 'move');
