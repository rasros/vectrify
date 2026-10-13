import {createEditorState} from '../../src/vectrify/ui/static/app_state.js';
import {createFramesController} from '../../src/vectrify/ui/static/app_frames.js';
import {createOverlayController} from '../../src/vectrify/ui/static/app_overlay.js';
import {createCanvasController} from '../../src/vectrify/ui/static/app_canvas.js';
import {createHitsController} from '../../src/vectrify/ui/static/app_hits.js';
import {createSelectionController} from '../../src/vectrify/ui/static/app_selection.js';
import assert from 'node:assert/strict';

import {clippingFrames} from '../../src/vectrify/ui/static/clipping.js';
import {pickTarget} from '../../src/vectrify/ui/static/selection.js';

class Matrix {
  constructor(a = 1, b = 0, c = 0, d = 1, e = 0, f = 0) { Object.assign(this, {a, b, c, d, e, f}); }
  static fromMatrix(m) { return new Matrix(m.a, m.b, m.c, m.d, m.e, m.f); }
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
}
globalThis.DOMMatrix = Matrix;
const components = m => [m.a, m.b, m.c, m.d, m.e, m.f];
const transform = matrix => ({baseVal: {consolidate: () => ({matrix})}});
const clip = {
  id: 'art-clip', dataset: {objectId: 'clip'}, localName: 'clipPath',
  transform: transform(new Matrix().translate(5, 6)),
  getAttribute: name => name === 'clipPathUnits' ? units : null,
  closest: selector => selector === 'clipPath' ? clip : null,
  querySelectorAll: () => [contour],
};
let units = 'userSpaceOnUse';
const contour = {
  id: 'art-contour', dataset: {objectId: 'contour'}, parentElement: clip,
  transform: transform(new Matrix().scale(.5, 2)),
  closest: selector => selector === 'clipPath' || selector === 'defs' ? clip : null,
};
const consumer = (id, matrix, bounds) => ({
  dataset: {objectId: id}, getScreenCTM: () => matrix, getBBox: () => bounds,
  getAttribute: () => 'url(#art-clip)', closest: () => null,
});
const first = consumer('first', new Matrix().translate(100, 40).scale(2, 3), {x:10, y:20, width:100, height:50});
const second = consumer('second', new Matrix().translate(200, 300).scale(3, 2), {x:10, y:20, width:100, height:50});
const hidden = {...first, closest: () => clip};
let consumers = [first, second, hidden];
const root = consumer('root', new Matrix(), {x:0, y:0, width:600, height:360});
const drawing = {querySelectorAll: () => consumers, querySelector: () => root};
let frames = clippingFrames(contour, drawing, new Matrix());
assert.equal(frames.length, 2, 'ignore consumers stored inside definitions');
assert.deepEqual(components(frames[0].matrix), [1, 0, 0, 6, 110, 58]);
assert.deepEqual(components(frames[1].matrix), [1.5, 0, 0, 4, 215, 312]);
assert.equal(clippingFrames(contour, drawing, new Matrix(), 'second')[0].consumer, second);
// Coordinates must also account for the editor's screen transform.
assert.deepEqual(components(clippingFrames(contour, drawing, new Matrix().translate(10, 20).scale(2))[0].matrix), [.5, 0, 0, 3, 50, 19]);
units = 'objectBoundingBox'; clip.transform = transform(new Matrix().translate(.1, .2));
frames = clippingFrames(contour, drawing, new Matrix());
assert.deepEqual(components(frames[1].matrix), [150, 0, 0, 200, 260, 360]);
consumers = [consumer('empty', new Matrix(), {x:0, y:0, width:0, height:5})];
assert.equal(clippingFrames(contour, drawing, new Matrix()).length, 0, 'empty object bounds have no clip frame');
units = 'userSpaceOnUse'; consumers = [];
assert.equal(clippingFrames(contour, drawing, new Matrix())[0].consumer, root, 'unused clips remain editable');
assert.equal(clippingFrames(contour, drawing, null).length, 0);

// Run the editor's actual selection and coordinate helpers. Clip definitions
// must select their own contour, never their painted consumers or defs group.

const items = [
  {id:'defs', parent:'root', tag:'defs', resource:true},
  {id:'clip', parent:'defs', tag:'clipPath', resource:true},
  {id:'contour', parent:'clip', tag:'path', resource:true},
  {id:'group', parent:'root', tag:'g', resource:false},
  {id:'paint', parent:'group', tag:'path', resource:false},
  {id:'gradient', parent:'defs', tag:'linearGradient', resource:true},
];
const byId = new Map(items.map(item => [item.id, item]));
const paint = {dataset: {objectId:'paint'}, closest: () => null};
const gradient = {dataset: {objectId:'gradient'}, closest: () => null};
const elements = new Map([['clip', clip], ['contour', contour], ['paint', paint], ['gradient', gradient]]);
const guide = {dataset: {clipObject:'contour'}};
const hit = {closest: () => guide};
const overlay = {
  contains: element => element === hit,
  getScreenCTM: () => ({...new Matrix().scale(2), inverse() { return Matrix.fromMatrix(this).inverse(); }, multiply() { throw Error('legacy SVGMatrix cannot multiply DOMMatrix'); }}),
};
const context = Object.assign(createEditorState(), {
  DOMMatrix: Matrix, clippingFrames, drawing, overlay, clipConsumers: new Map(), pickTarget,
  svgElement: id => elements.get(id), object: id => byId.get(id),
  state: {root:'root'}, scope: null, level: () => 'objects',
  parents: () => new Map(items.map(item => [item.id, item.parent])),
  document: {elementsFromPoint: () => [hit, paint, gradient]},
});
drawing.contains = element => [paint, gradient].includes(element);
const context_frames = createFramesController(context, {
  DOMMatrix: context.DOMMatrix,
  document: context.document,
  drawOverlayContent: (...args) => context.drawOverlayContent(...args),
  drawing: context.drawing,
  overlay: context.overlay,
});
context.localToOverlay = context_frames.localToOverlay;
context.screenMatrix = context_frames.screenMatrix;
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
context.selectionTargets = context_overlay.selectionTargets;
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
context.targetId = context_canvas.targetId;
const context_hits = createHitsController(context, {
  document: context.document,
  object: (...args) => context.object(...args),
  svgElement: (...args) => context.svgElement(...args),
  targetId: (...args) => context.targetId(...args),
});
context.hitStack = context_hits.hitStack;
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
context.clickTargets = context_selection.clickTargets;
assert.equal((context.selectionTargets("contour")[0]), contour);
assert.equal((context.selectionTargets("clip")[0]), contour);
assert.deepEqual(Array.from((context.hitStack(0, 0))), ['contour', 'paint']);
assert.deepEqual(Array.from((context.clickTargets(context.hitStack(0, 0)))), ['contour', 'paint']);
assert.ok((context.localToOverlay(context.svgElement("clip"))), 'clipPath has a frame without getScreenCTM');
assert.ok((context.screenMatrix(context.svgElement("contour"))) instanceof Matrix, 'convert legacy SVGMatrix before multiplying');
