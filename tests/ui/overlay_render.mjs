import {createEditorState} from '../../src/vectrify/ui/static/app_state.js';
import {createFramesController} from '../../src/vectrify/ui/static/app_frames.js';
import {createAgentsController} from '../../src/vectrify/ui/static/app_agents.js';
import {createGeometryController} from '../../src/vectrify/ui/static/app_geometry.js';
import {createMarkersController} from '../../src/vectrify/ui/static/app_markers.js';
import {createSelectionController} from '../../src/vectrify/ui/static/app_selection.js';
import {createOverlayController} from '../../src/vectrify/ui/static/app_overlay.js';
// Geometry reads must precede one overlay commit, and node work stays bounded.
import assert from 'node:assert/strict';

let commits = 0, reads = 0, overlayReads = 0, frame = 0;
const element = {getScreenCTM() { reads++; return {frame}; }};
const matrix = {inverse() {return this;}, multiply(from) {return from;}};
const context = Object.assign(createEditorState(), {
  state: {}, zoom: 1,
  DOMMatrix: {fromMatrix: m => m},
  document: {createDocumentFragment: () => ({children: [], append(...items) {this.children.push(...items);}})},
  overlay: {
    style: {setProperty() {}},
    getScreenCTM() {overlayReads++; return matrix;},
    replaceChildren(content) {
      assert.equal(reads, commits + 1, 'read the drawing before committing any shapes');
      assert.equal(content.children.length, 2); commits++;
    },
  },
  element,
});
context.drawOverlayContent = function drawOverlayContent() {
 const a = context.localToOverlay(context.element);
 context.overlayFrame.content.append({shape: 1});
 const b = context.localToOverlay(context.element);
 if (a !== b) throw Error('reread an unchanged frame');
 context.overlayFrame.content.append({shape: 2});
};
const context_frames = createFramesController(context, {
  DOMMatrix: context.DOMMatrix,
  document: context.document,
  drawOverlayContent: (...args) => context.drawOverlayContent(...args),
  drawing: context.drawing,
  overlay: context.overlay,
});
context.localToOverlay = context_frames.localToOverlay;
context.screenMatrix = context_frames.screenMatrix;
context.drawOverlay = context_frames.drawOverlay;
const drawOverlay = (context.drawOverlay);
drawOverlay(); frame++; drawOverlay();
assert.equal(commits, 2); assert.equal(reads, 2); assert.equal(overlayReads, 2);
// Outside a redraw, live optimistic transforms must always be read again.
const localToOverlay = (context.localToOverlay);
assert.equal(localToOverlay(element).frame, 1);
frame++;
assert.equal(localToOverlay(element).frame, 2);
(context.drawOverlayContent = () => { throw Error("render failed"); });
assert.throws(drawOverlay, /render failed/);
assert.equal((context.overlayFrame), null, 'a failed redraw must release its cache');

// An agent edit may touch both a path and its gradient. Paint resources have
// no screen CTM: ignore them while still flashing drawable changed objects.

const flashContent = {children: [], append(child) {this.children.push(child);}};
const path = {
 getScreenCTM: () => matrix,
 getBBox: () => ({x: 0, y: 0, width: 10, height: 20}),
};
const resources = new Map([
 ['gradient', {localName: 'linearGradient'}],
 ['stop', {localName: 'stop'}],
 ['path', path],
]);
const flashContext = Object.assign(createEditorState(), {
 DOMMatrix: {fromMatrix: m => m},
 DOMPoint: class {
  constructor(x, y) {this.x = x; this.y = y;}
  matrixTransform() {return this;}
 },
 overlay: {getScreenCTM: () => matrix},
 svgElement: id => resources.get(id),
 getComputedStyle: () => ({stroke: 'none'}),
 xmlElement: (name, attrs) => ({name, ...attrs, style: {}}),
 performance: {now: () => 100},
 clearTimeout() {}, setTimeout: () => 1,
 drawOverlay: () => (flashContext.drawFlash()),
});
flashContext.FLASH_MS = 1500;
  flashContext.agent.flash = {ids: [], start: 0, timer: null, seen: 0};
const flashContext_frames = createFramesController(flashContext, {
  DOMMatrix: flashContext.DOMMatrix,
  document: flashContext.document,
  drawOverlayContent: (...args) => flashContext.drawOverlayContent(...args),
  drawing: flashContext.drawing,
  overlay: flashContext.overlay,
});
flashContext.localToOverlay = flashContext_frames.localToOverlay;
flashContext.screenMatrix = flashContext_frames.screenMatrix;
const flashContext_agents = createAgentsController(flashContext, {
  DOMPoint: flashContext.DOMPoint,
  applyState: (...args) => flashContext.applyState(...args),
  drawOverlay: (...args) => flashContext.drawOverlay(...args),
  getComputedStyle: flashContext.getComputedStyle,
  localToOverlay: (...args) => flashContext.localToOverlay(...args),
  request: (...args) => flashContext.request(...args),
  showAgent: (...args) => flashContext.showAgent(...args),
  stage: flashContext.stage,
  svgElement: (...args) => flashContext.svgElement(...args),
  toast: (...args) => flashContext.toast(...args),
  window: flashContext.window,
  xmlElement: (...args) => flashContext.xmlElement(...args),
});
flashContext.flashTouched = flashContext_agents.flashTouched;
flashContext.drawFlash = flashContext_agents.drawFlash;
flashContext.content = flashContent;
(flashContext.overlayFrame = {content: flashContext.content, to: flashContext.overlay.getScreenCTM(), matrices: new Map()});
const flashTouched = (flashContext.flashTouched);
flashTouched([{change: 1, ids: ['gradient', 'stop', 'path', 'deleted']}]);
assert.equal(flashContent.children.length, 1);
assert.equal(flashContent.children[0]['data-object'], 'path');
assert.equal(flashContent.children[0].class, 'agent-flash');
assert.equal(flashContent.children[0].points, '0,0 10,0 10,20 0,20');
const resourceFrame = (flashContext.localToOverlay);
for (const value of [undefined, null, ...resources.values()].filter(value => value !== path)) {
 assert.equal(resourceFrame(value), null);
}

// Reuse node and contour lookups while values are previewed; reindex only
// when the backend supplies a replacement geometry, including shared users.

let contourReads = 0;
const node = {id: 'n', values: [1, 2]}, contour = {nodes: [node]};
const geometry = {get subpaths() {contourReads++; return [contour];}};
const geometries = new Map([['p', geometry], ['shared', geometry]]);
const lookupContext = Object.assign(createEditorState(), {geometries, splitKey: key => key.split(' ')});
const lookupContext_geometry = createGeometryController(lookupContext, {
  action: (...args) => lookupContext.action(...args),
  request: (...args) => lookupContext.request(...args),
  schedulePoll: (...args) => lookupContext.schedulePoll(...args),
});
lookupContext.geometryIndexes = lookupContext_geometry.geometryIndexes;
lookupContext.geometryIndex = lookupContext_geometry.geometryIndex;
lookupContext.geometryNodes = lookupContext_geometry.geometryNodes;
lookupContext.nodeAt = lookupContext_geometry.nodeAt;
lookupContext.contourAt = lookupContext_geometry.contourAt;
const lookup = name => lookupContext[name];
const nodes = lookup('geometryNodes')('p');
for (let i = 0; i < 1000; i++) {
 assert.equal(lookup('nodeAt')('p n'), node);
 assert.equal(lookup('contourAt')('p n'), contour);
}
assert.equal(contourReads, 1);
assert.equal(lookup('geometryNodes')('shared'), nodes);
node.values = [3, 4];
assert.deepEqual(lookup('nodeAt')('p n').values, [3, 4]);
const replacement = {id: 'new', values: [5, 6]};
geometries.set('p', {subpaths: [{nodes: [replacement]}]});
assert.equal(lookup('nodeAt')('p n'), undefined);
assert.equal(lookup('nodeAt')('p new'), replacement);
assert.equal(lookup('nodeAt')('missing n'), undefined);

// Stop creating unselected markers at 1200 while preserving the true count,
// and continue to show selected points beyond that cap.

const allNodes = Array.from({length: 2000}, (_, i) => ({id: `n${i}`, values: [i*10, i*10]}));
const content = {children: [], append(child) {this.children.push(child);}};
let picked = [], handles = 0;
const pointContext = Object.assign(createEditorState(), {
 stage: {getBoundingClientRect: () => ({left: -1, top: -1, right: 50000, bottom: 50000})},
 shownPoints: () => ({strong: picked, twins: []}), pointPaths: () => ['p'],
 tool: 'nodes', hoverPath: null, nearPoint: null, nearAnchor: null, zoom: 1,
 geometries: new Map([['p', {}]]),
 svgElement: () => ({getScreenCTM: () => ({a:1,b:0,c:0,d:1,e:0,f:0})}),
 screenMatrix: element => element.getScreenCTM(),
 localToOverlay: () => ({a:2,b:1,c:3,d:4,e:5,f:6}),
 geometryNodes: () => allNodes, pointKey: (id, node) => `${id} ${node}`,
 xmlElement: (_name, attrs) => ({...attrs, dataset: {}}),
 overlayFrame: {content}, drawHandles: () => handles++,
 $: () => pointContext.counter,
 counter: {},
});
const pointContext_markers = createMarkersController(pointContext, {
  $: (...args) => pointContext.$(...args),
  drawHandles: (...args) => pointContext.drawHandles(...args),
  geometryNodes: (...args) => pointContext.geometryNodes(...args),
  localToOverlay: (...args) => pointContext.localToOverlay(...args),
  pointPaths: (...args) => pointContext.pointPaths(...args),
  screenMatrix: (...args) => pointContext.screenMatrix(...args),
  shownPoints: (...args) => pointContext.shownPoints(...args),
  stage: pointContext.stage,
  svgElement: (...args) => pointContext.svgElement(...args),
  xmlElement: (...args) => pointContext.xmlElement(...args),
});
pointContext.drawPoints = pointContext_markers.drawPoints;
const drawPoints = (pointContext.drawPoints);
drawPoints();
assert.equal(content.children.length, 1200);
assert.equal(pointContext.counter.textContent, '1,200 / 2,000');
assert.equal(content.children[1].cx, 55); assert.equal(content.children[1].cy, 56);
picked = ['p n1999']; content.children = [];
drawPoints();
assert.equal(content.children.length, 1201);
assert.ok(content.children.at(-1).class.includes('selected'));
assert.equal(content.children.at(-1).dataset.node, 'n1999');
assert.equal(handles, 1);

// Either node at a closed join must count and draw both of its handles.

const joinNodes = [
 {id: 'start', command: 'M', values: [0, 0]},
 {id: 'out', command: 'C', values: [10, -10, 30, -10, 20, 0]},
 {id: 'end', command: 'C', values: [30, 10, -10, 10, 0, 0]},
];
const joinContour = {nodes: joinNodes, closed: true};
const handleContent = {children: [], append(child) {this.children.push(child);}};
class Point {
 constructor(x, y) {this.x = x; this.y = y;}
 matrixTransform() {return this;}
}
const handleContext = Object.assign(createEditorState(), {
 nodeAt: key => joinNodes.find(n => n.id === key.split(' ')[1]),
 contourAt: () => joinContour, splitKey: key => key.split(' '),
 localToOverlay: () => ({}), svgElement: () => ({}), DOMPoint: Point,
 zoom: 1, HANDLE_SPREAD: 16, nearPoint: null, activeHandle: null,
 overlayFrame: {content: handleContent},
 xmlElement: (name, attrs) => ({name, ...attrs, dataset: {}}),
});
const handleContext_selection = createSelectionController(handleContext, {
  $: (...args) => handleContext.$(...args),
  action: (...args) => handleContext.action(...args),
  contourAt: (...args) => handleContext.contourAt(...args),
  document: handleContext.document,
  focusSelection: (...args) => handleContext.focusSelection(...args),
  geometryNodes: (...args) => handleContext.geometryNodes(...args),
  later: (...args) => handleContext.later(...args),
  level: (...args) => handleContext.level(...args),
  nodeAt: (...args) => handleContext.nodeAt(...args),
  object: (...args) => handleContext.object(...args),
  overlay: handleContext.overlay,
  parents: (...args) => handleContext.parents(...args),
  pointPaths: (...args) => handleContext.pointPaths(...args),
  pointTargetsNow: (...args) => handleContext.pointTargetsNow(...args),
  renderCommands: (...args) => handleContext.renderCommands(...args),
  request: (...args) => handleContext.request(...args),
  revealObject: (...args) => handleContext.revealObject(...args),
  runCommand: (...args) => handleContext.runCommand(...args),
  scheduleStrip: (...args) => handleContext.scheduleStrip(...args),
  selectPoints: (...args) => handleContext.selectPoints(...args),
  selectedPoints: (...args) => handleContext.selectedPoints(...args),
  sharingPaths: (...args) => handleContext.sharingPaths(...args),
  svgElement: (...args) => handleContext.svgElement(...args),
  toast: (...args) => handleContext.toast(...args),
});
handleContext.nodeHandles = handleContext_selection.nodeHandles;
handleContext.handleCount = handleContext_selection.handleCount;
handleContext.selectedHandle = handleContext_selection.selectedHandle;
handleContext.straightenHandles = handleContext_selection.straightenHandles;
const handleContext_overlay = createOverlayController(handleContext, {
  $: (...args) => handleContext.$(...args),
  CSS: handleContext.CSS,
  DOMPoint: handleContext.DOMPoint,
  drawFlash: (...args) => handleContext.drawFlash(...args),
  drawKnife: (...args) => handleContext.drawKnife(...args),
  drawOverlay: (...args) => handleContext.drawOverlay(...args),
  drawPathDraft: (...args) => handleContext.drawPathDraft(...args),
  drawPoints: (...args) => handleContext.drawPoints(...args),
  drawRedraw: (...args) => handleContext.drawRedraw(...args),
  drawSnap: (...args) => handleContext.drawSnap(...args),
  drawing: handleContext.drawing,
  getComputedStyle: handleContext.getComputedStyle,
  later: (...args) => handleContext.later(...args),
  level: (...args) => handleContext.level(...args),
  localToOverlay: (...args) => handleContext.localToOverlay(...args),
  nodeAt: (...args) => handleContext.nodeAt(...args),
  nodeHandles: (...args) => handleContext.nodeHandles(...args),
  object: (...args) => handleContext.object(...args),
  overlay: handleContext.overlay,
  renderStatus: (...args) => handleContext.renderStatus(...args),
  resizeRefusal: (...args) => handleContext.resizeRefusal(...args),
  svgElement: (...args) => handleContext.svgElement(...args),
  topSelection: (...args) => handleContext.topSelection(...args),
  xmlElement: (...args) => handleContext.xmlElement(...args),
});
handleContext.drawHandles = handleContext_overlay.drawHandles;
const handleCount = (handleContext.handleCount);
const renderHandles = (handleContext.drawHandles);
for (const key of ['p start', 'p end']) {
 assert.equal(handleCount(key), 2);
 handleContent.children = [];
 renderHandles(key);
 const circles = handleContent.children.filter(n => n.name === 'circle');
 assert.deepEqual(circles.map(n => [n.dataset.node, n.dataset.part]), [['end', '2'], ['out', '0']]);
 assert.ok(handleContent.children.filter(n => n.name === 'line').every(n => n.x1 === 0 && n.y1 === 0));
}
joinNodes[2].values.splice(2, 2, 0, 0);
assert.equal(handleCount('p start'), 1);
assert.equal(handleCount('p end'), 1);
handleContent.children = [];
renderHandles('p end');
assert.equal(handleContent.children.filter(n => n.name === 'circle').length, 1);
// Coincident endpoints of an open path remain separate points.
joinContour.closed = false;
assert.equal(handleCount('p start'), 1);
assert.equal(handleCount('p end'), 0);
joinNodes[2].values.splice(2, 2, -10, 10);
assert.equal(handleCount('p end'), 1);
// An implicit close has no controls until the backend materializes it.
joinContour.closed = true;
joinNodes.pop();
assert.equal(handleCount('p start'), 1);
assert.equal(handleCount('p out'), 1);
