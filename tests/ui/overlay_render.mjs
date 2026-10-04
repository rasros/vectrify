// Geometry reads must precede one overlay commit, and node work stays bounded.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const source = readFileSync(new URL('../../src/vectrify/ui/static/app.js', import.meta.url), 'utf8');
const matrices = source.slice(source.indexOf('let overlayFrame ='), source.indexOf('function selectionContour('));
const overlay = source.slice(source.indexOf('function drawOverlay()'), source.indexOf('function drawOverlayContent()'));
let commits = 0, reads = 0, overlayReads = 0, frame = 0;
const element = {getScreenCTM() { reads++; return {frame}; }};
const matrix = {inverse() {return this;}, multiply(from) {return from;}};
const context = vm.createContext({
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
vm.runInContext(`${matrices}\n${overlay}\n
function drawOverlayContent() {
 const a = localToOverlay(element);
 overlayFrame.content.append({shape: 1});
 const b = localToOverlay(element);
 if (a !== b) throw Error('reread an unchanged frame');
 overlayFrame.content.append({shape: 2});
}
`, context);
const drawOverlay = vm.runInContext('drawOverlay', context);
drawOverlay(); frame++; drawOverlay();
assert.equal(commits, 2); assert.equal(reads, 2); assert.equal(overlayReads, 2);
// Outside a redraw, live optimistic transforms must always be read again.
const localToOverlay = vm.runInContext('localToOverlay', context);
assert.equal(localToOverlay(element).frame, 1);
frame++;
assert.equal(localToOverlay(element).frame, 2);
vm.runInContext('drawOverlayContent = () => { throw Error("render failed"); }', context);
assert.throws(drawOverlay, /render failed/);
assert.equal(vm.runInContext('overlayFrame', context), null, 'a failed redraw must release its cache');

// Reuse node and contour lookups while values are previewed; reindex only
// when the backend supplies a replacement geometry, including shared users.
const lookupSource = source.slice(source.indexOf('const geometryIndexes ='), source.indexOf('// The selected points, from'));
let contourReads = 0;
const node = {id: 'n', values: [1, 2]}, contour = {nodes: [node]};
const geometry = {get subpaths() {contourReads++; return [contour];}};
const geometries = new Map([['p', geometry], ['shared', geometry]]);
const lookupContext = vm.createContext({geometries, splitKey: key => key.split(' ')});
vm.runInContext(lookupSource, lookupContext);
const lookup = name => vm.runInContext(name, lookupContext);
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
const pointSource = source.slice(source.indexOf('function drawPoints()'), source.indexOf('function drawHandles('));
const allNodes = Array.from({length: 2000}, (_, i) => ({id: `n${i}`, values: [i*10, i*10]}));
const content = {children: [], append(child) {this.children.push(child);}};
let picked = [], handles = 0;
const pointContext = vm.createContext({
 stage: {getBoundingClientRect: () => ({left: -1, top: -1, right: 50000, bottom: 50000})},
 shownPoints: () => ({strong: picked, twins: []}), pointPaths: () => ['p'],
 tool: 'nodes', hoverPath: null, nearPoint: null, nearAnchor: null, zoom: 1,
 geometries: new Map([['p', {}]]),
 svgElement: () => ({getScreenCTM: () => ({a:1,b:0,c:0,d:1,e:0,f:0})}),
 localToOverlay: () => ({a:2,b:1,c:3,d:4,e:5,f:6}),
 geometryNodes: () => allNodes, pointKey: (id, node) => `${id} ${node}`,
 xmlElement: (_name, attrs) => ({...attrs, dataset: {}}),
 overlayFrame: {content}, drawHandles: () => handles++,
 $: () => pointContext.counter,
 counter: {},
});
vm.runInContext(pointSource, pointContext);
const drawPoints = vm.runInContext('drawPoints', pointContext);
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
