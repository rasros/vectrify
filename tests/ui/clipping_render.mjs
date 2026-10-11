import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
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
const source = readFileSync(new URL('../../src/vectrify/ui/static/app.js', import.meta.url), 'utf8');
const section = (start, end) => source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start)));
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
const context = vm.createContext({
  DOMMatrix: Matrix, clippingFrames, drawing, overlay, clipConsumers: new Map(), pickTarget,
  svgElement: id => elements.get(id), object: id => byId.get(id),
  state: {root:'root'}, scope: null, level: () => 'objects',
  parents: () => new Map(items.map(item => [item.id, item.parent])),
  document: {elementsFromPoint: () => [hit, paint, gradient]},
});
drawing.contains = element => [paint, gradient].includes(element);
vm.runInContext(`
  ${section('let overlayFrame =', 'function selectionContour(')}
  ${section('function selectionTargets(', 'function clipSelection(')}
  ${section('function targetId(', 'function topSelection(')}
  ${section('function hitStack(', 'function sameClickSpot(')}
  ${section('function clickTargets(', 'async function selectAtPoint(')}
`, context);
assert.equal(vm.runInContext('selectionTargets("contour")[0]', context), contour);
assert.equal(vm.runInContext('selectionTargets("clip")[0]', context), contour);
assert.deepEqual(Array.from(vm.runInContext('hitStack(0, 0)', context)), ['contour', 'paint']);
assert.deepEqual(Array.from(vm.runInContext('clickTargets(hitStack(0, 0))', context)), ['contour', 'group']);
assert.ok(vm.runInContext('localToOverlay(svgElement("clip"))', context), 'clipPath has a frame without getScreenCTM');
assert.ok(vm.runInContext('screenMatrix(svgElement("contour"))', context) instanceof Matrix, 'convert legacy SVGMatrix before multiplying');
