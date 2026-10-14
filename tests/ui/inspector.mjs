import assert from 'node:assert/strict';
import {wireInspector, selectInspectorTab, revealInspectorField} from '../../src/vectrify/ui/static/inspector.js';
import {createSelectionController} from '../../src/vectrify/ui/static/app_selection.js';
import {createEditorState} from '../../src/vectrify/ui/static/app_state.js';

class Element {
  constructor() { this.hidden = false; this.value = ''; this.dataset = {}; this.attributes = {}; this.listeners = {}; this.open = false; }
  setAttribute(key, value) { this.attributes[key] = value; }
  getAttribute(key) { return this.attributes[key]; }
  addEventListener(key, fn) { this.listeners[key] = fn; }
  focus() { document.activeElement = this; }
}
const elements = new Map();
const $ = id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
const tabs = ['object', 'points', 'actions'].map(name => {
  const tab = $(`inspector-tab-${name}`); tab.dataset.inspectorTab = name;
  tab.setAttribute('aria-controls', `inspector-${name}`); return tab;
});
const coordinates = new Element(), x = new Element(), y = new Element();
coordinates.querySelector = selector => selector === '.point-x' ? x : y;
coordinates.querySelectorAll = () => [x, y];
const document = {
  activeElement: null, getElementById: $,
  querySelectorAll: selector => selector === '[data-inspector-tab]' ? tabs : selector === '.point-coordinates' ? [coordinates] : [],
};
wireInspector(document);
selectInspectorTab(document, 'points');
assert.equal($('inspector-points').hidden, false);
assert.equal(tabs[1].tabIndex, 0);
assert.equal(tabs[0].tabIndex, -1);
let prevented = 0;
tabs[1].listeners.keydown({key: 'ArrowRight', preventDefault() { prevented++; }, stopPropagation() {}});
assert.equal(document.activeElement, tabs[2]);
assert.equal($('inspector-actions').hidden, false);
assert.equal($('inspector-points').hidden, true);
assert.equal(prevented, 1);
tabs[2].listeners.keydown({key: 'Home', preventDefault() {}, stopPropagation() {}});
assert.equal(document.activeElement, tabs[0]);

// F2/palette field commands must open their tab and any folded ancestor.
const transform = $('transform-section'); transform.tagName = 'DETAILS';
x.parentElement = transform;
selectInspectorTab(document, 'actions');
revealInspectorField(document, x);
assert.equal(transform.open, true);
assert.equal(document.activeElement, x);
assert.equal($('inspector-object').hidden, false);
transform.open = false;
selectInspectorTab(document, 'points');
document.activeElement = null;

const editor = createEditorState();
const nodes = [{id: 'a', command: 'M', values: [10, 20]}, {id: 'b', command: 'L', values: [30, 40], pinned: true}];
let points = [], pointMode = true;
editor.state = {epoch: 'e', revision: 0, selection: {objects: ['p']}};
editor.tool = 'nodes'; editor.geometries.set('p', {});
const selection = createSelectionController(editor, {
  $, document, level: () => pointMode ? 'points' : 'objects',
  pointPaths: () => ['p'], selectedPoints: () => points,
  geometryNodes: () => nodes, nodeAt: key => nodes.find(n => key === `p ${n.id}`),
  pointTargetsNow: () => ({instances: []}), sharingPaths: () => [],
  contourAt: () => ({nodes, closed: false}), scheduleStrip() {}, renderCommands() {},
});
// Avoid an irrelevant hole lookup; all these points lie on the outer contour.
editor.pathHoles.set('e:0:p', new Set());
for (const chosen of [[], ['p a'], ['p a', 'p b'], ['p b'], []]) {
  points = chosen;
  selection.renderNodeInspector();
  assert.equal($('point-section').hidden, false);
  assert.equal($('point-properties').hidden, false);
  assert.equal(coordinates.hidden, false, 'coordinates never appear/disappear with selection');
  assert.equal($('inspector-points').hidden, false, 'selection never changes tabs');
  assert.equal(transform.open, false, 'selection never opens folded sections');
  assert.equal(x.disabled, chosen.length !== 1 || chosen[0] === 'p b');
  if (chosen.length !== 1) assert.equal(x.value, '', 'clear stale coordinates for zero/multiple points');
  else assert.equal(x.value, chosen[0] === 'p a' ? 10 : 30);
  assert.equal($('node-pin').disabled, chosen.length === 0);
}
pointMode = false;
selection.renderNodeInspector();
assert.equal($('node-pin').disabled, true);
assert.match($('point-selection').textContent, /Nodes tool/);
