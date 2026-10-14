import {createEditorState} from '../../src/vectrify/ui/static/app_state.js';
import {createSessionController} from '../../src/vectrify/ui/static/app_session.js';
import {createTreeController} from '../../src/vectrify/ui/static/app_tree.js';
// Run the editor's state application and tree rendering with a small DOM stand-in.
// Counting row/paint work makes this regression independent of machine load.
import assert from 'node:assert/strict';

class Element {
  constructor() {
    this.children = []; this.style = {}; this.dataset = {}; this.attributes = {};
    this.classes = new Set(); this.value = ''; this.writes = 0;
    this.scrollTop = 0; this.scrollLeft = 0;
    this.classList = {toggle: (name, on) => { this.writes++; if (on) this.classes.add(name); else this.classes.delete(name); }, add: name => this.classes.add(name)};
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) {
    this.children = children.flatMap(child => child.fragment ? child.children : [child]);
    // Model the browser shifting the viewport as its old rows disappear.
    if (this.scrollTop) this.scrollTop += 49;
    if (this.scrollLeft) this.scrollLeft = 0;
  }
  setAttribute(name, value) { this.writes++; this.attributes[name] = value; }
  querySelector(selector) {
    if (selector === '.tree-toggle') return this.children.find(child => child.className === 'tree-toggle');
    const id = selector.match(/^\[data-object="(.+)"\]$/)?.[1];
    return this.children.find(child => child.dataset.object === id);
  }
  focus() { this.focused = true; }
  scrollIntoView() { this.revealed = true; }
}
const elements = new Map();
const $ = id => {
  if (!elements.has(id)) elements.set(id, new Element());
  return elements.get(id);
};
const counts = {paint: 0, drawing: 0, inspector: 0, fit: 0, loads: 0};
const context = Object.assign(createEditorState(), {
  $, counts,
  document: {createElement: () => new Element(), createDocumentFragment: () => Object.assign(new Element(), {fragment: true})},
  overlay: new Element(),
  clipConsumers: new Map(),
  renderDrawing: () => counts.drawing++,
  renderInspector: () => counts.inspector++, drawOverlay: () => {},
  fit: () => counts.fit++,
  loadGeometries: async () => counts.loads++,
  loadReference: async () => {},
  missingGeometries: () => [],
  paintSwatch: swatch => { counts.paint++; swatch.title = 'Paint'; },
  objectContext: () => ({role: ''}),
  later: fn => fn(), pressTreeRow: () => {},
  CSS: {escape: id => id},
});

context.state = undefined, context.savedVersion = null, context.objectsById = new Map(), context.dirty = false, context.scope = null;
   context.focusPoint = null, context.pointMemory = null, context.geometries = new Map();
   context.clickCycle = null, context.lastPick = null, context.pathHoles = new Map();
   context.tree.drag = null, context.tree.dragEnded = false, context.treeDropLine = {};
   context.toolLevel = 'objects';
   context.level = () => context.toolLevel;
   context.object = id => context.objectsById.get(id);
   context.files = {currentName: () => context.state.name};
const context_session = createSessionController(context, {
  $: (...args) => context.$(...args),
  document: context.document,
  drawOverlay: (...args) => context.drawOverlay(...args),
  files: context.files,
  fit: (...args) => context.fit(...args),
  level: (...args) => context.level(...args),
  loadGeometries: (...args) => context.loadGeometries(...args),
  loadReference: (...args) => context.loadReference(...args),
  missingGeometries: (...args) => context.missingGeometries(...args),
  overlay: context.overlay,
  renderDrawing: (...args) => context.renderDrawing(...args),
  renderInspector: (...args) => context.renderInspector(...args),
  renderObjects: (...args) => context.renderObjects(...args),
  selectedHandle: (...args) => context.selectedHandle(...args),
  updateObjectSelection: (...args) => context.updateObjectSelection(...args),
});
context.applyState = context_session.applyState;
context.object = context_session.object;
const context_tree = createTreeController(context, {
  $: (...args) => context.$(...args),
  CSS: context.CSS,
  document: context.document,
  enterObject: (...args) => context.enterObject(...args),
  later: (...args) => context.later(...args),
  object: (...args) => context.object(...args),
  objectContext: (...args) => context.objectContext(...args),
  paintSwatch: (...args) => context.paintSwatch(...args),
  pressTreeRow: (...args) => context.pressTreeRow(...args),
  selectObject: (...args) => context.selectObject(...args),
});
context.setGroupExpanded = context_tree.setGroupExpanded;
context.updateObjectSelection = context_tree.updateObjectSelection;
context.renderObjects = context_tree.renderObjects;
context.revealObject = context_tree.revealObject;
const applyState = (context.applyState);
const rows = () => $('objects').children.slice(0, -1);
const objects = Array.from({length: 772}, (_, i) => ({id: `p${i}`, parent: 'root', tag: 'path', label: `Path ${i}`, depth: 0, inherited_locks: []}));
const initial = {epoch: 'first', revision: 0, name: 'large.svg', root: 'root', objects, bounds: [0, 0, 1000, 1000], selection: {objects: [], nodes: []}, undo: [], redo: [], svg: '<svg/>'};
await applyState(initial);
assert.equal(rows().length, 772);
assert.equal(counts.paint, 772);
const original = rows();
for (let i = 0; i < 20; i++) {
  // Responses carry fresh JSON arrays, but unchanged revision means the same tree.
  await applyState({...initial, objects: objects.map(item => ({...item})), svg: undefined, selection: {objects: [`p${i}`], nodes: []}});
  assert.deepEqual(rows(), original);
  assert.equal(original[i].attributes['aria-selected'], 'true');
  if (i) assert.equal(original[i - 1].attributes['aria-selected'], 'false');
}
assert.equal(counts.paint, 772, 'selection must not reread every path paint');
assert.equal(counts.drawing, 1);
assert.equal(counts.inspector, 21, 'object selection renders the inspector once');
// Only the old and new selection markers were changed; all other rows stayed untouched.
assert.equal(original[500].writes, original[771].writes);
await applyState({epoch: 'first', revision: 0, selection: {objects: [], nodes: []}});
assert.equal(original[19].attributes['aria-selected'], 'false');
// Search rebuilds the visible rows; selecting a hidden row must be harmless.
$('object-search').value = 'Path 771';
(context.renderObjects());
assert.equal(rows().length, 1);
await applyState({epoch: 'first', revision: 0, selection: {objects: ['p0'], nodes: []}});
$('object-search').value = '';
(context.renderObjects());
assert.equal(rows()[0].attributes['aria-selected'], 'true');
// Document edits and reopen at revision zero replace labels/paint/lookup entries.
const edited = objects.map((item, i) => i ? item : {...item, label: 'Renamed'});
await applyState({...initial, revision: 1, objects: edited});
assert.notEqual(rows()[0], original[0]);
assert.equal((context.object("p0").label), 'Renamed');
assert.equal(rows()[0].children[2].children[0].textContent, 'Renamed');
// Saving repeated canvas point/handle drags refreshes the tree's rows without
// changing where the user has scrolled, even far from the selected path.
$('objects').scrollTop = 6400;
$('objects').scrollLeft = 12;
for (let revision = 2; revision <= 4; revision++) {
  await applyState({...initial, revision, selection: {objects: ['p0'], nodes: ['n0']}});
  assert.equal($('objects').scrollTop, 6400, 'point edits must preserve the tree viewport');
  assert.equal($('objects').scrollLeft, 12);
  assert.equal(rows()[0].revealed, undefined, 'saving a drag must not reveal its object');
}
await applyState({...initial, epoch: 'second', objects: objects.slice(-1)});
assert.equal(rows().length, 1);
assert.equal((context.object("p0")), undefined);
assert.equal(counts.fit, 2);
// Point tools still load geometry and refresh their inspector after the load.
(context.toolLevel = "points");
await applyState({epoch: 'second', revision: 0, selection: {objects: ['p771'], nodes: []}});
assert.equal(counts.loads, 1);

// Folding nested groups hides whole branches, preserves selection and nested
// fold state, and never asks the editor to select or edit anything.
const nested = [
  ['g', 'root', 'g', 0], ['inner', 'g', 'g', 1],
  ['leaf', 'inner', 'path', 2], ['sibling', 'g', 'path', 1],
  ['outside', 'root', 'path', 0],
].map(([id, parent, tag, depth]) => ({id, parent, tag, depth, label: id, inherited_locks: []}));
const nestedState = {...initial, epoch: 'nested', objects: nested, selection: {objects: ['leaf'], nodes: []}};
await applyState(nestedState);
const ids = () => rows().map(row => row.dataset.object);
const toggle = id => rows().find(row => row.dataset.object === id).children[0];
let stopped = 0;
const click = {stopPropagation: () => stopped++};
toggle('inner').onpointerdown(click);
toggle('inner').ondblclick(click);
toggle('inner').onclick(click);
assert.equal(stopped, 3, 'disclosure does not select, enter or drag the group');
assert.deepEqual(ids(), ['g', 'inner', 'sibling', 'outside']);
assert.equal(toggle('inner').attributes['aria-expanded'], 'false');
assert.equal(toggle('inner').focused, true);
assert.deepEqual((context.state.selection.objects), ['leaf']);
toggle('g').onclick(click);
assert.deepEqual(ids(), ['g', 'outside']);
await applyState({...nestedState, revision: 1});
assert.deepEqual(ids(), ['g', 'outside'], 'fold state survives document edits');
toggle('g').onclick(click);
assert.deepEqual(ids(), ['g', 'inner', 'sibling', 'outside'], 'nested group stays folded');
toggle('g').onclick(click);
$('object-search').value = 'leaf';
(context.renderObjects());
assert.deepEqual(ids(), ['g', 'inner', 'leaf']);
assert.equal(toggle('g').disabled, true);
assert.equal(rows()[2].attributes['aria-selected'], 'true');
$('object-search').value = '';
(context.renderObjects());
assert.deepEqual(ids(), ['g', 'outside'], 'clearing search restores folds');
(context.revealObject("leaf"));
assert.deepEqual(ids(), ['g', 'inner', 'leaf', 'sibling', 'outside']);
assert.equal(rows()[2].revealed, true);
const key = {target: rows()[0], key: 'ArrowLeft', preventDefault() {}, stopPropagation() {}};
rows()[0].onkeydown(key);
assert.deepEqual(ids(), ['g', 'outside']);
rows()[0].onkeydown({...key, target: rows()[0], key: 'ArrowRight'});
assert.deepEqual(ids(), ['g', 'inner', 'leaf', 'sibling', 'outside']);
toggle('g').onclick(click);
await applyState({...nestedState, epoch: 'reopened'});
assert.deepEqual(ids(), ['g', 'inner', 'leaf', 'sibling', 'outside'], 'new drawing resets folds');
