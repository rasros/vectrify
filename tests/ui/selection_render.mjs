// Run the editor's state application and tree rendering with a small DOM stand-in.
// Counting row/paint work makes this regression independent of machine load.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

class Element {
  constructor() {
    this.children = []; this.style = {}; this.dataset = {}; this.attributes = {};
    this.classes = new Set(); this.value = ''; this.writes = 0;
    this.classList = {toggle: (name, on) => { this.writes++; if (on) this.classes.add(name); else this.classes.delete(name); }, add: name => this.classes.add(name)};
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children.flatMap(child => child.fragment ? child.children : [child]); }
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
const context = vm.createContext({
  $, counts,
  document: {createElement: () => new Element(), createDocumentFragment: () => Object.assign(new Element(), {fragment: true})},
  overlay: new Element(),
  renderDrawing: () => counts.drawing++,
  renderInspector: () => counts.inspector++, drawOverlay: () => {},
  fit: () => counts.fit++,
  loadGeometries: async () => counts.loads++,
  missingGeometries: () => [],
  paintSwatch: swatch => { counts.paint++; swatch.title = 'Paint'; },
  objectContext: () => ({role: ''}),
  later: fn => fn(), pressTreeRow: () => {},
  CSS: {escape: id => id},
});
const source = readFileSync(new URL('../../src/vectrify/ui/static/app.js', import.meta.url), 'utf8');
const apply = source.slice(source.indexOf('async function applyState('), source.indexOf('const level ='));
const tree = source.slice(source.indexOf('let objectRows ='), source.indexOf('// Dragging rows in the tree'));
const reveal = source.slice(source.indexOf('function revealObject('), source.indexOf('function hitStack('));
vm.runInContext(`
  let state, objectsById = new Map(), dirty = false, scope = null;
  let focusPoint = null, pointMemory = null, geometries = new Map();
  let clickCycle = null, lastPick = null, pathHoles = new Map();
  let treeDrag = null, treeDragEnded = false, treeDropLine = {};
  let toolLevel = 'objects';
  const level = () => toolLevel;
  const object = id => objectsById.get(id);
  ${apply}
  ${tree}
  ${reveal}
`, context);
const applyState = vm.runInContext('applyState', context);
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
vm.runInContext('renderObjects()', context);
assert.equal(rows().length, 1);
await applyState({epoch: 'first', revision: 0, selection: {objects: ['p0'], nodes: []}});
$('object-search').value = '';
vm.runInContext('renderObjects()', context);
assert.equal(rows()[0].attributes['aria-selected'], 'true');
// Document edits and reopen at revision zero replace labels/paint/lookup entries.
const edited = objects.map((item, i) => i ? item : {...item, label: 'Renamed'});
await applyState({...initial, revision: 1, objects: edited});
assert.notEqual(rows()[0], original[0]);
assert.equal(vm.runInContext('object("p0").label', context), 'Renamed');
assert.equal(rows()[0].children[2].children[0].textContent, 'Renamed');
await applyState({...initial, epoch: 'second', objects: objects.slice(-1)});
assert.equal(rows().length, 1);
assert.equal(vm.runInContext('object("p0")', context), undefined);
assert.equal(counts.fit, 2);
// Point tools still load geometry and refresh their inspector after the load.
vm.runInContext('toolLevel = "points"', context);
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
assert.deepEqual(vm.runInContext('state.selection.objects', context), ['leaf']);
toggle('g').onclick(click);
assert.deepEqual(ids(), ['g', 'outside']);
await applyState({...nestedState, revision: 1});
assert.deepEqual(ids(), ['g', 'outside'], 'fold state survives document edits');
toggle('g').onclick(click);
assert.deepEqual(ids(), ['g', 'inner', 'sibling', 'outside'], 'nested group stays folded');
toggle('g').onclick(click);
$('object-search').value = 'leaf';
vm.runInContext('renderObjects()', context);
assert.deepEqual(ids(), ['g', 'inner', 'leaf']);
assert.equal(toggle('g').disabled, true);
assert.equal(rows()[2].attributes['aria-selected'], 'true');
$('object-search').value = '';
vm.runInContext('renderObjects()', context);
assert.deepEqual(ids(), ['g', 'outside'], 'clearing search restores folds');
vm.runInContext('revealObject("leaf")', context);
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
