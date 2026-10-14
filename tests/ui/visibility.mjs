// Hiding objects: the eye on each row of the Objects tree, the commands, and
// what hidden objects are left out of on the canvas. Run by
// test_visibility.py.
import assert from 'node:assert/strict';
import {createEditorState} from '../../src/vectrify/ui/static/app_state.js';
import {createTreeController} from '../../src/vectrify/ui/static/app_tree.js';
import {createSelectionController} from '../../src/vectrify/ui/static/app_selection.js';
import {createHitsController} from '../../src/vectrify/ui/static/app_hits.js';
import {createResizingController, selectionResizeRefusal} from '../../src/vectrify/ui/static/app_resizing.js';
import {pointTargets} from '../../src/vectrify/ui/static/selection.js';
import {EXPORTED, eyeControl, hiddenRefusal, visibilityRefusal} from '../../src/vectrify/ui/static/visibility.js';

class Element {
  constructor(tag = '') {
    this.tag = tag; this.children = []; this.style = {}; this.dataset = {}; this.attributes = {};
    this.classes = new Set(); this.value = ''; this.title = ''; this.textContent = '';
    this.classList = {toggle: (name, on) => { if (on) this.classes.add(name); else this.classes.delete(name); }, add: name => this.classes.add(name)};
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children.flatMap(child => child.fragment ? child.children : [child]); }
  setAttribute(name, value) { this.attributes[name] = value; }
  querySelector() { return null; }
}
const elements = new Map();
const $ = id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
const document = {createElement: tag => new Element(tag), createDocumentFragment: () => Object.assign(new Element(), {fragment: true})};

// A layer group, hidden, holding a path; a shown rectangle; a hidden path
// whose paint is locked; definitions with a clipping boundary.
const objects = [
  {id: 'layer', tag: 'g', label: 'Layer', parent: 'root', depth: 0, hidden: true, inherited_hidden: true},
  {id: 'inner', tag: 'path', label: 'Inner', parent: 'layer', depth: 1, hidden: false, inherited_hidden: true},
  {id: 'box', tag: 'rect', label: 'Box', parent: 'root', depth: 0, hidden: false, inherited_hidden: false},
  {id: 'kept', tag: 'path', label: 'Kept', parent: 'root', depth: 0, hidden: true, inherited_hidden: true, locks: ['paint'], inherited_locks: ['paint']},
  {id: 'defs', tag: 'defs', label: 'Definitions', parent: 'root', depth: 0, resource: true},
  {id: 'clip', tag: 'clipPath', label: 'Clip', parent: 'defs', depth: 1, resource: true},
  {id: 'edge', tag: 'path', label: 'Edge', parent: 'clip', depth: 2, resource: true},
];
for (const item of objects) Object.assign(item, {resource: !!item.resource, hidden: !!item.hidden, inherited_hidden: !!item.inherited_hidden, locks: item.locks || [], inherited_locks: item.inherited_locks || [], attributes: {}});
const object = id => objects.find(item => item.id === id);

// The pure rules: what each row's eye says and does.
const eye = id => eyeControl(object(id), object);
assert.equal(eye('edge'), null, 'definitions and clipping boundaries have no eye');
assert.equal(eye('defs'), null);
assert.deepEqual({...eye('box'), title: undefined}, {hidden: false, concealed: false, label: 'Hide Box', title: undefined, refusal: ''});
assert.ok(eye('box').title.includes(EXPORTED), 'the eye says hiding is saved in the SVG');
assert.equal(eye('layer').label, 'Show Layer');
assert.match(eye('inner').title, /^Hidden with Layer: show the group to show it/);
assert.equal(eye('inner').concealed, true);
assert.equal(eye('inner').hidden, false);
assert.equal(eye('kept').refusal, 'Kept: paint is locked; unlock it to show it');
assert.equal(hiddenRefusal(['box', 'inner'], object, 'move'), 'Inner is hidden with Layer: show it to move it');
assert.equal(hiddenRefusal(['layer'], object, 'resize'), 'Layer is hidden: show it to resize it');
assert.equal(hiddenRefusal(['box'], object, 'move'), '');
assert.equal(visibilityRefusal([], object, true), 'Select objects first');
assert.equal(visibilityRefusal(['box'], object, true), '');
assert.equal(visibilityRefusal(['layer'], object, true), 'The selection is hidden already');
assert.equal(visibilityRefusal(['box', 'inner'], object, false), 'Nothing selected is hidden itself');
assert.equal(visibilityRefusal(['box', 'layer'], object, false), '');
assert.equal(visibilityRefusal(['edge'], object, true), 'Definitions and clipping boundaries cannot be hidden');
assert.equal(visibilityRefusal(['kept'], object, false), 'Kept: paint is locked; unlock it to show it');

// The tree: an eye at the end of every drawing row, apart from the fold
// arrow, crossed out on hidden rows; hidden rows stay listed, dimmed.
const editor = Object.assign(createEditorState(), {state: {epoch: 'e', revision: 0, root: 'root', selection: {objects: [], nodes: []}, objects}});
const actions = [], toasts = [], queued = [];
const tree = createTreeController(editor, {
  $, CSS: {escape: id => id}, document, object,
  action: async (...args) => { actions.push(args); return true; },
  later: fn => { queued.push(fn); return fn(); },
  enterObject: () => assert.fail('a hidden row is not entered'),
  objectContext: () => ({role: ''}), paintSwatch: swatch => { swatch.title = 'Paint'; },
  pressTreeRow: () => assert.fail('pressing the eye starts no row drag'),
  selectObject: () => assert.fail('the eye does not select its row'),
  toast: (message, error) => toasts.push({message, error: !!error}),
});
tree.renderObjects();
const rows = new Map($('objects').children.filter(row => row.dataset?.object).map(row => [row.dataset.object, row]));
const eyeOf = id => rows.get(id).children.find(child => child.className === 'tree-eye');
assert.equal(eyeOf('edge'), undefined);
assert.equal(eyeOf('defs'), undefined);
assert.equal(rows.get('layer').children[0].className, 'tree-toggle', 'the fold arrow stays first');
assert.equal(rows.get('layer').children.at(-1), eyeOf('layer'), 'the eye comes last');
assert.ok(eyeOf('layer').classes.has('off'));
assert.equal(eyeOf('layer').attributes['aria-pressed'], 'true');
assert.equal(eyeOf('layer').attributes['aria-label'], 'Show Layer');
assert.ok(eyeOf('inner').classes.has('inherited') && !eyeOf('inner').classes.has('off'));
assert.ok(rows.get('inner').classes.has('concealed') && rows.get('layer').classes.has('concealed'));
assert.ok(!rows.get('box').classes.has('concealed') && !eyeOf('box').classes.has('off'));
assert.equal(eyeOf('kept').attributes['aria-disabled'], 'true');

const event = {stopped: false, stopPropagation() { this.stopped = true; }};
eyeOf('layer').onpointerdown(event);
assert.ok(event.stopped, 'pressing the eye starts no tree drag');
eyeOf('box').onclick({...event});
eyeOf('layer').onclick({...event});
assert.deepEqual(actions, [
  ['visibility', {objects: ['box'], hidden: true}, 'Hiding…'],
  ['visibility', {objects: ['layer'], hidden: false}, 'Showing…'],
]);
eyeOf('kept').onclick({...event});
assert.equal(actions.length, 2, 'a paint lock refuses');
assert.deepEqual(toasts, [{message: 'Kept: paint is locked; unlock it to show it', error: true}]);
rows.get('inner').ondblclick();
assert.equal(toasts.at(-1).message, 'Inner is hidden with Layer: show it to edit it');

// The canvas: Select all, the browser's hits, snapping and the point tools
// leave hidden objects out; edits that need their place refuse.
const selection = createSelectionController(editor, {
  object, level: () => 'objects', action: async (...args) => { actions.push(args); return true; },
  topSelection: () => editor.state.selection.objects, toast: (message, error) => toasts.push({message, error: !!error}),
  moveLockRefusal: () => '',
});
assert.deepEqual(selection.selectAllTargets(), ['box']);
editor.state.selection.objects = ['inner'];
assert.equal(selection.selectAllRefusal(), '');
editor.state.selection.objects = [];
const hits = createHitsController(editor, {
  document: {elementsFromPoint: () => ['inner', 'box', 'kept'].map(id => ({id}))},
  object, svgElement: () => ({closest: () => null}), targetId: element => element.id,
});
assert.deepEqual(hits.hitStack(0, 0), ['box']);
editor.state.selection.objects = ['box'];
const resizing = createResizingController(editor, {object, parents: () => new Map(objects.map(item => [item.id, item.parent]))});
assert.deepEqual(resizing.snapCandidates().map(item => item.id), []);
editor.state.selection.objects = ['inner'];
assert.deepEqual(resizing.snapCandidates().map(item => item.id), ['box']);
assert.equal(selectionResizeRefusal(editor, {topSelection: () => ['box', 'inner'], object}, 'rotate'), 'Inner is hidden with Layer: show it to rotate it');
assert.deepEqual(pointTargets(['layer', 'kept'], objects).paths, []);
editor.state.selection.objects = ['inner'];
toasts.length = 0;
assert.equal(await selection.nudgeBy(1, 0, 'run'), false);
assert.equal(actions.length, 2, 'a hidden selection is not nudged');
assert.deepEqual(toasts, [{message: 'Inner is hidden with Layer: show it to move it', error: true}]);
editor.state.selection.objects = ['box'];
await selection.nudgeBy(1, 0, 'run');
assert.deepEqual(actions.at(-1), ['move', {dx: 1, dy: 0, nudge: 'run'}, 'Moving selection…']);
