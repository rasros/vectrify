// Set clip's roles and dialog: which selected object is clipped and which is
// the boundary, said before anything changes. Run by test_clip_dialog.py.
import assert from 'node:assert/strict';
import {boundaryRefusal, defaultRoles, removeClipRefusal, rolesRefusal, setClipRefusal, targetRefusal} from '../../src/vectrify/ui/static/clipping.js';
import {createClippingController} from '../../src/vectrify/ui/static/app_clipping.js';

// Objects in stacking order, back to front, as the session lists them.
const objects = [
  {id: 'shading', tag: 'g', label: 'Shading', parent: 'root', inherited_locks: [], attributes: {}},
  {id: 'shade', tag: 'path', label: 'Shade', parent: 'shading', inherited_locks: [], attributes: {}},
  {id: 'old', tag: 'path', label: 'Old', parent: 'root', inherited_locks: [], attributes: {'clip-path': 'url(#c)'}},
  {id: 'box', tag: 'rect', label: 'Rect 1', parent: 'root', inherited_locks: [], attributes: {}},
  {id: 'stroke', tag: 'path', label: 'Stroke', parent: 'root', open_path: true, inherited_locks: [], attributes: {}},
  {id: 'locked', tag: 'path', label: 'Locked', parent: 'root', inherited_locks: ['structure'], attributes: {}},
  {id: 'body', tag: 'path', label: 'Body', parent: 'root', inherited_locks: [], attributes: {}},
  {id: 'ring', tag: 'path', label: 'Ring', parent: 'root', inherited_locks: [], attributes: {}},
  {id: 'contour', tag: 'path', label: 'Clipping shape', parent: 'c', resource: true, inherited_locks: [], attributes: {}},
];
const object = id => objects.find(item => item.id === id);
const pick = (...ids) => objects.filter(item => ids.includes(item.id));

// Only closed paths are boundaries; basic shapes are pointed to Convert to path.
assert.equal(boundaryRefusal(object('body')), '');
assert.match(boundaryRefusal(object('box')), /^A rectangle must be a path to be a boundary: use Convert to path/);
assert.match(boundaryRefusal(object('stroke')), /open/);
assert.match(boundaryRefusal(object('shading')), /Only closed paths/);
assert.equal(targetRefusal(object('shading')), '');
assert.equal(targetRefusal(object('box')), '');
assert.match(targetRefusal(object('locked')), /structure is locked/);

// Set clip needs something to clip and a closed path; definitions take no part.
assert.match(setClipRefusal(pick('body')), /objects to clip and a closed path/);
assert.equal(setClipRefusal(pick('shading', 'body')), '');
assert.match(setClipRefusal(pick('shading', 'box')), /Convert to path/);
assert.match(setClipRefusal(pick('shading', 'stroke')), /Select a closed path/);
assert.match(setClipRefusal(pick('shading', 'contour')), /not definitions or clipping boundaries/);

// The frontmost closed path starts as the boundary; a group, a shape and an
// open path are clipped.
assert.deepEqual([...defaultRoles(pick('shading', 'body', 'box', 'stroke'))], [['shading', 'target'], ['box', 'target'], ['stroke', 'target'], ['body', 'boundary']]);
assert.deepEqual([...defaultRoles(pick('body', 'ring'))], [['body', 'target'], ['ring', 'boundary']]);

// Chosen roles: one of each at least, each fit for its role, and no clipped
// object inside another clipped one. A boundary inside its target is fine.
const roles = entries => new Map(Object.entries(entries));
assert.equal(rolesRefusal(pick('shading', 'shade'), roles({shading: 'target', shade: 'boundary'}), object), '');
assert.match(rolesRefusal(pick('shading', 'body'), roles({shading: 'boundary', body: 'boundary'}), object), /at least one object to clip/);
assert.match(rolesRefusal(pick('shading', 'body'), roles({shading: 'target', body: 'target'}), object), /at least one boundary/);
assert.match(rolesRefusal(pick('box', 'body'), roles({box: 'boundary', body: 'target'}), object), /^Rect 1: A rectangle must be a path/);
assert.match(rolesRefusal(pick('locked', 'body'), roles({locked: 'target', body: 'boundary'}), object), /^Locked: Its structure is locked/);
assert.equal(rolesRefusal(pick('shading', 'shade', 'body'), roles({shading: 'target', shade: 'target', body: 'boundary'}), object), 'Shade is inside Shading: clip one or the other');

// Remove clip needs a clipped object among the selection.
assert.equal(removeClipRefusal(pick('old', 'body')), '');
assert.equal(removeClipRefusal(pick('body')), 'None of the selected objects is clipped');
assert.match(removeClipRefusal(pick('contour')), /not their clipping boundaries/);

// The dialog: just enough DOM for its rows, radios and buttons.
class Node {
  constructor(tag = '') {
    this.tag = tag; this.children = []; this.dataset = {}; this.attributes = new Map(); this.listeners = new Map();
    this.hidden = false; this.disabled = false; this.checked = false; this.title = ''; this.ownText = ''; this.className = '';
  }
  get textContent() { return this.ownText + this.children.map(child => child.textContent).join(''); }
  set textContent(value) { this.ownText = String(value); this.children = []; }
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children.flatMap(child => child.fragment ? child.children : [child]); }
  addEventListener(name, handler) { this.listeners.set(name, [...(this.listeners.get(name) || []), handler]); }
  fire(name, event = {}) { for (const handler of this.listeners.get(name) || []) handler({preventDefault() {}, ...event}); }
  all(test) { return this.children.flatMap(child => child instanceof Node ? [...(test(child) ? [child] : []), ...child.all(test)] : []); }
  showModal() { this.open = true; } close() { this.open = false; this.fire('close'); }
}
class Text { constructor(text) { this.textContent = text; } }
const ids = new Map();
const $ = id => { if (!ids.has(id)) { const node = new Node('div'); node.id = id; ids.set(id, node); } return ids.get(id); };
const document = {
  createElement: tag => new Node(tag),
  createTextNode: text => new Text(text),
  createDocumentFragment: () => Object.assign(new Node(''), {fragment: true}),
};

const sent = [], toasts = [];
let succeed = true;
const editor = {pending: 0, state: {epoch: 'e', revision: 3, objects, selection: {objects: ['body', 'shading'], nodes: []}}};
const clipping = createClippingController(editor, {
  $, document, object,
  action: async (command, data) => { sent.push({command, data}); if (!succeed) $('toast-message').textContent = 'shade: locked'; return succeed; },
  noSelection: () => !editor.state.selection.objects.length && 'Select objects first',
  paintSwatch() {},
  toast: (message, error = false) => toasts.push({message, error}),
}).wire();

const rows = () => $('clip-roles').all(node => node.className === 'clip-role');
const role = id => rows().find(row => row.dataset.object === id).dataset.role;
const radio = (id, value) => $('clip-roles').all(node => node.tag === 'input' && node.dataset.object === id && node.value === value)[0];
const choose = (id, value) => { const input = radio(id, value); input.checked = true; input.fire('change'); };

// Opening names both objects in stacking order, each with its role, and says
// what will be clipped to what before anything is sent.
clipping.openSetClip();
assert.equal($('clip-dialog').open, true);
assert.deepEqual(rows().map(row => [row.dataset.object, row.dataset.role]), [['shading', 'target'], ['body', 'boundary']]);
assert.equal($('clip-summary').textContent, 'Clip Shading to Body.');
assert.equal($('clip-linked').checked, true);
assert.match($('clip-geometry-detail').textContent, /reshaping the path in Nodes reshapes the clip/);
assert.equal($('clip-confirm').disabled, false);
assert.equal(radio('shading', 'boundary').disabled, true, 'a group cannot be a boundary');
const choice = input => $('clip-roles').all(node => node.className === 'clip-role-choice' && node.children[0] === input)[0];
assert.match(choice(radio('shading', 'boundary')).title, /Only closed paths/, 'and says why');
assert.deepEqual(sent, []);

// Making both clipped leaves nothing to clip to: the dialog says so and
// cannot apply. Copied geometry says what it keeps.
choose('body', 'target');
assert.equal(role('body'), 'target');
assert.equal($('clip-summary').textContent, 'Choose at least one boundary');
assert.equal($('clip-confirm').disabled, true);
choose('body', 'boundary');
$('clip-copied').checked = true; $('clip-linked').checked = false; $('clip-copied').fire('change');
assert.match($('clip-geometry-detail').textContent, /keeps a copy of the outline/);
await clipping.confirmSetClip();
assert.deepEqual(sent.at(-1), {command: 'set_clip', data: {targets: ['shading'], boundaries: ['body'], linked: false}});
assert.equal($('clip-dialog').open, false);
assert.match(toasts.at(-1).message, /^Clipped 1 object to Body\. The clip keeps a copy/);

// Several boundaries clip to their combined area; an object already clipped
// is told its clip is replaced.
editor.state.selection.objects = ['old', 'body', 'ring'];
clipping.openSetClip();
assert.deepEqual(rows().map(row => row.dataset.role), ['target', 'target', 'boundary']);
choose('body', 'boundary');
assert.equal($('clip-summary').textContent, 'Clip Old to the combined area of Body and Ring.');
assert.match(rows()[0].textContent, /replaces its current clip/);
assert.equal($('clip-linked').checked, true, 'each opening starts linked');
await clipping.confirmSetClip();
assert.deepEqual(sent.at(-1).data, {targets: ['old'], boundaries: ['body', 'ring'], linked: true});

// A selection that changed while the dialog was open sends nothing.
editor.state.selection.objects = ['shading', 'body'];
clipping.openSetClip();
editor.state.revision = 4;
const before = sent.length;
await clipping.confirmSetClip();
assert.equal(sent.length, before);
assert.equal($('clip-error').hidden, false);
assert.match($('clip-error').textContent, /selection changed/);
$('clip-dialog').close();

// A refused edit keeps the dialog open with the reason.
succeed = false;
clipping.openSetClip();
await clipping.confirmSetClip();
assert.equal($('clip-dialog').open, true);
assert.equal($('clip-error').textContent, 'shade: locked');
assert.equal($('clip-confirm').disabled, false);
$('clip-dialog').close();
succeed = true;

// Set clip does not open for a selection it cannot clip, and says why.
editor.state.selection.objects = ['shading', 'box'];
clipping.openSetClip();
assert.equal($('clip-dialog').open, false);
assert.deepEqual(toasts.at(-1), {message: 'A rectangle must be a path to be a boundary: use Convert to path, under Path', error: true});

// Remove clip sends the edit and counts what it unclips.
editor.state.selection.objects = ['old', 'body'];
assert.equal(clipping.removeRefusal(), '');
await clipping.removeClip();
assert.deepEqual(sent.at(-1), {command: 'remove_clip', data: {}});
assert.equal(toasts.at(-1).message, 'Removed the clip from 1 object. Boundary paths stay in the drawing.');
