// New drawing's and the Artboard's dialogs, and how the canvas keeps the
// artwork and the reference in place when the artboard changes. Run by
// test_document_dialogs.py.
import assert from 'node:assert/strict';
import {artboardRefusal, createDocumentController, MAX_SIDE} from '../../src/vectrify/ui/static/app_document.js';
import {createEditorState} from '../../src/vectrify/ui/static/app_state.js';
import {createSessionController} from '../../src/vectrify/ui/static/app_session.js';
import {createReferenceController} from '../../src/vectrify/ui/static/app_reference.js';

// An artboard needs a positive, bounded size; its corner may be anywhere.
assert.equal(artboardRefusal([0, 0, 640, 480]), '');
assert.equal(artboardRefusal([-50, 12.5, 1, MAX_SIDE]), '');
assert.equal(artboardRefusal([0, 0, NaN, 480]), 'Enter a number in every field');
assert.equal(artboardRefusal([0, 0, 0, 480]), 'The width and height must be more than 0');
assert.match(artboardRefusal([0, 0, 640, MAX_SIDE + 1]), /at most 100,000/);

// Just enough DOM for fields, buttons and dialogs.
class Node {
  constructor() { this.value = ''; this.textContent = ''; this.hidden = false; this.disabled = false; this.open = false; this.style = {}; this.listeners = new Map(); }
  addEventListener(name, handler) { this.listeners.set(name, [...(this.listeners.get(name) || []), handler]); }
  showModal() { this.open = true; } close() { this.open = false; }
  setAttribute() {} removeAttribute() {}
}
const nodes = new Map();
const $ = id => { if (!nodes.has(id)) nodes.set(id, new Node()); return nodes.get(id); };

const editor = createEditorState();
editor.state = {epoch: 'e', revision: 2, bounds: [0, 0, 1200, 800.25], selection: {objects: [], nodes: []}};
const actions = [], created = [], toasts = [];
let succeed = true;
const controller = createDocumentController(editor, {
  $,
  action: async (command, data) => {
    actions.push({command, ...data});
    if (!succeed) { $('toast-message').textContent = 'Nothing is drawn to fit the artboard to'; return false; }
    editor.state = {...editor.state, bounds: data.box || [10, 10, 50, 60]};
    return true;
  },
  files: {newDrawing: async (width, height) => { created.push([width, height]); return succeed; }},
  toast: message => toasts.push(message),
}).wire();

// New drawing starts from the current artboard's size.
$('new-drawing').onclick();
assert.equal($('new-dialog').open, true);
assert.deepEqual([$('new-width').value, $('new-height').value], ['1200', '800.25']);
$('new-width').value = '0';
await $('new-confirm').onclick();
assert.equal($('new-error').textContent, 'The width and height must be more than 0');
assert.equal($('new-error').hidden, false);
assert.deepEqual(created, []);
$('new-width').value = '640'; $('new-height').value = '480';
// Declining to replace unsaved edits keeps the dialog open.
succeed = false;
await $('new-confirm').onclick();
assert.deepEqual(created, [[640, 480]]);
assert.equal($('new-dialog').open, true);
succeed = true;
await $('new-confirm').onclick();
assert.equal($('new-dialog').open, false);
assert.equal($('new-confirm').disabled, false);
// Reopened, it shows no stale error.
$('new-drawing').onclick();
assert.equal($('new-error').hidden, true);
$('new-cancel').onclick();
assert.equal($('new-dialog').open, false);

// The footer's size opens the Artboard dialog with the artboard's corner and size.
$('dimensions').onclick();
assert.equal($('artboard-dialog').open, true);
assert.deepEqual(['x', 'y', 'width', 'height'].map(name => $(`artboard-${name}`).value), ['0', '0', '1200', '800.25']);
$('artboard-width').value = '-1';
await $('artboard-confirm').onclick();
assert.equal(actions.length, 0);
assert.match($('artboard-error').textContent, /more than 0/);
$('artboard-x').value = '-20'; $('artboard-width').value = '1600';
await $('artboard-confirm').onclick();
assert.deepEqual(actions.at(-1), {command: 'artboard', box: [-20, 0, 1600, 800.25]});
assert.equal($('artboard-dialog').open, false);
assert.match(toasts.at(-1), /^Artboard 1600 × 800.25\. The artwork and the reference stay where they are\.$/);

// Fit to artwork asks the session for the artwork's bounds; a refusal stays
// in the dialog, saying why.
$('dimensions').onclick();
succeed = false;
await $('artboard-fit').onclick();
assert.deepEqual(actions.at(-1), {command: 'artboard', fit: 'artwork'});
assert.equal($('artboard-dialog').open, true);
assert.equal($('artboard-error').textContent, 'Nothing is drawn to fit the artboard to');
assert.equal($('artboard-fit').disabled, false);
succeed = true;
await controller.fitArtboard();
assert.equal($('artboard-dialog').open, false);
assert.match(toasts.at(-1), /^Artboard 50 × 60\./);

// Enter in a field applies the dialog.
$('dimensions').onclick();
const keydown = $('artboard-dialog').listeners.get('keydown')[0];
let prevented = false;
keydown({key: 'Enter', target: {tagName: 'INPUT'}, preventDefault() { prevented = true; }});
await editor.queue;
assert.equal(prevented, true);
assert.equal(actions.at(-1).command, 'artboard');

// The reference lies over its own box: after the artboard moves right by
// half, it shows from the artboard's left edge, half off the page and clipped.
const canvas = createEditorState();
canvas.state = {bounds: [50, 0, 100, 100]};
canvas.reference = {name: 'r.png', data_url: 'data:image/png;base64,', opacity: .5, box: [0, 0, 100, 100]};
const reference = createReferenceController(canvas, {$, errorView: () => ({update() {}})});
reference.placeReference();
const image = $('reference-image');
assert.deepEqual([image.style.left, image.style.top, image.style.width, image.style.height], ['-50%', '0%', '100%', '100%']);
assert.equal(image.style.clipPath, 'inset(0% 0% 0% 50%)');
// Over the whole artboard it fills it as before.
canvas.state.bounds = [0, 0, 100, 100];
reference.placeReference();
assert.deepEqual([image.style.left, image.style.width, image.style.clipPath], ['0%', '100%', '']);
canvas.reference = null;
reference.placeReference();
assert.equal(image.style.left, '');

// A changed artboard corner pans the canvas so the artwork stays put on
// screen, and places the reference again; a replaced drawing does not.
const session = createEditorState();
session.zoom = 2; session.pan = {x: 10, y: 20};
let views = 0, placed = 0;
const {applyState} = createSessionController(session, {
  $, files: {currentName: () => 'drawing.svg'}, overlay: {setAttribute() {}},
  fit() {}, renderDrawing() {}, renderObjects() {}, updateObjectSelection() {},
  level: () => 'objects', renderInspector() {}, drawOverlay() {}, loadReference: async () => {},
  placeReference: () => placed++, updateView: () => views++,
});
const state = {epoch: 'e', revision: 0, reference_revision: 0, project_version: '0:0', objects: [], bounds: [0, 0, 100, 100], undo: [], redo: [], selection: {objects: [], nodes: []}, reference: null};
await applyState(state);
assert.deepEqual([views, placed], [0, 0]);
await applyState({...state, revision: 1, project_version: '1:0', bounds: [-5, 10, 300, 100]});
assert.deepEqual(session.pan, {x: 0, y: 40});
assert.deepEqual([views, placed], [1, 1]);
assert.equal($('dimensions').textContent, '300 × 100');
// Only the size changed: no pan, but the reference is placed again.
await applyState({...state, revision: 2, project_version: '2:0', bounds: [-5, 10, 500, 100]});
assert.deepEqual(session.pan, {x: 0, y: 40});
assert.deepEqual([views, placed], [1, 2]);
await applyState({...state, epoch: 'other', bounds: [0, 0, 640, 480]});
assert.deepEqual([views, placed], [1, 2]);
