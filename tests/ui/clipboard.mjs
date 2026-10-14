import {createKeysController} from '../../src/vectrify/ui/static/app_keys.js';
import {createEditorState} from '../../src/vectrify/ui/static/app_state.js';

import assert from 'node:assert/strict';

const document = {activeElement: {tagName: 'BODY'}, querySelector: () => null};
const queued = [], commands = [];
const saves = [];
let listener;
const context = Object.assign(createEditorState(), {
  document, window: {addEventListener: (_, fn) => { listener = fn; }},
  pending: 0, input: {waiting: 0}, tool: 'select',
  later: fn => queued.push(fn), runCommand: id => commands.push(id),
  files: {save: saveAs => saves.push(!!saveAs)},
});
listener = createKeysController(context, {document, window:context.window, later:context.later, runCommand:context.runCommand, files:context.files, input:context.input}).keyDown;
function press(key, extra = {}) {
  const event = {key, code: `Key${key.toUpperCase()}`, ctrlKey: true, preventDefault() { this.prevented = true; }, ...extra};
  listener(event);
  return event;
}
assert.ok(press('c').prevented);
assert.ok(press('v', {ctrlKey: false, metaKey: true}).prevented);
assert.deepEqual(commands, [], 'object clipboard commands must wait their turn');
queued.splice(0).forEach(fn => fn());
assert.deepEqual(commands, ['copy', 'paste']);
for (const tagName of ['INPUT', 'TEXTAREA', 'SELECT']) {
  document.activeElement = {tagName};
  assert.ok(!press('c').prevented);
  assert.ok(!press('v').prevented);
}
document.activeElement = {tagName: 'DIV', isContentEditable: true};
assert.ok(!press('c').prevented);
assert.ok(!press('v').prevented);
document.activeElement = {tagName: 'BODY'};
document.querySelector = () => ({});
assert.ok(!press('v').prevented);
document.querySelector = () => null;
assert.ok(!press('c', {altKey: true}).prevented);
assert.ok(!press('v', {shiftKey: true}).prevented);
context.pending = 1;
assert.ok(press('v', {repeat: true}).prevented);
assert.equal(queued.length, 0, 'held shortcuts must not pile up during an edit');
assert.ok(press('s').prevented);
assert.ok(press('S', {shiftKey:true}).prevented);
assert.ok(press('s', {ctrlKey:false,metaKey:true}).prevented);
assert.ok(press('S', {ctrlKey:false,metaKey:true,shiftKey:true}).prevented);
assert.deepEqual(saves, [false,true,false,true]);
document.activeElement = {tagName:'INPUT'};
assert.ok(press('s').prevented, 'Save works while editing a field');
assert.ok(press('s', {repeat:true}).prevented);
assert.equal(saves.length, 5, 'held Save must not open repeated dialogs');

// Ctrl/⌘ Z with nothing to undo or redo does nothing, rather than asking the
// backend for an empty set of history entries; the stack is read when the
// queued key runs, so an edit that lands meanwhile can still be undone.
const sent = [];
const history = Object.assign(createEditorState(), {pending: 0, tool: 'select', state: {undo: [], redo: []}});
const undoKeys = createKeysController(history, {document: {activeElement: {tagName: 'BODY'}, querySelector: () => null}, window: {addEventListener() {}}, later: fn => queued.push(fn), action: command => sent.push(command), input: {waiting: 0}}).keyDown;
const undoKey = (extra = {}) => { const event = {key: 'z', code: 'KeyZ', ctrlKey: true, preventDefault() { this.prevented = true; }, ...extra}; undoKeys(event); return event; };
queued.length = 0;
assert.ok(undoKey().prevented, 'the browser must not undo in the page');
assert.ok(undoKey({shiftKey: true, key: 'Z'}).prevented);
queued.splice(0).forEach(fn => fn());
assert.deepEqual(sent, [], 'an empty history sends nothing');
undoKey();
history.state = {undo: ['Change paint'], redo: []};
queued.splice(0).forEach(fn => fn());
assert.deepEqual(sent, ['undo'], 'an edit that lands before the key runs is undone');
