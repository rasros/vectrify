import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const source = readFileSync(new URL('../../src/vectrify/ui/static/app.js', import.meta.url), 'utf8');
const keys = source.slice(source.indexOf("window.addEventListener('keydown',event=>{"), source.indexOf("window.addEventListener('keyup',event=>"));
const document = {activeElement: {tagName: 'BODY'}, querySelector: () => null};
const queued = [], commands = [];
let listener;
const context = vm.createContext({
  document, window: {addEventListener: (_, fn) => { listener = fn; }},
  pending: 0, input: {waiting: 0}, tool: 'select', treeDrag: null,
  later: fn => queued.push(fn), runCommand: id => commands.push(id),
});
vm.runInContext(keys, context);
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
