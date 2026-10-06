import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {gzipSync, gunzipSync} from 'node:zlib';

const source = readFileSync(new URL('../../src/vectrify/ui/static/app.js', import.meta.url), 'utf8');
const fileControls = source.slice(source.indexOf("$('restore-saved').onclick"), source.indexOf('// The Reference panel'));
const original = JSON.stringify({vectrify_editor: 1, document: {
  root: {children: [{children: []}]}, geometries: [{subpaths: [{nodes: [1, 2, 3]}]}],
}, reference: {name: 'räven.png', data_url: 'data:image/png;base64,abcd', opacity: 0.35}});
const bytes = gzipSync(original), content = bytes.toString('base64');
const result = {content, encoding: 'base64', summary: {objects: 1, nodes: 3}};
const controls = new Map(), requests = [], actions = [], recovery = [], errors = [];
let blob, filename, stored = [];
function element() {
  return {textContent: '', children: [], replaceChildren() {this.children = [];},
    append(child) {this.children.push(child);}, showModal() {}, close() {},
    click() {filename = this.download;}};
}
const context = vm.createContext({
  $: id => {if (!controls.has(id)) controls.set(id, element()); return controls.get(id);},
  state: {name: 'drawing.svg', epoch: 'e', revision: 3}, session: 's', dirty: true,
  queue: Promise.resolve(), bridge: null, setBusy() {},
  request: async (path, body) => {requests.push({path, ...body}); return body.project ? result : {content: '<svg/>'};},
  recoveryStore: async (mode, key, value) => {if (mode === 'readonly') return stored; recovery.push(value);},
  toast: (text, error) => {if (error) errors.push(text);},
  document: {createElement: () => element()}, window: {confirm: () => true},
  Blob, atob, setTimeout: fn => fn(),
  URL: {createObjectURL: value => {blob = value; return 'blob:download';}, revokeObjectURL() {}},
  action: async (command, body) => {actions.push({command, ...body}); return true;},
  loadReference: async () => {}, fit() {},
  FileReader: class {
    readAsDataURL(file) {this.result = `data:application/octet-stream;base64,${file.bytes.toString('base64')}`; this.onload();}
  },
});
vm.runInContext(fileControls, context);

await context.download(true);
assert.equal(filename, 'drawing.vectrify');
assert.equal(blob.type, 'application/gzip');
assert.equal(gunzipSync(Buffer.from(await blob.arrayBuffer())).toString(), original);
assert.equal(requests.at(-1).compressed, true);
assert.equal(recovery.at(-1).source, content);
assert.equal(recovery.at(-1).encoding, 'base64');
assert.equal(context.dirty, false);

await context.download(false);
assert.equal(filename, 'drawing.svg');
assert.equal(await blob.text(), '<svg/>');
assert.equal(blob.type, 'image/svg+xml');
assert.equal(recovery.length, 1, 'SVG export must not replace project recovery');

const native = [];
context.bridge = Promise.resolve({save: async (...args) => {native.push(args); return 'saved.vectrify';}});
await context.download(true);
assert.deepEqual(native[0], ['drawing.vectrify', content, 'base64']);
context.dirty = true;
context.bridge = Promise.resolve({save: async () => null});
await context.download(true);
assert.equal(context.dirty, true, 'cancelled native saves must keep unsaved edits');
assert.equal(recovery.length, 2, 'cancelled saves must not overwrite recovery');

// The picker carries binary gzip, legacy JSON and SVG bytes unchanged.
for (const [name, data] of [['saved.vectrify', bytes], ['old.vectrify', Buffer.from(original)], ['drawing.svg', Buffer.from('<svg/>')]]) {
  const input = {files: [{name, bytes: data, size: data.length}], value: name};
  await controls.get('svg-file').onchange({target: input});
  assert.equal(actions.at(-1).encoding, 'base64');
  assert.deepEqual(Buffer.from(actions.at(-1).source, 'base64'), data);
  assert.equal(input.value, '');
}

// Recovery lists can display and restore both old and compressed entries.
stored = [{source: original, name: 'old.vectrify'}, {...recovery[0], name: 'saved.vectrify'}];
await controls.get('restore-saved').onclick();
const buttons = controls.get('recovery-list').children;
assert.equal(buttons.length, 2);
for (const button of buttons) {
  assert.match(button.textContent, /1 objects · 3 points/);
  await button.onclick();
  const entry = stored.find(item => item.name === actions.at(-1).name);
  assert.equal(actions.at(-1).source, entry.source);
  assert.equal(actions.at(-1).encoding, entry.encoding);
}
assert.deepEqual(errors, []);
