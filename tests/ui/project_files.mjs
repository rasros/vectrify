import assert from 'node:assert/strict';
import {gzipSync, gunzipSync} from 'node:zlib';

import {projectFiles} from '../../src/vectrify/ui/static/app_files.js';
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
const context = {
  $: id => {if (!controls.has(id)) controls.set(id, element()); return controls.get(id);},
  state: {name: 'drawing.svg', epoch: 'e', revision: 3, project_version:'3:0'}, session: 's', dirty: true,
  queue: Promise.resolve(), bridge: null, setBusy() {},
  request: async (path, body) => {requests.push({path, ...body}); return {...context.state, ...(body.project ? result : {content: '<svg/>'})};},
  toast: (text, error) => {if (error) errors.push(text);},
  document: {getElementById: id => context.$(id), createElement: () => element()}, window: {confirm: () => true},
  Blob, atob, setTimeout: fn => fn(),
  URL: {createObjectURL: value => {blob = value; return 'blob:download';}, revokeObjectURL() {}},
  action: async (command, body) => {actions.push({command, ...body}); return true;},
  loadReference: async () => {}, fit() {},
  FileReader: class {
    readAsDataURL(file) {this.result = `data:application/octet-stream;base64,${file.bytes.toString('base64')}`; this.onload();}
  },
};
// Exercise the controller's IndexedDB transactions, including the session key.
const savedBySession = new Map();
let closedDatabases = 0;
globalThis.indexedDB = {open(name, version) {
  assert.equal(name, 'vectrify-recovery'); assert.equal(version, 1);
  const opening = {result: {
    close() {closedDatabases++;},
    transaction(storeName, mode) {
      assert.equal(storeName, 'projects');
      const tx = {objectStore() {return {
        getAll: () => ({result:stored}),
        get: key => ({result:savedBySession.get(key)}),
        put(value, key) {
          assert.equal(mode, 'readwrite');
          recovery.push(value); savedBySession.set(key, value);
          return {result:key};
        },
      };}};
      queueMicrotask(() => tx.oncomplete());
      return tx;
    },
  }};
  queueMicrotask(() => opening.onsuccess());
  return opening;
}};
for (const name of ['document', 'window', 'FileReader', 'URL', 'setTimeout']) globalThis[name] = context[name];
const files = projectFiles({
  getState: () => context.state, getSession: () => context.session,
  getBridge: () => context.bridge, waitForEdits: () => context.queue,
  isDirty: () => context.dirty, markSaved: (version = context.state) => {context.dirty = version.epoch !== context.state.epoch || version.revision !== context.state.revision || version.project_version !== context.state.project_version;},
  request: context.request, action: (...args) => context.action(...args), setBusy: context.setBusy,
  toast: context.toast, refreshDrawing: context.loadReference,
});
context.download = files.download;

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
context.bridge = Promise.resolve({save_target: async () => 'saved.vectrify', save: async (...args) => {native.push(args); return 'saved.vectrify';}});
await context.download(true);
assert.deepEqual(native[0], ['drawing.vectrify', content, 'base64', false]);
context.dirty = true;
context.bridge = Promise.resolve({save_target: async () => null});
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

// A queued save reads the drawing left by the edit instead of a copied state.
context.bridge = null;
let releaseEdit;
context.queue = new Promise(resolve => {releaseEdit = resolve;});
const requestCount = requests.length;
const saving = files.download(false);
await Promise.resolve();
assert.equal(requests.length, requestCount);
context.state = {name:'edited.json', epoch:'new', revision:9};
releaseEdit();
await saving;
assert.equal(filename, 'edited.svg');
assert.equal(requests.at(-1).epoch, 'new');
assert.equal(requests.at(-1).revision, 9);

// A restart restores only the previous tab's saved bytes into the new session.
const saved = recovery[0];
savedBySession.set('old-window', saved);
context.session = 'new-window';
const initial = {epoch:'fresh', revision:0};
await files.recover('old-window', initial);
assert.equal(requests.at(-1).path, '/api/action');
assert.equal(requests.at(-1).source, content);
assert.equal(requests.at(-1).encoding, 'base64');
assert.equal(requests.at(-1).epoch, 'fresh');
assert.equal(requests.at(-1).revision, 0);
assert.equal(savedBySession.get('new-window'), saved);
const before = requests.length;
assert.equal(await files.recover('new-window', initial), initial);
assert.equal(await files.recover('missing-window', initial), initial);
assert.equal(requests.length, before);
assert.ok(closedDatabases > 0);

// Cancelled picking and refused opens retain unsaved edits and skip recovery.
context.dirty = true;
context.window.confirm = () => false;
const previousActions = actions.length;
await controls.get('svg-file').onchange({target:{files:[{name:'other.svg', size:1}], value:'other.svg'}});
assert.equal(actions.length, previousActions);
assert.equal(context.dirty, true);
context.window.confirm = () => true;
await controls.get('svg-file').onchange({target:{files:[{name:'large.svg', size:128*1024*1024+1}], value:'large.svg'}});
assert.equal(actions.length, previousActions);
assert.match(errors.at(-1), /128 MB/);

// Save overwrites the selected file; Save As picks a different destination.
errors.length = 0;
context.state = {name:'drawing.svg',epoch:'e',revision:3};
const targets = [], writes = [], pickerCalls = [];
function handle(name) {
  return {name, async createWritable() {return {
    async write(data) {writes.push({name,data});}, async close() {}, async abort() {},
  };}};
}
context.window.showSaveFilePicker = async options => {pickerCalls.push(options); return handle(pickerCalls.length === 1 ? 'drawing.svg' : 'copy.vectrify');};
await controls.get('save-project').onclick();
assert.equal(pickerCalls.length, 1);
assert.equal(writes.at(-1).name, 'drawing.svg');
assert.equal(writes.at(-1).data, '<svg/>');
assert.equal(recovery.at(-1).source, content, 'SVG saves retain full project recovery');
await files.save();
assert.equal(pickerCalls.length, 1, 'ordinary saves reuse the destination');
await controls.get('save-as').onclick();
assert.equal(pickerCalls.length, 2);
assert.equal(writes.at(-1).name, 'copy.vectrify');
assert.equal(gunzipSync(writes.at(-1).data).toString(), original);
assert.equal(files.currentName(), 'copy.vectrify');
await files.save();
assert.equal(pickerCalls.length, 2);
assert.equal(writes.at(-1).name, 'copy.vectrify');
context.dirty = true;
const beforeCancel = writes.length;
context.window.showSaveFilePicker = async () => {throw Object.assign(new Error('cancelled'), {name:'AbortError'});};
await files.save(true);
assert.equal(writes.length, beforeCancel);
assert.equal(context.dirty, true);
assert.equal(files.currentName(), 'copy.vectrify');
assert.deepEqual(errors, []);

// Opening another file replaces the destination only after a successful open.
context.window.showOpenFilePicker = async () => [{...handle('opened.svg'),getFile: async () => ({name:'opened.svg',bytes:Buffer.from('<svg/>'),size:6})}];
await controls.get('open-file').onclick();
assert.equal(controls.get('filename').textContent, 'opened.svg');
await files.save();
assert.equal(writes.at(-1).name, 'opened.svg');
assert.equal(pickerCalls.length, 2);
context.dirty = true;
const originalAction=context.action;
context.action=async()=>false;
await controls.get('open-file').onclick();
assert.equal(files.currentName(), 'opened.svg', 'a refused open keeps the save destination');
assert.equal(context.dirty, true);
context.action=originalAction;
context.window.confirm = () => false;
await controls.get('open-file').onclick();
context.window.confirm = () => true;
await files.save();
assert.equal(writes.at(-1).name, 'opened.svg');

// The desktop chooses the target before export, so its suffix selects the format.
context.bridge = Promise.resolve({
  save_target: async (name, saveAs) => {targets.push({name,saveAs});return saveAs?'copy.vectrify':'opened.svg';},
  save: async (...args) => {native.push(args);return args[0];},
});
await files.save();
assert.deepEqual(native.at(-1), ['opened.svg','<svg/>',null]);
await files.save(true);
assert.deepEqual(native.at(-1), ['copy.vectrify',content,'base64']);
assert.equal(targets.at(-1).saveAs, true);

// Saving waits for pending edits and does not clear edits made during a write.
let releasePendingEdit;
context.queue=new Promise(resolve=>{releasePendingEdit=resolve;});
const beforeQueuedSave=requests.length;
const queuedSave=files.save();
await new Promise(resolve=>queueMicrotask(resolve));
assert.equal(requests.length,beforeQueuedSave);
context.state={name:'opened.svg',epoch:'edited',revision:4};
releasePendingEdit();
await queuedSave;
assert.equal(requests.at(-1).epoch,'edited');
assert.equal(requests.at(-1).revision,4);
context.dirty=true;
context.bridge=Promise.resolve({save_target:async()=> 'opened.svg',save:async()=>{context.state.revision++;return 'opened.svg';}});
await files.save();
assert.equal(context.dirty,true);
