// Exercise the real command and geometry-loading code with requests racing MCP edits.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const source = readFileSync(new URL('../../src/vectrify/ui/static/app.js', import.meta.url), 'utf8');
const actionSource = source.slice(source.indexOf('function action('), source.indexOf('let svgElements ='));
const geometrySource = source.slice(source.indexOf('async function loadGeometries('), source.indexOf('function paintReference('));
const calls = [], errors = [], loaded = [], polls = [];
let conflict = false;
const context = vm.createContext({
  $: () => ({}),
  setBusy: () => {},
  renderDrawing: () => {}, renderInspector: () => {}, drawOverlay: () => {},
  missingGeometries: () => ['a'], schedulePoll: delay => polls.push(delay),
  toast: message => errors.push(message),
  request: async (path, payload) => {
    calls.push({path, payload: JSON.parse(JSON.stringify(payload))});
    if (path === '/api/action' && conflict) throw new Error('This edit conflicts with another edit at fill. Nothing was changed.');
    if (path === '/api/nodes') return {epoch: 'drawing', revision: 2, geometries: {a: {id: 'geometry'}}};
    return {epoch: 'drawing', revision: 2, svg: '<svg/>', selection: {objects: ['a'], nodes: []}};
  },
  applyState: async state => { loaded.push(state); },
});
vm.runInContext(`
  let queue = Promise.resolve(), clickCycle = null, dirty = false, session = 'window';
  let state = {epoch: 'drawing', revision: 0, svg: '<svg/>', selection: {objects: ['a'], nodes: []}, undo_ids: ['manual-edit'], redo_ids: ['undone-edit']};
  const geometries = new Map();
  ${actionSource}
  ${geometrySource}
`, context);
const action = vm.runInContext('action', context);
assert.equal(await action('paint', {changes: {fill: 'green'}}), true);
assert.deepEqual(calls[0].payload.selection, {objects: ['a'], nodes: []});
assert.equal(calls[0].payload.revision, 0);
assert.equal(loaded[0].revision, 2);
assert.equal(await action('undo'), true);
assert.deepEqual(calls.at(-1).payload.ids, ['manual-edit']);
assert.equal(await action('redo'), true);
assert.deepEqual(calls.at(-1).payload.ids, ['undone-edit']);
conflict = true;
const count = calls.length;
assert.equal(await action('paint', {changes: {fill: 'black'}}), false);
assert.equal(calls.length, count + 2, 'a refused edit must not be replayed');
assert.equal(calls.at(-1).path, '/api/session', 'automatically load the live state after a conflict');
assert.equal(loaded.at(-1).revision, 2);
assert.equal(errors.length, 1);
assert.ok(!errors[0].toLowerCase().includes('refresh'));
await vm.runInContext('loadGeometries()', context);
assert.equal(vm.runInContext('geometries.size', context), 0, 'do not cache newer geometry under an older revision');
assert.deepEqual(polls, [0]);
