import {createEditorState} from '../../src/vectrify/ui/static/app_state.js';
import {createActionsController} from '../../src/vectrify/ui/static/app_actions.js';
import {createGeometryLoader} from '../../src/vectrify/ui/static/app_geometry.js';
// Exercise the real command and geometry-loading code with requests racing MCP edits.
import assert from 'node:assert/strict';

const calls = [], errors = [], loaded = [], polls = [];
let conflict = false, pinnedDelete = false, confirmed = false;
const confirmations = [];
const context = Object.assign(createEditorState(), {
  window: {confirm: message => { confirmations.push(message); return confirmed; }},
  $: () => ({}),
  setBusy: () => {},
  renderDrawing: () => {}, renderInspector: () => {}, drawOverlay: () => {},
  missingGeometries: () => ['a'], schedulePoll: delay => polls.push(delay),
  toast: message => errors.push(message),
  request: async (path, payload) => {
    calls.push({path, payload: JSON.parse(JSON.stringify(payload))});
    if (path === '/api/action' && conflict) throw new Error('This edit conflicts with another edit at fill. Nothing was changed.');
    if (path === '/api/action' && payload.command === 'delete' && pinnedDelete && !payload.confirm_pinned) return {confirmation: 'delete_pinned', message: 'Delete objects with pinned points?'};
    if (path === '/api/nodes') return {epoch: 'drawing', revision: 2, geometries: {a: {id: 'geometry'}}};
    return {epoch: 'drawing', revision: 2, svg: '<svg/>', selection: {objects: ['a'], nodes: []}};
  },
  applyState: async state => { loaded.push(state); },
});
context.queue = Promise.resolve(), context.clickCycle = null, context.dirty = false, context.session = 'window';
   context.state = {epoch: 'drawing', revision: 0, svg: '<svg/>', selection: {objects: ['a'], nodes: []}, undo_ids: ['manual-edit'], redo_ids: ['undone-edit']};
   context.geometries = new Map();
const context_actions = createActionsController(context, {
  $: (...args) => context.$(...args),
  applyState: (...args) => context.applyState(...args),
  drawOverlay: (...args) => context.drawOverlay(...args),
  renderDrawing: (...args) => context.renderDrawing(...args),
  renderInspector: (...args) => context.renderInspector(...args),
  request: (...args) => context.request(...args),
  setBusy: (...args) => context.setBusy(...args),
  toast: (...args) => context.toast(...args),
  window: context.window,
});
context.action = context_actions.action;
const context_geometry = createGeometryLoader(context, {
  missingGeometries: (...args) => context.missingGeometries(...args),
  request: (...args) => context.request(...args),
  schedulePoll: (...args) => context.schedulePoll(...args),
});
context.loadGeometries = context_geometry.loadGeometries;
const action = (context.action);
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
await (context.loadGeometries());
assert.equal((context.geometries.size), 0, 'do not cache newer geometry under an older revision');
assert.deepEqual(polls, [0]);

conflict = false;
pinnedDelete = true;
const cancelledAt = calls.length, loadedAt = loaded.length;
assert.equal(await action('delete'), false);
assert.equal(calls.length, cancelledAt + 1, 'cancel must not issue a deletion');
assert.equal(loaded.length, loadedAt, 'a confirmation response is not drawing state');
assert.equal(errors.length, 1, 'cancelling is not an error');
confirmed = true;
assert.equal(await action('delete'), true);
const attempts = calls.slice(-2);
assert.equal(attempts[0].payload.confirm_pinned, undefined);
assert.deepEqual(attempts[1].payload, {...attempts[0].payload, confirm_pinned: true}, 'retry the same targets and revision that were confirmed');
assert.equal(confirmations.length, 2);
pinnedDelete = false;
assert.equal(await action('delete'), true);
assert.equal(confirmations.length, 2, 'ordinary deletion needs no confirmation');
