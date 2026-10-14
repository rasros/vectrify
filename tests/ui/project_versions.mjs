import assert from 'node:assert/strict';
import {createEditorState} from '../../src/vectrify/ui/static/app_state.js';
import {createSessionController} from '../../src/vectrify/ui/static/app_session.js';
import {createAgentsController} from '../../src/vectrify/ui/static/app_agents.js';
import {createReferenceDataController} from '../../src/vectrify/ui/static/app_reference_data.js';

const controls = new Map(), requests = [], referenceImages = [];
let reply;
const context = createEditorState();
context.session = 's'; context.agent.flash.seen = 0;
const $ = id => {if (!controls.has(id)) controls.set(id, {style:{}}); return controls.get(id);};
context.request = async (path, body) => {requests.push({path, body}); return reply;};
const referenceController = createReferenceDataController(context, {
  request: (...args) => context.request(...args),
  showReference: () => referenceImages.push(context.reference),
});
const sessionController = createSessionController(context, {
  $, files:{currentName:() => 'drawing.svg'}, overlay:{setAttribute() {}},
  fit() {}, renderDrawing() {}, renderObjects() {}, updateObjectSelection() {},
  level:() => 'objects', renderInspector() {}, drawOverlay() {},
  loadReference: referenceController.loadReference,
});
const {applyState} = sessionController;
const {pulse} = createAgentsController(context, {
  applyState, request:(...args) => context.request(...args),
  showAgent() {}, drawOverlay() {}, toast() {},
});
const initial = {epoch:'e',revision:0,reference_revision:1,project_version:'0:1',objects:[],bounds:[0,0,100,100],undo:[],redo:[],selection:{objects:[],nodes:[]},reference:{name:'same.png',opacity:.5}};
reply = {...initial,reference:{...initial.reference,data_url:'red'}};
await applyState(initial);
assert.equal(context.dirty, false);
assert.equal(referenceImages.at(-1).data_url, 'red');

// Selection-only state refreshes do not change content or fetch the reference.
const requestCount = requests.length;
await applyState({...initial,selection:{objects:['shape'],nodes:[]}});
assert.equal(context.dirty, false);
assert.equal(requests.length, requestCount);

// Poll versions catch same-name pixels even without a different agent counter.
const changed = {...initial,reference_revision:2,project_version:'0:2'};
context.request = async path => {
  requests.push({path});
  return path === '/api/session' ? changed : {...changed,reference:{...initial.reference,data_url:'blue'}};
};
await pulse({...changed,agent:{changes:0}});
assert.equal(context.dirty, true);
assert.equal(referenceImages.at(-1).data_url, 'blue');
sessionController.markSaved();
assert.equal(context.dirty, false);

// A save completes before the next agent pulse: only its serialized version
// becomes the clean marker, so the later unseen replacement is still dirty.
sessionController.markSaved({epoch:'e',project_version:'0:2'});
context.request = async path => path === '/api/session'
  ? {...changed,reference_revision:3,project_version:'0:3'}
  : {...changed,reference_revision:3,reference:{...initial.reference,data_url:'green'}};
await pulse({...changed,reference_revision:3,project_version:'0:3',agent:{changes:0}});
assert.equal(context.dirty, true);
assert.equal(referenceImages.at(-1).data_url, 'green');

// A live export may already include changes newer than the page's last state.
sessionController.markSaved({epoch:'e',project_version:'0:4'});
assert.equal(context.dirty, true);
context.request = async() => ({epoch:'e',reference_revision:4,reference:null});
await applyState({...changed,reference_revision:4,project_version:'0:4',reference:null});
assert.equal(context.dirty, false);
assert.equal(referenceImages.at(-1), null);

// Late reference responses belonging to earlier epochs or versions are ignored.
const images = referenceImages.length;
context.request = async() => ({epoch:'old',reference_revision:9,reference:{data_url:'stale'}});
await referenceController.loadReference();
context.request = async() => ({epoch:'e',reference_revision:1,reference:{data_url:'stale'}});
await referenceController.loadReference();
assert.equal(referenceImages.length, images);
