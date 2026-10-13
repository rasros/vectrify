import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const source = readFileSync(new URL('../../src/vectrify/ui/static/app.js', import.meta.url), 'utf8');
const applySource = source.slice(source.indexOf('async function applyState('), source.indexOf('const level ='));
const pulseSource = source.slice(source.indexOf('let pulsing ='), source.indexOf('function schedulePoll('));
const referenceSource = source.slice(source.indexOf('async function loadReference('), source.indexOf("$('add-reference').onclick"));
const markerSource = source.slice(source.indexOf('  markSaved:'), source.indexOf('  request, action, setBusy, toast,', source.indexOf('  markSaved:')));
const controls = new Map(), requests = [], referenceImages = [];
let reply;
const context = vm.createContext({
  $: id => {if (!controls.has(id)) controls.set(id, {style:{}}); return controls.get(id);},
  files: {currentName: () => 'drawing.svg'}, overlay: {setAttribute() {}},
  fit() {}, renderDrawing() {}, renderObjects() {}, updateObjectSelection() {},
  level: () => 'objects', renderInspector() {}, drawOverlay() {},
  object: () => null, showAgent() {}, flashTouched() {}, toast() {}, schedulePoll() {},
  showReference: () => referenceImages.push(vm.runInContext('reference', context)),
  request: async (path, body) => {requests.push({path, body}); return reply;},
});
vm.runInContext(`
  let state, savedVersion = null, dirty = false, reference = null;
  let activeHandle, focusPoint, pointMemory, scope, objectsById, geometries;
  let clickCycle, lastPick;
  const collapsedGroups = new Set(), clipConsumers = new Map(), pathHoles = new Map();
  let session = 's', agentStatus, agentSeen = 0, drag = null, pending = 0;
  const flash = {seen:0}; const AGENT_RETRY_MS = 1;
  let queue = Promise.resolve();
  ${applySource}
  ${pulseSource}
  ${referenceSource}
  const markSaved = ({${markerSource}}).markSaved;
`, context);
const applyState = vm.runInContext('applyState', context), pulse = vm.runInContext('pulse', context);
const initial = {epoch:'e',revision:0,reference_revision:1,project_version:'0:1',objects:[],bounds:[0,0,100,100],undo:[],redo:[],selection:{objects:[],nodes:[]},reference:{name:'same.png',opacity:.5}};
reply = {...initial,reference:{...initial.reference,data_url:'red'}};
await applyState(initial);
assert.equal(vm.runInContext('dirty', context), false);
assert.equal(referenceImages.at(-1).data_url, 'red');

// Selection-only state refreshes do not change content or fetch the reference.
const requestCount = requests.length;
await applyState({...initial,selection:{objects:['shape'],nodes:[]}});
assert.equal(vm.runInContext('dirty', context), false);
assert.equal(requests.length, requestCount);

// Poll versions catch same-name pixels even without a different agent counter.
const changed = {...initial,reference_revision:2,project_version:'0:2'};
context.request = async path => {
  requests.push({path});
  return path === '/api/session' ? changed : {...changed,reference:{...initial.reference,data_url:'blue'}};
};
await pulse({...changed,agent:{changes:0}});
assert.equal(vm.runInContext('dirty', context), true);
assert.equal(referenceImages.at(-1).data_url, 'blue');
vm.runInContext('markSaved()', context);
assert.equal(vm.runInContext('dirty', context), false);

// A save completes before the next agent pulse: only its serialized version
// becomes the clean marker, so the later unseen replacement is still dirty.
vm.runInContext("markSaved({epoch:'e',project_version:'0:2'})", context);
context.request = async path => path === '/api/session'
  ? {...changed,reference_revision:3,project_version:'0:3'}
  : {...changed,reference_revision:3,reference:{...initial.reference,data_url:'green'}};
await pulse({...changed,reference_revision:3,project_version:'0:3',agent:{changes:0}});
assert.equal(vm.runInContext('dirty', context), true);
assert.equal(referenceImages.at(-1).data_url, 'green');

// A live export may already include changes newer than the page's last state.
vm.runInContext("markSaved({epoch:'e',project_version:'0:4'})", context);
assert.equal(vm.runInContext('dirty', context), true);
context.request = async() => ({epoch:'e',reference_revision:4,reference:null});
await applyState({...changed,reference_revision:4,project_version:'0:4',reference:null});
assert.equal(vm.runInContext('dirty', context), false);
assert.equal(referenceImages.at(-1), null);

// Late reference responses belonging to earlier epochs or versions are ignored.
const images = referenceImages.length;
context.request = async() => ({epoch:'old',reference_revision:9,reference:{data_url:'stale'}});
await vm.runInContext('loadReference()', context);
context.request = async() => ({epoch:'e',reference_revision:1,reference:{data_url:'stale'}});
await vm.runInContext('loadReference()', context);
assert.equal(referenceImages.length, images);
