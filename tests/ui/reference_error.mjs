import {createEditorState} from '../../src/vectrify/ui/static/app_state.js';
import {createReferenceController} from '../../src/vectrify/ui/static/app_reference.js';
import assert from 'node:assert/strict';

import {errorView} from '../../src/vectrify/ui/static/reference_error.js';
const deferred = () => {
  let resolve, reject;
  const result = new Promise((yes, no) => {resolve = yes; reject = no;});
  return {result, resolve, reject};
};
const calls = [], statuses = [], failures = [], painted = [];
let visible;
const swaps = [];
const makeCanvas = () => ({hidden: true,
  cloneNode: () => makeCanvas(),
  getContext() { return {drawImage: pixels => { assert.notEqual(this, visible, 'paint offscreen before replacing the displayed canvas'); this.pixels = pixels; painted.push(pixels); }}; },
  replaceWith(next) { assert.equal(this, visible); assert.ok(next.pixels, 'publish only a complete frame'); swaps.push(next); visible = next; },
});
const canvas = makeCanvas(); visible = canvas;
const controller = errorView(canvas, value => statuses.push(value), error => failures.push(error), (...args) => {
  const call = {...deferred(), args}; calls.push(call); return call.result;
});
const bounds = [40, 20, 320, 240];
const update = (svg = 'first', ref = 'reference') => controller.update(svg, bounds, ref);
const flush = () => new Promise(resolve => setImmediate(resolve));
update(); update();
assert.equal(calls.length, 1, 'selection and opacity changes do not rerasterize');
update('intermediate'); update('edited');
assert.equal(calls.length, 1, 'only one comparison can be in flight');
calls[0].resolve({width: 320, height: 240, name: 'old'}); await flush();
assert.equal(painted.length, 0, 'an older drawing cannot replace the current one');
assert.equal(calls.length, 2);
assert.equal(calls[1].args[0], 'edited', 'skip superseded edits');
calls[1].resolve({width: 320, height: 240, name: 'edited'}); await flush();
assert.equal(painted.at(-1).name, 'edited');
assert.equal(visible.hidden, false);
update('edited', null);
assert.equal(visible.hidden, true);
update('edited');
assert.equal(visible.hidden, false);
assert.equal(calls.length, 2, 'returning to the same view reuses its pixels');
update('edited', 'replacement');
assert.equal(visible.hidden, false, 'retain the old frame while replacing the reference');
update('edited', null);
calls[2].resolve({width: 320, height: 240, name: 'replacement'}); await flush();
assert.equal(visible.hidden, true, 'leaving the view hides late completions');
update('edited', 'replacement'); await flush();
assert.equal(painted.at(-1).name, 'replacement');
assert.equal(calls.length, 3, 'reuse a completed render when returning');
update('broken');
assert.equal(visible.hidden, false, 'retain the old frame until a new one is complete');
calls[3].reject(new Error('failed')); await flush();
assert.equal(statuses.at(-1), 'failed');
assert.equal(failures.length, 1);
assert.equal(visible.hidden, false, 'a failed refresh must not blank the previous frame');
update('broken');
assert.equal(calls.length, 5, 'a failed render can be retried');
update('broken', null);
calls[4].reject(new Error('obsolete')); await flush();
assert.equal(failures.length, 1, 'obsolete failures are ignored');

update('refresh', 'replacement');
assert.equal(visible.hidden, false, 'keep the comparison visible during edits');
update('refresh', null);
update('refresh', 'replacement');
assert.equal(calls.length, 6, 'returning reuses the in-flight request');
calls[5].resolve({width: 320, height: 240, name: 'refresh'}); await flush();
assert.equal(painted.at(-1).name, 'refresh');
assert.equal(calls.length, 6);
controller.update('resized', [0, 0, 160, 120], 'replacement');
assert.equal(visible.hidden, false, 'keep the old frame during an artboard resize');
update('latest', 'replacement');
calls[6].reject(new Error('obsolete size')); await flush();
assert.equal(failures.length, 1);
assert.equal(calls[7].args[0], 'latest', 'an obsolete failure still starts the latest comparison');
calls[7].resolve({width: 320, height: 240, name: 'latest'}); await flush();
assert.equal(painted.at(-1).name, 'latest');
assert.equal(visible.pixels.name, 'latest');
assert.equal(swaps.length, painted.length, 'each painted frame publishes in a single swap');
assert.equal(canvas.width, undefined, 'the original displayed canvas is never resized');

// Exercise the app's actual view controls and wraparound in both directions.

const controls = new Map();
const $ = id => {
  if (!controls.has(id)) controls.set(id, {style: {}, setAttribute() {}, removeAttribute() {}});
  return controls.get(id);
};
const buttons = ['drawing', 'overlay', 'reference', 'error'].map(view => ({dataset: {view}, setAttribute(name, value) {this[name] = value;}}));
const updates = [];
const classes = new Set();
const context = Object.assign(createEditorState(), {$, stage: {classList: {toggle(name, enabled) {if (enabled) classes.add(name); else classes.delete(name);}}}, document: {querySelectorAll: () => buttons},
  reportView() {}, scheduleStrip() {}, toast() {}, errorView: () => ({update: (...args) => updates.push(args)})});
context.reference = {name: 'ref', data_url: 'ref-url', opacity: 0};  context.state = {svg: '<svg/>', bounds: [40,20,320,240]};
const context_reference = createReferenceController(context, {
  $: (...args) => context.$(...args),
  errorView: context.errorView,
  FileReader: context.FileReader,
  action: (...args) => context.action(...args),
  document: context.document,
  fit: (...args) => context.fit(...args),
  loadReference: (...args) => context.loadReference(...args),
  localStorage: context.localStorage,
  renderInspector: (...args) => context.renderInspector(...args),
  reportView: (...args) => context.reportView(...args),
  request: (...args) => context.request(...args),
  scheduleStrip: (...args) => context.scheduleStrip(...args),
  stage: context.stage,
  toast: (...args) => context.toast(...args),
  viewReport: (...args) => context.viewReport(...args),
});
for (const button of buttons) button.onclick = () => context_reference.setView(button.dataset.view);
context.VIEWS = context_reference.VIEWS;
context.errorStatus = context_reference.errorStatus;
context.referenceError = context_reference.referenceError;
context.refreshErrorView = context_reference.refreshErrorView;
context.showReference = context_reference.showReference;
context.setView = context_reference.setView;
context.cycleReference = context_reference.cycleReference;

context.setView('drawing'); context.cycleReference(-1);
assert.equal(context.referenceView, 'error');
assert.equal($('reference-image').hidden, true);
assert.equal($('drawing').style.opacity, '0');
assert.equal(buttons[3]['aria-pressed'], 'true');
assert.deepEqual(updates.at(-1)[2], {data_url: 'ref-url', box: undefined}, 'error ignores overlay opacity');
context.cycleReference(); assert.equal(context.referenceView, 'drawing');
assert.equal($('drawing').style.opacity, '');
assert.equal(updates.at(-1)[2], null);
for (const expected of ['overlay', 'reference', 'error', 'drawing']) {
  context.cycleReference(); assert.equal(context.referenceView, expected);
}
for (const expected of ['error', 'reference', 'overlay', 'drawing']) {
  context.cycleReference(-1); assert.equal(context.referenceView, expected);
}
context.setView('reference');
assert.equal($('drawing').style.opacity, '0', 'hide artwork without hiding its clipping resources');
assert.equal($('drawing').style.visibility, undefined);
assert.equal($('drawing').style.pointerEvents, 'none');
assert.equal($('reference-image').style.opacity, 1);
assert.ok(classes.has('reference-only-visible'), 'reference-only contours use the faint style');
for (const view of ['drawing', 'overlay', 'error']) {
  context.setView(view);
  assert.ok(!classes.has('reference-only-visible'));
  assert.equal($('drawing').style.pointerEvents, '');
}
context.setView('error'); context.reference = null; context.showReference(); context.cycleReference();
assert.equal($('drawing').style.opacity, '');
assert.equal($('reference-toggle').hidden, true);
assert.equal(updates.at(-1)[2], null);
