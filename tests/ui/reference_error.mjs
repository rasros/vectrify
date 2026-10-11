import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const source = readFileSync(new URL('../../src/vectrify/ui/static/reference_error.js', import.meta.url), 'utf8');
const {errorView} = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const deferred = () => {
  let resolve, reject;
  const result = new Promise((yes, no) => {resolve = yes; reject = no;});
  return {result, resolve, reject};
};
const calls = [], statuses = [], failures = [], painted = [];
const canvas = {hidden: true, getContext: () => ({drawImage: pixels => painted.push(pixels)})};
const controller = errorView(canvas, value => statuses.push(value), error => failures.push(error), (...args) => {
  const call = {...deferred(), args}; calls.push(call); return call.result;
});
const bounds = [40, 20, 320, 240];
const update = (svg = 'first', ref = 'reference') => controller.update(svg, bounds, ref);
const flush = async () => {await Promise.resolve(); await Promise.resolve();};
update(); update();
assert.equal(calls.length, 1, 'selection and opacity changes do not rerasterize');
update('edited');
calls[0].resolve({width: 320, height: 240, name: 'old'}); await flush();
assert.equal(painted.length, 0, 'an older drawing cannot replace the current one');
calls[1].resolve({width: 320, height: 240, name: 'edited'}); await flush();
assert.equal(painted.at(-1).name, 'edited');
assert.equal(canvas.hidden, false);
update('edited', null);
assert.equal(canvas.hidden, true);
update('edited');
assert.equal(canvas.hidden, false);
assert.equal(calls.length, 2, 'returning to the same view reuses its pixels');
update('edited', 'replacement');
update('edited', null);
calls[2].resolve({width: 320, height: 240, name: 'replacement'}); await flush();
assert.equal(canvas.hidden, true, 'leaving the view hides late completions');
update('edited', 'replacement'); await flush();
assert.equal(painted.at(-1).name, 'replacement');
assert.equal(calls.length, 3, 'reuse an in-flight render when returning');
update('broken');
calls[3].reject(new Error('failed')); await flush();
assert.equal(statuses.at(-1), 'failed');
assert.equal(failures.length, 1);
update('broken');
assert.equal(calls.length, 5, 'a failed render can be retried');
update('broken', null);
calls[4].reject(new Error('obsolete')); await flush();
assert.equal(failures.length, 1, 'obsolete failures are ignored');

// Exercise the app's actual view controls and wraparound in both directions.
const app = readFileSync(new URL('../../src/vectrify/ui/static/app.js', import.meta.url), 'utf8');
const controls = new Map();
const $ = id => {
  if (!controls.has(id)) controls.set(id, {style: {}, setAttribute() {}, removeAttribute() {}});
  return controls.get(id);
};
const buttons = ['drawing', 'overlay', 'reference', 'error'].map(view => ({dataset: {view}, setAttribute(name, value) {this[name] = value;}}));
const updates = [];
const context = vm.createContext({$, stage: {classList: {toggle() {}}}, document: {querySelectorAll: () => buttons},
  reportView() {}, scheduleStrip() {}, toast() {}, errorView: () => ({update: (...args) => updates.push(args)})});
vm.runInContext(`let reference = {name: 'ref', data_url: 'ref-url', opacity: 0}; let state = {svg: '<svg/>', bounds: [40,20,320,240]};
  ${app.slice(app.indexOf('const VIEWS ='), app.indexOf('// The panel folds'))}`, context);
const run = code => vm.runInContext(code, context);
run("setView('drawing'); cycleReference(-1)");
assert.equal(run('referenceView'), 'error');
assert.equal($('reference-image').hidden, true);
assert.equal($('drawing').style.opacity, '0');
assert.equal(buttons[3]['aria-pressed'], 'true');
assert.equal(updates.at(-1)[2], 'ref-url', 'error ignores overlay opacity');
run('cycleReference()'); assert.equal(run('referenceView'), 'drawing');
assert.equal($('drawing').style.visibility, '');
assert.equal(updates.at(-1)[2], null);
for (const expected of ['overlay', 'reference', 'error', 'drawing']) {
  run('cycleReference()'); assert.equal(run('referenceView'), expected);
}
for (const expected of ['error', 'reference', 'overlay', 'drawing']) {
  run('cycleReference(-1)'); assert.equal(run('referenceView'), expected);
}
run("setView('error'); reference = null; showReference(); cycleReference()");
assert.equal($('drawing').style.visibility, '');
assert.equal($('reference-toggle').hidden, true);
assert.equal(updates.at(-1)[2], null);
