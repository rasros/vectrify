// Exercise the actual Tidy dialog request and change handlers with its HTML defaults.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const app = readFileSync(new URL('../../src/vectrify/ui/static/app.js', import.meta.url), 'utf8');
const html = readFileSync(new URL('../../src/vectrify/ui/static/index.html', import.meta.url), 'utf8');
const inputs = new Map();
for (const [tag, id] of Array.from(html.matchAll(/<input\b[^>]*id="(nodes-[^"]+)"[^>]*>/g), m => [m[0], m[1]])) {
  const value = tag.match(/\bvalue="([^"]*)"/)?.[1] || '';
  const checked = /\bchecked\b/.test(tag);
  const handlers = {};
  inputs.set(id, {id, value, defaultValue:value, checked, defaultChecked:checked,
    disabled:/\bdisabled\b/.test(tag), handlers,
    addEventListener:(event, handler) => { handlers[event] = handler; }});
}
const $ = id => {
  if (!inputs.has(id)) inputs.set(id, {id});
  return inputs.get(id);
};
let options;
const state = {reference:true};
const region = [10, 20, 80, 60];
const context = vm.createContext({$, state, viewReport:() => ({region}),
  document:{querySelectorAll:() => []},
  jobDialog:(_prefix, config) => { options = config; return {wire:() => ({})}; }});
vm.runInContext(app.slice(app.indexOf('const NODE_STEPS ='), app.indexOf('// Whether paths, or groups')), context);
const request = () => JSON.parse(JSON.stringify(options.start()));
const toggle = checked => {
  $('nodes-custom-run').checked = checked;
  $('nodes-custom-run').handlers.change();
};

vm.runInContext('syncNodeSteps()', context);
assert.deepEqual(request().settings, {shape:true, detail:true, simplify:false, snap:true});
assert.deepEqual(request().budget, {});

// Reproduce a custom run that cannot move anything or accept modest gains.
toggle(true);
$('nodes-movement').value = '0';
$('nodes-gain').value = '50';
$('nodes-seconds').value = '0.5';
$('nodes-rounds').value = '1';
$('nodes-steps').value = '1';
$('nodes-shared').checked = false;
assert.equal(request().settings.movement, 0);
assert.equal(request().settings.gain, 50);
assert.equal(request().settings.seconds, 0.5);
assert.equal(request().settings.steps, 1);
assert.equal(request().settings.shared, false);
assert.deepEqual(request().budget, {steps:1});

// Turning overrides off restores the displayed values AND omits every custom
// advanced option from the request, so backend defaults really take effect.
toggle(false);
assert.deepEqual(request().settings, {shape:true, detail:true, simplify:false, snap:true});
assert.deepEqual(request().budget, {});
for (const input of inputs.values()) {
  if (input.defaultValue) {
    assert.equal(input.value, input.defaultValue, input.id);
    assert.equal(input.disabled, true, input.id);
  }
}
assert.equal($('nodes-shared').checked, true);
assert.equal($('nodes-shared').disabled, true);
assert.equal($('nodes-detail').disabled, false);

// Defaults do not discard the person's enabled steps or view restriction.
$('nodes-detail').checked = false;
$('nodes-simplify').checked = true;
$('nodes-in-view').checked = true;
assert.deepEqual(request().settings, {shape:true, detail:false, simplify:true, snap:true, region});
toggle(true);
assert.equal($('nodes-detail-gain').disabled, true);
$('nodes-detail').checked = true;
vm.runInContext('syncNodeSteps()', context);
assert.equal($('nodes-detail-gain').disabled, false);

// The no-reference simplification default is tighter than reference fitting.
state.reference = false;
toggle(false);
assert.equal($('nodes-tolerance').value, '1');
