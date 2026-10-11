// Exercise the actual Tidy dialog request and change handlers with its HTML defaults.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';

const app = readFileSync(new URL('../../src/vectrify/ui/static/app_jobs.js', import.meta.url), 'utf8');
const {operationDialogs} = await import(`data:text/javascript;base64,${Buffer.from(app).toString('base64')}`);
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
  if (!inputs.has(id)) inputs.set(id, {id, textContent:'Run', handlers:{},
    addEventListener(event, handler) {this.handlers[event] = handler;},
    showModal() {}, removeAttribute() {}});
  return inputs.get(id);
};
let state = {reference:true, epoch:'e', revision:0, selection:{objects:['path']}};
const region = [10, 20, 80, 60];
const calls = [];
globalThis.document = {getElementById:$, querySelectorAll:() => []};
await operationDialogs({getState:() => state, getObject:() => ({tag:'path'}),
  waitForEdits:() => Promise.resolve(), selectionBounds:() => [0, 0, 100, 100],
  viewReport:() => ({region}), applyResult:async () => {}, toast() {},
  operation:async (command, body) => {
    calls.push({command, body});
    return command === 'start' ? {id:'job', status:'running'} : {status:'cancelled'};
  },
}).openTidy();
const request = async () => {
  await $('nodes-run').onclick();
  return [...calls].reverse().find(call => call.command === 'start').body;
};
const toggle = checked => {
  $('nodes-custom-run').checked = checked;
  $('nodes-custom-run').handlers.change();
};

$('nodes-shape').handlers.change();
assert.deepEqual((await request()).settings, {shape:true, detail:true, simplify:false, snap:true});
assert.deepEqual((await request()).budget, {});

// Reproduce a custom run that cannot move anything or accept modest gains.
toggle(true);
$('nodes-movement').value = '0';
$('nodes-gain').value = '50';
$('nodes-seconds').value = '0.5';
$('nodes-rounds').value = '1';
$('nodes-steps').value = '1';
$('nodes-shared').checked = false;
assert.equal((await request()).settings.movement, 0);
assert.equal((await request()).settings.gain, 50);
assert.equal((await request()).settings.seconds, 0.5);
assert.equal((await request()).settings.steps, 1);
assert.equal((await request()).settings.shared, false);
assert.deepEqual((await request()).budget, {steps:1});

// Turning overrides off restores the displayed values AND omits every custom
// advanced option from the request, so backend defaults really take effect.
toggle(false);
assert.deepEqual((await request()).settings, {shape:true, detail:true, simplify:false, snap:true});
assert.deepEqual((await request()).budget, {});
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
assert.deepEqual((await request()).settings, {shape:true, detail:false, simplify:true, snap:true, region});
toggle(true);
assert.equal($('nodes-detail-gain').disabled, true);
$('nodes-detail').checked = true;
$('nodes-shape').handlers.change();
assert.equal($('nodes-detail-gain').disabled, false);

// The no-reference simplification default is tighter than reference fitting.
state.reference = false;
toggle(false);
assert.equal($('nodes-tolerance').value, '1');

// The accessors must read a replacement editor state, not a startup snapshot.
state = {reference:true, epoch:'replacement', revision:12, selection:{objects:['path']}};
const live = await request();
assert.equal(live.epoch, 'e', 'an open dialog keeps the epoch it captured');
assert.equal(live.revision, 0);
assert.equal($('nodes-detail').disabled, true);
$('nodes-shape').handlers.change();
assert.equal($('nodes-detail').disabled, false, 'settings read the new reference');
