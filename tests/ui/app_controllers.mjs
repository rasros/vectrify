// Test the exported job controller against async replies and real control events.
import assert from 'node:assert/strict';

import {createJobDialog} from '../../src/vectrify/ui/static/app_jobs.js';
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => {resolve = yes; reject = no;});
  return {promise, resolve, reject};
};
const timers = new Map();
let timerId = 0;
globalThis.setTimeout = callback => {timers.set(++timerId, callback); return timerId;};
globalThis.clearTimeout = id => timers.delete(id);
const result = (name, changed = true) => ({changed, previews:{before:`${name}-before`, after:`${name}-after`}, metrics:{name}});
function harness() {
  const controls = new Map(), calls = [], applied = [], toasts = [];
  const element = () => ({hidden:false, disabled:false, textContent:'Run', children:[], handlers:new Map(),
    append(...children) {this.children.push(...children);},
    replaceChildren() {this.children = [];}, removeAttribute() {},
    addEventListener(name, handler) {
      const listeners = this.handlers.get(name) || [];
      listeners.push(handler); this.handlers.set(name, listeners);
    },
    showModal() {this.open = true;},
    close() {this.open = false; for (const handler of this.handlers.get('close') || []) handler();},
  });
  const $ = id => {if (!controls.has(id)) controls.set(id, element()); return controls.get(id);};
  globalThis.document = {getElementById:$, createElement:element};
  const env = {state:{epoch:'drawing', revision:3},
    respond:async command => command === 'start' ? {id:'job', status:'running'}
      : command === 'status' ? {status:'ready', result:result('first'), alternatives:[result('second'), result('unchanged', false)]}
        : {},
  };
  const dialog = createJobDialog('test', {
    start:() => ({action:'improve', settings:{gain:2}}), describe:entry => entry.metrics.name,
    applied:'Applied', choiceLabel:entry => entry.metrics.name,
  }, {
    getState:() => env.state,
    operation:async (command, body) => {calls.push({command, body}); return env.respond(command, body);},
    applyResult:async state => applied.push(state), toast:(...args) => toasts.push(args),
  }).wire();
  const show = id => $(`test-${id}`);
  return {dialog, show, calls, applied, toasts, env};
}

// Open captures epoch/revision; settings remain evaluated when Run is pressed.
{
  const h = harness();
  h.dialog.open('Selected path');
  h.env.state = {epoch:'other', revision:10};
  await h.dialog.run();
  assert.deepEqual(h.calls[0], {command:'start', body:{epoch:'drawing', revision:3, selection:undefined, action:'improve', settings:{gain:2}}});
  assert.equal(h.show('summary').textContent, 'Selected path');
  assert.equal(h.show('previews').hidden, false);
  assert.equal(h.show('before').src, 'first-before');
  assert.equal(h.show('choices').children.length, 2, 'unchanged alternatives are omitted');
  h.show('choices').children[1].children[0].onchange();
  assert.equal(h.show('after').src, 'second-after');
  assert.equal(h.show('metrics').textContent, 'second');
  await h.show('apply').onclick();
  assert.deepEqual(h.calls.at(-1), {command:'apply', body:{job:'job', choice:1}});
  assert.equal(h.applied.length, 1);
  assert.equal(h.show('dialog').open, false);
  assert.equal(h.calls.filter(call => call.command === 'discard').length, 0);
  assert.equal(h.show('apply').disabled, false);
  for (let i = 0; i < 3; i++) {h.dialog.open(); h.show('dialog').close();}
  assert.equal(h.show('dialog').handlers.get('close').length, 1, 'opening does not duplicate listeners');
}

// Polling stays alive after Stop so the last, usable result can be applied.
{
  const h = harness();
  let status = 'running';
  h.env.respond = async command => command === 'start' ? {id:'job', status:'running'}
    : command === 'status' ? {status, steps:9, step:4, message:'Fitting', result:result('stopped')} : {};
  h.dialog.open(); await h.dialog.run();
  assert.equal(h.show('stop').hidden, false);
  assert.equal(h.show('meter').value, 4);
  const timer = [...timers.keys()].at(-1);
  await h.show('stop').onclick();
  assert.deepEqual(h.calls.at(-1), {command:'stop', body:{job:'job'}});
  assert.equal(h.show('stop').hidden, true);
  status = 'ready';
  const poll = timers.get(timer); timers.delete(timer); await poll();
  assert.equal(h.show('metrics').textContent, 'stopped');
  await h.dialog.run();
  assert.deepEqual(h.calls.slice(-3).map(call => call.command), ['discard', 'start', 'status']);
  h.show('dialog').close();
  assert.equal(h.calls.at(-1).command, 'discard');
}

// A job created after its dialog was cancelled is discarded without changing
// controls belonging to the next open. Late errors are equally irrelevant.
for (const reject of [false, true]) {
  const h = harness(), starting = deferred();
  h.env.respond = command => command === 'start' ? starting.promise : {};
  h.dialog.open(); const running = h.dialog.run();
  h.show('dialog').close(); h.dialog.open('New open');
  if (reject) starting.reject(new Error('Old start failed'));
  else starting.resolve({id:'orphan', status:'running'});
  await running;
  assert.equal(h.show('error').hidden, true);
  assert.equal(h.show('settings').disabled, false);
  assert.equal(h.show('summary').textContent, 'New open');
  if (!reject) assert.deepEqual(h.calls.at(-1), {command:'discard', body:{job:'orphan'}});
}

// Status replies, including failures, must not display a previous run's preview.
for (const reject of [false, true]) {
  const h = harness(), status = deferred();
  h.env.respond = command => command === 'start' ? {id:'job', status:'running'}
    : command === 'status' ? status.promise : {};
  h.dialog.open(); const running = h.dialog.run();
  await Promise.resolve(); await Promise.resolve();
  assert.equal(h.calls.at(-1).command, 'status');
  h.show('dialog').close(); h.dialog.open('New open');
  if (reject) status.reject(new Error('Old poll failed'));
  else status.resolve({status:'ready', result:result('obsolete')});
  await running;
  assert.equal(h.show('previews').hidden, true);
  assert.equal(h.show('error').hidden, true);
}

// Closing during discard must not start the replacement run afterwards.
{
  const h = harness(), discarding = deferred();
  h.dialog.open(); await h.dialog.run();
  h.env.respond = command => command === 'discard' ? discarding.promise : {};
  const rerun = h.dialog.run();
  h.show('dialog').close(); h.dialog.open();
  discarding.resolve({}); await rerun;
  assert.equal(h.calls.filter(call => call.command === 'start').length, 1);
}

// Stop and Apply use the job captured at click time. An old completion cannot
// show an error on or close a newer dialog; successful Apply still refreshes art.
for (const command of ['stop', 'apply']) {
  for (const reject of [false, true]) {
    const h = harness(), reply = deferred();
    h.dialog.open(); await h.dialog.run();
    h.env.respond = name => name === command ? reply.promise : {};
    const clicked = h.show(command).onclick();
    h.show('dialog').close(); h.dialog.open('Next dialog');
    if (reject) reply.reject(new Error('Obsolete error'));
    else reply.resolve({epoch:'drawing', revision:4});
    await clicked;
    assert.equal(h.show('dialog').open, true);
    assert.equal(h.show('summary').textContent, 'Next dialog');
    assert.equal(h.show('error').hidden, true);
    assert.equal(h.show('apply').disabled, false);
    assert.equal(h.applied.length, command === 'apply' && !reject ? 1 : 0);
  }
}

// Current failures allow another run and leave the editor drawing unchanged.
{
  const h = harness();
  h.env.respond = async () => {throw new Error('Refused');};
  h.dialog.open(); await h.dialog.run();
  assert.equal(h.show('error').textContent, 'Refused');
  assert.equal(h.show('progress').hidden, true);
  assert.equal(h.show('run').disabled, false);
  assert.equal(h.applied.length, 0);
}
assert.equal(timers.size, 0, 'closed dialogs cancel their poll');
