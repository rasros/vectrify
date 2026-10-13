import assert from 'node:assert/strict';
import {createEditorState} from '../../src/vectrify/ui/static/app_state.js';
import {createTransportController} from '../../src/vectrify/ui/static/app_transport.js';

const document = {body:{setAttribute() {}}};
const editor = createEditorState(); editor.session = 'first';
const calls = [];
const browser = createTransportController(editor, {
  document, location:{search:''},
  fetch:async (path, options) => {calls.push({path,options}); return {ok:true,json:async () => ({result:'browser'})};},
});
assert.deepEqual(await browser.request('/api/session', {value:2}), {result:'browser'});
assert.equal(calls[0].options.headers['X-Vectrify-Session'], 'first');
assert.deepEqual(JSON.parse(calls[0].options.body), {value:2});
editor.session = 'second'; await browser.request('/api/session');
assert.equal(calls[1].options.headers['X-Vectrify-Session'], 'second');
let ready;
const window = {addEventListener: (name, handler) => {assert.equal(name,'pywebviewready'); ready = handler;}};
const desktop = createTransportController(editor, {document, window, location:{search:'?desktop'}});
const waiting = desktop.request('/api/action', {command:'select'});
window.pywebview = {api:{request:async (...args) => {calls.push(args); return {status:200,body:{result:'desktop'}};}}};
editor.session = 'third'; ready();
assert.deepEqual(await waiting, {result:'desktop'});
assert.deepEqual(calls.at(-1), ['/api/action',{command:'select'},'third']);
window.pywebview.api.request = async () => ({status:409,body:{error:'Conflict'}});
await assert.rejects(desktop.request('/api/action'), /Conflict/);
const failed = createTransportController(editor, {
  document, location:{search:''}, fetch:async () => ({ok:false,status:400,json:async () => ({error:'Invalid edit'})}),
});
await assert.rejects(failed.request('/api/action'), /Invalid edit/);
let ran = false;
await browser.later(() => {ran = true;});
assert.equal(ran, true);
