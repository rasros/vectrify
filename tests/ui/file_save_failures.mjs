import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';

const source = readFileSync(new URL('../../src/vectrify/ui/static/app_files.js', import.meta.url), 'utf8');
const {projectFiles} = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

function setup({readError = false, writeError = false, editDuringSave = null, queueError = false, restoreError = false} = {}) {
  const controls = new Map(), requests = [], errors = [], writes = [], recovery = [];
  const saved = {name:'saved.vectrify',source:'saved project'};
  const context = {state:{name:'drawing.svg',epoch:'e',revision:3,project_version:'3:0'},dirty:true,busy:0,queueError};
  globalThis.document = {
    getElementById(id) {
      if (!controls.has(id)) controls.set(id, {});
      return controls.get(id);
    },
  };
  globalThis.window = {};
  globalThis.indexedDB = {open() {
    const opening = {error:new Error('Recovery storage unavailable'),result:{
      close() {},
      transaction(_store,mode) {
        const tx = {error:new Error('Recovery storage full'),objectStore:() => ({
          get:() => ({result:saved}),
          put(value) {recovery.push(value);return {};},
        })};
        queueMicrotask(() => writeError && mode === 'readwrite' ? tx.onabort() : tx.oncomplete());
        return tx;
      },
    }};
    queueMicrotask(() => readError ? opening.onerror() : opening.onsuccess());
    return opening;
  }};
  const files = projectFiles({
    getState:() => context.state,getSession:() => 'new-session',
    getBridge:() => Promise.resolve({
      save_target:async() => 'drawing.vectrify',
      save:async(...args) => {
        writes.push(args);
        if (editDuringSave) context.state = editDuringSave(context.state);
        return 'drawing.vectrify';
      },
    }),
    waitForEdits:async() => {if (context.queueError) throw new Error('Pending edit failed');},
    isDirty:() => context.dirty,markSaved:(version = context.state) => {context.dirty=version.epoch!==context.state.epoch||version.revision!==context.state.revision||version.project_version!==context.state.project_version;},
    setBusy:(_label,delta) => {context.busy+=delta;},
    toast:(message,error) => {if (error) errors.push(message);},
    request:async(path,body) => {
      requests.push({path,...body});
      if (path === '/api/action' && restoreError) throw new Error('Invalid saved project');
      return path === '/api/action' ? {epoch:'restored',revision:0} : {...context.state,content:'exported project'};
    },
  });
  return {files,context,requests,errors,writes,recovery,saved};
}

for (const [change,editDuringSave] of [
  ['an edit',state => ({...state,revision:state.revision+1})],
  ['a reference replacement',state => ({...state,project_version:'3:1'})],
  ['a replacement drawing',state => ({...state,epoch:'other',name:'other.svg'})],
]) {
  test(`project download retains unsaved status after ${change} during its write`, async() => {
    const {files,context,requests,recovery} = setup({editDuringSave});
    await files.download(true);
    assert.equal(requests[0].epoch,'e');
    assert.equal(requests[0].revision,3);
    assert.equal(recovery[0].source,'exported project');
    assert.equal(recovery[0].name,'drawing.svg','recovery metadata belongs to the exported drawing');
    assert.equal(context.dirty,true,'the downloaded version does not contain the later change');
    assert.equal(context.busy,0);
  });
}

test('a failed edit wait releases the save controls and allows retry', async() => {
  const {files,context,errors,writes} = setup({queueError:true});
  await assert.doesNotReject(files.download(true));
  assert.deepEqual(errors,['Pending edit failed']);
  assert.equal(context.busy,0);
  assert.equal(context.dirty,true);
  context.queueError=false;
  await files.download(true);
  assert.equal(writes.length,1,'retry must still write the file');
  assert.equal(context.dirty,false);
  assert.equal(context.busy,0);
});

test('unavailable recovery storage still lets the new session open', async() => {
  const {files,requests,errors} = setup({readError:true});
  const initial = {epoch:'fresh',revision:0};
  assert.equal(await files.recover('old-session',initial),initial);
  assert.equal(requests.length,0);
  assert.match(errors[0],/Recovery storage unavailable/);
});

test('failure to copy recovery never hides a successfully restored drawing', async() => {
  const {files,requests,errors,saved} = setup({writeError:true});
  assert.deepEqual(await files.recover('old-session',{epoch:'fresh',revision:0}),{epoch:'restored',revision:0});
  assert.equal(requests[0].source,saved.source);
  assert.match(errors[0],/Recovery storage full/);
});

test('an invalid recovery project leaves the initial session usable', async() => {
  const {files,errors,recovery} = setup({restoreError:true});
  const initial = {epoch:'fresh',revision:0};
  assert.equal(await files.recover('old-session',initial),initial);
  assert.equal(recovery.length,0,'a refused restore must not be recorded as successful');
  assert.match(errors[0],/Invalid saved project/);
});

for (const method of ['save', 'download']) {
  test(`${method} retains a reference replacement made during its write`, async() => {
    const {files,context,recovery} = setup({editDuringSave:state => ({...state,project_version:'3:1'})});
    await files[method](method === 'download');
    assert.equal(context.dirty,true);
    assert.equal(recovery[0].source,'exported project');
  });
}
