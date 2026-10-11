const $ = id => document.getElementById(id);

// Keep a local copy of saved projects so a backend restart can restore this tab.
export function recoveryStore(mode, key, value) {
  return new Promise((resolve, reject) => {
    const opening = indexedDB.open('vectrify-recovery', 1);
    opening.onupgradeneeded = () => opening.result.createObjectStore('projects');
    opening.onerror = () => reject(opening.error);
    opening.onsuccess = () => {
      const db = opening.result, tx = db.transaction('projects', mode), store = tx.objectStore('projects');
      const request = mode === 'readonly' ? (key === undefined ? store.getAll() : store.get(key)) : store.put(value, key);
      tx.oncomplete = () => {resolve(request.result);db.close();};
      tx.onerror = () => {reject(tx.error);db.close();};
      tx.onabort = () => {reject(tx.error);db.close();};
    };
  });
}

// Saving waits for the editor queue, then reads the live drawing. Recovery and
// picker handlers share exactly the same bytes as exports and native saves.
export function projectFiles({getState, getSession, getBridge, waitForEdits,
  isDirty, markSaved, request, action, setBusy, toast, refreshDrawing}) {
  $('restore-saved').onclick=async()=>{
    try {
      const saved=await recoveryStore('readonly');
      $('recovery-list').replaceChildren();
      const unique=[...new Map(saved.map(item=>[item.source,item])).values()];
      unique.sort((a,b)=>(b.savedAt||0)-(a.savedAt||0));
      for(const item of unique){
        let summary=item.summary;
        if(!summary){
          const data=JSON.parse(item.source), doc=data.document||data;
          let objects=0; const walk=e=>{objects++;(e.children||[]).forEach(walk);};walk(doc.root);
          const nodes=(doc.geometries||[]).reduce((total,g)=>total+g.subpaths.reduce((n,s)=>n+s.nodes.length,0),0);
          summary={objects:objects-1,nodes};
        }
        const button=document.createElement('button');button.className='wide-button';
        button.textContent=`${item.name} · ${summary.objects} objects · ${summary.nodes.toLocaleString()} points${item.savedAt?' · '+new Date(item.savedAt).toLocaleString():''}`;
        button.onclick=async()=>{
          if(isDirty()&&!window.confirm('Replace this tab’s drawing with the saved copy?'))return;
          if(await action('open',{name:item.name,source:item.source,encoding:item.encoding},'Restoring saved project…')){
            await recoveryStore('readwrite',getSession(),item);markSaved();
            $('recovery-dialog').close();await refreshDrawing();
          }
        };
        $('recovery-list').append(button);
      }
      if(!unique.length)$('recovery-list').textContent='No saved browser recovery copies yet.';
      $('recovery-dialog').showModal();
    }catch(error){toast(error.message,true);}
  };
  $('recovery-close').onclick=()=>$('recovery-dialog').close();
  async function download(project) {
    await waitForEdits(); setBusy(project?'Saving project…':'Exporting SVG…',1);
    try {
      const result=await request('/api/export',{project,compressed:project,epoch:getState().epoch,revision:getState().revision});
      const filename=getState().name.replace(/\.(svg|json|vectrify)$/i,'')+(project?'.vectrify':'.svg');
      if(getBridge()){
        // A native save dialog; cancelling it saves nothing.
        if(!await (await getBridge()).save(filename,result.content,result.encoding||null))return;
      }else{
        const content=result.encoding==='base64'?Uint8Array.from(atob(result.content),char=>char.charCodeAt(0)):result.content;
        const blob=new Blob([content],{type:project?'application/gzip':'image/svg+xml'}),url=URL.createObjectURL(blob);
        const link=document.createElement('a');link.href=url;link.download=filename;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
      }
      if(project){
        await recoveryStore('readwrite',getSession(),{source:result.content,encoding:result.encoding,summary:result.summary,name:getState().name,savedAt:Date.now()});
        markSaved();
      }
      toast(project?'Project saved, with a browser recovery copy.':'SVG exported.');
    }catch(error){toast(error.message,true);}finally{setBusy('',-1);}
  }
  $('save-project').onclick=()=>download(true);$('export-svg').onclick=()=>download(false);
  $('open-file').onclick=()=>$('svg-file').click();
  function fileSource(file) {
    return new Promise((resolve,reject)=>{
      const reader=new FileReader();
      reader.onload=()=>resolve({source:reader.result.split(',')[1],encoding:'base64'});
      reader.onerror=()=>reject(reader.error);
      reader.readAsDataURL(file);
    });
  }
  $('svg-file').onchange=async event=>{
    const file=event.target.files[0];event.target.value='';if(!file)return;
    if(isDirty()&&!window.confirm('Open another drawing? Save your project first if you want to keep these edits.'))return;
    try {
      if(file.size>128*1024*1024)throw new Error('File exceeds the 128 MB editor limit');
      const success=await action('open',{name:file.name,...await fileSource(file)},'Opening drawing…');
      if(success){markSaved();await refreshDrawing();}
    }catch(error){toast(error.message,true);}
  };
  async function recover(previous, result) {
    if (!previous || getSession() === previous) return result;
    const saved = await recoveryStore('readonly', previous);
    if (!saved) return result;
    const restored = await request('/api/action', {command:'open', ...saved, epoch:result.epoch, revision:result.revision});
    await recoveryStore('readwrite', getSession(), saved);
    toast('Restored your saved project after the server restart. Earlier undo history is unavailable.');
    return restored;
  }
  return {download, recover};
}
