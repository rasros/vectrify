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
  let fileHandle = null, fileName = null, saving = false;
  const currentName = () => fileName || getState()?.name || 'Untitled.svg';
  const types = [
    {description:'SVG drawing',accept:{'image/svg+xml':['.svg']}},
    {description:'Vectrify project',accept:{'application/gzip':['.vectrify']}},
    {description:'JSON project',accept:{'application/json':['.json']}},
  ];
  const forgetFile = () => {fileHandle = null; fileName = null; $('filename').textContent=currentName();};
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
            forgetFile();
            const version = getState();
            await recoveryStore('readwrite',getSession(),item);markSaved(version);
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
  function exportFile(project, name, version = getState(), exact = false) {
    return request('/api/export',{project,compressed:project&&!/\.json$/i.test(name),epoch:version.epoch,revision:version.revision,...(exact ? {project_version:version.project_version} : {})});
  }
  function fileContent(result) {
    return result.encoding==='base64'?Uint8Array.from(atob(result.content),char=>char.charCodeAt(0)):result.content;
  }
  function browserDownload(result, name, project) {
    const blob=new Blob([fileContent(result)],{type:project?'application/gzip':'image/svg+xml'}),url=URL.createObjectURL(blob);
    const link=document.createElement('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  async function save(saveAs = false) {
    if(saving)return;
    saving=true;
    let busy=false;
    try {
      // Ask while the click/shortcut still has browser user activation.
      const api=getBridge()?await getBridge():null;
      let target, handle=fileHandle;
      if(api){
        target=await api.save_target(currentName(),saveAs);
        if(!target)return;
      }else if(window.showSaveFilePicker){
        if(saveAs||!handle)handle=await window.showSaveFilePicker({suggestedName:currentName(),types});
        if(handle.requestPermission&&await handle.requestPermission({mode:'readwrite'})!=='granted')throw new Error('Permission to save this file was denied');
        target=handle.name;
      }else target=currentName();
      const name=target.split(/[\\/]/).pop(), project=/\.(vectrify|json)$/i.test(name);
      await waitForEdits();setBusy('Saving file…',1);busy=true;
      const result=await exportFile(project,name);
      // Even SVG saves keep a full project for recovery after a restart.
      const recovery=project?result:await exportFile(true,'recovery.vectrify',result,true);
      if(api){
        if(!await api.save(name,result.content,result.encoding||null))return;
      }else if(handle){
        const writable=await handle.createWritable();
        try {await writable.write(fileContent(result));await writable.close();}
        catch(error){await writable.abort().catch(()=>{});throw error;}
      }else browserDownload(result,name,project);
      fileHandle=handle;fileName=name;
      await recoveryStore('readwrite',getSession(),{source:recovery.content,encoding:recovery.encoding,summary:recovery.summary,name,savedAt:Date.now()});
      if(getState().epoch===result.epoch)markSaved(result);
      $('filename').textContent=name;
      toast('File saved.');
    }catch(error){if(error.name!=='AbortError')toast(error.message,true);}
    finally{saving=false;if(busy)setBusy('',-1);}
  }
  async function download(project) {
    if(saving)return;
    saving=true;
    let busy=false;
    try {
      await waitForEdits();setBusy(project?'Saving project…':'Exporting SVG…',1);busy=true;
      const {epoch,revision,name}=getState();
      const result=await request('/api/export',{project,compressed:project,epoch,revision});
      const filename=currentName().replace(/\.(svg|json|vectrify)$/i,'')+(project?'.vectrify':'.svg');
      if(getBridge()){
        const api=await getBridge();
        if(!await api.save_target(filename,true))return;
        if(!await api.save(filename,result.content,result.encoding||null,false))return;
      }else{
        browserDownload(result,filename,project);
      }
      if(project){
        await recoveryStore('readwrite',getSession(),{source:result.content,encoding:result.encoding,summary:result.summary,name,savedAt:Date.now()});
        if(getState().epoch===result.epoch)markSaved(result);
      }
      toast(project?'Project saved, with a browser recovery copy.':'SVG exported.');
    }catch(error){toast(error.message,true);}finally{saving=false;if(busy)setBusy('',-1);}
  }
  $('save-project').onclick=()=>save();$('save-as').onclick=()=>save(true);$('export-svg').onclick=()=>download(false);
  $('open-file').onclick=async()=>{
    if(saving)return;
    if(!getBridge()&&!window.showOpenFilePicker){$('svg-file').click();return;}
    if(isDirty()&&!window.confirm('Open another drawing? Save your file first if you want to keep these edits.'))return;
    try {
      let picked,handle=null;
      if(getBridge())picked=await (await getBridge()).open_file();
      else if(window.showOpenFilePicker){
        [handle]=await window.showOpenFilePicker({types,multiple:false});
        const file=await handle.getFile();
        if(file.size>128*1024*1024)throw new Error('File exceeds the 128 MB editor limit');
        picked={name:file.name,...await fileSource(file)};
      }
      if(!picked)return;
      if(await action('open',picked,'Opening drawing…')){
        fileHandle=handle;fileName=picked.name;$('filename').textContent=picked.name;markSaved();await refreshDrawing();
      }
    }catch(error){if(error.name!=='AbortError')toast(error.message,true);}
  };
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
      if(success){forgetFile();markSaved();await refreshDrawing();}
    }catch(error){toast(error.message,true);}
  };
  async function recover(previous, result) {
    if (!previous || getSession() === previous) return result;
    let restored = result;
    try {
      const saved = await recoveryStore('readonly', previous);
      if (!saved) return result;
      restored = await request('/api/action', {command:'open', ...saved, epoch:result.epoch, revision:result.revision});
      forgetFile();
      await recoveryStore('readwrite', getSession(), saved);
      toast('Restored your saved project after the server restart. Earlier undo history is unavailable.');
    } catch (error) {
      toast(`Browser recovery failed: ${error.message}`, true);
    }
    // The backend may already have restored a new epoch before storing its
    // recovery copy fails. Always show that drawing so the tab stays usable.
    return restored;
  }
  return {save, download, recover, currentName};
}
