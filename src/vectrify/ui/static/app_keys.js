import {NUDGE_STEP, NUDGE_STEP_LARGE} from './app_config.js';

// Keys owns its controls; editor contains the live state shared with other controllers.
export function createKeysController(editor, {
  $,
  action,
  cycleReference,
  document = globalThis.document,
  drawOverlay,
  endTreeDrag,
  files,
  finishPath,
  fit,
  focusSelection,
  gestures,
  input,
  later,
  level,
  nudge,
  oneObject,
  openPalette,
  regionBack,
  regionEnter,
  renameObject,
  renderDrawing,
  renderInspector,
  runCommand,
  selectedPoints,
  setTool,
  stage,
  stepUp,
  window = globalThis.window,
} = {}) {

  // Arrow keys nudge the selection; with Shift, ten times as far.
  const NUDGES = {arrowleft: [-1, 0], arrowright: [1, 0], arrowup: [0, -1], arrowdown: [0, 1]};

  // Where arrow keys keep their own meaning: the object tree, menus, the
  // palette's list and choices. Text fields and dialogs keep all keys.
  const NAVIGATES = '[role="tree"], [role="menu"], [role="listbox"], [role="radiogroup"], [role="dialog"], [role="slider"]';

  // Tool shortcuts; Space held pans.
  const TOOL_KEYS = {a: 'select', v: 'move', s: 'nodes', d: 'path', m: 'rect', e: 'ellipse', l: 'line', p: 'pencil', i: 'eyedropper', c: 'knife', x: 'region', g: 'gradient', r: 'redraw'};

  // A tool strip's numbered controls, by digit.
  function stripKeys(name) {
    const keys = {};
    for (const button of document.querySelectorAll('[data-key][data-strip]')) {
      // A strip shared by tools numbers only the controls of the tool in use.
      const only = button.closest('.strip-item[data-tools]')?.dataset.tools;
      if (button.dataset.strip.split(' ').includes(name) && (!only || only.split(' ').includes(name))) keys[button.dataset.key] = button;
    }
    return keys;
  }

  // Cancel the gesture under way: a drag (put back as it was), a path or a
  // region being drawn or a held gesture. Whether there was one.
  function cancelGesture() {
    const busy = !!(editor.drag || editor.pathDraft.length || editor.region || (editor.holding && !editor.holding.released));
    if (editor.holding && !editor.holding.released) { editor.holding.cancel(); editor.holding = null; }
    if (editor.drag) {
      gestures[editor.drag.kind].cancel?.(editor.drag);
      editor.drag = null; if (editor.state) renderDrawing();
    }
    editor.pathDraft=[];editor.pathHover=null;editor.region=null;editor.regionHover=null;stage.classList.remove('panning');
    if (editor.state) renderInspector();
    drawOverlay();
    return busy;
  }

  // Modifiers change a shape being dragged out (Shift and Alt), a moving
  // selection or gradient end (Shift, Alt and Ctrl/⌘) as they are pressed and released,
  // without waiting for the pointer to move.
  const GESTURE_MODIFIERS = {shape: ['Shift', 'Alt'], move: ['Shift', 'Alt', 'Control', 'Meta'], gradient: ['Shift', 'Alt', 'Control', 'Meta']};
  function gestureModifiers(event) {
    if (!GESTURE_MODIFIERS[editor.drag?.kind]?.includes(event.key)) return false;
    event.preventDefault(); gestures[editor.drag.kind].modifiers(editor.drag, event);
    return true;
  }

  function keyDown(event) {
    if (gestureModifiers(event)) return;
    const typing=['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)||document.activeElement.isContentEditable, dialog=document.querySelector('dialog[open]');
    const mod=event.ctrlKey||event.metaKey, key=event.key.toLowerCase();
    if(mod&&!event.altKey&&key==='s'&&!dialog){event.preventDefault();if(!event.repeat)files.save(event.shiftKey);return;}
    if(event.key==='F2'&&!typing&&!dialog&&oneObject()){event.preventDefault();later(()=>{if(oneObject())renameObject();});return;}
    if(event.key==='Escape'&&editor.tree.drag){event.preventDefault();endTreeDrag(false);return;}
    if(mod&&!event.altKey&&key==='k'){event.preventDefault();if(!dialog)openPalette();return;}
    // Ctrl/⌘ Alt N: browsers keep Ctrl/⌘ N and Shift N for windows. The key's
    // code, as Option N types a character on a Mac; not in a field, where
    // Ctrl Alt is AltGr and may type one.
    if(mod&&event.altKey&&!event.shiftKey&&event.code==='KeyN'&&!typing&&!dialog){event.preventDefault();if(!event.repeat)runCommand('new-drawing');return;}
    if(typing||dialog)return;
    if(event.code==='Space'){event.preventDefault();editor.space=true;return;}
    if(event.key==='Escape'){
      // Escape cancels what is under way at once, else steps the selection up
      // a level once any edit is done.
      const busy = cancelGesture();
      if (!busy && editor.state && !event.repeat) later(stepUp);
      return;
    }
    // Held keys repeat; while an edit runs the repeats would pile up. Held
    // arrows add up into the nudge waiting instead.
    if(event.repeat&&(editor.pending||input.waiting)&&!NUDGES[key]){event.preventDefault();return;}
    if (editor.tool==='path' && editor.pathDraft.length) {
      if (event.key==='Enter') {event.preventDefault();later(()=>finishPath(false));return;}
      if (event.key==='Backspace' || event.key==='Delete' || (mod&&key==='z')) {
        event.preventDefault();if(!editor.drag){editor.pathDraft.pop();editor.pathHover=null;drawOverlay();}return;
      }
    }
    // In Region, Enter closes the polygon being drawn or applies the region;
    // Backspace takes back the polygon's last corner.
    if (editor.tool==='region' && editor.region && !mod) {
      if (event.key==='Enter') {event.preventDefault();later(regionEnter);return;}
      if ((event.key==='Backspace' || event.key==='Delete') && !editor.region.closed) {event.preventDefault();if(!editor.drag)regionBack();return;}
    }
    // Ctrl/⌘ A selects everything at the editing level, or every point of
    // the selected paths in a point tool; text fields keep their own.
    if(mod&&!event.altKey&&!event.shiftKey&&key==='a'){event.preventDefault();if(!event.repeat)later(()=>runCommand('select-all'));return;}
    // With nothing to undo or redo the key does nothing; the stack is read
    // when the queued key runs, so an edit still in flight counts.
    if(mod&&key==='z'){event.preventDefault();const redo=event.shiftKey;later(()=>(redo?editor.state.redo:editor.state.undo).length?action(redo?'redo':'undo',{},redo?'Redoing…':'Undoing…'):undefined);return;}
    if(mod&&!event.altKey&&!event.shiftKey&&['c','v'].includes(key)){
      event.preventDefault();later(()=>runCommand(key==='c'?'copy':'paste'));return;
    }
    // Ctrl/⌘ G groups (with Shift, ungroups) and J joins what the selection
    // allows: two points, shapes to merge or line ends.
    if(mod&&!event.altKey&&['g','j'].includes(key)){
      event.preventDefault();
      const command=key==='j'?'join':event.shiftKey?'ungroup':'group';
      later(()=>runCommand(command));return;
    }
    if(mod&&['BracketLeft','BracketRight'].includes(event.code)){
      // Ctrl/⌘ ] and [ step forward and backward; with Shift, to the front and back.
      event.preventDefault();
      const command=event.shiftKey?(event.code==='BracketRight'?'to-front':'to-back'):(event.code==='BracketRight'?'forward':'backward');
      later(()=>{if(editor.state?.selection.objects.length)runCommand(command);});return;
    }
    if(mod||event.altKey)return;
    if(NUDGES[key]){
      // Not while a gesture is under way, nor with nothing selected, when
      // the key keeps its default.
      if(document.activeElement?.closest?.(NAVIGATES)||editor.drag||editor.pathDraft.length||editor.region||!editor.state?.selection.objects.length)return;
      event.preventDefault();
      const step=event.shiftKey?NUDGE_STEP_LARGE:NUDGE_STEP;
      nudge(NUDGES[key][0]*step,NUDGES[key][1]*step);return;
    }
    if(event.key==='?'){event.preventDefault();$('help-dialog').showModal();return;}
    // Tools and the view are on the left hand. A tool key switches at once,
    // even while an edit runs, and cancels a gesture under way.
    if(TOOL_KEYS[key]&&!event.shiftKey){event.preventDefault();if(!event.repeat){cancelGesture();setTool(TOOL_KEYS[key]);}return;}
    // Shift H and Shift V flip the selection; V alone is the Move tool.
    if(event.shiftKey&&(key==='h'||key==='v')){event.preventDefault();if(!event.repeat)later(()=>runCommand(key==='h'?'flip-horizontal':'flip-vertical'));return;}
    if(key==='w'){event.preventDefault();if(!event.repeat)cycleReference(event.shiftKey?-1:1);return;}
    if(key==='f'&&!event.shiftKey){fit();return;}
    if(key==='z'&&!event.shiftKey){event.preventDefault();if(editor.state?.selection.objects.length)focusSelection();return;}
    // 1-9 press the tool strip's controls, as numbered on them.
    if(/^[1-9]$/.test(event.key)){
      // A command's control runs it (saying why when it cannot); another
      // control is pressed, as a click would.
      const button=stripKeys(editor.tool)[event.key];
      if(button){event.preventDefault();const command=button.dataset.command;later(()=>command?runCommand(command):button.disabled||button.click());}
      return;
    }
    if(event.key==='Delete'||event.key==='Backspace'){
      event.preventDefault();
      const contours=event.shiftKey;
      later(()=>{
        if(!editor.state?.selection.objects.length)return;
        if (!contours) runCommand('delete-selection');
        else if (level() === 'points' && selectedPoints().length) runCommand('delete-contour');
      });
    }
  }

  function wire() {
    // Each numbered strip control remembers its strip, since the overflow menu
    // may move it out of the strip.
    for (const controls of document.querySelectorAll('.tool-controls[data-tools]'))
      for (const button of controls.querySelectorAll('[data-key]')) {
        button.dataset.strip = controls.dataset.tools;
        button.title = `${button.title || button.getAttribute('aria-label') || ''} (${button.dataset.key})`.trim();
      }

    // Keys that edit wait for any edit under way, and run in order once it is
    // done; keys that only change the view act at once.
    window.addEventListener('keydown', keyDown);
    window.addEventListener('keyup',event=>{if(event.code==='Space')editor.space=false;gestureModifiers(event);});

    window.addEventListener('blur',()=>editor.space=false);

    window.addEventListener('beforeunload',event=>{if(editor.dirty||editor.pathDraft.length){event.preventDefault();event.returnValue='';}});
      return api;
  }
  const api = {TOOL_KEYS, NUDGES, stripKeys, cancelGesture, gestureModifiers, keyDown, wire};
  return api;
}
