import {matchCommands, moveHighlight} from './palette.js';
import {clickPointPath, pointKey, splitKey} from './selection.js';
import {segmentAmong} from './lines.js';

// Commands owns its controls; editor contains the live state shared with other controllers.
export function createCommandsController(editor, {
  $,
  COMMANDS,
  action,
  clickTargets,
  colorHex,
  commandById,
  commandName,
  commandRefusal,
  commandTitle,
  document = globalThis.document,
  drawOverlay,
  hitStack,
  later,
  level,
  object,
  paintSwatch,
  pointContours,
  pointPairs,
  pointPaths,
  renderInspector,
  resolvedPaint,
  selectObject,
  selectPoints,
  selectedHandle,
  selectedPoints,
  stage,
  toast,
  window = globalThis.window,
} = {}) {
  let paletteIndex = 0;

  let paletteRows = [];

  let joinContext = null;

  // The reasons shared by the commands of the registry: each says what to
  // select for the command to run.
  const noSelection = () => !editor.state?.selection.objects.length && 'Select objects first';

  const noPoints = () => !selectedPoints().length && (level() === 'points' ? 'Select points first' : 'Select points in Nodes (S) first');

  const visiblePaths = () => editor.state.selection.objects.every(id => object(id)?.tag === 'path' && !object(id)?.resource);

  const noReference = () => !editor.state.reference && 'Load a reference image first, under Reference below the objects';

  // Lines are paths that paint no fill; fills are paths that do.
  const filledPath = id => resolvedPaint(id, 'fill', 'black') !== 'none';

  const strokedPath = id => resolvedPaint(id, 'stroke', 'none') !== 'none';

  const linePaths = () => joinCandidates().filter(item => !filledPath(item.id) && strokedPath(item.id));

  const fillPaths = () => joinCandidates().filter(item => filledPath(item.id));

  const twoEnds = () => selectedPoints().length === 2;

  // Run a command now; event handlers queue it with later() instead, so it runs
  // once any edit under way is done.
  function runCommand(id) {
    const command = commandById.get(id), reason = command && commandRefusal(command);
    if (!command) return;
    if (reason) { toast(reason, true); return; }
    return command.run();
  }

  function pointHandles(count) { return action('node_handles', {points: pointPairs(), count}, 'Changing handles…'); }

  function renameObject() { $('object-name').focus(); $('object-name').select(); }

  // The inspector's command lists: a container naming commands in
  // data-commands gets a button for each, labelled as the registry names it.
  function buildCommandLists() {
    for (const list of document.querySelectorAll('[data-commands]')) {
      const buttons = list.dataset.commands.split(' ').filter(id => commandById.has(id)).map(id => {
        const button = document.createElement('button');
        button.dataset.command = id; button.dataset.label = 'command';
        if (commandById.get(id).danger) button.className = 'danger';
        button.append(document.createTextNode(commandById.get(id).name));
        return button;
      });
      list.replaceChildren(...buttons);
    }
  }

  // Buttons naming a command in data-command take its state and run it. One
  // that cannot run now stays in place, dimmed, with the reason as its
  // tooltip; pressing it shows the reason.
  function renderCommands() {
    for (const button of document.querySelectorAll('[data-command]')) {
      const command = commandById.get(button.dataset.command); if (!command) continue;
      button.dataset.title ??= button.title || '';
      const reason = commandRefusal(command);
      button.setAttribute('aria-disabled', String(!!reason));
      button.title = reason ? commandTitle(command) : button.dataset.title || commandTitle(command);
      if (button.dataset.label === 'command') button.firstChild.textContent = commandName(command);
    }
  }

  // The context menu offers the commands for what is selected: object
  // commands for objects, Nodes for points. Those that cannot run now are
  // dimmed with the reason, as everywhere else.
  const MENU_GROUPS = {objects: ['Select', 'Object', 'Arrange', 'Path', 'Repair'], points: ['Nodes']};

  function openContextMenu(x, y) {
    const menu = $('context-menu'), groups = MENU_GROUPS[level() === 'points' ? 'points' : 'objects'];
    menu.replaceChildren();
    for (const group of groups) {
      const shown = COMMANDS.filter(command => command.group === group && !command.hidden?.() && command.id !== 'step-up');
      if (!shown.length) continue;
      if (menu.children.length) menu.append(Object.assign(document.createElement('div'), {className: 'menu-divider', role: 'separator'}));
      menu.append(Object.assign(document.createElement('div'), {className: 'menu-heading', textContent: group, role: 'presentation'}));
      for (const command of shown) {
        const item = document.createElement('button');
        item.className = 'menu-item'; item.setAttribute('role', 'menuitem'); item.dataset.command = command.id;
        const reason = commandRefusal(command);
        item.setAttribute('aria-disabled', String(!!reason)); item.title = commandTitle(command);
        const name = document.createElement('span'); name.textContent = commandName(command);
        const keys = document.createElement('kbd'); keys.textContent = command.keys || '';
        item.append(name, keys);
        // One that cannot run says why, as the palette and the inspector do.
        item.onclick = () => { closeContextMenu(); later(() => runCommand(command.id)); };
        menu.append(item);
      }
    }
    if (!menu.children.length) return;
    menu.hidden = false;
    const box = menu.getBoundingClientRect();
    menu.style.left = `${Math.min(x, innerWidth - box.width - 8)}px`;
    menu.style.top = `${Math.max(8, Math.min(y, innerHeight - box.height - 8))}px`;
    (menu.querySelector('.menu-item[aria-disabled=false]') || menu.querySelector('.menu-item'))?.focus();
  }

  function closeContextMenu() { $('context-menu').hidden = true; }

  function openPalette() {
    if (!editor.state || document.querySelector('dialog[open]')) return;
    closeContextMenu();
    $('palette-input').value = '';
    renderPalette();
    $('palette').showModal();
    $('palette-input').focus();
  }

  function renderPalette() {
    paletteRows = matchCommands(COMMANDS.filter(command => !command.hidden?.()), $('palette-input').value);
    paletteIndex = Math.min(paletteIndex, Math.max(0, paletteRows.length - 1));
    if ($('palette-input').dataset.query !== $('palette-input').value) paletteIndex = 0;
    $('palette-input').dataset.query = $('palette-input').value;
    const list = document.createDocumentFragment();
    paletteRows.forEach((command, i) => {
      const row = document.createElement('div');
      row.className = 'palette-row'; row.id = `palette-row-${i}`; row.setAttribute('role', 'option');
      row.setAttribute('aria-selected', String(i === paletteIndex));
      const reason = commandRefusal(command);
      row.classList.toggle('disabled', !!reason); row.setAttribute('aria-disabled', String(!!reason));
      const text = document.createElement('span'); text.className = 'palette-text';
      const name = document.createElement('span'); name.className = 'palette-name'; name.textContent = commandName(command);
      const detail = document.createElement('small'); detail.textContent = reason || command.group;
      text.append(name, detail);
      const keys = document.createElement('kbd'); keys.textContent = command.keys || ''; keys.hidden = !command.keys;
      row.append(text, keys);
      row.onpointermove = () => { if (paletteIndex !== i) { paletteIndex = i; highlightPalette(); } };
      row.onclick = () => runPaletteRow(i);
      list.append(row);
    });
    if (!paletteRows.length) list.append(Object.assign(document.createElement('p'), {className: 'palette-empty', textContent: 'No command matches.'}));
    $('palette-list').replaceChildren(list);
    highlightPalette();
  }

  function highlightPalette() {
    for (const row of $('palette-list').querySelectorAll('.palette-row')) row.setAttribute('aria-selected', String(row.id === `palette-row-${paletteIndex}`));
    const row = $(`palette-row-${paletteIndex}`);
    $('palette-input').setAttribute('aria-activedescendant', row?.id || '');
    row?.scrollIntoView({block: 'nearest'});
  }

  async function runPaletteRow(i) {
    const command = paletteRows[i]; if (!command) return;
    const reason = commandRefusal(command);
    // A command that cannot run keeps the palette open, with its reason shown.
    if (reason) return;
    $('palette').close();
    later(() => runCommand(command.id));
  }

  function joinSelectionKey() { return JSON.stringify([editor.state.epoch,editor.state.revision,editor.state.selection.objects]); }

  function joinCandidates() {
    const selected = new Set(editor.state.selection.objects), candidates = [];
    for (const item of editor.state.objects) {
      let included = selected.has(item.id);
      for (let parent = object(item.parent); !included && parent; parent = object(parent.parent)) included = selected.has(parent.id);
      if (!included) continue;
      if (item.resource || !['g','path'].includes(item.tag)) return [];
      if (item.tag === 'path') candidates.push(item);
    }
    return candidates;
  }

  function openJoin() {
    if (editor.pending) return;
    const candidates = joinCandidates();
    if (candidates.length < 2) {
      toast('Select paths or groups containing at least two paths in total.',true); return;
    }
    joinContext = {key:joinSelectionKey(), candidates};
    $('join-summary').textContent = `Merge ${candidates.length} paths into one shape. Overlapping filled areas merge into their union.`;
    $('join-error').hidden = true;
    const options = document.createDocumentFragment();
    function option(value, title, detail, item = null) {
      const label = document.createElement('label'); label.className = 'join-color-option';
      const radio = document.createElement('input'); radio.type = 'radio'; radio.name = 'join-colors'; radio.value = value; radio.checked = value === 'mix';
      label.append(radio);
      if (item) { const swatch = document.createElement('span'); swatch.className = 'swatch'; paintSwatch(swatch,item); label.append(swatch); }
      const text = document.createElement('span'), name = document.createElement('strong'), description = document.createElement('small');
      name.textContent = title; description.textContent = detail; text.append(name,description); label.append(text); options.append(label);
    }
    option('mix','Area-weighted mix','Larger shapes contribute more to the mixed colors.');
    const colorChoices = new Map();
    for (const item of candidates) {
      const fill = resolvedPaint(item.id,'fill','black'), stroke = resolvedPaint(item.id,'stroke','none');
      const fillAlpha = resolvedPaint(item.id,'fill-opacity','1'), strokeAlpha = resolvedPaint(item.id,'stroke-opacity','1');
      const key = JSON.stringify([fill,stroke,fillAlpha,strokeAlpha]);
      if (colorChoices.has(key)) colorChoices.get(key).count++;
      else colorChoices.set(key,{item,fill,stroke,fillAlpha,strokeAlpha,count:1});
    }
    for (const {item,fill,stroke,fillAlpha,strokeAlpha,count} of colorChoices.values()) {
      const alpha = value => Number(value) < 1 ? ` at ${Math.round(Number(value)*100)}%` : '';
      const detail = `Fill ${colorHex(fill)||fill}${alpha(fillAlpha)} · Stroke ${colorHex(stroke)||stroke}${alpha(strokeAlpha)}` +
        (count > 1 ? ` · ${count} paths share these colors` : '');
      option(item.id,`Use ${item.label} colors`,detail,item);
    }
    $('join-color-options').replaceChildren(options);
    $('join-dialog').showModal();
  }

  async function splitParts() {
    const before = editor.state.selection.objects.length;
    if (await action('split_disconnected', {}, 'Splitting disconnected parts…')) {
      toast(editor.state.selection.objects.length > before ? `Split into ${editor.state.selection.objects.length} independently editable paths.` : 'No disconnected parts found. Holes and touching contours stay together.');
    }
  }

  // Delete works at the current level: the selected points in a point tool,
  // else the selected objects.
  function deleteSelection() {
    const handle = selectedHandle();
    if (handle) return action('delete_handle', {object: handle.object, node: handle.node, offset: handle.offset}, 'Deleting handle…');
    return level() === 'points' ? action('delete_node', {points: pointPairs()}, 'Deleting points…') : action('delete');
  }

  const segmentPicked = () => pointContours().some(({contour, ids}) => segmentAmong(contour, ids));

  function breakPoints() {
    return segmentPicked() ? action('delete_segment', {points: pointPairs()}, 'Deleting segments…') : action('break_points', {points: pointPairs()}, 'Breaking lines…');
  }

  async function cutHole() {
    if (await action('cut_hole', {}, 'Cutting out the hole…')) toast('Cut the shape out as a hole. Undo restores both paths.');
  }

  function wire() {
    document.addEventListener('click', event => {
      const button = event.target.closest('[data-command]');
      if (button && !button.closest('#context-menu')) later(() => runCommand(button.dataset.command));
    });

    buildCommandLists();

    $('palette-input').addEventListener('input', renderPalette);

    $('palette-input').addEventListener('keydown', event => {
      if (['ArrowDown', 'ArrowUp'].includes(event.key)) {
        event.preventDefault();
        paletteIndex = moveHighlight(paletteIndex, event.key === 'ArrowDown' ? 1 : -1, paletteRows.length);
        highlightPalette();
      } else if (event.key === 'Enter') { event.preventDefault(); runPaletteRow(paletteIndex); }
    });

    $('palette').addEventListener('click', event => { if (event.target === $('palette')) $('palette').close(); });

    window.addEventListener('pointerdown', event => { if (!event.target.closest('#context-menu')) closeContextMenu(); }, true);

    window.addEventListener('blur', closeContextMenu);

    $('context-menu').addEventListener('keydown', event => {
      const items = [...$('context-menu').querySelectorAll('.menu-item')], i = items.indexOf(document.activeElement);
      if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); closeContextMenu(); stage.focus(); }
      if (['ArrowDown', 'ArrowUp'].includes(event.key) && items.length) { event.preventDefault(); items[(i + (event.key === 'ArrowDown' ? 1 : items.length - 1)) % items.length].focus(); }
    });

    // Right-clicking something unselected selects it first, as a click would.
    stage.addEventListener('contextmenu', event => {
      event.preventDefault();
      if (!editor.state) return;
      const x = event.clientX, y = event.clientY;
      later(async () => {
        if (level() === 'objects' || level() === 'points') {
          // Found again now: an edit before it may have changed what is there.
          const hits = hitStack(x, y), targets = hits.length ? clickTargets(hits) : [];
          const at = document.elementFromPoint(x, y);
          const node = at?.dataset?.node && (at.dataset.anchor || pointKey(at.dataset.object, at.dataset.node));
          if (node && level() === 'points') {
            if (!selectedPoints().includes(node)) await selectPoints(clickPointPath(editor.state.selection.objects, pointPaths(), splitKey(node)[0], false), [node]);
            editor.activeHandle = at.dataset.part !== 'endpoint' ? {object: at.dataset.object, node: at.dataset.node, offset: Number(at.dataset.part), anchor: node} : null;
            renderInspector(); drawOverlay();
          }
          else if (!node && targets.length && !targets.some(id => editor.state.selection.objects.includes(id) || pointPaths().includes(id))) await selectObject(targets[0]);
        }
        await editor.queue;
        openContextMenu(x, y);
      });
    });

    $('objects').addEventListener('contextmenu', event => {
      const row = event.target.closest('.object-row'); if (!row) return;
      event.preventDefault();
      if (!editor.state) return;
      const id = row.dataset.object, x = event.clientX, y = event.clientY;
      later(async () => {
        if (!object(id)) return;
        if (!editor.state.selection.objects.includes(id)) await selectObject(id);
        await editor.queue;
        openContextMenu(x, y);
      });
    });

    $('undo').onclick=()=>later(()=>action('undo',{},'Undoing…'));
     $('redo').onclick=()=>later(()=>action('redo',{},'Redoing…'));

    $('join-cancel').onclick = () => $('join-dialog').close();

    $('join-dialog').addEventListener('close',()=>joinContext=null);

    $('join-dialog').addEventListener('cancel',event=>{if(editor.pending)event.preventDefault();});

    $('join-confirm').onclick = async () => {
      if(editor.pending || !joinContext) return;
      if(joinSelectionKey() !== joinContext.key) { $('join-error').textContent = 'The selection changed. Close this dialog and open Merge shapes again.'; $('join-error').hidden = false; return; }
      const choice = $('join-color-options').querySelector('input:checked')?.value;
      if(!choice) return;
      const source = joinContext.candidates.find(item=>item.id === choice);
      const options = choice === 'mix' ? {colors:'mix'} : {colors:'source',color_source:choice};
      $('join-confirm').disabled = true; $('join-cancel').disabled = true; $('join-error').hidden = true;
      try {
        if(await action('join_paths',{options},'Merging shapes…')) {
          $('join-dialog').close();
          toast(source ? `Merged using ${source.label}'s colors at the frontmost selected position.` : 'Merged at the frontmost selected position with area-weighted paint.');
        } else { $('join-error').textContent = $('toast-message').textContent; $('join-error').hidden = false; }
      } finally { $('join-confirm').disabled = false; $('join-cancel').disabled = false; }
    };

    $('palette-open').onclick=openPalette;

    $('help').onclick=()=>$('help-dialog').showModal();
    $('help-close').onclick=()=>$('help-dialog').close();
      return api;
  }
  const api = {noSelection, noPoints, visiblePaths, noReference, filledPath, strokedPath, linePaths, fillPaths, twoEnds, runCommand, pointHandles, renameObject, buildCommandLists, renderCommands, MENU_GROUPS, openContextMenu, closeContextMenu, openPalette, renderPalette, highlightPalette, runPaletteRow, joinSelectionKey, joinCandidates, openJoin, splitParts, deleteSelection, segmentPicked, breakPoints, cutHole, wire};
  return api;
}
