import {TREE_INDENT} from './app_config.js';
import {dropRefusal, dropTarget} from './tree.js';
import {eyeControl, hiddenRefusal} from './visibility.js';

// Tree owns its controls; editor contains the live state shared with other controllers.
export function createTreeController(editor, {
  $,
  action,
  CSS = globalThis.CSS,
  document = globalThis.document,
  enterObject,
  later,
  object,
  objectContext,
  paintSwatch,
  pressTreeRow,
  selectObject,
  toast,
} = {}) {
  let treeSelection = new Set();

  let objectRows = new Map();

  function setGroupExpanded(id, expanded, disclosure = false) {
    if (expanded) editor.collapsedGroups.delete(id);
    else editor.collapsedGroups.add(id);
    renderObjects();
    const row = objectRows.get(id);
    (disclosure ? row?.querySelector('.tree-toggle') : row)?.focus();
  }

  function updateObjectSelection() {
    const selected = new Set(editor.state.selection.objects);
    for (const id of new Set([...treeSelection, ...selected])) {
      if (treeSelection.has(id) === selected.has(id)) continue;
      const row = objectRows.get(id);
      if (!row) continue;
      row.classList.toggle('selected', selected.has(id));
      row.setAttribute('aria-selected', String(selected.has(id)));
    }
    treeSelection = selected;
  }

  function renderObjects() {
    const list = $('objects'), scrollTop = list.scrollTop, scrollLeft = list.scrollLeft;
    const search = $('object-search').value.trim().toLowerCase();
    const fragment = document.createDocumentFragment();
    objectRows = new Map();
    treeSelection = new Set(editor.state.selection.objects);
    const contexts = new Map(editor.state.objects.map(item => [item.id, objectContext(item)]));
    const visible = new Set();
    for (const item of editor.state.objects) {
      if (!`${item.label} ${item.tag} ${contexts.get(item.id).role}`.toLowerCase().includes(search)) continue;
      // Retain parents in filtered results so indentation never implies a false parent.
      for (let ancestor = item; ancestor; ancestor = object(ancestor.parent)) visible.add(ancestor.id);
    }
    const containers = new Set(editor.state.objects.map(item => item.parent));
    const folded = new Set();
    let count = 0;
    for (const item of editor.state.objects) {
      // Objects arrive in tree order. Search temporarily opens matching branches.
      if (!search && (editor.collapsedGroups.has(item.parent) || folded.has(item.parent))) {
        folded.add(item.id); continue;
      }
      if (!visible.has(item.id)) continue;
      const context = contexts.get(item.id);
      count++;
      const row = document.createElement('div'); row.className = 'object-row'; row.tabIndex = 0;
      row.classList.toggle('selected', treeSelection.has(item.id));
      row.classList.toggle('resource', item.resource); row.setAttribute('role', 'treeitem');
      row.setAttribute('aria-level', String(item.depth + 1));
      row.setAttribute('aria-label', item.label);
      row.setAttribute('aria-selected', String(treeSelection.has(item.id)));
      row.dataset.object = item.id; row.title = `${item.tag} · ${item.id}`;
      row.style.paddingLeft = `${9 + item.depth * 9}px`;
      const expandable = containers.has(item.id);
      const expanded = !!search || !editor.collapsedGroups.has(item.id);
      const toggle = document.createElement(expandable ? 'button' : 'span');
      toggle.className = expandable ? 'tree-toggle' : 'tree-toggle-spacer';
      if (expandable) {
        row.setAttribute('aria-expanded', String(expanded));
        toggle.type = 'button'; toggle.textContent = expanded ? '▾' : '▸';
        toggle.setAttribute('aria-label', `${expanded ? 'Collapse' : 'Expand'} ${item.label}`);
        toggle.setAttribute('aria-expanded', String(expanded));
        toggle.disabled = !!search;
        toggle.title = search ? 'Clear search to collapse groups' : `${expanded ? 'Collapse' : 'Expand'} group`;
        toggle.onpointerdown = event => event.stopPropagation();
        toggle.ondblclick = event => event.stopPropagation();
        toggle.onkeydown = event => { if (event.key === ' ' || event.key === 'Enter') event.stopPropagation(); };
        toggle.onclick = event => {
          event.stopPropagation();
          if (!editor.treeDragEnded) setGroupExpanded(item.id, !expanded, true);
        };
      } else toggle.setAttribute('aria-hidden', 'true');
      const swatch = document.createElement('span'); swatch.className = 'swatch';
      paintSwatch(swatch, item);
      row.title += ` · ${swatch.title}`;
      const label = document.createElement('span'); label.className = 'object-label';
      const name = document.createElement('span'); name.textContent = (item.tag === 'g' ? '▱ ' : '') + item.label;
      label.append(name);
      if (context.role) {
        row.classList.add('has-description');
        const detail = document.createElement('small'); detail.textContent = context.role;
        label.append(detail); row.title += ` · ${context.role}`;
      }
      row.append(toggle, swatch, label);
      if (item.inherited_locks.length) { const mark = document.createElement('span'); mark.className = 'lock-mark'; mark.textContent = '◆'; mark.title = `Locked: ${item.inherited_locks.map(lock => lock === 'transform' ? 'position' : lock).join(', ')}`; row.append(mark); }
      const eye = visibilityEye(item);
      if (eye) row.append(eye);
      // Hidden rows stay listed, dimmed, so they can be selected and shown.
      row.classList.toggle('concealed', !!item.inherited_hidden);
      if (item.inherited_hidden) row.title += item.hidden ? ' · hidden' : ' · hidden with its group';
      row.onclick = event => {
        const additive = event.shiftKey || event.ctrlKey || event.metaKey;
        if (!editor.treeDragEnded) later(() => object(item.id) && selectObject(item.id, additive, true, true));
      };
      row.ondblclick = () => {
        if (item.inherited_hidden) toast(hiddenRefusal([item.id], object, 'edit'), true);
        else if (!item.resource) later(() => object(item.id) && enterObject(item.id));
      };
      row.onpointerdown = event => pressTreeRow(event, item);
      row.onkeydown = event => {
        if (event.target !== row) return;
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault(); event.stopPropagation(); row.onclick(event);
        } else if (expandable && !search && ['ArrowLeft', 'ArrowRight'].includes(event.key)) {
          event.preventDefault(); event.stopPropagation();
          setGroupExpanded(item.id, event.key === 'ArrowRight');
        }
      };
      objectRows.set(item.id, row);
      fragment.append(row);
    }
    list.replaceChildren(fragment, treeDropLine); $('object-count').textContent = count;
    // Saving a point drag refreshes these rows too. Keep the tree where the
    // user left it instead of letting row replacement shift its viewport.
    list.scrollTop = scrollTop; list.scrollLeft = scrollLeft;
    // A drag held across an edit drops where the pointer is on the new rows.
    if (editor.treeDrag?.active) {
      editor.treeDrag.ids = new Set([...editor.treeDrag.ids].filter(id => object(id)));
      editor.treeDrag.revision = `${editor.state.epoch}:${editor.state.revision}`;
      if (editor.treeDrag.point) Object.assign(editor.treeDrag, treeDropAt(editor.treeDrag.point, editor.treeDrag.ids));
      showTreeDrop();
    }
  }

  // The eye at the end of a row hides or shows its object, as one undoable
  // edit saved with the drawing; the selection stays as it is. It is apart
  // from the fold arrow at the start of the row and from the lock mark.
  function visibilityEye(item) {
    const control = eyeControl(item, object);
    if (!control) return null;
    const eye = document.createElement('button');
    eye.type = 'button'; eye.className = 'tree-eye';
    eye.classList.toggle('off', control.hidden);
    eye.classList.toggle('inherited', control.concealed && !control.hidden);
    eye.setAttribute('aria-label', control.label);
    eye.setAttribute('aria-pressed', String(control.hidden));
    if (control.refusal) eye.setAttribute('aria-disabled', 'true');
    eye.title = control.title;
    eye.onpointerdown = event => event.stopPropagation();
    eye.ondblclick = event => event.stopPropagation();
    eye.onkeydown = event => { if (event.key === ' ' || event.key === 'Enter') event.stopPropagation(); };
    eye.onclick = event => {
      event.stopPropagation();
      if (editor.treeDragEnded) return;
      if (control.refusal) { toast(control.refusal, true); return; }
      later(() => object(item.id) && action('visibility', {objects: [item.id], hidden: !control.hidden}, control.hidden ? 'Showing…' : 'Hiding…'));
    };
    return eye;
  }

  const treeDropLine = document.createElement('div');
   treeDropLine.className = 'tree-drop-line';
   treeDropLine.hidden = true;

  function treeRows() {
    return [...$('objects').querySelectorAll('.object-row')].map(row => {
      const item = object(row.dataset.object), box = row.getBoundingClientRect();
      return {id: item.id, parent: item.parent, depth: item.depth, container: item.tag === 'g' && !item.resource, top: box.top, height: box.height, left: box.left};
    });
  }

  function showTreeDrop() {
    const {target, refusal} = editor.treeDrag;
    for (const row of $('objects').querySelectorAll('.object-row')) {
      row.classList.toggle('drop-into', !refusal && !!target?.into && row.dataset.object === target.parent);
      row.classList.toggle('dragged', editor.treeDrag.ids.has(row.dataset.object));
    }
    treeDropLine.hidden = !!refusal || !target?.line;
    $('objects').classList.toggle('drop-refused', !!refusal);
    $('objects').title = refusal;
    if (treeDropLine.hidden) return;
    const list = $('objects'), box = list.getBoundingClientRect();
    treeDropLine.style.top = `${target.line.y - box.top + list.scrollTop - 1}px`;
    treeDropLine.style.left = `${8 + TREE_INDENT * (1 + target.line.depth)}px`;
  }

  // Where rows dropped at *point* land, and why they cannot, if so.
  function treeDropAt(point, ids) {
    const rows = treeRows(), box = $('objects').getBoundingClientRect();
    const target = dropTarget(rows, point.x - (rows[0]?.left ?? box.left) - TREE_INDENT, point.y, TREE_INDENT);
    const parents = new Map(editor.state.objects.map(item => [item.id, item.parent]));
    const resources = new Set(editor.state.objects.filter(item => item.resource).map(item => item.id));
    return {target, refusal: dropRefusal(target, ids, parents, resources)};
  }

  function revealObject(id) {
    let unfolded = false;
    for (let ancestor = object(object(id)?.parent); ancestor; ancestor = object(ancestor.parent)) {
      if (editor.collapsedGroups.delete(ancestor.id)) unfolded = true;
    }
    if (unfolded) renderObjects();
    let row = $('objects').querySelector(`[data-object="${CSS.escape(id)}"]`);
    if (!row && $('object-search').value) {
      $('object-search').value = ''; renderObjects();
      row = $('objects').querySelector(`[data-object="${CSS.escape(id)}"]`);
    }
    row?.scrollIntoView({block: 'nearest', inline: 'nearest'});
  }

  function wire() {
    $('object-search').oninput=renderObjects;
      return api;
  }
  const api = {setGroupExpanded, updateObjectSelection, renderObjects, treeDropLine, treeRows, showTreeDrop, treeDropAt, revealObject, wire};
  return api;
}
