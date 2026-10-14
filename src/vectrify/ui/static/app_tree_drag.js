import {dropIndex, replayDrop} from './tree.js';

// Tree_drag owns its controls; editor contains the live state shared with other controllers.
export function createTreeDragController(editor, {
  $,
  action,
  later,
  object,
  showTreeDrop,
  toast,
  treeDropAt,
  treeDropLine,
  window = globalThis.window,
} = {}) {

  function pressTreeRow(event, item) {
    if (event.button !== 0 || item.resource) return;
    editor.tree.drag = {item: item.id, x: event.clientX, y: event.clientY, active: false, target: null};
  }

  function moveTreeDrag(event) {
    if (!editor.tree.drag) return;
    if (!editor.tree.drag.active) {
      if (Math.hypot(event.clientX - editor.tree.drag.x, event.clientY - editor.tree.drag.y) < 5) return;
      const selected = editor.state.selection.objects;
      const ids = selected.includes(editor.tree.drag.item) ? selected.filter(id => !object(id)?.resource) : [editor.tree.drag.item];
      editor.tree.drag = {...editor.tree.drag, active: true, ids: new Set(ids), revision: `${editor.state.epoch}:${editor.state.revision}`};
      $('objects').classList.add('dragging');
    }
    const list = $('objects'), box = list.getBoundingClientRect();
    // Only scroll at an edge while the pointer is inside the list. Dragging
    // outside the tree must not keep moving it in the background.
    if (event.clientX >= box.left && event.clientX <= box.right && event.clientY >= box.top && event.clientY <= box.bottom) {
      if (event.clientY < box.top + 24) list.scrollTop -= 12;
      else if (event.clientY > box.bottom - 24) list.scrollTop += 12;
    }
    editor.tree.drag.point = {x: event.clientX, y: event.clientY};
    Object.assign(editor.tree.drag, treeDropAt(editor.tree.drag.point, editor.tree.drag.ids));
    showTreeDrop();
  }

  function endTreeDrag(drop) {
    const finished = editor.tree.drag;
    editor.tree.drag = null; treeDropLine.hidden = true;
    $('objects').classList.remove('dragging', 'drop-refused'); $('objects').title = '';
    for (const row of $('objects').querySelectorAll('.drop-into, .dragged')) row.classList.remove('drop-into', 'dragged');
    if (!finished?.active) return;
    // The pointer is released over a row: that is not a click on it.
    editor.tree.dragEnded = true; setTimeout(() => editor.tree.dragEnded = false);
    if (!drop) return;
    // Dropped while an edit runs, it lands once the edit is done, next to the
    // rows it was dropped by, if they are still there.
    later(() => {
      const {ids, target, refusal} = finished.refusal || finished.revision === `${editor.state.epoch}:${editor.state.revision}` ? finished : replayDrop(finished.target, finished.ids, editor.state.objects);
      if (refusal) { toast(refusal, true); return; }
      const children = editor.state.objects.filter(item => item.parent === target.parent).map(item => item.id);
      return action('move_objects', {objects: [...ids], parent: target.parent, index: dropIndex(target, children, ids)}, 'Moving objects…');
    });
  }

  function wire() {
    window.addEventListener('pointermove', moveTreeDrag);

    window.addEventListener('pointerup', () => endTreeDrag(true));

    window.addEventListener('pointercancel', () => endTreeDrag(false));
      return api;
  }
  const api = {pressTreeRow, moveTreeDrag, endTreeDrag, wire};
  return api;
}
