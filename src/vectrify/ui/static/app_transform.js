import {FIELD_NAMES, boxFields, fieldEdit, formatLength, sizeRefusal} from './transform.js';

const FIELDS = ['x', 'y', 'w', 'h'];

// What each field sets, shown when it can be edited.
const DESCRIPTIONS = {
  x: 'Left edge of the selection frame, in document units. Moves the selection.',
  y: 'Top edge of the selection frame, in document units. Moves the selection.',
  w: 'Width of the selection frame, in document units. Resizes from the top-left corner; strokes keep their width.',
  h: 'Height of the selection frame, in document units. Resizes from the top-left corner; strokes keep their width.',
};

// Transform owns its controls; editor contains the live state shared with other controllers.
// Its fields edit the selection frame through the move and resize commands a
// drag sends, so they keep the same locks, stroke widths and undo steps.
export function createTransformController(editor, {
  $,
  action,
  later,
  object,
  resizeRefusal,
  selectionBounds,
  toast,
  topSelection,
} = {}) {

  // Why nothing in the section can run now, or ''.
  function boundsRefusal(box) {
    if (!editor.state?.selection.objects.length) return 'Select objects first';
    return box ? '' : 'The selection has no bounds to edit: select shapes or groups, not definitions or clipping boundaries';
  }

  function moveRefusal() {
    for (const id of topSelection()) {
      const item = object(id);
      if (item?.inherited_locks.includes('transform')) return `${item.label}: position is locked; unlock it to move`;
    }
    return '';
  }

  // Why the field cannot be edited now, or ''.
  function fieldRefusal(field, box = selectionBounds()) {
    const none = boundsRefusal(box);
    if (none) return none;
    if (field === 'x' || field === 'y') return moveRefusal();
    return resizeRefusal() || sizeRefusal(boxFields(box), field);
  }

  // The fields show the selection frame, except one being typed in.
  function renderTransform() {
    const box = editor.state.selection.objects.length ? selectionBounds() : null, values = box && boxFields(box);
    for (const field of FIELDS) {
      const input = $(`transform-${field}`), reason = fieldRefusal(field, box);
      if (!input.dataset.editing) input.value = values ? formatLength(values[field]) : '';
      input.disabled = !!reason;
      input.parentElement.title = reason ? `${FIELD_NAMES[field]}: ${reason}` : DESCRIPTIONS[field];
    }
    const ratio = $('transform-ratio'), sizing = boundsRefusal(box) || resizeRefusal();
    ratio.setAttribute('aria-pressed', String(!!editor.keepRatio));
    ratio.disabled = !!sizing;
    ratio.title = sizing ? `Keep proportions: ${sizing}` : editor.keepRatio ? 'Width and height change together; click to change them separately' : 'Width and height change separately; click to keep their proportions';
  }

  // A committed field moves or resizes the selection as one undoable edit,
  // measured once earlier edits have landed.
  function commit(field) {
    const input = $(`transform-${field}`), text = input.value.trim();
    delete input.dataset.editing;
    return later(async () => {
      const box = selectionBounds();
      if (!box || text === formatLength(boxFields(box)[field])) { renderTransform(); return false; }
      const edit = fieldEdit(box, field, text === '' ? NaN : Number(text), {keepRatio: !!editor.keepRatio});
      if (!edit || edit.refusal) {
        if (edit) toast(edit.refusal, true);
        renderTransform(); return false;
      }
      return action(edit.command, edit.data, edit.command === 'move' ? 'Moving selection…' : 'Resizing…');
    });
  }

  // Move by offsets the selection in document units, like X and Y.
  function moveBy() {
    const dx = Number($('move-x').value), dy = Number($('move-y').value);
    $('move-x').value = '0'; $('move-y').value = '0';
    if (!Number.isFinite(dx) || !Number.isFinite(dy)) { toast('Move by needs numbers', true); return null; }
    return later(() => action('move', {dx, dy}));
  }

  function wire() {
    for (const field of FIELDS) {
      const input = $(`transform-${field}`);
      input.addEventListener('input', () => { input.dataset.editing = 'true'; });
      input.addEventListener('change', () => commit(field));
      // Leaving a field without a change shows the frame again.
      input.addEventListener('blur', () => { if (input.dataset.editing) { delete input.dataset.editing; renderTransform(); } });
      input.addEventListener('keydown', event => {
        if (event.key !== 'Escape') return;
        // Escape puts the field back, without also stepping the selection up.
        event.preventDefault(); event.stopPropagation();
        delete input.dataset.editing; renderTransform(); input.blur();
      });
    }
    $('transform-ratio').onclick = () => { editor.keepRatio = !editor.keepRatio; renderTransform(); };
    $('move-apply').onclick = moveBy;
    return api;
  }
  const api = {fieldRefusal, renderTransform, commit, moveBy, wire};
  return api;
}
