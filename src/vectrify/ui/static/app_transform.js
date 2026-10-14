import {FIELD_NAMES, boxFields, fieldEdit, formatLength, rotationEdit, sizeRefusal} from './transform.js';

const FIELDS = ['x', 'y', 'w', 'h'];

// What each field sets, shown when it can be edited.
const DESCRIPTIONS = {
  x: 'Left edge of the selection frame, in document units. Moves the selection.',
  y: 'Top edge of the selection frame, in document units. Moves the selection.',
  w: 'Width of the selection frame, in document units. Resizes from the top-left corner; strokes keep their width.',
  h: 'Height of the selection frame, in document units. Resizes from the top-left corner; strokes keep their width.',
};

// What Rotation sets, when the objects share an angle and when they do not.
const ROTATION = 'Angle of the selection in degrees, clockwise. Turns it about the pivot shown in Move; strokes keep their width.';
const MIXED = 'The objects are turned differently: a value turns them all by that many degrees, clockwise, about the pivot shown in Move.';

// Transform owns its controls; editor contains the live state shared with other controllers.
// Its fields edit the selection frame through the move, resize and rotate
// commands a drag sends, so they keep the same locks, stroke widths and undo
// steps.
export function createTransformController(editor, {
  $,
  action,
  later,
  object,
  resizeRefusal,
  rotateBy,
  rotateRefusal,
  selectionAngle,
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
    if (field === 'angle') return rotateRefusal();
    return resizeRefusal() || sizeRefusal(boxFields(box), field);
  }

  // Rotation shows the angle the objects share, or nothing when they differ.
  function renderRotation(box) {
    const input = $('transform-angle'), reason = fieldRefusal('angle', box), angle = box ? selectionAngle() : null;
    if (!input.dataset.editing) input.value = angle === null ? '' : formatLength(angle);
    input.placeholder = box && angle === null ? 'Mixed' : '';
    input.disabled = !!reason;
    input.parentElement.title = reason ? `Rotation: ${reason}` : angle === null ? MIXED : ROTATION;
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
    renderRotation(box);
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

  // A committed angle turns the selection by the difference, about the pivot,
  // as one undoable edit.
  function commitRotation() {
    const input = $('transform-angle'), text = input.value.trim();
    delete input.dataset.editing;
    return later(async () => {
      const current = selectionAngle();
      if (current !== null && text === formatLength(current)) { renderTransform(); return false; }
      const edit = rotationEdit(current, text === '' ? NaN : Number(text));
      if (!edit || edit.refusal) {
        if (edit) toast(edit.refusal, true);
        renderTransform(); return false;
      }
      return rotateBy(edit.angle);
    });
  }

  function wireField(input, commitField) {
    input.addEventListener('input', () => { input.dataset.editing = 'true'; });
    input.addEventListener('change', commitField);
    // Leaving a field without a change shows the frame again.
    input.addEventListener('blur', () => { if (input.dataset.editing) { delete input.dataset.editing; renderTransform(); } });
    input.addEventListener('keydown', event => {
      if (event.key !== 'Escape') return;
      // Escape puts the field back, without also stepping the selection up.
      event.preventDefault(); event.stopPropagation();
      delete input.dataset.editing; renderTransform(); input.blur();
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
    for (const field of FIELDS) wireField($(`transform-${field}`), () => commit(field));
    wireField($('transform-angle'), commitRotation);
    $('transform-ratio').onclick = () => { editor.keepRatio = !editor.keepRatio; renderTransform(); };
    $('move-apply').onclick = moveBy;
    return api;
  }
  const api = {fieldRefusal, renderTransform, commit, commitRotation, moveBy, wire};
  return api;
}
