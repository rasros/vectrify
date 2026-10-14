import {ALIGNMENTS, REFERENCES, SPACINGS, alignMoves, distributeMoves, distributeName, distributeOrder, unionBox} from './align.js';
import {hiddenRefusal} from './visibility.js';

const REFERENCE_NAMES = {selection: 'Selection', artboard: 'Artboard', key: 'Key object'};

// Align owns Arrange's alignment reference and key object; editor contains the
// live state shared with other controllers. Align and Distribute measure each
// top selected object's geometric bounds, the selection frame's, and send one
// move per object together, so each command is one undoable edit.
//
// The key object is the object last added to the selection on its own (a
// click, a Shift-click or a row in the tree), or the one picked beside the
// reference; the selection's order on the server is not used. It keeps its
// place and has its own frame on the canvas.
export function createAlignController(editor, {
  $,
  action,
  document = globalThis.document,
  drawOverlay,
  later,
  level,
  object,
  objectBounds,
  renderCommands,
  toast,
  topSelection,
  xmlElement,
} = {}) {
  let measured = null;

  // The key object follows the selection: an object added alone becomes the
  // key, and the key is forgotten once it is no longer selected.
  function trackKey() {
    const selected = editor.state?.selection.objects.length ? topSelection() : [];
    const seen = new Set(editor.alignSeen), added = selected.filter(id => !seen.has(id));
    if (added.length === 1) editor.alignKey = added[0];
    else if (!selected.includes(editor.alignKey)) editor.alignKey = null;
    editor.alignSeen = selected;
  }

  const keyObject = () => editor.state?.selection.objects.length && topSelection().includes(editor.alignKey) ? editor.alignKey : null;

  // Each top selected object with its bounds, measured once per state.
  function measure() {
    if (measured?.state === editor.state) return measured;
    const items = [];
    let refusal = '';
    for (const id of editor.state.selection.objects.length ? topSelection() : []) {
      const item = object(id);
      if (!item || item.resource || ['defs', 'clipPath', 'svg'].includes(item.tag)) { refusal = 'Definitions and clipping boundaries have no place to align: select shapes or groups'; break; }
      refusal = hiddenRefusal([id], object, 'align');
      if (refusal) break;
      const box = objectBounds(id);
      if (!box) { refusal = `${item.label} has no bounds to align`; break; }
      items.push({id, box});
    }
    measured = {state: editor.state, items, refusal};
    return measured;
  }

  // Why an object that would move cannot: its position, or that of a group
  // around it or an object inside it, is locked.
  function lockRefusal(moves) {
    const ids = Object.keys(moves);
    if (!ids.length) return '';
    const moving = new Set(ids);
    for (const id of ids) {
      const item = object(id);
      if (item.inherited_locks.includes('transform')) return `${item.label}: position is locked; unlock it to move`;
    }
    for (const inner of editor.state.objects) {
      if (!inner.locks.includes('transform')) continue;
      for (let parent = inner.parent; parent; parent = object(parent)?.parent) {
        if (moving.has(parent)) return `${inner.label} in ${object(parent).label}: position is locked; unlock it to move`;
      }
    }
    return '';
  }

  // The box the selection lines up to, or {refusal}.
  function target(items) {
    const reference = editor.alignReference;
    if (reference === 'artboard') {
      const [x, y, w, h] = editor.state.bounds;
      return {box: {left: x, top: y, right: x + w, bottom: y + h}};
    }
    if (reference === 'key') {
      const key = keyObject();
      if (items.length < 2) return {refusal: 'Select two or more objects to align them to a key object'};
      if (!key) return {refusal: 'Choose the key object: add it to the selection last, or pick it beside Key object'};
      return {box: items.find(item => item.id === key).box, fixed: key};
    }
    if (items.length < 2) return {refusal: 'Select two or more objects to align them to each other, or align to the artboard'};
    return {box: unionBox(items.map(item => item.box))};
  }

  // What *alignment* would move now: {moves} or {refusal}.
  function alignPlan(alignment) {
    if (!editor.state?.selection.objects.length) return {refusal: 'Select objects to align'};
    const {items, refusal} = measure();
    if (refusal) return {refusal};
    const goal = target(items);
    if (goal.refusal) return goal;
    const moves = alignMoves(items, alignment, goal.box, {fixed: goal.fixed});
    return {moves, refusal: lockRefusal(moves)};
  }

  // What distributing along *axis* would move now: {moves} or {refusal}. The
  // reference does not apply: the outer two objects stay, and a key object
  // must be one of them.
  function distributePlan(axis, spacing) {
    if (!editor.state?.selection.objects.length) return {refusal: 'Select three or more objects to distribute'};
    const {items, refusal} = measure();
    if (refusal) return {refusal};
    if (items.length < 3) return {refusal: 'Select three or more objects to distribute'};
    const key = editor.alignReference === 'key' && keyObject();
    if (key) {
      const order = distributeOrder(items, axis);
      if (![order[0].id, order.at(-1).id].includes(key)) return {refusal: `The key object, ${object(key).label}, lies between the others: Distribute keeps the outer two in place, so it would move`};
    }
    const moves = distributeMoves(items, axis, spacing);
    return {moves, refusal: lockRefusal(moves)};
  }

  const alignRefusal = alignment => alignPlan(alignment).refusal || '';
  const distributeRefusal = (axis, spacing) => distributePlan(axis, spacing).refusal || '';

  // The tooltip of an alignment, naming the current reference.
  function alignDescription(alignment) {
    const {edge} = ALIGNMENTS[alignment];
    return `Line up the selected objects' ${edge} with ${REFERENCES[editor.alignReference]}'s, moving them without resizing`;
  }

  const distributeDescription = (axis, spacing) => `${distributeName(axis, spacing)}: move the objects between the outer two, which stay, to put ${SPACINGS[spacing].description}, without resizing`;

  // Run a plan measured once earlier edits have landed, as one undoable edit.
  function run(command, plan, done) {
    return later(async () => {
      const {moves, refusal} = plan();
      if (refusal) { toast(refusal, true); return false; }
      if (!Object.keys(moves).length) { toast(done); return false; }
      return action(command, {moves}, command === 'align' ? 'Aligning…' : 'Distributing…');
    });
  }

  const align = alignment => run('align', () => alignPlan(alignment), `Already lined up: ${ALIGNMENTS[alignment].edge} match`);
  const distribute = (axis, spacing) => run('distribute', () => distributePlan(axis, spacing), `Already spaced with ${SPACINGS[spacing].name}`);

  // The reference choice, and the key object's picker when it is chosen.
  function renderAlign() {
    trackKey();
    $('align-reference').value = editor.alignReference;
    const picker = $('align-key'), key = keyObject();
    $('align-key-field').hidden = editor.alignReference !== 'key';
    if (editor.alignReference !== 'key') return;
    const options = topSelection().map(id => {
      const option = document.createElement('option');
      option.value = id; option.textContent = object(id)?.label || id;
      return option;
    });
    const none = document.createElement('option');
    none.value = ''; none.textContent = 'Choose…'; none.disabled = true;
    picker.replaceChildren(...(key ? [] : [none]), ...options);
    picker.value = key || '';
    picker.title = key ? `${object(key).label} stays in place; the others line up to it` : 'Choose the object the others line up to';
  }

  // The key object's own frame, while it is the reference.
  function drawKeyObject() {
    const key = editor.alignReference === 'key' && level() === 'objects' && topSelection().length > 1 && keyObject();
    const box = key && objectBounds(key);
    if (!box) return;
    editor.overlayFrame.content.append(xmlElement('rect', {x: box.left, y: box.top, width: box.right - box.left, height: box.bottom - box.top, class: 'key-object'}));
  }

  function refresh() { renderAlign(); renderCommands(); drawOverlay(); }

  function wire() {
    $('align-reference').addEventListener('change', event => {
      editor.alignReference = REFERENCE_NAMES[event.target.value] ? event.target.value : 'selection';
      if (editor.state) refresh();
    });
    $('align-key').addEventListener('change', event => {
      editor.alignKey = event.target.value || null;
      if (editor.state) refresh();
    });
    return api;
  }
  const api = {trackKey, keyObject, lockRefusal, alignPlan, distributePlan, alignRefusal, distributeRefusal, alignDescription, distributeDescription, align, distribute, renderAlign, drawKeyObject, wire};
  return api;
}
