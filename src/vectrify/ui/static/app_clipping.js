import {boundaryRefusal, defaultRoles, isClipped, removeClipRefusal, rolesRefusal, setClipRefusal, targetRefusal} from './clipping.js';
import {plural, selectedObjects as selectedItems} from './util.js';

// What each kind of boundary geometry does, shown under the choice.
const GEOMETRY = {
  linked: 'The clip shares the boundary path\'s points: reshaping the path in Nodes reshapes the clip. Moving the path does not move the clip.',
  copied: 'The clip keeps a copy of the outline the boundary has now: later edits to the path, or deleting it, leave the clip as it is.',
};

const names = items => items.length < 2 ? items.map(item => item.label).join('')
  : `${items.slice(0, -1).map(item => item.label).join(', ')} and ${items.at(-1).label}`;

// Clipping owns Set clip's dialog; editor contains the live state shared with other controllers.
// The dialog names every selected object's role, clipped or boundary, before
// anything changes: which is which is never left to the stacking order.
export function createClippingController(editor, {
  $,
  action,
  document = globalThis.document,
  noSelection,
  object,
  paintSwatch,
  toast,
} = {}) {
  let context = null;

  const selectedObjects = () => selectedItems(editor.state);

  const selectionKey = () => JSON.stringify([editor.state.epoch, editor.state.revision, editor.state.selection.objects]);

  const setRefusal = () => noSelection() || setClipRefusal(selectedObjects());

  const removeRefusal = () => noSelection() || removeClipRefusal(selectedObjects());

  function openSetClip() {
    if (editor.pending) return;
    const reason = setRefusal();
    if (reason) { toast(reason, true); return; }
    const items = selectedObjects();
    context = {key: selectionKey(), items, roles: defaultRoles(items)};
    $('clip-linked').checked = true; $('clip-copied').checked = false;
    $('clip-error').hidden = true;
    renderSetClip();
    $('clip-dialog').showModal();
  }

  function roleControl(item, role, title, reason) {
    const label = document.createElement('label'), radio = document.createElement('input');
    radio.type = 'radio'; radio.name = `clip-role-${item.id}`; radio.value = role;
    radio.checked = context.roles.get(item.id) === role; radio.disabled = !!reason && !radio.checked;
    radio.dataset.object = item.id;
    radio.addEventListener('change', () => { context.roles.set(item.id, role); renderSetClip(); });
    label.className = 'clip-role-choice'; label.title = reason || '';
    label.append(radio, document.createTextNode(title));
    return label;
  }

  // One row per selected object, with its two roles, and a sentence saying
  // what will be clipped to what.
  function renderSetClip() {
    const {items, roles} = context, rows = document.createDocumentFragment();
    for (const item of items) {
      const row = document.createElement('div'); row.className = 'clip-role'; row.dataset.role = roles.get(item.id); row.dataset.object = item.id;
      const swatch = document.createElement('span'); swatch.className = 'swatch'; paintSwatch(swatch, item);
      const text = document.createElement('span'), name = document.createElement('strong'), detail = document.createElement('small');
      name.textContent = item.label;
      detail.textContent = roles.get(item.id) === 'boundary' ? 'Boundary: what lies outside it is hidden'
        : isClipped(item) ? 'Clipped: replaces its current clip' : 'Clipped';
      text.append(name, detail);
      const choices = document.createElement('span'); choices.className = 'clip-role-choices'; choices.setAttribute('role', 'radiogroup'); choices.setAttribute('aria-label', `Role of ${item.label}`);
      choices.append(roleControl(item, 'target', 'Clipped', targetRefusal(item)), roleControl(item, 'boundary', 'Boundary', boundaryRefusal(item)));
      row.append(swatch, text, choices);
      rows.append(row);
    }
    $('clip-roles').replaceChildren(rows);
    const targets = items.filter(item => roles.get(item.id) === 'target'), boundaries = items.filter(item => roles.get(item.id) === 'boundary');
    const reason = rolesRefusal(items, roles, object);
    $('clip-summary').textContent = reason || `Clip ${names(targets)} to ${boundaries.length > 1 ? `the combined area of ${names(boundaries)}` : names(boundaries)}.`;
    $('clip-geometry-detail').textContent = GEOMETRY[$('clip-copied').checked ? 'copied' : 'linked'];
    $('clip-confirm').disabled = !!reason;
  }

  async function confirmSetClip() {
    if (editor.pending || !context) return;
    if (selectionKey() !== context.key) { showError('The selection changed. Close this dialog and open Set clip again.'); return; }
    const {items, roles} = context;
    if (rolesRefusal(items, roles, object)) return;
    const pick = role => items.filter(item => roles.get(item.id) === role);
    const targets = pick('target'), boundaries = pick('boundary'), linked = !$('clip-copied').checked;
    $('clip-confirm').disabled = true; $('clip-cancel').disabled = true; $('clip-error').hidden = true;
    try {
      if (await action('set_clip', {targets: targets.map(item => item.id), boundaries: boundaries.map(item => item.id), linked}, 'Setting clip…')) {
        $('clip-dialog').close();
        toast(`Clipped ${plural(targets.length, 'object')} to ${names(boundaries)}. ${linked ? 'Reshaping the boundary reshapes the clip.' : 'The clip keeps a copy of the boundary\'s outline.'}`);
      } else showError($('toast-message').textContent);
    } finally { $('clip-confirm').disabled = !!context && !!rolesRefusal(items, roles, object); $('clip-cancel').disabled = false; }
  }

  function showError(message) { $('clip-error').textContent = message; $('clip-error').hidden = false; }

  async function removeClip() {
    const clipped = selectedObjects().filter(isClipped).length;
    if (await action('remove_clip', {}, 'Removing clip…')) toast(`Removed the clip from ${plural(clipped, 'object')}. Boundary paths stay in the drawing.`);
  }

  function wire() {
    $('clip-cancel').onclick = () => $('clip-dialog').close();
    $('clip-dialog').addEventListener('close', () => { context = null; });
    $('clip-dialog').addEventListener('cancel', event => { if (editor.pending) event.preventDefault(); });
    for (const id of ['clip-linked', 'clip-copied']) $(id).addEventListener('change', () => { if (context) renderSetClip(); });
    $('clip-confirm').onclick = confirmSetClip;
    return api;
  }
  const api = {openSetClip, confirmSetClip, removeClip, setRefusal, removeRefusal, wire};
  return api;
}
