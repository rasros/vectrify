// Hiding objects. Pure functions over the session's object rows, so they can
// be checked outside the browser.
//
// Visibility is document state: a hidden object is saved as display="none",
// so it stays hidden in the exported SVG, the project and for agents, and
// undo brings it back. A hidden group hides everything it holds; each row
// says whether it is hidden itself (hidden) and whether it is hidden at all,
// itself or by a group around it (inherited_hidden). Hidden objects are not
// hit, box selected, selected by Select all or snapped to on the canvas; the
// Objects tree still selects them, and the panel edits their paint, name,
// locks and stacking. What needs their place on the canvas, moving, resizing,
// turning, aligning and their points, waits until they are shown.

// What can be hidden: drawing objects and groups, not definitions, gradients
// or clipping boundaries.
export const HIDEABLE = new Set(['g', 'path', 'rect', 'circle', 'ellipse', 'line', 'use']);

// Said wherever hiding is offered, so it is not taken for a view setting.
export const EXPORTED = 'Hidden objects stay hidden in the saved and exported SVG';

export const hideable = item => !!item && !item.resource && HIDEABLE.has(item.tag);

// Why *item*'s visibility cannot change, or '': it is not a drawing object,
// or its paint, or a group's around it, is locked.
export function visibilityLock(item, verb) {
  if (!hideable(item)) return 'Definitions and clipping boundaries cannot be hidden';
  if (item.inherited_locks.includes('paint')) return `${item.label}: paint is locked; unlock it to ${verb} it`;
  return '';
}

// The group that hides *item*, when it is not hidden itself, or null.
export function hidingGroup(item, object) {
  if (!item?.inherited_hidden || item.hidden) return null;
  for (let at = object(item.parent); at; at = object(at.parent)) if (at.hidden) return at;
  return null;
}

// The eye on *item*'s row in the Objects tree, or null for a row without one:
// {hidden, concealed, label, title, refusal}. *hidden* is its own state, which
// pressing it turns over; *concealed* that it is not drawn, by itself or by
// a hidden group.
export function eyeControl(item, object) {
  if (!hideable(item)) return null;
  const verb = item.hidden ? 'show' : 'hide', label = `${item.hidden ? 'Show' : 'Hide'} ${item.label}`;
  const refusal = visibilityLock(item, verb), group = hidingGroup(item, object);
  const title = refusal ? `${label}: ${refusal}`
    : item.hidden ? `Hidden. ${label}. ${EXPORTED}`
      : group ? `Hidden with ${group.label}: show the group to show it. ${label} itself. ${EXPORTED}`
        : `${label}. ${EXPORTED}`;
  return {hidden: !!item.hidden, concealed: !!item.inherited_hidden, label, title, refusal};
}

// Why the objects *ids* cannot be hidden (*hidden* true) or shown, or ''.
export function visibilityRefusal(ids, object, hidden) {
  const verb = hidden ? 'hide' : 'show';
  if (!ids.length) return 'Select objects first';
  const items = ids.map(object).filter(Boolean);
  for (const item of items) {
    const refusal = visibilityLock(item, verb);
    if (refusal) return refusal;
  }
  if (items.every(item => !!item.hidden === hidden)) return hidden ? 'The selection is hidden already' : 'Nothing selected is hidden itself';
  return '';
}

// Why an edit that needs the objects *ids* on the canvas (*verb*: move,
// resize, rotate...) cannot run while one of them is hidden, or ''.
export function hiddenRefusal(ids, object, verb) {
  const item = ids.map(object).find(one => one?.inherited_hidden);
  if (!item) return '';
  const group = hidingGroup(item, object);
  return group ? `${item.label} is hidden with ${group.label}: show it to ${verb} it` : `${item.label} is hidden: show it to ${verb} it`;
}

// The objects that are not drawn: hidden, or inside a hidden group.
export const concealed = item => !!item?.inherited_hidden;
