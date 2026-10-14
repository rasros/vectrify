// A clip is drawn in each consumer's coordinate system, rather than where
// its definition happens to live. These frames also position editable nodes.
export function clippingFrames(element, drawing, to, preferred = null) {
  const clip = element?.closest?.('clipPath');
  if (!clip || !to) return [];
  const consumers = [...drawing.querySelectorAll('[clip-path]')].filter(item =>
    !item.closest('defs, clipPath') && item.getAttribute('clip-path') === `url(#${clip.id})`);
  // An unused definition can still be selected in the tree and edited in
  // artboard coordinates until it has a consumer.
  if (!consumers.length) consumers.push(drawing.querySelector('svg'));
  consumers.sort((a, b) => Number(b.dataset.objectId === preferred) - Number(a.dataset.objectId === preferred));
  let local = new DOMMatrix();
  for (let at = element; at; at = at.parentElement) {
    const transform = at.transform?.baseVal.consolidate()?.matrix;
    if (transform) local = DOMMatrix.fromMatrix(transform).multiply(local);
    if (at === clip) break;
  }
  return consumers.flatMap(consumer => {
    const screen = consumer?.getScreenCTM();
    if (!screen) return [];
    try {
      let matrix = DOMMatrix.fromMatrix(to.inverse().multiply(screen));
      if (clip.getAttribute('clipPathUnits') === 'objectBoundingBox') {
        const b = consumer.getBBox();
        if (!b.width || !b.height) return [];
        matrix = matrix.translate(b.x, b.y).scale(b.width, b.height);
      }
      return [{consumer, matrix: matrix.multiply(local)}];
    } catch { return []; }
  });
}

// The basic shapes, which Convert to path turns into paths.
export const SHAPES = new Set(['rect', 'circle', 'ellipse', 'line']);
const SHAPE_NAMES = {rect: 'rectangle', circle: 'circle', ellipse: 'ellipse', line: 'line'};
const CLIPPABLE = new Set(['path', ...SHAPES, 'g', 'use']);

export const isClipped = item => (item?.attributes?.['clip-path'] ?? 'none') !== 'none';

// Why *item* cannot be a clipping boundary, or ''. A boundary is a closed
// path: the outline of what stays visible.
export function boundaryRefusal(item) {
  if (SHAPES.has(item.tag)) return `A ${SHAPE_NAMES[item.tag]} must be a path to be a boundary: use Convert to path, under Path`;
  if (item.tag !== 'path') return 'Only closed paths can be boundaries';
  if (item.open_path) return 'This path is open: close it to clip to it';
  return '';
}

// Why *item* cannot be clipped, or ''.
export function targetRefusal(item) {
  if (!CLIPPABLE.has(item.tag)) return 'Only paths, shapes, groups and instances can be clipped';
  if (item.inherited_locks?.includes('structure')) return 'Its structure is locked: unlock it to clip it';
  return '';
}

// The roles Set clip starts with: the frontmost object that can be a
// boundary is the boundary, and every other one is clipped. *items* are in
// stacking order, back to front.
export function defaultRoles(items) {
  const boundary = items.findLast(item => !boundaryRefusal(item));
  return new Map(items.map(item => [item.id, item === boundary ? 'boundary' : 'target']));
}

// Why Set clip cannot start with these selected objects, or ''.
export function setClipRefusal(items) {
  if (items.some(item => item.resource)) return 'Select drawn objects, not definitions or clipping boundaries';
  if (items.length < 2) return 'Select the objects to clip and a closed path to clip them to';
  if (items.every(item => boundaryRefusal(item))) {
    const shape = items.find(item => SHAPES.has(item.tag));
    return shape ? boundaryRefusal(shape) : 'Select a closed path to be the boundary';
  }
  return '';
}

// Why the chosen roles cannot be applied, or ''. *object* finds an object by
// id, to tell a clipped group from a clipped object inside it.
export function rolesRefusal(items, roles, object) {
  const targets = items.filter(item => roles.get(item.id) === 'target');
  const boundaries = items.filter(item => roles.get(item.id) === 'boundary');
  if (!targets.length) return 'Choose at least one object to clip';
  if (!boundaries.length) return 'Choose at least one boundary';
  for (const item of boundaries) { const reason = boundaryRefusal(item); if (reason) return `${item.label}: ${reason}`; }
  for (const item of targets) { const reason = targetRefusal(item); if (reason) return `${item.label}: ${reason}`; }
  const clipped = new Set(targets.map(item => item.id));
  for (const item of targets) {
    for (let parent = object(item.parent); parent; parent = object(parent.parent)) {
      if (clipped.has(parent.id)) return `${item.label} is inside ${parent.label}: clip one or the other`;
    }
  }
  return '';
}

// Why Remove clip cannot run on these selected objects, or ''.
export function removeClipRefusal(items) {
  if (items.some(isClipped)) return '';
  if (items.some(item => item.resource)) return 'Select the clipped objects, not their clipping boundaries';
  return 'None of the selected objects is clipped';
}
