// A region drawn on the canvas for Extract region and Delete region, and what
// the preview says each will do. Pure, so it can be checked outside the
// browser; app_region.js draws the region and sends it. Corners are [x, y]
// in the overlay's frame, which is the document's.

// A dragged rectangle's corners, clockwise from its top left.
export function rectangleCorners(a, b) {
  const left = Math.min(a.x, b.x), right = Math.max(a.x, b.x), top = Math.min(a.y, b.y), bottom = Math.max(a.y, b.y);
  return [[left, top], [right, top], [right, bottom], [left, bottom]];
}

// Twice the signed area the corners enclose: none for a line or a point.
export function polygonArea(corners) {
  let area = 0;
  corners.forEach(([x, y], i) => { const [nx, ny] = corners[(i + 1) % corners.length]; area += x * ny - nx * y; });
  return area;
}

// Whether a click at *p* closes a polygon: on its first corner, within
// *reach*, once it has three corners.
export function closesPolygon(corners, p, reach) {
  return corners.length >= 3 && Math.hypot(p.x - corners[0][0], p.y - corners[0][1]) <= reach;
}

const plural = (count, one, many = `${one}s`) => `${count} ${count === 1 ? one : many}`;

// What each operation does to an object, by the preview's outcome for it.
const DETAILS = {
  extract: {
    cut: 'Cut along the edge: the part inside becomes a path of its own',
    contours: 'Its contours inside become a path of their own',
  },
  delete: {
    cut: 'Cut along the edge: the part inside is deleted',
    contours: 'Its contours inside are deleted',
    whole: 'Wholly inside: deleted',
  },
};

// What *mode*'s preview (see Session.region) says: a sentence, a row for each
// object it changes and for each it leaves although the region reaches it,
// each with its outcome as the overlay colours it, or the reason it is
// refused. *cut* is whether crossing objects are cut; *label* names an object.
export function regionPreview(result, mode, cut, label) {
  const operation = result[mode];
  if (operation.error) return {summary: operation.error, rows: [], refusal: operation.error};
  const uncut = new Set(cut ? [] : result.crossing || []);
  const row = (id, outcome, detail) => ({id, label: label(id), outcome, detail});
  const rows = operation.targets.map(target => row(target.id, target.outcome,
    DETAILS[mode][target.outcome] + (uncut.has(target.id) ? '; its contours crossing the edge stay' : '')));
  const changed = new Set(rows.map(item => item.id));
  // A path already wholly inside is left as it is by Extract (Delete deletes it).
  if (mode === 'extract' && !result.delete.error) {
    for (const target of result.delete.targets) if (target.outcome === 'whole') rows.push(row(target.id, 'inside', 'Already wholly inside: stays as it is'));
  }
  for (const id of uncut) if (!changed.has(id)) rows.push(row(id, 'uncut', 'Crosses the edge: stays whole, as cutting is off'));
  const count = operation.targets.length, cuts = operation.targets.filter(target => target.outcome === 'cut').length;
  let summary = `${mode === 'extract' ? 'Extracts from' : 'Deletes from'} ${plural(count, 'object')}`;
  summary += cuts ? `, cutting ${cuts === count ? (count === 1 ? 'it' : 'each') : cuts} along the edge.` : '; nothing is cut.';
  if (uncut.size) summary += ` ${plural(uncut.size, 'object')} crossing the edge ${uncut.size === 1 ? 'is' : 'are'} not cut.`;
  return {summary, rows, refusal: ''};
}
