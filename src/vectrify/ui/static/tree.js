// Where rows dragged in the Objects tree land. Pure functions over the rows'
// positions, so they can be checked outside the browser; app.js reads the rows
// and draws the insertion line.
//
// The tree lists objects in paint order, back first: a row paints over the
// rows above it, and a group's children follow its row, one level deeper.
// Rows are {id, parent, depth, container, top, height}, top to bottom.

// Where a drop at (x, y) goes, or null over no row. x is measured from the
// rows' left edge; *indent* is the width of one tree level. The middle of a
// group row drops into the group, at its front. Between rows, where a group
// ends, the pointer's depth chooses how many groups to leave.
// Returns {parent, after|before, into, line: {y, depth}} or {parent, into}.
export function dropTarget(rows, x, y, indent) {
  if (!rows.length) return null;
  let i = rows.findIndex(row => y < row.top + row.height);
  if (i < 0) i = rows.length - 1;
  const row = rows[i], share = Math.min(1, Math.max(0, (y - row.top) / row.height));
  if (row.container && share >= 0.25 && share < 0.75) return {parent: row.id, into: true};
  const below = share >= 0.5;
  const above = below ? row : rows[i - 1], next = below ? rows[i + 1] : row;
  const lineY = below ? row.top + row.height : row.top;
  if (!above) return {parent: next.parent, before: next.id, line: {y: lineY, depth: next.depth}};
  if (next && next.depth > above.depth) return {parent: above.id, before: next.id, line: {y: lineY, depth: next.depth}};
  const byId = new Map(rows.map(r => [r.id, r]));
  const lowest = next ? next.depth : 0;
  const depth = Math.max(lowest, Math.min(above.depth, Math.round(x / indent)));
  let sibling = above;
  while (sibling.depth > depth && byId.has(sibling.parent)) sibling = byId.get(sibling.parent);
  return {parent: sibling.parent, after: sibling.id, line: {y: lineY, depth: sibling.depth}};
}

// Why the dragged objects cannot drop on *target*, or '' if they can.
// *parents* maps an id to its parent's id; *resources* holds defs and clipPaths.
export function dropRefusal(target, dragged, parents, resources) {
  if (!target) return 'Drop between rows or onto a group';
  for (let id = target.parent; id; id = parents.get(id)) {
    if (dragged.has(id)) return 'Cannot move a group into itself';
    if (resources.has(id)) return 'Definitions and clipping boundaries cannot take objects';
  }
  return '';
}

// A drop made on an earlier tree, while an edit ran, checked against the
// tree the edit left: it lands by the rows it was next to, not where the
// pointer was, or is refused if those are gone or have moved. *objects* are
// the current {id, parent, resource}. Returns {ids, target, refusal}.
export function replayDrop(target, dragged, objects) {
  const byId = new Map(objects.map(item => [item.id, item]));
  const ids = new Set([...dragged].filter(id => byId.has(id)));
  const refuse = refusal => ({ids, target, refusal});
  if (!ids.size) return refuse('The dragged objects were removed meanwhile');
  if (!target) return refuse(dropRefusal(target, ids, new Map(), new Set()));
  const parents = new Map(objects.map(item => [item.id, item.parent]));
  if (!byId.has(target.parent) && !objects.some(item => item.parent === target.parent)) return refuse('The group they were dropped into was removed meanwhile');
  const anchor = target.after || target.before;
  if (anchor && parents.get(anchor) !== target.parent) return refuse('The tree changed where they were dropped; drag them again');
  const resources = new Set(objects.filter(item => item.resource).map(item => item.id));
  return refuse(dropRefusal(target, ids, parents, resources));
}

// The target's index among the parent's children that are not dragged,
// from the parent's children in paint order: what move_objects expects.
export function dropIndex(target, children, dragged) {
  let stop = children.length;
  if (target.after) stop = children.indexOf(target.after) + 1;
  else if (target.before) stop = children.indexOf(target.before);
  return children.slice(0, stop).filter(id => !dragged.has(id)).length;
}
