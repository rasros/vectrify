// The two-level selection: objects, and optionally points inside them. Pure
// functions over ids, so they can be checked outside the browser; app.js
// holds the state and sends it to the server.
//
// A point is named by its path and node, as `${object} ${node}`: SVG ids hold
// no spaces, and a node of shared geometry is a point of each path using it.

// Which level each tool works on: object tools pick whole objects, point tools
// show and pick the points of every selected path, and the rest ignore the
// selection.
export const TOOL_LEVEL = {select: 'objects', knife: 'objects', trace: 'objects', nodes: 'points', redraw: 'points', path: 'create', hand: 'view'};

export const pointKey = (object, node) => `${object} ${node}`;
export function splitKey(key) {
  const space = key.indexOf(' ');
  return [key.slice(0, space), key.slice(space + 1)];
}

const objectsKey = objects => [...objects].sort().join(' ');

// The points to show after switching from tool *from* to *to*, and what to
// remember. Leaving the point level hides the points and remembers them with
// the objects they were in; coming back restores them if the objects are the
// same, and clears them otherwise.
export function switchTool({objects, points, memory = null}, from, to) {
  const was = TOOL_LEVEL[from] === 'points', will = TOOL_LEVEL[to] === 'points';
  if (was && !will) return {points: [], memory: points.length ? {objects: objectsKey(objects), points: [...points]} : null};
  if (!was && will) return {points: memory?.objects === objectsKey(objects) ? [...memory.points] : [], memory: null};
  return {points: [...points], memory};
}

// The chain of ids from *id* up to, not including, *root*.
function ancestry(id, parents, root) {
  const chain = [];
  for (let at = id; at && at !== root; at = parents.get(at)) chain.push(at);
  return chain;
}

// What a click on *hit* (a painted leaf) picks in an object tool: the
// outermost object inside the entered group *scope*, or inside the drawing.
// A hit outside the entered group leaves it. Returns {id, scope}.
export function pickTarget(hit, scope, parents, root) {
  const chain = ancestry(hit, parents, root);
  if (!chain.length) return {id: null, scope};
  const inside = scope ? chain.indexOf(scope) : chain.length;
  if (inside < 0) return {id: chain.at(-1), scope: null};
  return {id: chain[Math.max(0, inside - 1)], scope};
}

// The groups from the drawing down to the entered group, outermost first.
export function scopeChain(scope, parents, root) {
  return ancestry(scope, parents, root).reverse();
}

// One step up for Escape: points to their paths, objects to the group they
// are in, top-level objects to nothing; with nothing selected, out of the
// entered group. Stepping up to the entered group leaves it.
// Returns {objects, points, scope}.
export function escapeStep({objects, points, scope}, parents, root) {
  if (points.length) return {objects: [...objects], points: [], scope};
  if (!objects.length) {
    const parent = scope ? parents.get(scope) : null;
    return {objects: [], points: [], scope: parent && parent !== root ? parent : null};
  }
  const up = [...new Set(objects.map(id => parents.get(id)).filter(id => id && id !== root))];
  let next = scope;
  if (scope && up.includes(scope)) {
    const parent = parents.get(scope);
    next = parent && parent !== root ? parent : null;
  }
  if (!up.length) next = null;
  return {objects: up, points: [], scope: next};
}

// The selection after a click on a point: that point alone, or with *toggle*
// (Shift) the current points with it added or removed.
export function clickPoint(points, key, toggle) {
  if (!toggle) return [key];
  return points.includes(key) ? points.filter(k => k !== key) : [...points, key];
}

// A drag's box from its two corners, as {left, top, right, bottom}.
export function dragBox(a, b) {
  return {left: Math.min(a.x, b.x), top: Math.min(a.y, b.y), right: Math.max(a.x, b.x), bottom: Math.max(a.y, b.y)};
}

// Whether *rect* ({left, top, right, bottom}) lies wholly inside *box*.
export function rectInside(rect, box) {
  return rect.left >= box.left && rect.right <= box.right && rect.top >= box.top && rect.bottom <= box.bottom;
}

export function pointInside(x, y, box) {
  return x >= box.left && x <= box.right && y >= box.top && y <= box.bottom;
}

// The selection after a box select finds *found*: those alone, or with
// *toggle* (Shift) the current selection with each of them flipped.
export function boxSelect(current, found, toggle) {
  if (!toggle) return [...new Set(found)];
  const result = new Set(current);
  for (const id of new Set(found)) if (result.has(id)) result.delete(id); else result.add(id);
  return [...result];
}

const plural = (count, one, many = `${one}s`) => `${count.toLocaleString()} ${count === 1 ? one : many}`;

// The tool strip's status: the level and what is selected at it.
export function selectionStatus(level, objects, points) {
  if (level === 'points') {
    const paths = new Set(points.map(key => splitKey(key)[0])).size;
    if (points.length) return `Points · ${plural(points.length, 'point')} in ${plural(paths, 'path')}`;
    return objects.length ? `Points · none selected in ${plural(objects.length, 'path')}` : 'Points · select a path';
  }
  return objects.length ? `Objects · ${objects.length.toLocaleString()} selected` : 'Objects · nothing selected';
}
