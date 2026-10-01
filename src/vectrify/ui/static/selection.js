// The two-level selection: objects, and optionally points inside them. Pure
// functions over ids, so they can be checked outside the browser; app.js
// holds the state and sends it to the server.
//
// A point is named by its path and node, as `${object} ${node}`: SVG ids hold
// no spaces, and a node of shared geometry is a point of each path using it.

// Which level each tool works on: object tools pick whole objects, point tools
// show and pick the points of every selected path, and the rest ignore the
// selection.
export const TOOL_LEVEL = {select: 'objects', knife: 'objects', nodes: 'points', redraw: 'points', path: 'create', hand: 'view'};

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

// The paths a point tool shows and edits for the selected objects: the
// selected paths, and every path inside a selected group, however deep, in
// drawing order. *objects* lists {id, tag, parent, resource} in drawing order.
// Instances (use) have no points of their own: they are counted apart, to ask
// for them to be detached.
export function pointTargets(selected, objects) {
  const chosen = new Set(selected), parent = new Map(objects.map(item => [item.id, item.parent]));
  const inside = id => {
    for (let at = id; at !== undefined && at !== null; at = parent.get(at)) if (chosen.has(at)) return true;
    return false;
  };
  const paths = [], instances = [];
  for (const item of objects) {
    if (item.resource || !inside(item.id)) continue;
    if (item.tag === 'path') paths.push(item.id);
    else if (item.tag === 'use') instances.push(item.id);
  }
  return {paths, instances};
}

// Which path each selected point was picked in, from the picked point keys:
// per node the paths it was picked in, and per geometry the path picked in
// last. *geometryOf* gives a path's geometry.
export function pointOwners(keys, geometryOf) {
  const nodes = new Map(), geometries = new Map();
  for (const key of keys) {
    const [object, node] = splitKey(key);
    if (!nodes.has(node)) nodes.set(node, []);
    if (!nodes.get(node).includes(object)) nodes.get(node).push(object);
    geometries.set(geometryOf(object), object);
  }
  return {nodes, geometries};
}

// The selected points to show, as point keys. The server holds node ids, and
// paths drawing one geometry share them: such a node is selected (*strong*)
// in the path it was picked in, else in the path picked in last for that
// geometry, else in the first path showing it. The other paths showing it
// get it as a *twin*, drawn faintly, since an edit moves them too.
// *paths* lists {id, geometry, nodes: [node ids]}.
export function instancePoints(paths, selectedNodes, owners) {
  const users = new Map();
  for (const path of paths) {
    if (!users.has(path.geometry)) users.set(path.geometry, []);
    users.get(path.geometry).push(path.id);
  }
  const strong = [], twins = [];
  for (const path of paths) {
    const shown = users.get(path.geometry);
    for (const node of path.nodes) {
      if (!selectedNodes.has(node)) continue;
      let picked = (owners.nodes.get(node) || []).filter(id => shown.includes(id));
      if (!picked.length) {
        const last = owners.geometries.get(path.geometry);
        picked = [shown.includes(last) ? last : shown[0]];
      }
      (picked.includes(path.id) ? strong : twins).push(pointKey(path.id, node));
    }
  }
  return {strong, twins};
}

// The path a tool acts on under the pointer: the frontmost of *hits* that is a
// drawing path whose geometry and structure are unlocked, inside the entered
// group *scope* if there is one, and among the selected paths (or the paths
// in selected groups) when anything is selected. Null when none qualifies.
export function pointerTarget(hits, objects, selected, scope = null) {
  const byId = new Map(objects.map(item => [item.id, item]));
  const eligible = selected.length ? new Set(pointTargets(selected, objects).paths) : null;
  const inside = id => {
    for (let at = id; at !== undefined && at !== null; at = byId.get(at)?.parent) if (at === scope) return true;
    return false;
  };
  return hits.find(id => {
    const item = byId.get(id);
    if (!item || item.tag !== 'path' || item.resource) return false;
    if ([...item.locks, ...item.inherited_locks].some(lock => lock === 'geometry' || lock === 'structure')) return false;
    if (scope && !inside(id)) return false;
    return !eligible || eligible.has(id);
  }) ?? null;
}
