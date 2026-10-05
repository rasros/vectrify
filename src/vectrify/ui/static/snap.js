// Snapping for node drags. Pure functions over plain points, so they can be
// checked outside the browser; app.js gathers the points and draws the hint.

const ARITY = {M: 2, L: 2, T: 2, H: 1, V: 1, C: 6, S: 4, Q: 4, A: 7, Z: 0};

// The on-curve points of SVG path data, absolute or relative.
export function pathEndpoints(d) {
  const tokens = String(d).match(/[A-Za-z]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?/g) || [];
  const points = [];
  let command = null, x = 0, y = 0, startX = 0, startY = 0, i = 0;
  while (i < tokens.length) {
    if (/[A-Za-z]/.test(tokens[i])) command = tokens[i++];
    if (!command) break;
    const upper = command.toUpperCase(), relative = command !== upper, arity = ARITY[upper];
    if (arity === undefined) break;
    if (upper === 'Z') { x = startX; y = startY; command = null; continue; }
    const values = tokens.slice(i, i + arity).map(Number);
    if (values.length < arity || values.some(v => !Number.isFinite(v))) break;
    i += arity;
    if (upper === 'H') x = values[0] + (relative ? x : 0);
    else if (upper === 'V') y = values[0] + (relative ? y : 0);
    else { x = values[arity-2] + (relative ? x : 0); y = values[arity-1] + (relative ? y : 0); }
    if (upper === 'M') { startX = x; startY = y; command = relative ? 'l' : 'L'; }
    points.push([x, y]);
  }
  return points;
}

// Bucket points on a grid of *cell*-sized squares for nearby lookups.
export function snapIndex(points, cell) {
  const size = cell > 0 && Number.isFinite(cell) ? cell : 1, grid = new Map();
  for (const [x, y] of points) {
    const key = `${Math.floor(x/size)},${Math.floor(y/size)}`;
    const bucket = grid.get(key);
    if (bucket) bucket.push([x, y]); else grid.set(key, [[x, y]]);
  }
  return {cell: size, grid};
}

// The closest indexed point within *radius* of (x, y), or null.
export function nearestPoint(index, x, y, radius, lockedAxis = null) {
  const reach = Math.ceil(radius / index.cell), cx = Math.floor(x/index.cell), cy = Math.floor(y/index.cell);
  let best = null, bestDistance = radius;
  for (let i = cx - reach; i <= cx + reach; i++) for (let j = cy - reach; j <= cy + reach; j++) {
    for (const [px, py] of index.grid.get(`${i},${j}`) || []) {
      if (lockedAxis === 'x' && px !== x || lockedAxis === 'y' && py !== y) continue;
      const distance = Math.hypot(px - x, py - y);
      if (distance <= bestDistance) { best = {x: px, y: py}; bestDistance = distance; }
    }
  }
  return best;
}

// Snap (x, y) to the nearest point target, else to the artboard's edges, each
// axis on its own so an edge holds while sliding along it. *bounds* is
// [x, y, width, height]; its corners count as points. Returns null when
// nothing is within *radius*, otherwise the snapped point and its target:
// {kind: 'point'} or {kind: 'edge', x?: edge x, y?: edge y}. With a locked
// axis, only targets that preserve that coordinate can snap.
export function snapPoint(x, y, index, bounds, radius, lockedAxis = null) {
  const [left, top, width, height] = bounds, right = left + width, bottom = top + height;
  let point = nearestPoint(index, x, y, radius, lockedAxis);
  for (const [cx, cy] of [[left, top], [right, top], [right, bottom], [left, bottom]]) {
    if (lockedAxis === 'x' && cx !== x || lockedAxis === 'y' && cy !== y) continue;
    const distance = Math.hypot(cx - x, cy - y);
    if (distance <= radius && (!point || distance < Math.hypot(point.x - x, point.y - y))) point = {x: cx, y: cy};
  }
  if (point) return {x: point.x, y: point.y, target: {kind: 'point'}};
  // An edge is a side of the artboard, not its whole line.
  const nearest = (value, edges) => edges.reduce((best, edge) =>
    Math.abs(value - edge) <= radius && (best === null || Math.abs(value - edge) < Math.abs(value - best)) ? edge : best, null);
  const ex = lockedAxis !== 'x' && y >= top - radius && y <= bottom + radius ? nearest(x, [left, right]) : null;
  const ey = lockedAxis !== 'y' && x >= left - radius && x <= right + radius ? nearest(y, [top, bottom]) : null;
  if (ex === null && ey === null) return null;
  const target = {kind: 'edge'};
  if (ex !== null) target.x = ex;
  if (ey !== null) target.y = ey;
  return {x: ex ?? x, y: ey ?? y, target};
}
