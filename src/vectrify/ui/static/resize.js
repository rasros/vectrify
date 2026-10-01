// Resizing the selection by its bounding box, like a window by its frame.
// Pure functions over plain boxes ({left, top, right, bottom}, y down), so
// they can be checked outside the browser; app.js finds the box, previews
// the scale and sends it to the server.

// The cursor over each part of the frame.
export const CURSORS = {n: 'ns-resize', s: 'ns-resize', e: 'ew-resize', w: 'ew-resize', nw: 'nwse-resize', se: 'nwse-resize', ne: 'nesw-resize', sw: 'nesw-resize', inside: 'move'};

// Which part of *box* (x, y) is on: an edge ('n', 'e', 's', 'w') or corner
// ('nw', 'ne', 'se', 'sw') within *reach* of it, inside it, or null outside.
// Where a small box puts two opposite edges within reach, the nearer wins.
export function frameHandle(box, x, y, reach) {
  const {left, top, right, bottom} = box;
  if (x < left - reach || x > right + reach || y < top - reach || y > bottom + reach) return null;
  const side = (value, low, high, before, after) => {
    const a = Math.abs(value - low), b = Math.abs(value - high);
    if (Math.min(a, b) > reach) return '';
    return a < b || (a === b && value < low) ? before : after;
  };
  const handle = side(y, top, bottom, 'n', 's') + side(x, left, right, 'w', 'e');
  return handle || 'inside';
}

// The scale a drag of *handle* to *point* gives *box*: {sx, sy, anchor:
// [x, y], box}. An edge scales its axis only, a corner both; the opposite
// edge or corner stays, or with *fromCentre* the centre. *keepRatio* scales
// both axes alike: by the axis that changed most at a corner, about the
// middle of the other axis at an edge. The box keeps at least *minimum* in
// size and never turns inside out; an axis with no size does not scale.
export function resizeScale(box, handle, point, {keepRatio = false, fromCentre = false, minimum = 0} = {}) {
  const width = box.right - box.left, height = box.bottom - box.top;
  const dx = handle.includes('w') ? -1 : handle.includes('e') ? 1 : 0;
  const dy = handle.includes('n') ? -1 : handle.includes('s') ? 1 : 0;
  const anchor = (d, low, high) => fromCentre || !d ? (low + high) / 2 : d < 0 ? high : low;
  const ax = anchor(dx, box.left, box.right), ay = anchor(dy, box.top, box.bottom);
  const factor = (d, size, from, to) => {
    if (!d || !(size > 0)) return 1;
    const span = fromCentre ? size / 2 : size;
    return Math.max(minimum / size, d * (to - from) / span);
  };
  let sx = factor(dx, width, ax, point.x), sy = factor(dy, height, ay, point.y);
  if (keepRatio) {
    const s = dx && dy ? (Math.abs(sx - 1) >= Math.abs(sy - 1) ? sx : sy) : dx ? sx : sy;
    sx = width > 0 ? s : 1; sy = height > 0 ? s : 1;
  }
  return {sx, sy, anchor: [ax, ay], box: {left: ax + (box.left - ax) * sx, top: ay + (box.top - ay) * sy, right: ax + (box.right - ax) * sx, bottom: ay + (box.bottom - ay) * sy}};
}

// The nearest of *edges* within *radius* of *value*, or null.
export function nearestEdge(value, edges, radius) {
  let best = null;
  for (const edge of edges) if (Math.abs(edge - value) <= radius && (best === null || Math.abs(edge - value) < Math.abs(best - value))) best = edge;
  return best;
}
