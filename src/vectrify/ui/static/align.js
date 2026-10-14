// Align and Distribute in Arrange. Pure functions over plain boxes ({left,
// top, right, bottom}, y down), so they can be checked outside the browser;
// app_align.js measures the boxes and sends the moves.
//
// Each box is one top selected object's geometric bounds in document
// coordinates, the same bounds as the selection frame and the Transform
// fields: its outline without stroke width, markers or clipping, mapped
// through its transforms. A move is an offset in document coordinates; only
// positions change, never sizes.

// Offsets shorter than this, in document units, are no move.
const EPSILON = 1e-9;

// What each alignment lines up: the axis, and where on the box.
export const ALIGNMENTS = {
  left: {axis: 'x', at: 0, name: 'Align left', edge: 'left edges'},
  centre: {axis: 'x', at: .5, name: 'Align centre', edge: 'horizontal centres'},
  right: {axis: 'x', at: 1, name: 'Align right', edge: 'right edges'},
  top: {axis: 'y', at: 0, name: 'Align top', edge: 'top edges'},
  middle: {axis: 'y', at: .5, name: 'Align middle', edge: 'vertical centres'},
  bottom: {axis: 'y', at: 1, name: 'Align bottom', edge: 'bottom edges'},
};

// Equal gaps put the same space between neighbours' edges; equal centres put
// their centres the same distance apart.
export const SPACINGS = {
  gaps: {name: 'equal gaps', description: 'the same space between neighbouring objects'},
  centres: {name: 'equal centres', description: 'their centres the same distance apart'},
};

export const REFERENCES = {
  selection: 'the selection frame',
  artboard: 'the artboard',
  key: 'the key object',
};

const DIRECTIONS = {x: 'horizontally', y: 'vertically'};

export const distributeName = (axis, spacing) => `Distribute ${DIRECTIONS[axis]}: ${SPACINGS[spacing].name}`;

// A box's extent along *axis*: [start, end].
const span = (box, axis) => axis === 'x' ? [box.left, box.right] : [box.top, box.bottom];

// The box around *boxes*.
export function unionBox(boxes) {
  return {
    left: Math.min(...boxes.map(b => b.left)), top: Math.min(...boxes.map(b => b.top)),
    right: Math.max(...boxes.map(b => b.right)), bottom: Math.max(...boxes.map(b => b.bottom)),
  };
}

// The moves as {id: [dx, dy]}, leaving out objects that stay.
function movesOf(axis, offsets) {
  const moves = {};
  for (const [id, offset] of offsets) {
    if (Math.abs(offset) < EPSILON) continue;
    moves[id] = axis === 'x' ? [offset, 0] : [0, offset];
  }
  return moves;
}

// The moves that line up *items* ([{id, box}]) to *target*, a box: the
// selection frame, the artboard or the key object's bounds. An item whose id
// is *fixed* (the key object) stays.
export function alignMoves(items, alignment, target, {fixed = null} = {}) {
  const {axis, at} = ALIGNMENTS[alignment];
  const point = box => { const [start, end] = span(box, axis); return start + (end - start) * at; };
  const goal = point(target);
  return movesOf(axis, items.filter(item => item.id !== fixed).map(item => [item.id, goal - point(item.box)]));
}

// The moves that space *items* evenly along *axis*. They keep their order by
// centre, and the first and last stay where they are: the others move between
// them, with equal gaps between neighbours' edges or equal steps between
// centres. A gap may be negative when the objects are wider than the room
// between the outer two; they then overlap evenly.
export function distributeMoves(items, axis, spacing) {
  if (items.length < 3) return {};
  const centre = box => { const [start, end] = span(box, axis); return (start + end) / 2; };
  const order = distributeOrder(items, axis);
  const first = order[0].box, last = order.at(-1).box, inner = order.slice(1, -1), offsets = [];
  if (spacing === 'centres') {
    const step = (centre(last) - centre(first)) / (order.length - 1);
    inner.forEach((item, i) => offsets.push([item.id, centre(first) + step * (i + 1) - centre(item.box)]));
  } else {
    const size = box => { const [start, end] = span(box, axis); return end - start; };
    const room = span(last, axis)[0] - span(first, axis)[1] - inner.reduce((sum, item) => sum + size(item.box), 0);
    const gap = room / (order.length - 1);
    let next = span(first, axis)[1] + gap;
    for (const item of inner) {
      offsets.push([item.id, next - span(item.box, axis)[0]]);
      next += size(item.box) + gap;
    }
  }
  return movesOf(axis, offsets);
}

// *items* in the order Distribute keeps along *axis*: by centre, then by
// starting edge. The first and last stay in place.
export function distributeOrder(items, axis) {
  const centre = box => { const [start, end] = span(box, axis); return (start + end) / 2; };
  return [...items].sort((a, b) => centre(a.box) - centre(b.box) || span(a.box, axis)[0] - span(b.box, axis)[0]);
}
