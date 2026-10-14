// The shape tools draw native SVG shapes: a drag from its press to the
// pointer, both in the overlay's frame, is the rect, circle, ellipse or line it
// draws. Shift constrains (a square, a circle, a line at 45° steps) and Alt
// draws from the centre, alike in every shape tool. Pure, so it can be checked
// outside the browser.

export const DRAW_TOOLS = ['path', 'rect', 'ellipse', 'line', 'pencil'];

export const SHAPE_TOOLS = ['rect', 'ellipse', 'line'];

// What each tool's shapes are called, by tag.
const SHAPE_NAMES = {rect: 'Rectangle', circle: 'Circle', ellipse: 'Ellipse', line: 'Line'};

// A pointer position to the decimal place finer than a tenth of a screen
// pixel at *zoom*: a shape's numbers come out as round as the pointer can
// place them, 50 rather than 50.000001.
export function roundPoint(p, zoom) {
  const digits = Math.max(0, Math.ceil(Math.log10(10 * zoom)));
  return {x: Number(p.x.toFixed(digits)), y: Number(p.y.toFixed(digits))};
}

// The box a drag spans: square with *square*, centred on the press with
// *centre*.
export function shapeBox(start, end, {square = false, centre = false} = {}) {
  let dx = end.x - start.x, dy = end.y - start.y;
  if (square) {
    const side = Math.max(Math.abs(dx), Math.abs(dy));
    dx = dx < 0 ? -side : side; dy = dy < 0 ? -side : side;
  }
  if (centre) return {x: start.x - Math.abs(dx), y: start.y - Math.abs(dy), width: 2 * Math.abs(dx), height: 2 * Math.abs(dy)};
  return {x: Math.min(start.x, start.x + dx), y: Math.min(start.y, start.y + dy), width: Math.abs(dx), height: Math.abs(dy)};
}

// A line's ends: turned to the nearest 45° with *snap*, centred on the press
// with *centre*.
export function lineEnds(start, end, {snap = false, centre = false} = {}) {
  let dx = end.x - start.x, dy = end.y - start.y;
  if (snap) {
    const step = Math.PI / 4, angle = Math.round(Math.atan2(dy, dx) / step) * step, length = Math.hypot(dx, dy);
    // Level and upright lines come out exact, not a rounding error off.
    const exact = v => Math.abs(v) < 1e-12 * length ? 0 : v;
    dx = exact(Math.cos(angle) * length); dy = exact(Math.sin(angle) * length);
  }
  if (centre) return {x1: start.x - dx, y1: start.y - dy, x2: start.x + dx, y2: start.y + dy};
  return {x1: start.x, y1: start.y, x2: start.x + dx, y2: start.y + dy};
}

// The shape *tool* draws from *start* to *end*, as {tag, attributes}: a
// rectangle with corners rounded by *radius* (at most to half a side), an
// ellipse (a circle when its radii are equal) or a line.
export function shapeFrom(tool, start, end, {square = false, centre = false, radius = 0} = {}) {
  if (tool === 'line') return {tag: 'line', attributes: lineEnds(start, end, {snap: square, centre})};
  const box = shapeBox(start, end, {square, centre});
  if (tool === 'rect') {
    const rx = Math.min(radius, box.width / 2, box.height / 2);
    return {tag: 'rect', attributes: rx > 0 ? {...box, rx} : box};
  }
  const rx = box.width / 2, ry = box.height / 2, cx = box.x + rx, cy = box.y + ry;
  return rx === ry ? {tag: 'circle', attributes: {cx, cy, r: rx}} : {tag: 'ellipse', attributes: {cx, cy, rx, ry}};
}

// Whether a shape shows: a width and a height, or a line some length.
export function drawable({tag, attributes: a}) {
  if (tag === 'line') return a.x1 !== a.x2 || a.y1 !== a.y2;
  if (tag === 'rect') return a.width > 0 && a.height > 0;
  if (tag === 'circle') return a.r > 0;
  return a.rx > 0 && a.ry > 0;
}

// A shape's size as the tool strip shows it: a width and a height, or a
// line's length and its angle in degrees, counterclockwise from rightwards.
export function shapeSize({tag, attributes: a}) {
  if (tag === 'line') {
    const dx = a.x2 - a.x1, dy = a.y2 - a.y1;
    return {length: Math.hypot(dx, dy), angle: Math.atan2(-dy, dx) * 180 / Math.PI + 0};
  }
  if (tag === 'rect') return {width: a.width, height: a.height};
  if (tag === 'circle') return {width: 2 * a.r, height: 2 * a.r};
  return {width: 2 * a.rx, height: 2 * a.ry};
}

// Where a drag from *start* ends to draw *tool*'s shape at *size*: a click
// places one at the size in the tool strip, from its corner or, with
// *centre*, around it.
export function sizedEnd(tool, start, size, centre = false) {
  const share = centre ? 0.5 : 1;
  if (tool === 'line') {
    const angle = size.angle * Math.PI / 180;
    return {x: start.x + Math.cos(angle) * size.length * share, y: start.y - Math.sin(angle) * size.length * share};
  }
  return {x: start.x + size.width * share, y: start.y + size.height * share};
}

// A number as the strip and status line show it: to two decimals.
export const shown = value => String(Math.round(value * 100) / 100 + 0);

// What the status line says while a shape is dragged out.
export function shapeStatus(shape) {
  const size = shapeSize(shape);
  const measure = shape.tag === 'line' ? `${shown(size.length)} at ${shown(size.angle)}°` : `${shown(size.width)} × ${shown(size.height)}`;
  return `${SHAPE_NAMES[shape.tag]} · ${measure}`;
}
