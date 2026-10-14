// A linear gradient's endpoints as the canvas handles place them. Pure
// functions over attribute values and plain points, so they can be checked
// outside the browser; app_gradients.js reads the gradient and sends the edit.
//
// x1, y1, x2 and y2 are in the gradient's own coordinates: with
// gradientUnits objectBoundingBox, the default, fractions of the painted
// shape's bounds; with userSpaceOnUse, the shape's user space. Either is
// then mapped by the gradientTransform. A percentage is of the bounds, or in
// drawing coordinates of the viewport's width (x) or height (y).

// The attributes of each end, and the values SVG uses for missing ones.
export const ENDPOINTS = {start: ['x1', 'y1'], end: ['x2', 'y2']};
export const ENDPOINT_DEFAULTS = {x1: '0%', y1: '0%', x2: '100%', y2: '0%'};

export const unitsOf = value => value === 'userSpaceOnUse' ? 'userSpaceOnUse' : 'objectBoundingBox';

// An endpoint attribute's value in the gradient's coordinates, or NaN.
// *extent* is the viewport's size along the attribute's axis.
export function endpointNumber(text, units, extent) {
  const value = String(text ?? '').trim();
  if (!value.endsWith('%')) return value ? Number(value) : NaN;
  const fraction = Number(value.slice(0, -1)) / 100;
  return unitsOf(units) === 'userSpaceOnUse' ? fraction * extent : fraction;
}

// *value* as an endpoint attribute: a percentage of the bounds when the
// attribute was one, else a number, without trailing zeros.
export function endpointText(value, units, percent) {
  const tidy = (number, places) => String(+number.toFixed(places) + 0);
  if (unitsOf(units) === 'userSpaceOnUse') return tidy(value, 3);
  return percent ? `${tidy(value * 100, 3)}%` : tidy(value, 5);
}

// Whether the attribute *key* is written as a percentage, a missing one
// being its default.
export const isPercent = (attributes, key) => String(attributes[key] ?? ENDPOINT_DEFAULTS[key]).trim().endsWith('%');

// The matrix {a, b, c, d, e, f} from gradient coordinates, before the
// gradientTransform, to the shape's user space, given its bounds {x, y,
// width, height}; null for bounds with no area, which SVG paints nothing
// with.
export function unitsMatrix(units, box) {
  if (unitsOf(units) === 'userSpaceOnUse') return {a: 1, b: 0, c: 0, d: 1, e: 0, f: 0};
  if (!(box?.width > 0 && box?.height > 0)) return null;
  return {a: box.width, b: 0, c: 0, d: box.height, e: box.x, f: box.y};
}

// Shift turns a dragged end about the fixed one in steps of this many degrees.
export const ENDPOINT_STEP = 15;

// *point* turned about *fixed* to the nearest whole step, keeping its distance.
export function constrainEndpoint(fixed, point, step = ENDPOINT_STEP) {
  const dx = point.x - fixed.x, dy = point.y - fixed.y, length = Math.hypot(dx, dy);
  if (!length) return {x: point.x, y: point.y};
  const radians = step * Math.PI / 180, angle = Math.round(Math.atan2(dy, dx) / radians) * radians;
  return {x: fixed.x + Math.cos(angle) * length, y: fixed.y + Math.sin(angle) * length};
}

// The corners, edge middles and centre of the bounds {x, y, width, height},
// where a dragged end snaps.
export function boxPoints({x, y, width, height}) {
  const points = [];
  for (const u of [0, 0.5, 1]) for (const v of [0, 0.5, 1]) points.push({x: x + u * width, y: y + v * height});
  return points;
}
