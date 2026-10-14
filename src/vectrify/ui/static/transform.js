// The Transform section's numeric position and size. Pure functions over
// plain boxes ({left, top, right, bottom}, y down), so they can be checked
// outside the browser; app_transform.js reads the box and sends the edit.
//
// The box is the selection frame, in document coordinates: the geometric
// bounds of the selected objects, their outlines without stroke width,
// markers or clipping. X and Y are its top-left corner and W and H its size.
// Moving keeps the size; resizing keeps the top-left corner where it is.

// A side shorter than this, in document units, has no size: a horizontal
// line has no height. It shows as 0 and cannot be scaled, and no edit may
// make a side that has a size any shorter.
export const MIN_SIZE = 0.001;

export const FIELD_NAMES = {x: 'X', y: 'Y', w: 'Width', h: 'Height'};

// The four fields' values for *box*.
export function boxFields(box) {
  const size = value => value < MIN_SIZE ? 0 : value;
  return {x: box.left, y: box.top, w: size(box.right - box.left), h: size(box.bottom - box.top)};
}

// A field's value as shown: at most three decimals, without trailing zeros.
export const formatLength = value => String(+value.toFixed(3) + 0);

// Why W or H cannot be edited for these *values*, or ''.
export function sizeRefusal(values, field) {
  return values[field] ? '' : `The selection has no ${FIELD_NAMES[field].toLowerCase()} to scale: it is a line or a point along that axis`;
}

// The edit that sets *field* of *box* to *value*: {command: 'move', data:
// {dx, dy}} for X and Y, {command: 'resize', data: {anchor, scale}} for W and
// H; {refusal} when it cannot be made, or null when nothing changes. With
// *keepRatio*, W and H scale together; a side with no size stays without.
export function fieldEdit(box, field, value, {keepRatio = false} = {}) {
  const name = FIELD_NAMES[field];
  if (!name) throw new Error(`Unknown transform field: ${field}`);
  if (!Number.isFinite(value)) return {refusal: `${name} needs a number`};
  const current = boxFields(box);
  if (field === 'x' || field === 'y') {
    const delta = value - current[field];
    if (Math.abs(delta) < 1e-9) return null;
    return {command: 'move', data: {dx: field === 'x' ? delta : 0, dy: field === 'y' ? delta : 0}};
  }
  const size = current[field], flat = sizeRefusal(current, field);
  if (flat) return {refusal: flat};
  if (value < MIN_SIZE) return {refusal: `${name} must be at least ${MIN_SIZE}`};
  const factor = value / size;
  if (!Number.isFinite(factor)) return {refusal: `${name} is too large`};
  if (Math.abs(factor - 1) < 1e-12) return null;
  const other = field === 'w' ? 'h' : 'w', scale = {w: 1, h: 1, [field]: factor};
  if (keepRatio && current[other]) {
    if (current[other] * factor < MIN_SIZE) return {refusal: `${FIELD_NAMES[other]} would be less than ${MIN_SIZE}`};
    scale[other] = factor;
  }
  return {command: 'resize', data: {anchor: [box.left, box.top], scale: [scale.w, scale.h]}};
}

// Rotation, in degrees clockwise on the page as SVG's rotate() turns (y is
// down). Shift turns a rotation drag in steps of this many degrees.
export const ROTATION_STEP = 15;

// *degrees* as the same turn in (-180, 180].
export function normalAngle(degrees) {
  const turn = ((degrees % 360) + 360) % 360;
  return turn > 180 ? turn - 360 : turn;
}

// The angle a matrix {a, b, c, d} turns its object by. A mirrored object
// is measured as flipped horizontally first, so a horizontal flip leaves its
// angle as it was (and a vertical flip, a horizontal one turned half round,
// adds 180°).
export function matrixAngle({a, b, c, d}) {
  const mirrored = a * d - b * c < 0;
  const angle = Math.atan2(mirrored ? -b : b, mirrored ? -a : a) * 180 / Math.PI;
  return normalAngle(+angle.toFixed(9));
}

// The angle shared by the selected objects' matrices, or null when they are
// turned differently.
export function sharedAngle(matrices) {
  const angles = matrices.map(matrixAngle);
  if (!angles.length) return null;
  return angles.every(angle => Math.abs(normalAngle(angle - angles[0])) < 1e-6) ? angles[0] : null;
}

// The turn that sets the Rotation field to *value*: {angle} in degrees, the
// difference from the *current* angle, or *value* itself when the objects
// are turned differently (current is null); {refusal} for no number, or
// null when nothing turns.
export function rotationEdit(current, value) {
  if (!Number.isFinite(value)) return {refusal: 'Rotation needs a number of degrees'};
  const angle = normalAngle(value - (current ?? 0));
  return Math.abs(angle) < 1e-9 ? null : {angle};
}

// How far a drag from *grab* to *point* turns about *pivot*, in degrees; with
// *snap*, in whole steps of ROTATION_STEP.
export function dragAngle(pivot, grab, point, {snap = false} = {}) {
  const at = p => Math.atan2(p.y - pivot.y, p.x - pivot.x);
  const angle = normalAngle((at(point) - at(grab)) * 180 / Math.PI);
  return snap ? normalAngle(Math.round(angle / ROTATION_STEP) * ROTATION_STEP) : angle;
}

// *point* turned by *degrees* about *pivot*.
export function turnPoint(point, pivot, degrees) {
  const radians = degrees * Math.PI / 180, cos = Math.cos(radians), sin = Math.sin(radians);
  const x = point.x - pivot.x, y = point.y - pivot.y;
  return {x: pivot.x + x * cos - y * sin, y: pivot.y + x * sin + y * cos};
}

// The corners of *box*, turned by *degrees* about *pivot*, clockwise from
// the top-left one.
export function turnedBox(box, pivot, degrees) {
  return [[box.left, box.top], [box.right, box.top], [box.right, box.bottom], [box.left, box.bottom]]
    .map(([x, y]) => turnPoint({x, y}, pivot, degrees));
}
