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
