import {SHAPES} from './clipping.js';

// Union, Subtract, Intersect and Exclude combine the filled areas of the
// selected paths. The backmost selected path is the target: it keeps its id,
// paint and place in the stacking order, and takes the result as its
// outline. The paths in front of it are the cutters, and go. Which is which
// follows the stacking order, never the order of selecting. app_booleans.js
// shows them on the canvas and sends the edit.
export const BOOLEANS = {
  union: {name: 'Union', verb: 'Uniting', keywords: 'boolean add join weld merge unite area together',
    general: 'Unite the selected filled paths into the backmost, which keeps its paint and place'},
  subtract: {name: 'Subtract', verb: 'Subtracting', keywords: 'boolean minus front difference cut remove punch knock out',
    general: 'Take the selected paths in front out of the backmost one, which keeps its paint and place'},
  intersect: {name: 'Intersect', verb: 'Intersecting', keywords: 'boolean overlap common shared area crop',
    general: 'Keep only where all the selected filled paths overlap, as the backmost, which keeps its paint and place'},
  exclude: {name: 'Exclude', verb: 'Excluding', keywords: 'boolean xor difference overlap remove shared area',
    general: 'Keep where the selected filled paths do not overlap, as the backmost, which keeps its paint and place'},
};

const LOCKS = ['geometry', 'structure'];

export const names = items => items.length < 2 ? items.map(item => item.label).join('')
  : `${items.slice(0, -1).map(item => item.label).join(', ')} and ${items.at(-1).label}`;

// The target and the cutters among *items*, in stacking order, back to front.
export const booleanOperands = items => ({target: items[0], cutters: items.slice(1)});

// Why these operations cannot combine *items*, or ''. *filled* tells whether
// an item paints a fill.
export function booleanRefusal(items, filled) {
  if (items.length < 2) return 'Select two or more filled paths';
  for (const item of items) {
    if (SHAPES.has(item.tag)) return `${item.label} is a basic shape: use Convert to path first`;
    if (item.tag !== 'path' || item.resource) return 'Select paths, not groups, instances or definitions';
    if (!filled(item)) return `${item.label} has no fill: these operations combine filled areas`;
    if (item.shared) return `${item.label} shares its geometry: use Detach first`;
    const lock = LOCKS.find(kind => item.inherited_locks?.includes(kind));
    if (lock) return `${item.label}: ${lock} is locked; unlock it first`;
  }
  return '';
}

// Where Exclude keeps the area of *items*.
const odd = items => items.length > 2 ? `an odd number of ${names(items)}` : `${names(items)} do not`;

// What *operation* does to *items*, naming the target and the cutters; in
// general terms without two or more of them.
export function booleanDescription(operation, items) {
  if (items.length < 2) return BOOLEANS[operation].general;
  const {target, cutters} = booleanOperands(items), front = names(cutters), all = names(items);
  const kept = `${target.label}, the backmost, keeps its paint and place`;
  return {
    union: `Unite ${front} with ${target.label}. ${kept}`,
    subtract: `Subtract ${front} from ${target.label}. ${kept}`,
    intersect: `Keep where ${all} ${items.length > 2 ? 'all ' : ''}overlap. ${kept}`,
    exclude: `Keep where ${odd(items)} overlap. ${kept}`,
  }[operation];
}

// What *operation* did, once it has.
export function booleanDone(operation, items) {
  const {target, cutters} = booleanOperands(items), front = names(cutters), all = names(items);
  return {
    union: `United ${front} with ${target.label}.`,
    subtract: `Subtracted ${front} from ${target.label}.`,
    intersect: `Kept where ${all} overlap, as ${target.label}.`,
    exclude: `Kept where ${odd(items)} overlap, as ${target.label}.`,
  }[operation] + ' Undo restores them.';
}
