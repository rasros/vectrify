// Union, Subtract, Intersect and Exclude in the browser: which selected path
// is the target and which are cutters, what the tooltips say before anything
// changes, the outlines drawn while a command is hovered, and the refusals.
// Run by test_boolean_commands.py.
import assert from 'node:assert/strict';
import {BOOLEANS, booleanDescription, booleanOperands, booleanRefusal} from '../../src/vectrify/ui/static/booleans.js';
import {createBooleansController} from '../../src/vectrify/ui/static/app_booleans.js';

const path = (id, label, extra = {}) => ({id, label, tag: 'path', locks: [], inherited_locks: [], ...extra});
const back = path('back', 'Back'), middle = path('middle', 'Middle'), front = path('front', 'Front');
const filled = () => true;

// The target is the first in stacking order; the rest cut.
assert.deepEqual(booleanOperands([back, middle, front]), {target: back, cutters: [middle, front]});
assert.deepEqual(Object.keys(BOOLEANS), ['union', 'subtract', 'intersect', 'exclude']);

// Each tooltip names the target, the cutters and the paint source.
assert.equal(booleanDescription('subtract', [back, front]), 'Subtract Front from Back. Back, the backmost, keeps its paint and place');
assert.equal(booleanDescription('subtract', [back, middle, front]), 'Subtract Middle and Front from Back. Back, the backmost, keeps its paint and place');
assert.equal(booleanDescription('union', [back, front]), 'Unite Front with Back. Back, the backmost, keeps its paint and place');
assert.equal(booleanDescription('intersect', [back, middle, front]), 'Keep where Back, Middle and Front all overlap. Back, the backmost, keeps its paint and place');
assert.equal(booleanDescription('exclude', [back, front]), 'Keep where Back and Front do not overlap. Back, the backmost, keeps its paint and place');
assert.equal(booleanDescription('exclude', [back, middle, front]), 'Keep where an odd number of Back, Middle and Front overlap. Back, the backmost, keeps its paint and place');
assert.equal(booleanDescription('subtract', []), BOOLEANS.subtract.general);

// Refusals: too few, basic shapes (pointing to Convert to path), groups,
// fills, shared geometry and locks.
assert.equal(booleanRefusal([back], filled), 'Select two or more filled paths');
assert.equal(booleanRefusal([back, front], filled), '');
assert.equal(booleanRefusal([back, {...front, tag: 'rect', label: 'Rect 1'}], filled), 'Rect 1 is a basic shape: use Convert to path first');
assert.equal(booleanRefusal([back, {...front, tag: 'g'}], filled), 'Select paths, not groups, instances or definitions');
assert.equal(booleanRefusal([back, {...front, resource: true}], filled), 'Select paths, not groups, instances or definitions');
assert.equal(booleanRefusal([back, front], item => item !== front), 'Front has no fill: these operations combine filled areas');
assert.equal(booleanRefusal([back, {...front, shared: true}], filled), 'Front shares its geometry: use Detach first');
assert.equal(booleanRefusal([{...back, inherited_locks: ['structure']}, front], filled), 'Back: structure is locked; unlock it first');

// The controller: the stacking order decides, whatever order the selection is in.
class Node {
  constructor(tag) { this.tag = tag; this.children = []; this.attributes = new Map(); this.dataset = {}; this.parent = null; }
  append(...children) { for (const child of children) { child.parent = this; this.children.push(child); } }
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  removeAttribute(name) { this.attributes.delete(name); }
  closest(selector) {
    for (let node = this; node; node = node.parent) if (selector === '[data-command]' && node.dataset.command) return node;
    return null;
  }
}
const listeners = new Map();
const document = {addEventListener: (name, handler) => listeners.set(name, [...(listeners.get(name) || []), handler])};
const fire = (name, target) => { for (const handler of listeners.get(name) || []) handler({target}); };
const objects = [back, path('other', 'Other'), front];
const editor = {state: {objects, selection: {objects: ['front', 'back']}}, overlayFrame: {content: new Node('g')}};
const actions = [], toasts = [];
let draws = 0;
const booleans = createBooleansController(editor, {
  action: async (...args) => { actions.push(args); return true; },
  document,
  drawOverlay: () => { draws++; editor.overlayFrame.content = new Node('g'); booleans.drawRoles(); },
  filledPath: () => true,
  localToOverlay: element => ({toString: () => `matrix(${element.id})`}),
  noSelection: () => !editor.state.selection.objects.length && 'Select objects first',
  selectionContour: element => Object.assign(new Node('path'), {source: element.id}),
  svgElement: id => ({id}),
  toast: message => toasts.push(message),
  xmlElement: (tag, attributes) => { const node = new Node(tag); for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value); return node; },
}).wire();
assert.deepEqual(booleans.selectedObjects().map(item => item.id), ['back', 'front']);
assert.equal(booleans.refusal(), '');
assert.equal(booleans.description('subtract'), 'Subtract Front from Back. Back, the backmost, keeps its paint and place');

// Hovering a command outlines the target and the cutter; leaving clears them.
const button = new Node('button'); button.dataset.command = 'subtract';
const label = new Node('span'); button.append(label);
fire('pointerover', label);
const roles = () => editor.overlayFrame.content.children.map(group => [group.attributes.get('class'), group.children[0].source, group.attributes.get('transform')]);
assert.deepEqual(roles(), [['boolean-target', 'back', 'matrix(back)'], ['boolean-cutter', 'front', 'matrix(front)']]);
fire('pointerover', label);
assert.equal(draws, 1, 'moving within the button draws nothing new');
fire('pointerout', button);
assert.deepEqual(roles(), []);
fire('focusin', button);
assert.equal(roles().length, 2, 'keyboard focus shows the operands too');
fire('pointerdown', button);
assert.deepEqual(roles(), [], 'pressing ends the preview');
const other = new Node('button'); other.dataset.command = 'cut-hole';
fire('pointerover', other);
assert.deepEqual(roles(), [], 'other commands outline nothing');

// A selection the commands refuse shows no operands.
fire('pointerover', button);
editor.state.selection.objects = ['back'];
editor.overlayFrame.content = new Node('g'); booleans.drawRoles();
assert.deepEqual(roles(), []);
assert.equal(booleans.description('union'), BOOLEANS.union.general);
fire('pointerout', button);

// Running sends the operation by name and says what it did.
editor.state.selection.objects = ['front', 'back'];
await booleans.run('subtract');
assert.deepEqual(actions, [['subtract', {}, 'Subtracting…']]);
assert.deepEqual(toasts, ['Subtracted Front from Back. Undo restores them.']);
