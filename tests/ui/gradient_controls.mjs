import {createEditorState} from '../../src/vectrify/ui/static/app_state.js';
import {createGradientsController} from '../../src/vectrify/ui/static/app_gradients.js';
import {createColorsController} from '../../src/vectrify/ui/static/app_colors.js';
import {createPaintController} from '../../src/vectrify/ui/static/app_paint.js';
// Manual gradient controls work without a reference and use local shape bounds.
import assert from 'node:assert/strict';

const fields = new Map();
const field = id => {
  if (!fields.has(id)) fields.set(id, {value: '', style: {}, children: [], querySelector: () => ({}), replaceChildren() {this.children = [];}, append(child) {this.children.push(child);}});
  return fields.get(id);
};
const directions = ['horizontal', 'vertical', 'diagonal'].map(direction => ({dataset: {gradientDirection: direction}}));
const bounds = {x: 10, y: 20, width: 120, height: 60};
let item = {id: 'shape', tag: 'rect', inherited_locks: []};
let value = '#33669980';
const edits = [], queued = [], errors = [];
const context = Object.assign(createEditorState(), {
  $: field,
  document: {querySelectorAll: () => directions, createElement: () => ({children: [], setAttribute() {}, append(...children) {this.children.push(...children);}, querySelectorAll() {return this.children.slice(0, 3);}})},
  state: {selection: {objects: ['shape']}},
  oneObject: () => item,
  svgElement: () => ({getBBox: () => bounds}),
  resolvedPaint: () => value,
  paintValue: () => value,
  paintColour: colour => colour,
  cssColour: colour => colour,
  colorHex: colour => colour,
  action: (command, payload) => edits.push({command, ...JSON.parse(JSON.stringify(payload))}),
  paint: changes => edits.push({command: 'paint', changes: JSON.parse(JSON.stringify(changes))}),
  later: fn => queued.push(fn),
  toast: message => errors.push(message),
  renderInspector() {},
  gradientCss: () => null,
  paintGradient: () => null,
});
const context_gradients = createGradientsController(context, {
  $: (...args) => context.$(...args),
  action: (...args) => context.action(...args),
  colorHex: (...args) => context.colorHex(...args),
  cssColour: (...args) => context.cssColour(...args),
  document: context.document,
  gradientCss: (...args) => context.gradientCss(...args),
  later: (...args) => context.later(...args),
  oneObject: (...args) => context.oneObject(...args),
  paintColour: (...args) => context.paintColour(...args),
  paintGradient: (...args) => context.paintGradient(...args),
  paintValue: (...args) => context.paintValue(...args),
  renderInspector: (...args) => context.renderInspector(...args),
  resolvedPaint: (...args) => context.resolvedPaint(...args),
  svgElement: (...args) => context.svgElement(...args),
  toast: (...args) => context.toast(...args),
});
context_gradients.wire();
context.gradientRefusal = context_gradients.gradientRefusal;
context.gradientEndpoints = context_gradients.gradientEndpoints;
context.createFillGradient = context_gradients.createFillGradient;
context.setFillType = context_gradients.setFillType;
context.renderFillGradient = context_gradients.renderFillGradient;
context.fillGradientValue = context_gradients.fillGradientValue;
context.gradientEdit = context_gradients.gradientEdit;
context.saveGradient = context_gradients.saveGradient;
context.saveFillGradient = context_gradients.saveFillGradient;
// Selecting the fill type waits behind running edits before reading the shape.
field('fill-type').onchange({target: {value: 'linear'}});
assert.equal(edits.length, 0);
queued.shift()();
let fill = edits.at(-1).changes.fill;
assert.deepEqual(fill.start, [10, 50]);
assert.deepEqual(fill.end, [130, 50]);
assert.equal(fill.stops[0].colour, '#336699');
assert.equal(fill.stops[0].opacity, 128 / 255);
assert.equal(fill.stops[1].colour, '#ffffff');
assert.equal(fill.stops[1].opacity, 128 / 255);

// Direction presets preserve colours, offsets and opacity.
item.fill_gradient = {id: 'ramp', private: true};
field('gradient-units').value = 'userSpaceOnUse';
field('gradient-spread').value = 'pad';
['x1', 'y1', 'x2', 'y2'].forEach((key, i) => field('gradient-' + key).value = String([...fill.start, ...fill.end][i]));
const stops = [{offset: 0, colour: '#ff0000', opacity: 0}, {offset: 0.2, colour: '#00ff00', opacity: 0.4}, {offset: 1, colour: '#0000ff', opacity: 1}];
field('gradient-stops').children = stops.map(stop => ({querySelectorAll: () => [{value: 100 * stop.offset}, {value: stop.colour}, {value: 100 * stop.opacity}]}));
directions[1].onclick(); queued.shift()();
fill = edits.at(-1).gradient;
assert.deepEqual([fill.attributes.x1, fill.attributes.y1], ['70', '20']);
assert.deepEqual([fill.attributes.x2, fill.attributes.y2], ['70', '80']);
assert.deepEqual(fill.stops, stops);
directions[2].onclick(); queued.shift()();
assert.deepEqual([edits.at(-1).gradient.attributes.x2, edits.at(-1).gradient.attributes.y2], ['130', '80']);
field('gradient-reverse').onclick(); queued.shift()();
assert.deepEqual(edits.at(-1).gradient.stops, [
  {offset: 0, colour: '#0000ff', opacity: 1},
  {offset: 0.8, colour: '#00ff00', opacity: 0.4},
  {offset: 1, colour: '#ff0000', opacity: 0},
]);

// Zero-width shapes still get distinct endpoints; none starts with black.
bounds.width = 0; value = 'none';
(context.createFillGradient());
fill = edits.at(-1).changes.fill;
assert.equal(fill.end[0] - fill.start[0], 1);
assert.equal(fill.stops[0].colour, '#000000');
value = '#ffffff'; (context.createFillGradient());
assert.equal(edits.at(-1).changes.fill.stops[1].colour, '#000000');

// Groups, multiple selections and locked paint cannot create a gradient.
const count = edits.length;
for (const rejected of [null, {...item, tag: 'g'}, {...item, resource: true}, {...item, inherited_locks: ['paint']}]) {
  item = rejected;
  (context.createFillGradient());
}
assert.equal(edits.length, count);
assert.equal(errors.length, 4);
item = {id: 'shape', tag: 'path', inherited_locks: []};
(context.setFillType('none'));
assert.equal(edits.at(-1).changes.fill, 'none');
value = 'none'; (context.setFillType('solid'));
assert.equal(edits.at(-1).changes.fill, '#000000');

// Shared resources show the same editor, retain imported settings, and edit
// the existing server rather than making a private copy for the selected shape.
item = {id: 'ramp', tag: 'linearGradient', inherited_locks: [], fill_gradient: {
  id: 'ramp', private: false, users: 2,
  attributes: {gradientTransform: 'rotate(20)', spreadMethod: 'reflect'},
  stops: [{'offset': '0', 'stop-color': '#ff0000', 'stop-opacity': '0'}, {'offset': '100%', 'stop-color': '#0000ff'}],
}};
(context.renderFillGradient(context.oneObject()));
assert.equal(field('fill-gradient').hidden, false);
assert.equal(field('fill-controls').hidden, true);
assert.equal(field('stroke-controls').hidden, true);
assert.match(field('gradient-scope').textContent, /all 2 shapes/);
assert.equal(field('gradient-x2').value, '100%');
assert.equal(field('gradient-units').value, 'objectBoundingBox');
(context.saveFillGradient()); queued.shift()();
assert.equal(edits.at(-1).gradient.id, 'ramp');
assert.equal(edits.at(-1).gradient.attributes.x2, '100%');
assert.equal(edits.at(-1).gradient.attributes.gradientTransform, 'rotate(20)');
assert.equal(edits.at(-1).gradient.attributes.spreadMethod, 'reflect');
assert.equal(edits.at(-1).gradient.stops[0].opacity, 0);
directions[1].onclick(); queued.shift()();
assert.equal(edits.at(-1).gradient.attributes.x1, '0.5');
assert.equal(edits.at(-1).gradient.attributes.y2, '1');

// Imported stops are shown at SVG's effective offsets, clamped and ordered.
item.fill_gradient.stops = ['-10%', '80%', '20%', '120%'].map(offset => ({offset, 'stop-color': '#000000'}));
(context.renderFillGradient(context.oneObject()));
assert.deepEqual(field('gradient-stops').children.map(row=>row.children[0].value), [0, 80, 80, 100]);

// The preview combines stop opacity with the colour's alpha, including zero.
const previews = Object.assign(createEditorState(), {
  paintGradient: () => ({localName:'linearGradient', querySelectorAll: () => [
    {attributes: {offset: '0', 'stop-opacity': '0'}, colour: 'rgb(255, 0, 0)'},
    {attributes: {offset: '100%', 'stop-opacity': '0.5'}, colour: 'rgba(0, 0, 255, 0.5)'},
  ].map(stop => ({...stop, getAttribute: name => stop.attributes[name] ?? null}))}),
  getComputedStyle: stop => ({stopColor: stop.colour}),
});
const previews_colors = createColorsController(previews, {
});
previews.colorHex = previews_colors.colorHex;
previews.drawing = {querySelector: () => previews.paintGradient()};
previews.CSS = {escape: id => id};
const previews_paint = createPaintController(previews, {
  $: (...args) => previews.$(...args),
  CSS: previews.CSS,
  action: (...args) => previews.action(...args),
  colorHex: (...args) => previews.colorHex(...args),
  document: previews.document,
  drawing: previews.drawing,
  getComputedStyle: previews.getComputedStyle,
  later: (...args) => previews.later(...args),
  object: (...args) => previews.object(...args),
  oneObject: (...args) => previews.oneObject(...args),
  renderCommands: (...args) => previews.renderCommands(...args),
  renderFillGradient: (...args) => previews.renderFillGradient(...args),
  renderNodeInspector: (...args) => previews.renderNodeInspector(...args),
  renderStrokeControls: (...args) => previews.renderStrokeControls(...args),
  scheduleStrip: (...args) => previews.scheduleStrip(...args),
  selectObject: (...args) => previews.selectObject(...args),
  svgElement: (...args) => previews.svgElement(...args),
});
previews.gradientStops = previews_paint.gradientStops;
previews.gradientCss = previews_paint.gradientCss;
assert.equal((previews.gradientCss('url(#ramp)')), 'linear-gradient(90deg, #ff000000 0.0%, #0000ff40 100.0%)');
