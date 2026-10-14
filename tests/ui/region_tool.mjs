// The Region tool: drawing a rectangle or polygon, the preview of what
// Extract region and Delete region would change, and applying or cancelling
// it. Run by test_region_tool.py.
import assert from 'node:assert/strict';
import {closesPolygon, polygonArea, rectangleCorners, regionPreview} from '../../src/vectrify/ui/static/region.js';
import {createRegionController} from '../../src/vectrify/ui/static/app_region.js';
import {canvasGestures} from '../../src/vectrify/ui/static/gestures.js';

// A dragged rectangle has its corners clockwise from the top left, whichever
// way it was dragged; one without area is no region.
assert.deepEqual(rectangleCorners({x: 30, y: 40}, {x: 10, y: 5}), [[10, 5], [30, 5], [30, 40], [10, 40]]);
assert.equal(polygonArea(rectangleCorners({x: 0, y: 0}, {x: 10, y: 0})), 0);
assert.notEqual(polygonArea([[0, 0], [10, 0], [0, 10]]), 0);
// A click closes a polygon on its first corner, once it has three.
assert.equal(closesPolygon([[0, 0], [10, 0]], {x: 1, y: 1}, 2), false);
assert.equal(closesPolygon([[0, 0], [10, 0], [10, 10]], {x: 1, y: 1}, 2), true);
assert.equal(closesPolygon([[0, 0], [10, 0], [10, 10]], {x: 5, y: 5}, 2), false);

// The preview's words: what each operation changes, whether crossing objects
// are cut, what it leaves, or why it is refused.
const names = {ink: 'Ink', face: 'Face', dot: 'Dot', line: 'Line'};
const label = id => names[id];
const preview = {
  crossing: ['ink', 'line'],
  extract: {targets: [{id: 'ink', outcome: 'cut'}, {id: 'face', outcome: 'contours'}], error: null},
  delete: {targets: [{id: 'ink', outcome: 'cut'}, {id: 'dot', outcome: 'whole'}], error: null},
};
let shown = regionPreview(preview, 'extract', true, label);
assert.equal(shown.summary, 'Extracts from 2 objects, cutting 1 along the edge.');
assert.deepEqual(shown.rows.map(row => [row.label, row.outcome]), [['Ink', 'cut'], ['Face', 'contours'], ['Dot', 'inside']]);
assert.match(shown.rows[0].detail, /^Cut along the edge: the part inside becomes a path of its own$/);
assert.match(shown.rows[2].detail, /wholly inside: stays as it is/);
shown = regionPreview(preview, 'delete', true, label);
assert.equal(shown.summary, 'Deletes from 2 objects, cutting 1 along the edge.');
assert.deepEqual(shown.rows.map(row => [row.label, row.outcome]), [['Ink', 'cut'], ['Dot', 'whole']]);
// Cutting off: the objects crossing the edge are said to stay whole.
const uncut = {...preview, extract: {targets: [{id: 'face', outcome: 'contours'}, {id: 'ink', outcome: 'contours'}], error: null}};
shown = regionPreview(uncut, 'extract', false, label);
assert.equal(shown.summary, 'Extracts from 2 objects; nothing is cut. 2 objects crossing the edge are not cut.');
assert.match(shown.rows.find(row => row.id === 'ink').detail, /its contours crossing the edge stay$/);
assert.deepEqual(shown.rows.find(row => row.id === 'line'), {id: 'line', label: 'Line', outcome: 'uncut', detail: 'Crosses the edge: stays whole, as cutting is off'});
shown = regionPreview({...preview, delete: {targets: [], error: 'Unpin the points before deleting them'}}, 'delete', true, label);
assert.deepEqual(shown, {summary: 'Unpin the points before deleting them', rows: [], refusal: 'Unpin the points before deleting them'});

// The controller, with just enough DOM for its panel.
class Node {
  constructor(tag = '', attributes = {}) {
    this.tag = tag; this.children = []; this.dataset = {}; this.attributes = new Map(Object.entries(attributes)); this.listeners = new Map();
    this.hidden = false; this.disabled = false; this.checked = false; this.title = ''; this.ownText = ''; this.className = ''; this.value = '';
    const classes = new Set();
    this.classList = {toggle: (name, on) => on ? classes.add(name) : classes.delete(name), contains: name => classes.has(name)};
  }
  get textContent() { return this.ownText + this.children.map(child => child.textContent).join(''); }
  set textContent(value) { this.ownText = String(value); this.children = []; }
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  getAttribute(name) { return this.attributes.get(name) ?? null; }
  removeAttribute(name) { this.attributes.delete(name); }
  append(...children) { this.children.push(...children.flatMap(child => child.fragment ? child.children : [child])); }
  replaceChildren(...children) { this.children = []; this.append(...children); }
  addEventListener(name, handler) { this.listeners.set(name, [...(this.listeners.get(name) || []), handler]); }
  fire(name, event = {}) { const sent = {preventDefault() { sent.prevented = true; }, stopPropagation() { sent.stopped = true; }, ...event}; for (const handler of this.listeners.get(name) || []) handler(sent); return sent; }
  all(test) { return this.children.flatMap(child => [...(test(child) ? [child] : []), ...child.all(test)]); }
  focus() {}
}
const ids = new Map();
const $ = id => { if (!ids.has(id)) { const node = new Node('div'); node.id = id; ids.set(id, node); } return ids.get(id); };
const control = (attributes, fields) => Object.assign(new Node('button'), fields, {dataset: attributes});
const modes = [control({regionMode: 'extract'}), control({regionMode: 'delete'})];
const shapes = [control({regionShape: 'rectangle'}), control({regionShape: 'polygon'})];
const scopes = [Object.assign(new Node('input'), {value: 'selection'}), Object.assign(new Node('input'), {value: 'region'})];
const document = {
  createElement: tag => new Node(tag),
  createDocumentFragment: () => Object.assign(new Node(''), {fragment: true}),
  querySelector: () => null,
  querySelectorAll: selector => selector === '[data-region-mode]' ? modes : selector === '[data-region-shape]' ? shapes : selector === 'input[name=region-scope]' ? scopes : [],
};

const objects = {ink: {label: 'Ink'}, face: {label: 'Face'}, dot: {label: 'Dot'}, line: {label: 'Line'}};
const sent = [], asked = [], toasts = [];
let answer = preview, succeed = true, hold = null;
const editor = {
  tool: 'region', zoom: 2, scope: null, drag: null, region: null, regionHover: null, overlayFrame: {content: new Node('g')},
  state: {epoch: 'e', revision: 4, selection: {objects: ['ink', 'face'], nodes: []}},
};
const region = createRegionController(editor, {
  $, document, later: run => run(), stage: new Node('div'),
  action: async (command, data) => { sent.push({command, data}); if (succeed) editor.state = {...editor.state, revision: editor.state.revision + 1}; return succeed; },
  drawOverlay: () => { editor.overlayFrame.content = new Node('g'); region.drawRegion(); },
  localToOverlay: () => ({toString: () => 'matrix(1 0 0 1 0 0)'}), object: id => objects[id],
  plural: (n, one) => `${n} ${one}${n === 1 ? '' : 's'}`, renderCommands() {},
  request: async (path, body) => { asked.push({path, body}); await hold; return {epoch: 'e', revision: body.revision, ...answer}; },
  runCommand: id => region.command(id === 'delete-region' ? 'delete' : 'extract'),
  selectionContour: element => new Node('path', {id: element.id}), setTool: async tool => { editor.tool = tool; },
  svgElement: id => objects[id] ? Object.assign(new Node('path'), {id}) : null,
  toast: (message, error = false) => toasts.push({message, error}),
  xmlElement: (tag, attributes = {}) => new Node(tag, attributes),
}).wire();
const settle = () => new Promise(resolve => setTimeout(resolve));
const drawn = () => editor.overlayFrame.content.all(node => true);
const area = () => drawn().find(node => /region-area/.test(node.getAttribute('class')));
const targets = () => drawn().filter(node => /region-target/.test(node.getAttribute('class'))).map(node => [node.children[0].getAttribute('id'), node.getAttribute('class')]);

// Before a region: the panel says how to draw one and nothing can apply.
region.drawRegion();
assert.match($('region-summary').textContent, /^Drag a rectangle/);
assert.equal($('region-apply').disabled, true);
assert.equal(region.refusal('extract'), '');
assert.equal(asked.length, 0);

// Dragging draws the rectangle; a click instead selects, as in Select.
const picked = [];
const gestures = canvasGestures({
  point: event => ({x: event.clientX / 2, y: event.clientY / 2}), drawOverlay: () => editor.overlayFrame && region.drawRegion(),
  selection: {pick: gesture => picked.push(gesture)}, region: {rectangle: region.rectangle, corner: region.corner},
});
const click = {moved: false};
gestures.region.press(click, {clientX: 20, clientY: 20});
await gestures.region.release(click);
assert.equal(picked.length, 1);
assert.equal(editor.region, null);
const drag = {moved: false};
editor.drag = Object.assign(drag, {kind: 'region'});
gestures.region.press(drag, {clientX: 20, clientY: 20});
drag.moved = true;
gestures.region.move(drag, {clientX: 100, clientY: 60});
assert.equal(area().getAttribute('points'), '10,10 50,10 50,30 10,30', 'the rectangle shows while dragged');
editor.drag = null;
let answered;
hold = new Promise(resolve => { answered = resolve; });
await gestures.region.release(drag);
assert.deepEqual(editor.region, {points: [[10, 10], [50, 10], [50, 30], [10, 30]], closed: true});

// The region previews at once: the session is asked, and until it answers
// nothing can apply.
assert.deepEqual(asked.at(-1), {path: '/api/region', body: {epoch: 'e', revision: 4, region: [[10, 10], [50, 10], [50, 30], [10, 30]], scope: 'selection', cut: true, within: null}});
assert.equal($('region-summary').textContent, 'Previewing the region…');
assert.equal(region.refusal('extract'), 'Previewing the region…');
hold = null; answered();
await settle();
// Then the panel and the canvas show the targets, and which are cut.
assert.equal($('region-summary').textContent, 'Extracts from 2 objects, cutting 1 along the edge.');
assert.deepEqual($('region-targets').children.map(row => [row.dataset.object, row.className]), [['ink', 'region-row cut'], ['face', 'region-row contours'], ['dot', 'region-row inside']]);
assert.equal($('region-targets').hidden, false);
assert.deepEqual(targets(), [['ink', 'region-target cut'], ['face', 'region-target contours'], ['dot', 'region-target inside']]);
assert.equal($('region-apply').disabled, false);
assert.equal($('region-apply').textContent, 'Extract region');
assert.equal(sent.length, 0, 'nothing changes before applying');

// Delete shows its own preview without asking again.
modes[1].onclick();
assert.equal(modes[1].getAttribute('aria-pressed'), 'true');
assert.equal($('region-summary').textContent, 'Deletes from 2 objects, cutting 1 along the edge.');
assert.deepEqual(targets(), [['ink', 'region-target cut'], ['dot', 'region-target whole']]);
assert.equal($('region-apply').textContent, 'Delete region');
assert.equal(asked.length, 1);

// Turning cutting off and acting on everything there asks again, with them.
answer = {...uncut, delete: {targets: [], error: 'Unpin the points before deleting them'}};
$('region-cut').checked = false; $('region-cut').onchange();
scopes[1].onchange();
await settle();
assert.deepEqual(asked.at(-1).body, {epoch: 'e', revision: 4, region: [[10, 10], [50, 10], [50, 30], [10, 30]], scope: 'region', cut: false, within: null});
// A refused preview says why, and the operation cannot apply.
assert.equal($('region-summary').textContent, 'Unpin the points before deleting them');
assert.equal($('region-summary').classList.contains('refused'), true);
assert.equal($('region-apply').disabled, true);
assert.equal(region.refusal('delete'), 'Unpin the points before deleting them');
await region.apply();
assert.deepEqual(toasts.at(-1), {message: 'Unpin the points before deleting them', error: true});
assert.equal(sent.length, 0);
modes[0].onclick();
assert.match($('region-summary').textContent, /2 objects crossing the edge are not cut\.$/);
assert.deepEqual(targets().find(([id]) => id === 'line'), ['line', 'region-target uncut']);

// A change to the drawing makes the preview stale until asked again.
editor.state = {...editor.state, revision: 5};
assert.equal(region.refusal('extract'), 'Previewing the region…');
region.drawRegion();
await settle();
assert.equal(asked.at(-1).body.revision, 5);
assert.equal(region.refusal('extract'), '');

// Enter applies the shown operation as one extract command.
region.enter();
await settle();
assert.deepEqual(sent, [{command: 'extract', data: {region: [[10, 10], [50, 10], [50, 30], [10, 30]], scope: 'region', cut: false, within: null, delete: false}}]);
assert.equal(editor.region, null);
assert.match(toasts.at(-1).message, /^Extracted the region from 2 objects into paths of their own/);

// A polygon: clicks place corners, the first corner closes it; Backspace
// takes one back and Escape (cancel) leaves the drawing as it was.
shapes[1].onclick();
assert.equal(shapes[1].getAttribute('aria-pressed'), 'true');
assert.equal(region.options.shape, 'polygon');
for (const [x, y] of [[0, 0], [40, 0], [40, 40]]) {
  const corner = {};
  gestures.regionCorner.press(corner, {clientX: x * 2, clientY: y * 2});
  gestures.regionCorner.release(corner);
}
assert.deepEqual(editor.region, {points: [[0, 0], [40, 0], [40, 40]], closed: false});
assert.match($('region-summary').textContent, /^3 corners placed/);
assert.equal(region.back(), true);
assert.deepEqual(editor.region.points, [[0, 0], [40, 0]]);
region.corner({x: 20, y: 30});
region.corner({x: 1, y: 1});
assert.equal(editor.region.closed, true, 'a click on the first corner closes it');
assert.equal(area().tag, 'polygon');
await settle();
assert.deepEqual(asked.at(-1).body.region, [[0, 0], [40, 0], [20, 30]]);
const before = sent.length;
$('region-panel').fire('keydown', {key: 'Escape'});
assert.equal(editor.region, null);
assert.equal(sent.length, before, 'cancelling sends nothing');
// Enter closes a polygon being drawn rather than applying anything.
for (const p of [{x: 0, y: 0}, {x: 30, y: 0}, {x: 0, y: 30}]) region.corner(p);
region.enter();
assert.equal(editor.region.closed, true);
assert.equal(sent.length, before);
// Keys pressing the panel's own controls are not the canvas's.
assert.equal($('region-panel').fire('keydown', {key: 'Enter'}).stopped, true);
assert.equal($('region-panel').fire('pointerdown').stopped, true);

// A refused edit keeps the region to adjust.
await settle();
succeed = false;
await region.apply();
assert.equal(sent.length, before + 1);
assert.equal(editor.region.closed, true);
