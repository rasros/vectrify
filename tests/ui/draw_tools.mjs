import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createEditorState} from '../../src/vectrify/ui/static/app_state.js';
import {createViewController} from '../../src/vectrify/ui/static/app_view.js';
import {createCanvasEventsController} from '../../src/vectrify/ui/static/app_canvas_events.js';
import {wireToolMenuKeys} from '../../src/vectrify/ui/static/tool_menu.js';
import {DRAW_TOOLS} from '../../src/vectrify/ui/static/shapes.js';
import {TOOL_LEVEL} from '../../src/vectrify/ui/static/selection.js';

class Element {
  constructor() {
    this.dataset = {}; this.style = {}; this.attributes = {};
    this.classList = {toggle: (key, value) => { this[key] = value; }};
  }
  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener() {}
}
const elements = new Map();
const $ = id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
const buttons = ['select', 'move', 'nodes', 'draw'].map(tool => {
  const button = $(`rail-${tool}`); button.dataset.tool = tool;
  if (tool === 'draw') button.dataset.secondary = 'true';
  return button;
});
const document = {querySelectorAll: selector => selector === '.tool[data-tool]' ? buttons : []};
const editor = createEditorState();
editor.state = {selection: {objects: [], nodes: []}};
const view = createViewController(editor, {
  $, document, stage: $('stage'), selectedPoints: () => [], level: () => TOOL_LEVEL[editor.tool],
  reportView() {}, renderInspector() {}, drawOverlay() {}, loadGeometries: async () => {}, setBusy() {},
});
const queued = [];
createCanvasEventsController(editor, {
  $, document, stage: $('stage'), ResizeObserver: class { observe() {} },
  later: fn => queued.push(fn), setTool: view.setTool,
}).wire();

// One entry activates Path first, then restores the last drawing type.
buttons[3].onclick();
await queued.shift()();
assert.equal(editor.tool, 'path');
assert.equal(buttons[3].attributes['aria-checked'], 'true');
assert.equal($('draw-type').value, 'path');
assert.equal($('tool-name').textContent, 'Draw');
for (const type of DRAW_TOOLS) {
  $('draw-type').value = type;
  $('draw-type').onchange({target: $('draw-type')});
  // Changing the field again must not change the already queued choice.
  $('draw-type').value = 'other';
  await queued.shift()();
  assert.equal(editor.tool, type);
  assert.equal(editor.drawTool, type);
  assert.equal($('draw-type').value, type);
  assert.match($('draw-type-icon').innerHTML, /<svg/);
  assert.equal(buttons[3].active, true);
  assert.equal($('general-tools').active, true);
  assert.equal($('general-tools-menu').hidden, true);
  await view.setTool('select');
  assert.equal(buttons[3].active, false);
  buttons[3].onclick(); await queued.shift()();
  assert.equal(editor.tool, type, 'Draw restores the last type');
}
// Shortcut and palette routes use the same tool setter and sync the picker.
await view.setTool('line');
assert.equal($('draw-type').value, 'line');
await view.setTool('select'); await view.setTool('draw');
assert.equal(editor.tool, 'line');
assert.equal($('tool-shortcut').textContent, 'L');

const html = readFileSync(new URL('../../src/vectrify/ui/static/index.html', import.meta.url), 'utf8');
const rail = html.split('<aside class="toolrail"')[1].split('</aside>')[0];
assert.ok(!rail.includes('data-tool="draw"'));
assert.equal([...html.matchAll(/data-tool="draw"/g)].length, 1);
for (const type of DRAW_TOOLS) assert.ok(!rail.includes(`data-tool="${type}"`), `${type} stays inside Draw`);

// Secondary menu navigation wraps, and prevents arrows from nudging artwork.
const menuItems = ['draw', 'knife', 'region', 'gradient', 'hand'].map(tool => ({focus() { document.activeElement = this; }, tool}));
let menuKey, prevented = 0;
wireToolMenuKeys({addEventListener: (_type, handler) => { menuKey = handler; }, querySelectorAll: () => menuItems}, document);
const key = value => menuKey({key: value, preventDefault() { prevented++; }, stopPropagation() {}});
document.activeElement = menuItems[0]; key('ArrowUp');
assert.equal(document.activeElement, menuItems[4]);
key('ArrowDown'); assert.equal(document.activeElement, menuItems[0]);
key('End'); assert.equal(document.activeElement, menuItems[4]);
key('Home'); assert.equal(document.activeElement, menuItems[0]);
assert.equal(prevented, 4);
for (const type of ['draw', 'knife', 'region', 'gradient', 'hand']) assert.ok(!rail.includes(`data-tool="${type}"`));
for (const type of ['select', 'move', 'nodes', 'redraw', 'eyedropper']) assert.ok(rail.includes(`data-tool="${type}"`));
assert.ok(html.indexOf('id="reference-panel"') < html.indexOf('id="objects"'));
