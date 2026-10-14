// Every entry point runs the registry's command and says the same thing about
// it: the inspector's command lists, a tool strip button, the context menu and
// the command palette. Run by test_command_entry_points.py.
import assert from 'node:assert/strict';
import {createEditorState} from '../../src/vectrify/ui/static/app_state.js';
import {createCommandRegistryController} from '../../src/vectrify/ui/static/app_command_registry.js';
import {createCommandsController} from '../../src/vectrify/ui/static/app_commands.js';
import {createJoinController} from '../../src/vectrify/ui/static/app_join.js';

// Just enough DOM: elements with attributes, children and simple selectors
// (a tag, an id, classes and [attribute] or [attribute=value] tests).
class Node {
  constructor(tag = '') {
    this.tag = tag.toUpperCase(); this.children = []; this.parent = null; this.dataset = {}; this.attributes = new Map();
    this.style = {}; this.hidden = false; this.title = ''; this.className = ''; this.ownText = ''; this.listeners = new Map();
  }
  get firstChild() { return this.children[0]; }
  get textContent() { return this.ownText + this.children.map(child => child.textContent).join(''); }
  set textContent(value) { this.ownText = String(value); this.children = []; }
  get classList() {
    const names = () => this.className.split(' ').filter(Boolean);
    return {contains: name => names().includes(name), toggle: (name, on) => { this.className = [...names().filter(n => n !== name), ...(on ? [name] : [])].join(' '); }};
  }
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  getAttribute(name) { return this.attributes.get(name) ?? null; }
  append(...children) { for (const child of children) { child.parent = this; this.children.push(child); } }
  replaceChildren(...children) { this.children = []; this.append(...children.flatMap(child => child.fragment ? child.children : [child])); }
  addEventListener(name, handler) { this.listeners.set(name, [...(this.listeners.get(name) || []), handler]); }
  closest(selector) { for (let node = this; node; node = node.parent) if (matches(node, selector)) return node; return null; }
  contains(node) { for (; node; node = node.parent) if (node === this) return true; return false; }
  querySelectorAll(selector) { return this.children.flatMap(child => [...(matches(child, selector) ? [child] : []), ...child.querySelectorAll(selector)]); }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  getBoundingClientRect() { return {left: 0, top: 0, width: 200, height: 300, bottom: 300}; }
  focus() { document.activeElement = this; } scrollIntoView() {} showModal() { this.open = true; } close() { this.open = false; }
}
function attribute(node, name) {
  if (name.startsWith('data-')) return node.dataset[name.slice(5).replace(/-(\w)/g, (_, c) => c.toUpperCase())];
  return node.getAttribute(name);
}
function matches(node, selector) {
  if (!(node instanceof Node) || !node.tag) return false;
  return selector.split(',').some(one => {
    const [, tag, id, classes, tests] = one.trim().match(/^([a-z]*)(?:#([\w-]+))?((?:\.[\w-]+)*)((?:\[[^\]]+\])*)$/i);
    if (tag && node.tag !== tag.toUpperCase()) return false;
    if (id && node.id !== id) return false;
    if (classes && !classes.split('.').filter(Boolean).every(name => node.classList.contains(name))) return false;
    return [...tests.matchAll(/\[([\w-]+)(?:=([^\]]+))?\]/g)].every(([, name, value]) => {
      const actual = name === 'id' ? node.id : attribute(node, name);
      return value === undefined ? actual !== undefined && actual !== null : actual === value;
    });
  });
}
class Text { constructor(text) { this.text = text; } querySelectorAll() { return []; } get textContent() { return this.text; } set textContent(value) { this.text = value; } }

const ids = new Map();
const $ = id => { if (!ids.has(id)) { const node = new Node('div'); node.id = id; ids.set(id, node); } return ids.get(id); };
const document = Object.assign(new Node('html'), {
  activeElement: null,
  createElement: tag => new Node(tag),
  createTextNode: text => new Text(text),
  createDocumentFragment: () => Object.assign(new Node(''), {fragment: true}),
  querySelector: selector => document.querySelectorAll(selector)[0] || null,
});
globalThis.innerWidth = 1000; globalThis.innerHeight = 800;

// The inspector's Path list and a Nodes strip as index.html has them.
const path = Object.assign(new Node('div'), {className: 'command-list'});
path.dataset.commands = 'merge-shapes join-line-ends stroke-to-path fill-to-centerline cut-hole';
const points = Object.assign(new Node('div'), {className: 'command-list'});
points.dataset.commands = 'break-points join-points delete-points';
const strip = new Node('div');
const stripJoin = Object.assign(new Node('button'), {title: 'Join the two selected points into one line, however far apart'});
Object.assign(stripJoin.dataset, {command: 'join-points', key: '5', label: 'command'}); stripJoin.append(new Text('Join'));
strip.append(stripJoin);
document.append(path, points, strip, $('context-menu'), $('palette'), $('palette-input'), $('palette-list'));

// A line and a filled shape; the line is selected.
const objects = [
  {id: 'line', tag: 'path', label: 'Line', paint: {fill: 'none', stroke: '#123'}},
  {id: 'shape', tag: 'path', label: 'Shape', paint: {fill: '#456', stroke: 'none'}},
];
let selectedPoints = [], pointLevel = false;
const actions = [], toasts = [];
const editor = Object.assign(createEditorState(), {
  state: {epoch: 'e', revision: 0, selection: {objects: ['line'], nodes: []}, objects, undo: [], redo: [], clipboard: null, reference: null},
  zoom: 1, tool: 'select', agentStatus: {enabled: false},
});
const object = id => objects.find(item => item.id === id);
const context = {};
const registry = createCommandRegistryController(editor, {
  $, action: (...args) => { actions.push(args); return true; }, document, object,
  oneObject: () => editor.state.selection.objects.length === 1 ? object(editor.state.selection.objects[0]) : null,
  level: () => pointLevel ? 'points' : 'objects', selectedPoints: () => selectedPoints, nodeAt: () => ({}),
  contourAt: () => ({closed: false, nodes: [{id: 'a'}, {id: 'b'}, {id: 'c'}]}), selectedHandle: () => null,
  pointPairs: () => selectedPoints.map(key => key.split(' ')), segmentPicked: () => false, holeContours: () => null,
  gradientRefusal: () => '', tidyTargets: () => 0,
  ...Object.fromEntries(['noSelection', 'noPoints', 'noReference', 'visiblePaths', 'linePaths', 'fillPaths', 'joinCandidates', 'twoEnds', 'deleteSelection', 'breakPoints', 'cutHole', 'splitParts', 'openPalette']
    .map(name => [name, (...args) => context.commands[name](...args)])),
  ...Object.fromEntries(['joinEnds', 'mergeShapes', 'joinOpensDialog'].map(name => [name, (...args) => context.join[name](...args)])),
});
context.commands = createCommandsController(editor, {
  $, COMMANDS: registry.COMMANDS, commandById: registry.commandById, commandName: registry.commandName,
  commandRefusal: registry.commandRefusal, commandTitle: registry.commandTitle, document, object,
  action: (...args) => { actions.push(args); return true; },
  later: run => run(), level: () => pointLevel ? 'points' : 'objects', selectedPoints: () => selectedPoints,
  resolvedPaint: (id, kind, fallback) => object(id)?.paint[kind] ?? fallback,
  pointPairs: () => selectedPoints.map(key => key.split(' ')), pointContours: () => [], selectedHandle: () => null,
  toast: (message, error) => toasts.push({message, error: !!error}), stage: new Node('div'), window: new Node('window'),
});
context.join = createJoinController(editor, {
  action: (...args) => { actions.push(args); return true; }, plural: (n, one) => `${n} ${one}`, toast() {},
  fillPaths: context.commands.fillPaths, joinCandidates: context.commands.joinCandidates, linePaths: context.commands.linePaths,
  openJoin: () => actions.push(['dialog']),
});
const {commandById, commandName, commandRefusal, commandTitle} = registry;
const {buildCommandLists, renderCommands, openContextMenu, renderPalette, runPaletteRow} = context.commands;

// The inspector's lists are built from the registry, named as it names them.
buildCommandLists();
renderCommands();
const button = (list, id) => list.querySelector(`button[data-command=${id}]`);
assert.deepEqual(path.children.map(child => child.textContent), ['Merge shapes', 'Join line ends', 'Stroke to path', 'Fill to centerline', 'Cut out as hole']);
assert.deepEqual(points.children.map(child => child.textContent), ['Break', 'Join points', 'Delete points']);
assert.equal(stripJoin.textContent, 'Join points', 'a strip button takes the registry name');

// The context menu: object commands for objects, by group, none hidden for
// being unavailable.
const menuItem = id => $('context-menu').querySelector(`.menu-item[data-command=${id}]`);
openContextMenu(10, 10);
const headings = $('context-menu').querySelectorAll('.menu-heading').map(heading => heading.textContent);
assert.deepEqual(headings, ['Select', 'Object', 'Arrange', 'Path', 'Repair']);
assert.ok(!menuItem('join'), 'the Ctrl/⌘ J shortcut is not a menu entry of its own');
assert.ok(!menuItem('join-points'), 'point operations stay with Nodes');

// The palette row for a command, found by typing its name.
function paletteRow(id) {
  $('palette-input').value = commandById.get(id).name;
  renderPalette();
  const row = $('palette-list').querySelectorAll('.palette-row').find(row => row.querySelector('.palette-name').textContent === commandName(commandById.get(id)));
  assert.ok(row, `the palette lists ${id}`);
  return row;
}

// Each entry point shows the command's name and its availability the same
// way: one reason, from the registry.
function agrees(id, list, reason) {
  const command = commandById.get(id);
  assert.equal(commandRefusal(command), reason, id);
  openContextMenu(10, 10);
  const inspector = button(list, id), item = menuItem(id);
  assert.equal(inspector.getAttribute('aria-disabled'), String(!!reason), `${id} in the inspector`);
  assert.equal(inspector.title, commandTitle(command), `${id} tooltip in the inspector`);
  if (item) {
    assert.equal(item.getAttribute('aria-disabled'), String(!!reason), `${id} in the context menu`);
    assert.equal(item.title, inspector.title, `${id} tooltip in the context menu`);
    assert.equal(item.querySelector('span').textContent, inspector.textContent);
  }
  const row = paletteRow(id);
  assert.equal(row.getAttribute('aria-disabled'), String(!!reason), `${id} in the palette`);
  if (reason) {
    assert.equal(row.querySelector('small').textContent, reason);
    assert.ok(inspector.title.endsWith(reason));
  }
}

// A line selected: its ends join, and it becomes a filled path; it is no shape
// to merge nor a fill to thin.
agrees('join-line-ends', path, '');
agrees('stroke-to-path', path, '');
agrees('merge-shapes', path, 'Select two or more paths, or groups that contain them, or one filled path');
agrees('fill-to-centerline', path, 'Select thin filled shapes');
agrees('cut-hole', path, 'Select two visible paths, one inside or overlapping the other');
agrees('join-points', points, 'Select points in Nodes (S) first');
assert.equal(stripJoin.getAttribute('aria-disabled'), 'true');
assert.equal(stripJoin.title, commandTitle(commandById.get('join-points')), 'the strip says why too');

// An unavailable command stays put and, pressed anywhere, says why and edits
// nothing.
const clicks = document.listeners;
assert.equal(clicks.size, 0);
context.commands.wire();
const press = target => { for (const handler of document.listeners.get('click')) handler({target}); };
press(button(path, 'merge-shapes'));
assert.deepEqual(toasts.at(-1), {message: commandRefusal(commandById.get('merge-shapes')), error: true});
openContextMenu(10, 10);
menuItem('merge-shapes').onclick();
assert.deepEqual(toasts.at(-1), {message: commandRefusal(commandById.get('merge-shapes')), error: true});
assert.equal($('context-menu').hidden, true);
press(stripJoin);
assert.deepEqual(toasts.at(-1), {message: 'Select points in Nodes (S) first', error: true});
$('palette').open = true;
paletteRow('merge-shapes');
runPaletteRow($('palette-list').querySelectorAll('.palette-row').findIndex(row => row.getAttribute('aria-disabled') === 'true'));
assert.equal($('palette').open, true, 'the palette keeps showing the reason');
assert.deepEqual(actions, []);

// An available command runs the same edit from each.
press(button(path, 'stroke-to-path'));
openContextMenu(10, 10);
menuItem('stroke-to-path').onclick();
paletteRow('stroke-to-path');
runPaletteRow(0);
assert.deepEqual(actions.map(([command]) => command), ['line_to_fill', 'line_to_fill', 'line_to_fill']);
assert.equal($('palette').open, false);
actions.length = 0;

// Both paths selected: they merge, through the colour dialog, and the labels
// say so.
editor.state.selection.objects = ['line', 'shape'];
renderCommands();
agrees('merge-shapes', path, '');
assert.equal(button(path, 'merge-shapes').textContent, 'Merge shapes…');
agrees('fill-to-centerline', path, '');
agrees('cut-hole', path, '');
press(button(path, 'merge-shapes'));
assert.deepEqual(actions.at(-1), ['dialog']);

// In a point tool the menu offers the Nodes commands only; two points join.
pointLevel = true; selectedPoints = ['line a', 'shape b'];
renderCommands();
openContextMenu(10, 10);
assert.deepEqual($('context-menu').querySelectorAll('.menu-heading').map(heading => heading.textContent), ['Nodes']);
assert.ok(!menuItem('merge-shapes'), 'object operations stay with objects');
agrees('join-points', points, '');
assert.equal(stripJoin.getAttribute('aria-disabled'), 'false');
assert.equal(stripJoin.title, 'Join the two selected points into one line, however far apart', 'an available strip button keeps its own tooltip');
press(stripJoin);
assert.deepEqual(actions.at(-1).slice(0, 2), ['join_two_ends', {points: [['line', 'a'], ['shape', 'b']]}]);
selectedPoints = ['line a'];
renderCommands();
agrees('join-points', points, 'Select exactly two points to join');

// The page names only registry commands, and each inspector list holds the
// commands of the group it is headed by: point operations under the points.
const {readFileSync} = await import('node:fs');
const html = readFileSync(new URL('../../src/vectrify/ui/static/index.html', import.meta.url), 'utf8');
for (const [, id] of html.matchAll(/data-command="([^"]+)"/g)) assert.ok(commandById.has(id), `index.html names unknown command ${id}`);
const lists = [...html.matchAll(/<h3[^>]*>([^<]+)<\/h3><div class="command-list" data-commands="([^"]+)"/g)].map(([, heading, list]) => [heading, list.split(' ')]);
assert.deepEqual(lists.map(([heading]) => heading), ['Arrange', 'Path', 'Clipping', 'Repair', 'Object']);
for (const [heading, list] of lists) for (const id of list) assert.equal(commandById.get(id)?.group, heading, `${id} under ${heading}`);
const pointList = html.match(/<div class="command-list" data-commands="([^"]+)" role="group" aria-label="Point operations">/)[1].split(' ');
for (const id of pointList) assert.equal(commandById.get(id)?.group, 'Nodes', `${id} beside the points`);
