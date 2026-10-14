// Every entry point runs the registry's command and says the same thing about
// it: the inspector's command lists, a tool strip button, the context menu and
// the command palette. Run by test_command_entry_points.py.
import assert from 'node:assert/strict';
import {createEditorState} from '../../src/vectrify/ui/static/app_state.js';
import {createCommandRegistryController} from '../../src/vectrify/ui/static/app_command_registry.js';
import {createCommandsController} from '../../src/vectrify/ui/static/app_commands.js';
import {createJoinController} from '../../src/vectrify/ui/static/app_join.js';
import {createClippingController} from '../../src/vectrify/ui/static/app_clipping.js';
import {createRegionController} from '../../src/vectrify/ui/static/app_region.js';
import {createAlignController} from '../../src/vectrify/ui/static/app_align.js';
import {createBooleansController} from '../../src/vectrify/ui/static/app_booleans.js';

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
path.dataset.commands = 'merge-shapes join-line-ends stroke-to-path fill-to-centerline cut-hole subtract convert-to-path detach extract-region delete-region';
const clipping = Object.assign(new Node('div'), {className: 'command-list'});
clipping.dataset.commands = 'set-clip remove-clip show-clips';
const arrange = Object.assign(new Node('div'), {className: 'command-list'});
arrange.dataset.commands = 'align-left align-top distribute-gaps-horizontal';
const points = Object.assign(new Node('div'), {className: 'command-list'});
points.dataset.commands = 'break-points join-points delete-points';
const strip = new Node('div');
const stripJoin = Object.assign(new Node('button'), {title: 'Join the two selected points into one line, however far apart'});
Object.assign(stripJoin.dataset, {command: 'join-points', key: '5', label: 'command'}); stripJoin.append(new Text('Join'));
strip.append(stripJoin);
document.append(path, clipping, arrange, points, strip, $('context-menu'), $('palette'), $('palette-input'), $('palette-list'));

// A line and a filled shape, a rectangle and an instance; the line is selected.
const objects = [
  {id: 'line', tag: 'path', label: 'Line', paint: {fill: 'none', stroke: '#123'}},
  {id: 'shape', tag: 'path', label: 'Shape', paint: {fill: '#456', stroke: 'none'}},
  {id: 'box', tag: 'rect', label: 'Rect 1', paint: {fill: '#789', stroke: 'none'}},
  {id: 'stamp', tag: 'use', label: 'Use 1', paint: {fill: '#abc', stroke: 'none'}},
  {id: 'clipped', tag: 'g', label: 'Group 1', paint: {}, attributes: {'clip-path': 'url(#c)'}},
];
for (const item of objects) Object.assign(item, {locks: [], inherited_locks: []});
let selectedPoints = [], pointLevel = false, turnLock = '';
const actions = [], toasts = [];
const editor = Object.assign(createEditorState(), {
  state: {epoch: 'e', revision: 0, selection: {objects: ['line'], nodes: []}, objects, undo: [], redo: [], clipboard: null, reference: null, bounds: [0, 0, 100, 100]},
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
  transformRefusal: () => turnLock, turnRefusal: verb => turnLock && `${turnLock}; unlock it to ${verb}`,
  flip: axis => { actions.push(['flip', axis]); return true; },
  ...Object.fromEntries(['noSelection', 'noPoints', 'noReference', 'visiblePaths', 'linePaths', 'fillPaths', 'joinCandidates', 'twoEnds', 'deleteSelection', 'breakPoints', 'cutHole', 'splitParts', 'openPalette']
    .map(name => [name, (...args) => context.commands[name](...args)])),
  ...Object.fromEntries(['joinEnds', 'mergeShapes', 'joinOpensDialog'].map(name => [name, (...args) => context.join[name](...args)])),
  openSetClip: () => context.clipping.openSetClip(), removeClip: () => context.clipping.removeClip(),
  setClipRefusal: () => context.clipping.setRefusal(), removeClipRefusal: () => context.clipping.removeRefusal(),
  regionCommand: mode => context.region.command(mode), regionRefusal: mode => context.region.refusal(mode),
  align: alignment => context.align.align(alignment), alignRefusal: alignment => context.align.alignRefusal(alignment),
  alignDescription: alignment => context.align.alignDescription(alignment),
  distribute: (axis, spacing) => context.align.distribute(axis, spacing), distributeRefusal: (axis, spacing) => context.align.distributeRefusal(axis, spacing),
  distributeDescription: (axis, spacing) => context.align.distributeDescription(axis, spacing),
  booleanDescription: operation => context.booleans.description(operation), booleanRefusal: () => context.booleans.refusal(),
  runBoolean: operation => context.booleans.run(operation),
  setTool: async tool => { editor.tool = tool; },
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
context.clipping = createClippingController(editor, {
  $, action: (...args) => { actions.push(args); return true; }, document, object, noSelection: () => context.commands.noSelection(),
  paintSwatch() {}, plural: (n, one) => `${n} ${one}`, toast() {},
});
// The region's preview, as the session answers it: Delete is refused.
const previews = [];
context.region = createRegionController(editor, {
  $, action: (...args) => { actions.push(args); return true; }, document, object, later: run => run(),
  drawOverlay: () => context.region.drawRegion(), renderCommands: () => context.commands.renderCommands(),
  plural: (n, one) => `${n} ${one}${n === 1 ? '' : 's'}`, runCommand: id => context.commands.runCommand(id),
  request: async (path, body) => {
    previews.push({path, body});
    return {epoch: 'e', revision: editor.state.revision, crossing: ['shape'], extract: {targets: [{id: 'shape', outcome: 'cut'}], error: null},
      delete: {targets: [], error: 'Unpin the points before deleting them'}};
  },
  localToOverlay: () => null, selectionContour: () => null, svgElement: () => null, stage: new Node('div'),
  setTool: async tool => { editor.tool = tool; }, toast: (message, error) => toasts.push({message, error: !!error}),
  xmlElement: tag => new Node(tag),
});
context.align = createAlignController(editor, {
  $, action: (...args) => { actions.push(args); return true; }, document, object, later: run => run(),
  drawOverlay() {}, level: () => 'objects', objectBounds: () => ({left: 0, top: 0, right: 10, bottom: 10}),
  renderCommands: () => context.commands.renderCommands(), toast: (message, error) => toasts.push({message, error: !!error}),
  topSelection: () => editor.state.selection.objects, xmlElement: tag => new Node(tag),
});
context.booleans = createBooleansController(editor, {
  action: (...args) => { actions.push(args); return true; }, document, drawOverlay() {},
  filledPath: id => context.commands.filledPath(id), noSelection: () => context.commands.noSelection(), toast() {},
});
const {commandById, commandName, commandRefusal, commandTitle} = registry;
const {buildCommandLists, renderCommands, openContextMenu, renderPalette, runPaletteRow} = context.commands;

// The inspector's lists are built from the registry, named as it names them.
buildCommandLists();
renderCommands();
const button = (list, id) => list.querySelector(`button[data-command=${id}]`);
assert.deepEqual(path.children.map(child => child.textContent), ['Merge shapes', 'Join line ends', 'Stroke to path', 'Fill to centerline', 'Cut out as hole', 'Subtract', 'Convert to path', 'Detach shared geometry', 'Extract region', 'Delete region']);
assert.deepEqual(points.children.map(child => child.textContent), ['Break', 'Join points', 'Delete points']);
assert.equal(stripJoin.textContent, 'Join points', 'a strip button takes the registry name');

// The context menu: object commands for objects, by group, none hidden for
// being unavailable.
const menuItem = id => $('context-menu').querySelector(`.menu-item[data-command=${id}]`);
openContextMenu(10, 10);
const headings = $('context-menu').querySelectorAll('.menu-heading').map(heading => heading.textContent);
assert.deepEqual(headings, ['Select', 'Object', 'Transform', 'Arrange', 'Path', 'Clipping', 'Repair']);
// Flips are offered beside the Transform fields and in the menu, and say
// why they cannot run as the rest do.
const transformList = Object.assign(new Node('div'), {className: 'command-list'});
transformList.dataset.commands = 'flip-horizontal flip-vertical';
document.append(transformList); buildCommandLists(); renderCommands();
assert.deepEqual(transformList.children.map(child => child.textContent), ['Flip horizontal', 'Flip vertical']);
assert.equal(menuItem('flip-vertical').querySelector('kbd').textContent, 'Shift V');
await commandById.get('flip-horizontal').run();
assert.deepEqual(actions.pop(), ['flip', 'horizontal']);
turnLock = 'Line: position is locked';
assert.equal(commandRefusal(commandById.get('flip-vertical')), 'Line: position is locked; unlock it to flip');
assert.equal(commandRefusal(commandById.get('rotate')), 'Line: position is locked');
turnLock = '';
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
agrees('subtract', path, 'Select two or more filled paths');
agrees('convert-to-path', path, 'Select rectangles, circles, ellipses or lines');
agrees('detach', path, 'Select one instance, or a path that shares its geometry');
agrees('join-points', points, 'Select points in Nodes (S) first');
// One object has nothing to align to but the artboard, nor to space out;
// Align's tooltip names the reference it lines up to.
agrees('align-left', arrange, 'Select two or more objects to align them to each other, or align to the artboard');
agrees('distribute-gaps-horizontal', arrange, 'Select three or more objects to distribute');
editor.alignReference = 'artboard';
renderCommands();
agrees('align-top', arrange, '');
assert.match(button(arrange, 'align-top').title, /top edges with the artboard's/);
assert.equal(button(arrange, 'distribute-gaps-horizontal').textContent, 'Distribute horizontally: equal gaps');
editor.alignReference = 'selection';
renderCommands();
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

// A rectangle converts to a path from each entry point; Detach, which is for
// instances and shared geometry, points to it.
editor.state.selection.objects = ['box'];
renderCommands();
agrees('convert-to-path', path, '');
agrees('detach', path, 'A basic shape shares nothing: use Convert to path');
press(button(path, 'convert-to-path'));
openContextMenu(10, 10);
menuItem('convert-to-path').onclick();
paletteRow('convert-to-path');
runPaletteRow(0);
assert.deepEqual(actions.map(([command]) => command), ['convert_to_path', 'convert_to_path', 'convert_to_path']);
actions.length = 0;

// An instance is detached, not converted, and a mixed selection converts
// nothing.
editor.state.selection.objects = ['stamp'];
renderCommands();
agrees('convert-to-path', path, 'Instances and shared geometry are detached: use Detach');
agrees('detach', path, '');
assert.equal(button(path, 'detach').textContent, 'Detach to editable path');
editor.state.selection.objects = ['box', 'line'];
renderCommands();
agrees('convert-to-path', path, 'Select rectangles, circles, ellipses or lines');
press(button(path, 'convert-to-path'));
assert.deepEqual(actions, []);

// Set clip needs a closed path among the selection, and says a basic shape
// becomes one with Convert to path; Remove clip needs a clipped object.
editor.state.selection.objects = ['shape'];
renderCommands();
agrees('set-clip', clipping, 'Select the objects to clip and a closed path to clip them to');
agrees('remove-clip', clipping, 'None of the selected objects is clipped');
editor.state.selection.objects = ['box', 'stamp'];
renderCommands();
agrees('set-clip', clipping, 'A rectangle must be a path to be a boundary: use Convert to path, under Path');
editor.state.selection.objects = ['clipped', 'shape'];
renderCommands();
agrees('set-clip', clipping, '');
agrees('remove-clip', clipping, '');
assert.equal(button(clipping, 'set-clip').textContent, 'Set clip…');
press(button(clipping, 'remove-clip'));
openContextMenu(10, 10);
menuItem('remove-clip').onclick();
paletteRow('remove-clip');
runPaletteRow(0);
assert.deepEqual(actions.map(([command]) => command), ['remove_clip', 'remove_clip', 'remove_clip']);
actions.length = 0;

// Without a region, Extract region and Delete region pick up the Region
// tool; the region applies only once its preview is shown, and a refused
// preview's reason is the command's everywhere.
editor.state.selection.objects = ['shape'];
renderCommands();
agrees('extract-region', path, '');
agrees('delete-region', path, '');
paletteRow('delete-region');
await runPaletteRow(0);
assert.equal(editor.tool, 'region');
assert.equal(context.region.options.mode, 'delete');
assert.deepEqual(actions, []);
editor.overlayFrame = {content: new Node('g')};
editor.region = {points: [[0, 0], [10, 0], [10, 10]], closed: true};
assert.equal(commandRefusal(commandById.get('delete-region')), 'Previewing the region…');
context.region.drawRegion();
await new Promise(resolve => setTimeout(resolve));
assert.deepEqual(previews.at(-1), {path: '/api/region', body: {epoch: 'e', revision: 0, region: [[0, 0], [10, 0], [10, 10]], scope: 'selection', cut: true, within: null}});
agrees('delete-region', path, 'Unpin the points before deleting them');
agrees('extract-region', path, '');
press(button(path, 'delete-region'));
assert.deepEqual(actions, [], 'a refused region edits nothing');
// The other operation shows its preview first, and applies when run again.
press(button(path, 'extract-region'));
assert.equal(context.region.options.mode, 'extract');
assert.deepEqual(actions, []);
paletteRow('extract-region');
await runPaletteRow(0);
assert.deepEqual(actions.at(-1).slice(0, 2), ['extract', {region: [[0, 0], [10, 0], [10, 10]], scope: 'selection', cut: true, within: null, delete: false}]);
actions.length = 0;
editor.tool = 'select'; editor.region = null;

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
// The line has no fill to subtract from; the shape behind a rectangle points
// to Convert to path. Two filled paths name the target before running.
agrees('subtract', path, 'Line has no fill: these operations combine filled areas');
editor.state.selection.objects = ['shape', 'box'];
renderCommands();
agrees('subtract', path, 'Rect 1 is a basic shape: use Convert to path first');
objects.push({id: 'patch', tag: 'path', label: 'Patch', paint: {fill: '#f00', stroke: 'none'}, locks: [], inherited_locks: []});
editor.state.selection.objects = ['patch', 'shape'];
renderCommands();
agrees('subtract', path, '');
assert.equal(button(path, 'subtract').title, 'Subtract Patch from Shape. Shape, the backmost, keeps its paint and place');
press(button(path, 'subtract'));
await new Promise(resolve => setTimeout(resolve));
assert.deepEqual(actions.at(-1), ['subtract', {}, 'Subtracting…']);
objects.pop();
editor.state.selection.objects = ['line', 'shape'];
renderCommands();

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
assert.deepEqual(lists.find(([heading]) => heading === 'Clipping')[1], ['set-clip', 'remove-clip', 'show-clips']);
for (const [heading, list] of lists) for (const id of list) assert.equal(commandById.get(id)?.group, heading, `${id} under ${heading}`);
const flips = html.match(/<div class="command-list transform-commands" data-commands="([^"]+)"/)[1].split(' ');
assert.deepEqual(flips, ['flip-horizontal', 'flip-vertical']);
for (const id of flips) assert.equal(commandById.get(id)?.group, 'Transform', `${id} under Transform`);
const pointList = html.match(/<div class="command-list" data-commands="([^"]+)" role="group" aria-label="Point operations">/)[1].split(' ');
for (const id of pointList) assert.equal(commandById.get(id)?.group, 'Nodes', `${id} beside the points`);
