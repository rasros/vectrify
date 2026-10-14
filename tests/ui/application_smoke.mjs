// Boots the whole application as the page does, against a fake DOM. Nothing
// type-checks the wiring between controllers, so this checks that every
// collaborator a controller names is given and resolves, then runs every
// registered command (label, description, refusal and run) and calls every
// collaborator once. Errors from the fake DOM are expected and ignored; an
// error is a wiring error when a bare collaborator name is not a function or
// the throw happens in app.js itself. Run by test_app_controllers.py.
import assert from 'node:assert/strict';
import {COLLABORATORS, CONTROLLERS, VALUES, createApplication} from '../../src/vectrify/ui/static/app.js';

class Element {
  constructor(tag = 'div') {
    this.tagName = tag.toUpperCase(); this.children = []; this.dataset = {}; this.attributes = new Map();
    this.style = {setProperty() {}, removeProperty() {}};
    this.classList = {toggle() {}, add() {}, remove() {}, contains() { return false; }};
    this.textContent = ''; this.value = ''; this.hidden = false; this.listeners = new Map();
    this.clientWidth = 800; this.clientHeight = 600; this.scrollTop = 0; this.parentNode = null;
  }
  addEventListener(name, handler) { const handlers = this.listeners.get(name) || []; handlers.push(handler); this.listeners.set(name, handlers); }
  removeEventListener() {}
  setAttribute(name, value) { this.attributes.set(name, String(value)); } getAttribute(name) { return this.attributes.get(name) ?? null; }
  removeAttribute(name) { this.attributes.delete(name); } hasAttribute(name) { return this.attributes.has(name); }
  querySelector() { return null; } querySelectorAll() { return []; }
  append(...children) { this.children.push(...children); } appendChild(child) { this.children.push(child); return child; }
  prepend(...children) { this.children.unshift(...children); } replaceChildren(...children) { this.children = children; }
  remove() {} before() {} after() {} insertBefore(child) { return child; }
  closest() { return null; } contains() { return false; } matches() { return false; }
  getBoundingClientRect() { return {x: 0, y: 0, left: 0, top: 0, right: 800, bottom: 600, width: 800, height: 600}; }
  getBBox() { return {x: 0, y: 0, width: 10, height: 10}; }
  getScreenCTM() { return null; } getCTM() { return null; }
  showModal() { this.open = true; } close() { this.open = false; } focus() {} blur() {} select() {} click() {} scrollIntoView() {}
  setPointerCapture() {} releasePointerCapture() {} hasPointerCapture() { return false; }
}
const elements = new Map();
const $ = id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
const document = Object.assign(new Element('html'), {
  body: new Element('body'), activeElement: {tagName: 'BODY'}, getElementById: $,
  createElement: tag => new Element(tag), createElementNS: (_, tag) => new Element(tag),
  createTextNode: text => Object.assign(new Element('#text'), {textContent: text}),
  createDocumentFragment: () => new Element(), createComment: () => new Element(),
});
const window = Object.assign(new Element(), {innerWidth: 1000, innerHeight: 800, devicePixelRatio: 1});
globalThis.document = document; globalThis.window = window;
globalThis.MutationObserver = class { observe() {} disconnect() {} };
globalThis.ResizeObserver = class { observe() {} disconnect() {} };
globalThis.requestAnimationFrame = () => 1; globalThis.cancelAnimationFrame = () => {};
globalThis.getComputedStyle = () => ({columnGap: '0', paddingLeft: '0', paddingRight: '0', display: 'block', getPropertyValue: () => ''});
globalThis.localStorage = {getItem() { return null; }, setItem() {}, removeItem() {}};
globalThis.sessionStorage = {getItem() { return null; }, setItem() {}, removeItem() {}};
globalThis.innerWidth = 1000; globalThis.innerHeight = 800;
// Every request fails as the server would refuse it, so commands stop there.
globalThis.fetch = async () => ({ok: false, status: 400, json: async () => ({error: 'offline'})});

const app = createApplication({document, window, location: {search: ''}});
app.wire();

const objects = [
  {id: 'line', tag: 'path', label: 'Line', paint: {fill: 'none', stroke: '#123'}},
  {id: 'shape', tag: 'path', label: 'Shape', paint: {fill: '#456', stroke: 'none'}},
  {id: 'box', tag: 'rect', label: 'Rect 1', paint: {fill: '#789', stroke: 'none'}},
  {id: 'group', tag: 'g', label: 'Group 1', paint: {}, children: ['line']},
];
for (const item of objects) Object.assign(item, {locks: [], inherited_locks: [], attributes: {}});

// What each controller names among its options, and which have no default:
// the destructured second parameter, split at its top-level commas.
function options(create) {
  const source = create.toString(), start = source.indexOf('{', source.indexOf(','));
  const parts = [''];
  for (let i = start + 1, depth = 0; ; i++) {
    const c = source[i];
    if (depth === 0 && c === '}') break;
    if ('{([' .includes(c)) depth++;
    if ('})]'.includes(c)) depth--;
    if (depth === 0 && c === ',') parts.push(''); else parts[parts.length - 1] += c;
  }
  return parts.map(part => part.trim()).filter(Boolean).map(part => ({name: part.split(/\s*=/)[0], required: !/=/.test(part)}));
}
const named = Object.fromEntries(Object.entries(CONTROLLERS).map(([key, create]) => [key, options(create)]));
const collaborators = new Set(Object.values(named).flat().map(({name}) => name));
assert.ok(collaborators.has('selectObject') && collaborators.has('toast'), 'the collaborator names were read');

// Every collaborator resolves to its controller's method or value, and every
// option a controller has no default for is given.
const flat = names => names.flatMap(name => typeof name === 'string' ? [[name, name]] : Object.entries(name));
const given = new Set(['$', 'app', 'document', 'location', 'window', 'drawing', 'overlay', 'stage', 'files', 'moveLockRefusal', 'regionShape']);
for (const [key, names] of Object.entries(COLLABORATORS)) for (const [name, method] of flat(names)) {
  assert.equal(typeof app[key]?.[method], 'function', `${name} is ${key}.${method}`);
  given.add(name);
}
for (const [key, names] of Object.entries(VALUES)) for (const name of names) {
  assert.notEqual(app[key]?.[name], undefined, `${name} is ${key}.${name}`);
  given.add(name);
}
for (const [key, options] of Object.entries(named))
  for (const {name, required} of options) if (required) assert.ok(given.has(name), `${key} is given ${name}`);

const failures = [];
function check(where, error) {
  if (!(error instanceof Error)) return;
  const bare = error.message.match(/^(\w+) is not a function$/);
  const inApp = /app\.js:\d+/.test((error.stack || '').split('\n').slice(1, 2).join(''));
  if (error instanceof ReferenceError || (bare && collaborators.has(bare[1])) || inApp)
    failures.push(`${where}: ${error.stack}`);
}
async function attempt(where, run) {
  try {
    const result = run();
    if (result && typeof result.then === 'function')
      await Promise.race([result, new Promise(resolve => setTimeout(resolve, 20))]);
  } catch (error) { check(where, error); }
}
process.on('unhandledRejection', error => check('unhandled', error));

// Every command, at the object level with one, two and no objects selected,
// and in Nodes with a point selected.
const {COMMANDS, commandTitle} = app.command_registry;
const selections = [
  {tool: 'select', objects: ['line'], nodes: []},
  {tool: 'select', objects: ['line', 'shape'], nodes: []},
  {tool: 'select', objects: [], nodes: []},
  {tool: 'nodes', objects: ['shape'], nodes: [['shape', 'n1']]},
];
await attempt('applyState', () => app.session.applyState({
  epoch: 'e', revision: 1, objects, selection: {objects: [], nodes: []}, undo: ['Move'], redo: [], clipboard: null,
  reference: null, bounds: [0, 0, 100, 100], root: 'root', project_version: 1,
}));
assert.ok(app.editor.state, 'the state applies');
for (const selection of selections) {
  for (const command of COMMANDS) {
    app.editor.tool = selection.tool;
    app.editor.state.selection = {objects: [...selection.objects], nodes: selection.nodes.map(pair => [...pair])};
    const where = `${command.id} (${selection.tool}, ${selection.objects.length} selected)`;
    await attempt(`${where} label`, () => command.label?.());
    await attempt(`${where} description`, () => command.description);
    await attempt(`${where} disabled`, () => command.disabled?.());
    await attempt(`${where} title`, () => commandTitle(command));
    await attempt(`${where} run`, () => command.run());
    app.editor.drag = null; app.editor.pending = 0;
  }
}
for (const name of ['commands', 'selection', 'paint', 'layout', 'tree'])
  for (const method of ['renderCommands', 'renderInspector', 'renderObjects', 'renderStatus', 'renderNodeInspector'])
    if (typeof app[name][method] === 'function') await attempt(`${name}.${method}`, () => app[name][method]());
await attempt('drawOverlay', () => app.frames.drawOverlay());

// Every collaborator once, as a controller calls it, with nothing selected.
app.editor.tool = 'select'; app.editor.state.selection = {objects: [], nodes: []};
for (const [key, names] of Object.entries(COLLABORATORS)) for (const [, method] of flat(names)) {
  await attempt(`${key}.${method}`, () => app[key][method]());
  app.editor.drag = null; app.editor.pending = 0;
}

if (failures.length) {
  console.error(failures.join('\n\n'));
  process.exit(1);
}
process.exit(0);
