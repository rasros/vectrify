import assert from 'node:assert/strict';
import {createApplication} from '../../src/vectrify/ui/static/app.js';

class Element {
  constructor() {
    this.children = []; this.dataset = {}; this.style = {setProperty() {}};
    this.classList = {toggle() {}, add() {}, remove() {}};
    this.textContent = ''; this.value = ''; this.hidden = false; this.listeners = new Map();
    this.clientWidth = 800; this.clientHeight = 600;
  }
  addEventListener(name, handler) {const handlers = this.listeners.get(name) || []; handlers.push(handler); this.listeners.set(name, handlers);}
  setAttribute() {} removeAttribute() {} querySelector() {return null;} querySelectorAll() {return [];}
  append(...children) {this.children.push(...children);} replaceChildren(...children) {this.children = children;}
  closest() {return null;} contains() {return false;}
  getBoundingClientRect() {return {x:0,y:0,left:0,top:0,right:800,bottom:600,width:800,height:600};}
  showModal() {this.open = true;} close() {this.open = false;} focus() {} select() {}
}
const elements = new Map();
const $ = id => {if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id);};
const document = Object.assign(new Element(), {
  body: new Element(), activeElement: {tagName:'BODY'}, getElementById: $, createElement: () => new Element(),
  createDocumentFragment: () => new Element(), createComment: () => new Element(),
});
const window = new Element();
globalThis.document = document; globalThis.window = window;
globalThis.MutationObserver = class {observe() {}};
globalThis.ResizeObserver = class {observe() {}};
globalThis.requestAnimationFrame = () => 1;
globalThis.getComputedStyle = () => ({columnGap:'0',paddingLeft:'0',paddingRight:'0',display:'block'});
globalThis.localStorage = {getItem() {return null;},setItem() {}};
const app = createApplication({document, window, location:{search:''}});
assert.equal(app.editor.state, undefined);
app.wire();
const keyListeners = window.listeners.get('keydown').length;
app.wire();
assert.equal(window.listeners.get('keydown').length, keyListeners, 'wiring is idempotent');
assert.ok($('stage').listeners.has('pointerdown'));
assert.ok($('stage').listeners.has('pointercancel'));
assert.ok(window.listeners.has('keydown'));
assert.equal(typeof window.vectrifyPulse, 'function');
assert.equal(typeof $('fill-type').onchange, 'function');
assert.equal(typeof $('reference-toggle').onclick, 'function');
assert.equal(typeof $('object-search').oninput, 'function');
assert.equal(typeof $('agent-enabled').onchange, 'function');
app.editor.state = {epoch:'e',revision:0,selection:{objects:[],nodes:[]},undo:[],redo:[],objects:[],bounds:[0,0,100,100],root:'root'};
const button = new Element(); button.dataset.command = 'help';
for (const handler of document.listeners.get('click')) handler({target:{closest:() => button}});
assert.equal($('help-dialog').open, true, 'delegated buttons reach the actual command registry');
$('help-dialog').close();
let prevented = false;
for (const handler of window.listeners.get('keydown')) handler({key:'?',code:'Slash',preventDefault() {prevented = true;}});
assert.equal(prevented, true);
assert.equal($('help-dialog').open, true, 'keyboard wiring reaches the same dialog');
