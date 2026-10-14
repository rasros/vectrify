import {overflowLayout} from './strip.js';
import {scopeChain, selectionStatus} from './selection.js';
import {shapeStatus} from './shapes.js';
import {strokeStatus} from './pencil.js';

// Layout owns its controls; editor contains the live state shared with other controllers.
export function createLayoutController(editor, {
  $,
  ResizeObserver = globalThis.ResizeObserver,
  document = globalThis.document,
  frameSize,
  getComputedStyle = globalThis.getComputedStyle,
  gradientStatus = () => '',
  level,
  object,
  parents,
  pointPaths,
  reportView,
  rotationStatus = () => '',
  requestAnimationFrame = globalThis.requestAnimationFrame,
  selectedPoints,
  window = globalThis.window,
} = {}) {
  let topbarSignature = '';

  let stripSignature = '';

  let stripFrame = 0;

  // The tool strip keeps to one row: the active tool's controls that do not fit
  // go, least important first, into the "⋯" menu, keeping the status line.
  const parked = new Map();

  function scheduleStrip() {
    if (!stripFrame) stripFrame = requestAnimationFrame(() => { stripFrame = 0; layoutStrip(); });
  }

  function layoutStrip(force = false) {
    const strip = $('tool-strip'), more = $('strip-more'), menu = $('strip-more-menu');
    const controls = strip.querySelector('.tool-controls:not([hidden])');
    const all = controls ? [...controls.querySelectorAll('[data-priority]'), ...[...parked.keys()].filter(item => controls.contains(parked.get(item)))] : [];
    const signature = [strip.clientWidth, editor.tool, $('selection-level').textContent, $('scope-trail').textContent, $('canvas-hint').textContent,
      ...all.map(item => `${item.hidden}${item.textContent.length}`)].join('|');
    if (!force && signature === stripSignature) return;
    stripSignature = signature;
    for (const [item, mark] of parked) mark.replaceWith(item);
    parked.clear();
    more.hidden = false;
    const moreWidth = more.getBoundingClientRect().width;
    more.hidden = true;
    const items = controls ? [...controls.querySelectorAll('[data-priority]')].filter(item => !item.hidden) : [];
    const style = getComputedStyle(strip), gap = parseFloat(style.columnGap) || 0;
    const inner = parseFloat(getComputedStyle(controls || strip).columnGap) || 0;
    // What the strip holds besides the controls, and the gaps between them all.
    const others = [...strip.children].filter(child => child !== controls && child !== more && child.id !== 'canvas-hint' && !child.hidden && getComputedStyle(child).display !== 'none');
    const fixed = others.reduce((sum, child) => sum + child.getBoundingClientRect().width + gap, 0) + ($('canvas-hint').textContent ? gap + 40 : 0);
    const available = strip.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight) - fixed;
    const widths = items.map(item => ({width: item.getBoundingClientRect().width + inner, priority: Number(item.dataset.priority)}));
    const hidden = overflowLayout(widths, available, moreWidth + gap);
    for (const index of hidden) {
      const item = items[index], mark = document.createComment('parked');
      item.before(mark); menu.append(item); parked.set(item, mark);
    }
    more.hidden = !hidden.length;
    if (!hidden.length) closeStripMenu();
    else if (!menu.hidden) placeStripMenu();
  }

  // A "⋯" button and the menu of what did not fit beside it. A command run from
  // the menu closes it; fields and choices keep it open.
  function moreMenu(button, menu) {
    const place = () => {
      const box = button.getBoundingClientRect(), size = menu.getBoundingClientRect();
      menu.style.left = `${Math.max(8, Math.min(box.left, innerWidth - size.width - 8))}px`;
      menu.style.top = `${box.bottom + 6}px`;
    };
    const close = () => { menu.hidden = true; button.setAttribute('aria-expanded', 'false'); };
    button.onclick = () => {
      if (!menu.hidden) { close(); return; }
      menu.hidden = false; button.setAttribute('aria-expanded', 'true');
      place();
      menu.querySelector('button:not(:disabled), input, select')?.focus();
    };
    menu.addEventListener('click', event => { if (event.target.closest('button')) close(); });
    menu.addEventListener('keydown', event => { if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); close(); button.focus(); } });
    window.addEventListener('pointerdown', event => { if (!menu.contains(event.target) && !button.contains(event.target)) close(); }, true);
    return {place, close};
  }

  const stripMenu = moreMenu($('strip-more'), $('strip-more-menu'));

  const placeStripMenu = stripMenu.place, closeStripMenu = stripMenu.close;

  // The top bar keeps to one row too: the title shrinks, then the file actions
  // that do not fit go into its "⋯" menu, least important first.
  const topbarPriority = {'export-svg': 1, 'palette-open': 1, 'save-as': 2, 'save-project': 2, 'open-file': 3};

  const topbarMenu = moreMenu($('topbar-more'), $('topbar-more-menu'));

  const topbarParked = new Map();

  function layoutTopbar() {
    const bar = document.querySelector('.topbar'), nav = bar.querySelector('nav'), more = $('topbar-more'), menu = $('topbar-more-menu');
    const signature = `${bar.clientWidth}|${$('filename').textContent}`;
    if (signature === topbarSignature) return;
    topbarSignature = signature;
    for (const [item, mark] of topbarParked) mark.replaceWith(item);
    topbarParked.clear();
    more.hidden = false;
    const moreWidth = more.getBoundingClientRect().width;
    more.hidden = true;
    const style = getComputedStyle(bar), gap = parseFloat(style.columnGap) || 0, inner = parseFloat(getComputedStyle(nav).columnGap) || 0;
    const title = bar.querySelector('.document-title'), name = $('filename');
    // The title keeps room for a short name while it is shown.
    const titleRoom = getComputedStyle(title).display === 'none' ? 0 : Math.min(name.scrollWidth + 30, 140) + gap;
    const available = bar.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight)
      - bar.querySelector('.brand').getBoundingClientRect().width - titleRoom - gap;
    const items = [...nav.children].filter(item => getComputedStyle(item).display !== 'none');
    const widths = items.map(item => ({width: item.getBoundingClientRect().width + inner, priority: topbarPriority[item.id] || 4}));
    const hidden = overflowLayout(widths, available + inner, moreWidth + gap);
    for (const index of hidden) {
      const item = items[index], mark = document.createComment('parked');
      item.before(mark); menu.append(item); topbarParked.set(item, mark);
    }
    more.hidden = !hidden.length;
    if (!hidden.length) topbarMenu.close();
    else if (!menu.hidden) topbarMenu.place();
  }

  // The level and count of the selection, and the entered group.
  function renderStatus() {
    reportView();
    if (!editor.state) return;
    scheduleStrip();
    const points = selectedPoints(), paths = level() === 'points' ? pointPaths() : editor.state.selection.objects;
    $('selection-level').textContent = selectionStatus(level() === 'points' ? 'points' : 'objects', paths, points);
    // Hidden objects have no frame on the canvas: say they are selected.
    const hidden = level() === 'points' ? 0 : editor.state.selection.objects.filter(id => object(id)?.inherited_hidden).length;
    if (hidden) $('selection-level').textContent += ` · ${hidden === paths.length ? (hidden === 1 ? 'hidden' : 'all hidden') : `${hidden} hidden`}`;
    // Show the selection's size, and the new size while resizing in Move.
    if (editor.drag?.kind === 'resize' && editor.drag.result) $('selection-level').textContent = `Resize · ${frameSize(editor.drag.result.box)}`;
    else if (editor.drag?.kind === 'rotate' && editor.drag.moved && editor.drag.result) $('selection-level').textContent = rotationStatus(editor.drag.result);
    else if (editor.drag?.kind === 'shape' && editor.drag.moved && editor.drag.shape) $('selection-level').textContent = shapeStatus(editor.drag.shape);
    else if (editor.drag?.kind === 'pencil' && editor.drag.moved && editor.drag.d) $('selection-level').textContent = strokeStatus(editor.drag.d);
    else if (editor.selectionBox) $('selection-level').textContent += ` · ${frameSize(editor.selectionBox)}`;
    // In Gradient, whose paint moving an end changes.
    else if (editor.tool === 'gradient') $('selection-level').textContent += ` · ${gradientStatus()}`;
    const trail = $('scope-trail');
    trail.replaceChildren();
    trail.hidden = !editor.scope;
    if (!editor.scope) return;
    const crumb = (label, id) => {
      const button = document.createElement('button'); button.className = 'crumb'; button.textContent = label;
      button.title = id ? `Pick within ${label}` : 'Leave the entered groups';
      button.onclick = () => { editor.scope = id; editor.clickCycle = null; renderStatus(); };
      trail.append(button);
    };
    crumb('Drawing', null);
    for (const id of scopeChain(editor.scope, parents(), editor.state.root)) { trail.append(' › '); crumb(object(id)?.label || id, id); }
  }

  function wire() {
    new ResizeObserver(() => scheduleStrip()).observe($('tool-strip'));

    new ResizeObserver(() => layoutTopbar()).observe(document.querySelector('.topbar'));

    new MutationObserver(() => layoutTopbar()).observe($('filename'), {childList: true, characterData: true, subtree: true});

    document.fonts?.ready.then(() => { topbarSignature = ''; layoutTopbar(); });
      return api;
  }
  const api = {parked, scheduleStrip, layoutStrip, moreMenu, stripMenu, placeStripMenu, closeStripMenu, topbarPriority, topbarMenu, topbarParked, layoutTopbar, renderStatus, wire};
  return api;
}
