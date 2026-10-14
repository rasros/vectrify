

import {taperPreview} from './taper.js';

// Arrows owns its controls; editor contains the live state shared with other controllers.
export function createArrowsController(editor, {
  drawing,
  getComputedStyle = globalThis.getComputedStyle,
  svgElement,
} = {}) {

  // Match the backend's endpoint triangles during a node/move/resize preview.
  function syncArrowPreview(element) {
    const id = element.dataset.objectId;
    if (!id || element !== svgElement(id) || element.hasAttribute('data-vectrify-arrow-owner') || element.hasAttribute('data-vectrify-taper-owner')) return;
    const companions = [...drawing.querySelectorAll('[data-vectrify-arrow-owner], [data-vectrify-taper-owner]')].filter(arrow => (arrow.getAttribute('data-vectrify-arrow-owner') || arrow.getAttribute('data-vectrify-taper-owner')) === id);
    if (!companions.length) return;
    for (const arrow of companions) {
      const transform = element.getAttribute('transform');
      if (transform === null) arrow.removeAttribute('transform'); else arrow.setAttribute('transform', transform);
    }
    const geometry = editor.geometries.get(id), width = parseFloat(getComputedStyle(element).strokeWidth);
    if (!geometry || !Number.isFinite(width)) return;
    const paths = [];
    for (const sub of geometry.subpaths) {
      if (sub.closed || sub.nodes.length < 2) continue;
      const points = sub.nodes.flatMap(node => { const p = []; for (let i = 0; i < node.values.length; i += 2) p.push(node.values.slice(i, i + 2)); return p; });
      for (const [side, tip, remaining] of [['start', points[0], points.slice(1)], ['end', points.at(-1), points.slice(0, -1).reverse()]]) {
        if (element.getAttribute('data-vectrify-arrow-' + side) !== 'arrow') continue;
        const next = remaining.find(p => p[0] !== tip[0] || p[1] !== tip[1]); if (!next) continue;
        const length = Math.hypot(next[0] - tip[0], next[1] - tip[1]);
        const dx = (next[0] - tip[0]) / length, dy = (next[1] - tip[1]) / length;
        const x = tip[0] + 4 * width * dx, y = tip[1] + 4 * width * dy;
        paths.push(`M${tip.join(' ')} L${x + 2 * width * dy} ${y - 2 * width * dx} L${x - 2 * width * dy} ${y + 2 * width * dx} Z`);
      }
    }
    const tapered = companions.find(arrow => arrow.getAttribute('data-vectrify-taper-owner') === id);
    if (tapered) {
      const css = getComputedStyle(element);
      tapered.setAttribute('d', taperPreview(geometry, width, {start: element.getAttribute('data-vectrify-taper-start'), end: element.getAttribute('data-vectrify-taper-end'), cap: css.strokeLinecap, join: css.strokeLinejoin, miter: css.strokeMiterlimit}) + ' ' + paths.join(' '));
    } else companions.forEach((arrow, i) => arrow.setAttribute('d', paths[i] || ''));
  }

  function wire() {
    new MutationObserver(records => {
      if (!editor.drag) return;
      for (const element of new Set(records.map(record => record.target))) syncArrowPreview(element);
    }).observe(drawing, {subtree: true, attributes: true, attributeFilter: ['d', 'transform']});
      return api;
  }
  const api = {syncArrowPreview, wire};
  return api;
}
