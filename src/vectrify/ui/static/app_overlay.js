import {clippingFrames} from './clipping.js';
import {FRAME_REACH, HANDLE_SPREAD} from './app_config.js';
import {splitKey} from './selection.js';
import {unionBox} from './align.js';

// Overlay owns its controls; editor contains the live state shared with other controllers.
export function createOverlayController(editor, {
  $,
  CSS = globalThis.CSS,
  DOMPoint = globalThis.DOMPoint,
  drawFlash,
  drawGradient = () => {},
  drawKeyObject,
  drawKnife,
  drawOverlay,
  drawPathDraft,
  drawPoints,
  drawRedraw,
  drawEyedropper,
  drawRegion,
  drawRotation = () => {},
  drawShape,
  drawSnap,
  drawing,
  getComputedStyle = globalThis.getComputedStyle,
  later,
  level,
  localToOverlay,
  nodeAt,
  nodeHandles,
  object,
  overlay,
  renderStatus,
  resizeRefusal,
  svgElement,
  topSelection,
  xmlElement,
} = {}) {

  function selectionContour(source, seen = new Set(), inheritedPaint = null) {
    if (seen.has(source) || ['defs', 'clipPath'].includes(source.localName)) return null;
    const branch = new Set(seen); branch.add(source);
    const style = inheritedPaint || getComputedStyle(source);
    const paint = {
      fill: source.getAttribute('fill') || style.fill,
      fillRule: source.getAttribute('fill-rule') || style.fillRule,
    };
    // Expand instances so explicit paint on the source cannot hide the highlight.
    const clone = source.localName === 'use' ? xmlElement('g') : source.cloneNode(false);
    if (source.localName === 'use') {
      for (const attr of source.attributes) clone.setAttributeNS(attr.namespaceURI, attr.name, attr.value);
      const href = source.getAttribute('href') || source.getAttributeNS('http://www.w3.org/1999/xlink', 'href');
      const target = href?.startsWith('#') ? drawing.querySelector(`#${CSS.escape(href.slice(1))}`) : null;
      const child = target && selectionContour(target, branch, paint);
      if (child) {
        const offset = xmlElement('g', {transform: `translate(${source.getAttribute('x') || 0} ${source.getAttribute('y') || 0})`});
        offset.append(child); clone.append(offset);
      }
    } else {
      for (const child of source.children) {
        const contour = selectionContour(child, branch, paint);
        if (contour) clone.append(contour);
      }
    }
    clone.removeAttribute('id'); clone.removeAttribute('data-object-id');
    clone.removeAttribute('class'); clone.removeAttribute('style');
    clone.removeAttribute('opacity');
    if (['path', 'rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon'].includes(source.localName)) {
      clone.setAttribute('class', `selection-contour${paint.fill !== 'none' && source.localName !== 'line' ? ' selection-filled' : ''}`);
      clone.setAttribute('fill-rule', paint.fillRule || 'nonzero');
    }
    return clone;
  }

  function selectionTargets(id) {
    const element = svgElement(id);
    if (!element) return [];
    const clip = element.closest('clipPath');
    if (clip) return element === clip ? [...clip.querySelectorAll('path, rect, circle, ellipse, line, use')] : [element];
    // Definitions are not painted themselves; highlight their visible instances.
    if (element.closest('defs')) {
      return [...drawing.querySelectorAll('use')].filter(use => !use.closest('defs') &&
        (use.getAttribute('href') || use.getAttributeNS('http://www.w3.org/1999/xlink', 'href')) === `#${element.id}`);
    }
    return [element, ...[...drawing.querySelectorAll('[data-vectrify-arrow-owner]')].filter(arrow => arrow.getAttribute('data-vectrify-arrow-owner') === id)];
  }

  function clipSelection(group, element) {
    // A selected child must keep the clipping inherited from its drawing groups.
    for (let ancestor = element.parentElement; ancestor && ancestor !== drawing; ancestor = ancestor.parentElement) {
      const clip = ancestor.getAttribute('clip-path'), matrix = localToOverlay(ancestor);
      if (!clip || !matrix) continue;
      const wrapper = xmlElement('g', {transform: matrix.toString(), 'clip-path': clip});
      const content = xmlElement('g', {transform: matrix.inverse().toString()});
      content.append(group);
      // Preserve the whole ancestor's bounds for objectBoundingBox clips.
      const b = ancestor.getBBox();
      wrapper.append(xmlElement('rect', {x:b.x,y:b.y,width:b.width,height:b.height,fill:'none',stroke:'none'}), content);
      group = wrapper;
    }
    return group;
  }

  function drawOverlayContent() {
    const targets = new Set(editor.state.selection.objects.flatMap(selectionTargets));
    editor.selectionBox = selectionFrame();
    for (const element of targets) {
      // Clip contours are drawn in all of their consumer frames below.
      if (element.closest('clipPath')) continue;
      const matrix = localToOverlay(element);
      if (!element?.getBBox || !matrix) continue;
      try {
        const contour = selectionContour(element);
        if (contour) {
          // The screen matrix already includes the selected object's transform.
          contour.removeAttribute('transform');
          let group = xmlElement('g', {transform: matrix.toString(), 'aria-hidden': 'true'});
          group.append(contour);
          group = clipSelection(group, element);
          const halo = group.cloneNode(true); halo.setAttribute('class', 'selection-halo');
          editor.overlayFrame.content.append(halo, group);
        }
        // Point tools show the points instead of the bounds, and one object's
        // frame stands for its bounds.
        if (level() === 'points' || (editor.selectionBox && targets.size === 1)) continue;
        const b = element.getBBox();
        const pts = [[b.x,b.y], [b.x+b.width,b.y], [b.x+b.width,b.y+b.height], [b.x,b.y+b.height]].map(([x,y]) => new DOMPoint(x,y).matrixTransform(matrix));
        editor.overlayFrame.content.append(xmlElement('polygon', {points: pts.map(p => `${p.x},${p.y}`).join(' '), class: 'selection-box'}));
      } catch { /* Resource elements may have no display bounds. */ }
    }
    drawClippingBoundaries();
    drawFrame();
    drawKeyObject();
    drawRotation();
    drawGradient();
    drawPathDraft();
    drawShape();
    drawKnife();
    drawRegion();
    drawEyedropper();
    drawRedraw();
    drawSnap();
    drawBox();
    if (level() === 'points') drawPoints();
    drawFlash();
    renderStatus();
  }

  function toggleClipping() {
    editor.showClipping = !editor.showClipping;
    $('clipping-toggle').setAttribute('aria-pressed', String(editor.showClipping));
    $('clipping-toggle').textContent = editor.showClipping ? 'Hide clips' : 'Show clips';
    editor.clickCycle = null;
    drawOverlay();
  }

  function drawClippingBoundaries() {
    const selected = new Set(editor.state.selection.objects);
    for (const item of editor.state.objects) {
      if (['defs', 'clipPath', 'g'].includes(item.tag)) continue;
      const element = svgElement(item.id), clip = element?.closest('clipPath');
      if (!clip) continue;
      let picked = false;
      for (let at = element; at; at = at.parentElement) {
        if (selected.has(at.dataset.objectId)) picked = true;
        if (at === clip) break;
      }
      if (!picked && !editor.showClipping) continue;
      const frames = clippingFrames(element, drawing, editor.overlayFrame.to, editor.clipConsumers.get(clip.dataset.objectId));
      for (const {consumer, matrix} of frames) {
        const contour = selectionContour(element);
        if (!contour) continue;
        contour.removeAttribute('transform');
        // The boundary is a guide, even when the clip's geometry has no fill.
        contour.removeAttribute('clip-path');
        for (const child of contour.querySelectorAll('[clip-path]')) child.removeAttribute('clip-path');
        const group = xmlElement('g', {transform: matrix.toString(), class: `clipping-boundary${picked ? ' selected' : ''}`,
          'data-clip-object': item.id, 'data-clip-consumer': consumer.dataset.objectId, 'aria-hidden': 'true'});
        const hit = contour.cloneNode(true);
        const hitGroup = xmlElement('g', {class: 'clipping-hit'});
        hitGroup.append(hit); group.append(hitGroup, contour);
        editor.overlayFrame.content.append(group);
      }
    }
  }

  // Both object tools show the selection bounds. Move adds resize handles;
  // Select keeps the frame passive so every drag can box-select.
  function selectionFrame() {
    return ['select', 'move'].includes(editor.tool) ? selectionBounds() : null;
  }

  // One object's geometric bounds in document coordinates: its own box (its
  // outline, without stroke, markers or clipping) mapped through its
  // transforms, or null when it has no size or is not drawn.
  function objectBounds(id) {
    const element = svgElement(id), matrix = element?.getBBox && localToOverlay(element);
    if (!matrix) return null;
    try {
      const b = element.getBBox();
      if (!b.width && !b.height) return null;
      const xs = [], ys = [];
      for (const [x, y] of [[b.x,b.y], [b.x+b.width,b.y], [b.x+b.width,b.y+b.height], [b.x,b.y+b.height]]) {
        const p = new DOMPoint(x, y).matrixTransform(matrix); xs.push(p.x); ys.push(p.y);
      }
      return {left: Math.min(...xs), top: Math.min(...ys), right: Math.max(...xs), bottom: Math.max(...ys)};
    } catch { return null; /* Unrendered objects have no bounds. */ }
  }

  // The selection's geometric bounds in document coordinates, whatever the
  // tool: the top selected objects' bounds joined. The frame shows it, the
  // Transform fields edit it, and Align and Distribute measure by the same
  // bounds. Objects with no size add nothing; definitions and clipping
  // boundaries have no bounds here.
  function selectionBounds() {
    if (!editor.state?.selection.objects.length) return null;
    const ids = topSelection();
    if (ids.some(id => !object(id) || object(id).resource || ['defs', 'clipPath', 'svg'].includes(object(id).tag))) return null;
    const boxes = ids.map(objectBounds).filter(Boolean);
    return boxes.length ? unionBox(boxes) : null;
  }

  function drawFrame() {
    // While turning, the rotation draws the frame turned with the selection.
    if (!editor.selectionBox || (editor.drag?.kind === 'rotate' && editor.drag.moved)) return;
    const {left, top, right, bottom} = editor.selectionBox, locked = editor.tool === 'move' && !!resizeRefusal();
    editor.overlayFrame.content.append(xmlElement('rect', {x: left, y: top, width: right - left, height: bottom - top, class: `resize-frame${locked ? ' locked' : ''}`}));
    if (locked || editor.tool !== 'move' || Math.min(right - left, bottom - top) * editor.zoom <= 2 * FRAME_REACH) return;
    const r = 2.5 / editor.zoom;
    for (const [x, y] of [[left, top], [right, top], [right, bottom], [left, bottom]]) editor.overlayFrame.content.append(xmlElement('rect', {x: x - r, y: y - r, width: 2 * r, height: 2 * r, class: 'resize-tick'}));
  }

  function drawHandles(key) {
    const [id] = splitKey(key), node = nodeAt(key), matrix = localToOverlay(svgElement(id));
    if (!node || !matrix) return;
    for (const handle of nodeHandles(key)) {
      let p = new DOMPoint(...handle.node.values.slice(handle.offset, handle.offset+2)).matrixTransform(matrix);
      const anchor = new DOMPoint(...node.values.slice(-2)).matrixTransform(matrix);
      // A handle on its point is retracted: none. One closer than
      // HANDLE_SPREAD is drawn that far out along its direction, on a dashed
      // line, so it shows clear of the point; a drag puts it at the pointer.
      const length = Math.hypot(p.x - anchor.x, p.y - anchor.y) * editor.zoom;
      if (length < 1e-6) continue;
      const short = length < HANDLE_SPREAD;
      if (short) p = new DOMPoint(anchor.x + (p.x - anchor.x) * HANDLE_SPREAD / length, anchor.y + (p.y - anchor.y) * HANDLE_SPREAD / length);
      editor.overlayFrame.content.append(xmlElement('line', {x1:anchor.x,y1:anchor.y,x2:p.x,y2:p.y,class:`handle-line${short ? ' short' : ''}`}));
      const near = editor.nearPoint === `${id} ${handle.node.id} ${handle.offset}`;
      const picked = editor.activeHandle?.object === id && editor.activeHandle.node === handle.node.id && editor.activeHandle.offset === handle.offset;
      const circle = xmlElement('circle', {cx:p.x,cy:p.y,r:(picked ? 4.8 : 3.8)*(near ? 1.5 : 1)/editor.zoom,class:`handle${picked ? ' selected' : ''}${near ? ' near' : ''}`});
      circle.dataset.object = id; circle.dataset.node = handle.node.id; circle.dataset.part = String(handle.offset); circle.dataset.anchor = key; editor.overlayFrame.content.append(circle);
    }
  }

  // The rubber band of a box select, in the overlay's frame.
  function drawBox() {
    if (editor.drag?.kind !== 'box' || !editor.drag.moved) return;
    const inverse = overlay.getScreenCTM()?.inverse(); if (!inverse) return;
    const a = new DOMPoint(editor.drag.x, editor.drag.y).matrixTransform(inverse), b = new DOMPoint(editor.drag.end.x, editor.drag.end.y).matrixTransform(inverse);
    editor.overlayFrame.content.append(xmlElement('rect', {x:Math.min(a.x,b.x), y:Math.min(a.y,b.y), width:Math.abs(a.x-b.x), height:Math.abs(a.y-b.y), class:'select-box'}));
  }

  function wire() {
    $('clipping-toggle').onclick = () => later(toggleClipping);
      return api;
  }
  const api = {selectionContour, selectionTargets, clipSelection, drawOverlayContent, toggleClipping, drawClippingBoundaries, selectionFrame, objectBounds, selectionBounds, drawFrame, drawHandles, drawBox, wire};
  return api;
}
