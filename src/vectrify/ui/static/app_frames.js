import {clippingFrames} from './clipping.js';

// Frames owns its controls; editor contains the live state shared with other controllers.
export function createFramesController(editor, {
  DOMMatrix = globalThis.DOMMatrix,
  document = globalThis.document,
  drawOverlayContent,
  drawing,
  overlay,
} = {}) {

  function localToOverlay(element) {
    // Agent updates include paint resources such as gradients, which have no
    // canvas frame. Only graphics elements can be positioned in the overlay.
    if (editor.overlayFrame?.matrices.has(element)) return editor.overlayFrame.matrices.get(element);
    const clip = element?.closest?.('clipPath');
    if (clip) {
      const matrix = clippingFrames(element, drawing, editor.overlayFrame ? editor.overlayFrame.to : overlay.getScreenCTM(), editor.clipConsumers.get(clip.dataset.objectId))[0]?.matrix || null;
      editor.overlayFrame?.matrices.set(element, matrix);
      return matrix;
    }
    if (typeof element?.getScreenCTM !== 'function') return null;
    const from = element?.getScreenCTM(), to = editor.overlayFrame ? editor.overlayFrame.to : overlay.getScreenCTM();
    if (!from || !to) return null;
    try {
      const matrix = DOMMatrix.fromMatrix(to.inverse().multiply(from));
      editor.overlayFrame?.matrices.set(element, matrix);
      return matrix;
    } catch { return null; }
  }

  function screenMatrix(element) {
    if (!element?.closest?.('clipPath')) return element?.getScreenCTM();
    const matrix = localToOverlay(element), screen = overlay.getScreenCTM();
    return matrix && screen ? DOMMatrix.fromMatrix(screen).multiply(matrix) : null;
  }

  function drawOverlay() {
    // The artboard zoom is a CSS transform, outside SVG's non-scaling-stroke.
    overlay.style.setProperty('--selection-scale', 1 / editor.zoom);
    if (!editor.state) { overlay.replaceChildren(); return; }
    // Read the drawing while all new overlay shapes are detached, then publish
    // them together. Appending each shape before the next path's CTM or bounds
    // read forced layout repeatedly during selection, node display and dragging.
    const content = document.createDocumentFragment();
    editor.overlayFrame = {content, to: overlay.getScreenCTM(), matrices: new Map()};
    try { drawOverlayContent(); overlay.replaceChildren(content); }
    finally { editor.overlayFrame = null; }
  }

  function wire() {
      return api;
  }
  const api = {localToOverlay, screenMatrix, drawOverlay, wire};
  return api;
}
