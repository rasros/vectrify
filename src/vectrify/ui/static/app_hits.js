

// Hits owns its controls; editor contains the live state shared with other controllers.
export function createHitsController(editor, {
  document = globalThis.document,
  object,
  svgElement,
  targetId,
} = {}) {

  function hitStack(x, y) {
    // Browser hit testing respects SVG fill/stroke, clipping, transforms and use
    // instances, including painted shapes hidden behind another painted shape.
    return [...new Set(document.elementsFromPoint(x, y).map(targetId).filter(id => {
      const item = object(id);
      return item && (!item.resource || svgElement(id)?.closest('clipPath')) && !['g', 'svg', 'defs', 'clipPath'].includes(item.tag);
    }))];
  }

  function wire() {
      return api;
  }
  const api = {hitStack, wire};
  return api;
}
