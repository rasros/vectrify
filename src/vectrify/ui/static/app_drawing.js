

// Drawing owns its controls; editor contains the live state shared with other controllers.
export function createDrawingController(editor, {
  DOMParser = globalThis.DOMParser,
  document = globalThis.document,
  drawing,
  refreshErrorView,
} = {}) {

  function renderDrawing() {
    const parsed = new DOMParser().parseFromString(editor.state.svg, 'image/svg+xml');
    const root = document.importNode(parsed.documentElement, true);
    editor.svgElements = new Map();
    // Isolate drawing IDs from editor controls while keeping local SVG references.
    for (const element of [root, ...root.querySelectorAll('*')]) {
      if (element.id) { editor.svgElements.set(element.id, element); element.dataset.objectId = element.id; element.id = `art-${element.id}`; }
      for (const attribute of [...element.attributes]) {
        if (attribute.localName === 'href' && attribute.value.startsWith('#')) {
          element.setAttributeNS(attribute.namespaceURI, attribute.name, `#art-${attribute.value.slice(1)}`);
        } else if (['clip-path', 'fill', 'stroke'].includes(attribute.localName) && attribute.value.startsWith('url(#')) {
          element.setAttribute(attribute.localName, `url(#art-${attribute.value.slice(5,-1)})`);
        }
      }
    }
    for (const arrow of root.querySelectorAll('[data-vectrify-arrow-owner], [data-vectrify-taper-owner]')) arrow.dataset.objectId = arrow.getAttribute('data-vectrify-arrow-owner') || arrow.getAttribute('data-vectrify-taper-owner');
    root.setAttribute('width', editor.state.bounds[2]); root.setAttribute('height', editor.state.bounds[3]);
    root.setAttribute('viewBox', editor.state.bounds.join(' ')); root.setAttribute('preserveAspectRatio', 'none');
    root.setAttribute('aria-label', editor.state.name);
    drawing.replaceChildren(root);
    refreshErrorView();
  }

  function wire() {
      return api;
  }
  const api = {renderDrawing, wire};
  return api;
}
