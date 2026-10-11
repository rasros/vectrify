// A clip is drawn in each consumer's coordinate system, rather than where
// its definition happens to live. These frames also position editable nodes.
export function clippingFrames(element, drawing, to, preferred = null) {
  const clip = element?.closest?.('clipPath');
  if (!clip || !to) return [];
  const consumers = [...drawing.querySelectorAll('[clip-path]')].filter(item =>
    !item.closest('defs, clipPath') && item.getAttribute('clip-path') === `url(#${clip.id})`);
  // An unused definition can still be selected in the tree and edited in
  // artboard coordinates until it has a consumer.
  if (!consumers.length) consumers.push(drawing.querySelector('svg'));
  consumers.sort((a, b) => Number(b.dataset.objectId === preferred) - Number(a.dataset.objectId === preferred));
  let local = new DOMMatrix();
  for (let at = element; at; at = at.parentElement) {
    const transform = at.transform?.baseVal.consolidate()?.matrix;
    if (transform) local = DOMMatrix.fromMatrix(transform).multiply(local);
    if (at === clip) break;
  }
  return consumers.flatMap(consumer => {
    const screen = consumer?.getScreenCTM();
    if (!screen) return [];
    try {
      let matrix = DOMMatrix.fromMatrix(to.inverse().multiply(screen));
      if (clip.getAttribute('clipPathUnits') === 'objectBoundingBox') {
        const b = consumer.getBBox();
        if (!b.width || !b.height) return [];
        matrix = matrix.translate(b.x, b.y).scale(b.width, b.height);
      }
      return [{consumer, matrix: matrix.multiply(local)}];
    } catch { return []; }
  });
}
