export function loadErrorImage(source) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const failed = () => reject(new Error('Could not render the reference error view'));
    if (img.decode) {
      img.src = source;
      img.decode().then(() => resolve(img), failed);
    } else {
      img.onload = () => resolve(img); img.onerror = failed; img.src = source;
    }
  });
}

// Render one comparison at a time. Edits during a render replace the waiting
// request, so only the latest drawing is rendered next.
export function errorView(canvas, status, failed, render, published = () => {}) {
  let current = null, cached = null, completed = null, pending = false, wanted = null;

  function sameKey(a, b) {
    if (!a || !b) return a === b;
    return a.svg === b.svg && a.bounds === b.bounds && a.reference === b.reference && a.box === b.box;
  }

  function paint(target, pixels) {
    // Prepare the complete replacement offscreen. The displayed canvas is
    // never cleared or resized; one DOM swap publishes the finished frame.
    const replacement = canvas.cloneNode(false);
    replacement.width = pixels.width; replacement.height = pixels.height;
    replacement.getContext('2d').drawImage(pixels, 0, 0);
    replacement.hidden = false;
    canvas.replaceWith(replacement); canvas = replacement;
    cached = {key: target.key, pixels}; status('ready');
    published(pixels);
  }

  function start() {
    if (pending || !wanted || !sameKey(wanted.key, current)) return;
    const target = wanted;
    wanted = null; pending = true;
    let result;
    try { result = render(...target.args); }
    catch (error) { result = Promise.reject(error); }
    Promise.resolve(result).then(pixels => {
      completed = {...target, pixels};
      if (!sameKey(target.key, current)) return;
      wanted = null; paint(target, pixels);
    }).catch(error => {
      if (!sameKey(target.key, current)) return;
      current = null; wanted = null;
      status('failed'); failed(error);
    }).finally(() => {
      pending = false; start();
    });
  }

  return {
    update(svg, bounds, reference) {
      // Compare existing strings directly instead of serializing the entire
      // SVG and base64 reference into a new multi-megabyte string per edit.
      const key = reference ? {svg, bounds: bounds.join(), reference: reference.data_url ?? reference, box: reference.box?.join() || ''} : null;
      if (sameKey(key, current)) {
        if (cached && sameKey(cached.key, key)) published(cached.pixels);
        return;
      }
      current = key; wanted = null;
      if (!key) { canvas.hidden = true; return; }
      if (sameKey(cached?.key, key)) { canvas.hidden = false; status('ready'); published(cached.pixels); return; }
      if (sameKey(completed?.key, key)) { paint(completed, completed.pixels); return; }
      // Keep the last complete comparison even if the next reference or
      // artboard has changed. Loading and failed refreshes never blank it.
      canvas.hidden = !cached;
      status('loading');
      wanted = {key, args: [svg, bounds, reference]};
      start();
    },
  };
}
