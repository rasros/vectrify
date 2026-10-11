export function loadErrorImage(source) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Could not render the reference error view'));
    img.src = source;
  });
}

// Cache only the current drawing/reference pair; ignore completions from a
// previous edit, replaced reference or a view the person has already left.
export function errorView(canvas, status, failed, render) {
  let generation = 0, current = null, cached = null, pending = null;
  return {
    update(svg, bounds, reference) {
      const key = reference ? JSON.stringify([svg, bounds, reference]) : null;
      if (key === current) return;
      current = key;
      const token = ++generation;
      canvas.hidden = true;
      if (!key) return;
      if (cached?.key === key) { canvas.hidden = false; status('ready'); return; }
      status('loading');
      // Leaving and returning to a view may reuse an in-flight rasterization.
      if (pending?.key !== key) pending = {key, result: render(svg, bounds, reference)};
      pending.result.then(pixels => {
        if (token !== generation) return;
        canvas.width = pixels.width; canvas.height = pixels.height;
        canvas.getContext('2d').drawImage(pixels, 0, 0);
        cached = {key}; canvas.hidden = false; status('ready');
      }).catch(error => {
        if (token !== generation) return;
        current = null; pending = null; status('failed'); failed(error);
      });
    },
  };
}
