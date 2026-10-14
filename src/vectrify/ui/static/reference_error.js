export function loadErrorImage(source) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Could not render the reference error view'));
    img.src = source;
  });
}

// Render one comparison at a time. Edits during a render replace the waiting
// request, so only the latest drawing is rendered next.
export function errorView(canvas, status, failed, render) {
  let current = null, cached = null, completed = null, pending = false, wanted = null;

  function paint(target, pixels) {
    canvas.width = pixels.width; canvas.height = pixels.height;
    canvas.getContext('2d').drawImage(pixels, 0, 0);
    cached = {key: target.key, frame: target.frame};
    canvas.hidden = false; status('ready');
  }

  function start() {
    if (pending || !wanted || wanted.key !== current) return;
    const target = wanted;
    wanted = null; pending = true;
    let result;
    try { result = render(...target.args); }
    catch (error) { result = Promise.reject(error); }
    Promise.resolve(result).then(pixels => {
      completed = {...target, pixels};
      if (target.key !== current) return;
      wanted = null; paint(target, pixels);
    }).catch(error => {
      if (target.key !== current) return;
      current = null; wanted = null; canvas.hidden = true;
      status('failed'); failed(error);
    }).finally(() => {
      pending = false; start();
    });
  }

  return {
    update(svg, bounds, reference) {
      const key = reference ? JSON.stringify([svg, bounds, reference]) : null;
      if (key === current) return;
      current = key; wanted = null;
      if (!key) { canvas.hidden = true; return; }
      if (cached?.key === key) { canvas.hidden = false; status('ready'); return; }
      if (completed?.key === key) { paint(completed, completed.pixels); return; }
      const frame = JSON.stringify([bounds, reference]);
      // Keep the last comparison visible during an edit, but never reuse its
      // placement after replacing the reference or changing the artboard.
      canvas.hidden = cached?.frame !== frame;
      status('loading');
      wanted = {key, frame, args: [svg, bounds, reference]};
      start();
    },
  };
}
