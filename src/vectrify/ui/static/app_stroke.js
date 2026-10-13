

// Stroke owns its controls; editor contains the live state shared with other controllers.
export function createStrokeController(editor, {
  $,
  enable,
  object,
  paint,
  resolvedPaint,
  toast,
} = {}) {

  // Stroke endpoint and corner controls share the existing queued paint edits.
  function renderStrokeControls() {
    const items = editor.state.selection.objects.map(object);
    const shapes = items.length && items.every(item => item && !item.resource && ['path', 'line', 'rect', 'circle', 'ellipse'].includes(item.tag));
    $('stroke-shape-controls').hidden = !shapes;
    const open = shapes && items.every(item => item.open_path);
    $('path-end-controls').hidden = !open;
    const fields = [['stroke-linecap', 'stroke-linecap', 'butt'], ['stroke-linejoin', 'stroke-linejoin', 'miter'], ['stroke-miterlimit', 'stroke-miterlimit', '4'], ['start-arrow', 'data-vectrify-arrow-start', 'none'], ['end-arrow', 'data-vectrify-arrow-end', 'none']];
    for (const [id, attr, fallback] of fields) {
      const values = items.map(item => attr.startsWith('data-') ? item?.attributes?.[attr] || fallback : resolvedPaint(item.id, attr, fallback));
      $(id).value = values.length && values.every(v => v === values[0]) ? values[0] : '';
      const locked = items.some(item => item?.inherited_locks?.some(lock => lock === 'paint' || lock === attr));
      enable($(id), !shapes ? 'Select a path or shape' : locked ? 'This stroke property is locked' : '');
    }
    $('miter-limit-field').hidden = !!$('stroke-linejoin').value && $('stroke-linejoin').value !== 'miter';
  }

  function wire() {
    for (const [id, attr] of [['stroke-linecap', 'stroke-linecap'], ['stroke-linejoin', 'stroke-linejoin'], ['start-arrow', 'data-vectrify-arrow-start'], ['end-arrow', 'data-vectrify-arrow-end']]) {
      $(id).onchange = event => { if (event.target.value) paint({[attr]: event.target.value}); };
    }

    $('stroke-miterlimit').onchange = event => {
      const value = Number(event.target.value);
      if (!Number.isFinite(value) || value < 1) { toast('Miter limit must be at least 1', true); renderStrokeControls(); return; }
      paint({'stroke-miterlimit': String(value)});
    };
      return api;
  }
  const api = {renderStrokeControls, wire};
  return api;
}
