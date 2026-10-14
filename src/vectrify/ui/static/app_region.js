import {closesPolygon, polygonArea, rectangleCorners, regionPreview} from './region.js';

// Each operation's command, its apply button and its busy label.
const OPERATIONS = {
  extract: {command: 'extract-region', name: 'Extract region', busy: 'Extracting region…'},
  delete: {command: 'delete-region', name: 'Delete region', busy: 'Deleting region…'},
};

// What to do before there is a region, by its shape.
const HINTS = {
  rectangle: 'Drag a rectangle over the part to take. A click selects objects.',
  polygon: 'Click the region\'s corners, then click the first corner or press Enter to close it.',
};

// Region owns the Region tool's panel and the region drawn with it; editor
// contains the live state shared with other controllers. Nothing changes
// until the person applies what the panel and the overlay show: the region,
// the objects it changes and whether those crossing its edge are cut. The
// preview is the session making each edit and dropping it, so it agrees
// with the edit, refusals included.
export function createRegionController(editor, {
  $,
  action,
  document = globalThis.document,
  drawOverlay,
  later,
  localToOverlay,
  object,
  plural,
  renderCommands,
  request,
  runCommand,
  selectionContour,
  setTool,
  stage,
  svgElement,
  toast,
  xmlElement,
} = {}) {
  // The panel's choices, kept while the tool is put down and picked up again.
  const options = {mode: 'extract', shape: 'rectangle', scope: 'selection', cut: true};
  // The last preview asked for: what it depends on, and its result or error.
  let preview = {key: null, result: null, error: ''};
  let shownPanel = '';

  const label = id => object(id)?.label || id;

  const closed = () => editor.region?.closed ? editor.region.points : null;

  // What the preview depends on: the drawing's version, the region, the
  // choices and what the scope reaches (the selection, or the entered group).
  function previewKey() {
    const points = closed();
    if (!points || !editor.state) return null;
    const reach = options.scope === 'selection' ? editor.state.selection.objects : editor.scope;
    return JSON.stringify([editor.state.epoch, editor.state.revision, points, options.scope, options.cut, reach]);
  }

  const payload = () => ({region: closed(), scope: options.scope, cut: options.cut, within: editor.scope});

  async function refreshPreview() {
    const key = previewKey();
    if (key === preview.key) return;
    preview = {key, result: null, error: ''};
    if (!key) return;
    const {epoch, revision} = editor.state;
    try {
      const result = await request('/api/region', {epoch, revision, ...payload()});
      // A preview of a later version waits for the page to show that version.
      if (preview.key === key && result.revision === revision) preview.result = result;
    } catch (error) { if (preview.key === key) preview.error = error.message; }
    // The commands' reasons, wherever they are listed, follow the preview.
    if (preview.key === key) { renderCommands(); drawOverlay(); }
  }

  const current = () => preview.key !== null && preview.key === previewKey() && (preview.result || preview.error) ? preview : null;

  // Why *mode* cannot apply the region now, or ''. Without a region, or for
  // the other operation, running it picks up the tool or switches to it.
  function refusal(mode) {
    if (!closed() || editor.tool !== 'region' || mode !== options.mode) return '';
    const shown = current();
    if (!shown) return 'Previewing the region…';
    return shown.error || shown.result[mode].error || '';
  }

  // Extract region and Delete region: pick up the tool with that operation,
  // show its preview when another was shown, and apply it once shown.
  async function command(mode) {
    if (editor.tool !== 'region') { options.mode = mode; await setTool('region'); drawOverlay(); return; }
    if (mode !== options.mode || !closed()) {
      options.mode = mode; drawOverlay();
      if (!closed()) toast(HINTS[options.shape]);
      return;
    }
    return apply();
  }

  async function apply() {
    const mode = options.mode, reason = refusal(mode);
    if (!closed()) return;
    if (reason) { toast(reason, true); return; }
    const count = current().result[mode].targets.length;
    if (await action('extract', {...payload(), delete: mode === 'delete'}, OPERATIONS[mode].busy)) {
      editor.region = null; editor.regionHover = null; drawOverlay();
      toast(mode === 'extract'
        ? `Extracted the region from ${plural(count, 'object')} into paths of their own, now selected. Undo puts it back.`
        : `Deleted the region from ${plural(count, 'object')}. Undo restores it.`);
    }
  }

  // A dragged rectangle is a region at once.
  function rectangle({start, end}) {
    const corners = rectangleCorners(start, end);
    if (!polygonArea(corners)) return;
    editor.region = {points: corners, closed: true}; editor.regionHover = null;
    drawOverlay();
  }

  // A click places a polygon's next corner, or on its first corner closes it.
  function corner(p) {
    if (!editor.region || editor.region.closed) editor.region = {points: [], closed: false};
    if (closesPolygon(editor.region.points, p, 8 / editor.zoom)) { close(); return; }
    editor.region.points.push([p.x, p.y]);
    drawOverlay();
  }

  // Close the polygon being drawn; whether there was one to close.
  function close() {
    const region = editor.region;
    if (!region || region.closed || region.points.length < 3) return false;
    if (!polygonArea(region.points)) { toast('The region needs corners that enclose an area', true); return true; }
    region.closed = true; editor.regionHover = null;
    drawOverlay();
    return true;
  }

  // Enter closes the polygon being drawn, or else applies the region.
  function enter() {
    if (!close() && closed()) return runCommand(OPERATIONS[options.mode].command);
  }

  // Backspace takes back the last corner of the polygon being drawn.
  function back() {
    const region = editor.region;
    if (!region || region.closed) return false;
    region.points.pop();
    if (!region.points.length) editor.region = null;
    drawOverlay();
    return true;
  }

  function cancel() { editor.region = null; editor.regionHover = null; drawOverlay(); }

  function setOption(name, value) {
    options[name] = value;
    // A new shape starts a new region.
    if (name === 'shape') { editor.region = null; editor.regionHover = null; }
    renderCommands(); drawOverlay();
  }

  // The overlay: the region (or the rectangle being dragged, or the polygon
  // being drawn up to the pointer) over the objects the preview says change,
  // coloured by what happens to each.
  function drawRegion() {
    if (editor.tool !== 'region' || !editor.state) return;
    refreshPreview();
    renderPanel();
    const dragged = editor.drag?.kind === 'region' && editor.drag.moved ? rectangleCorners(editor.drag.start, editor.drag.end) : null;
    const points = dragged || editor.region?.points;
    if (!points?.length) return;
    const group = xmlElement('g', {'pointer-events': 'none', 'aria-hidden': 'true'});
    const shown = !dragged && current();
    if (shown?.result) {
      for (const row of regionPreview(shown.result, options.mode, options.cut, label).rows) {
        const element = svgElement(row.id), matrix = element && localToOverlay(element), contour = matrix && selectionContour(element);
        if (!contour) continue;
        contour.removeAttribute('transform');
        const target = xmlElement('g', {transform: matrix.toString(), class: `region-target ${row.outcome}`});
        target.append(contour); group.append(target);
      }
    }
    const open = !dragged && !editor.region.closed;
    const outline = open && editor.regionHover ? [...points, [editor.regionHover.x, editor.regionHover.y]] : points;
    group.append(xmlElement(open ? 'polyline' : 'polygon', {points: outline.map(p => p.join(',')).join(' '), class: `region-area ${options.mode}`}));
    if (open) points.forEach(([x, y], i) => group.append(xmlElement('circle', {cx: x, cy: y, r: (i ? 3.5 : 5) / editor.zoom, class: 'region-corner'})));
    editor.overlayFrame.content.append(group);
  }

  // The panel: the choices, what the preview says and the button applying it.
  function renderPanel() {
    const points = closed(), shown = points && current(), mode = options.mode;
    const view = !points ? {summary: editor.region ? `${plural(editor.region.points.length, 'corner')} placed. ${HINTS.polygon}` : HINTS[options.shape], rows: []}
      : !shown ? {summary: 'Previewing the region…', rows: []}
        : shown.error ? {summary: shown.error, rows: []}
          : regionPreview(shown.result, mode, options.cut, label);
    const reason = points ? refusal(mode) : 'Draw a region first';
    const signature = JSON.stringify([options, view, reason]);
    if (signature === shownPanel) return;
    shownPanel = signature;
    for (const button of document.querySelectorAll('[data-region-mode]')) button.setAttribute('aria-pressed', String(button.dataset.regionMode === mode));
    for (const button of document.querySelectorAll('[data-region-shape]')) button.setAttribute('aria-pressed', String(button.dataset.regionShape === options.shape));
    for (const input of document.querySelectorAll('input[name=region-scope]')) input.checked = input.value === options.scope;
    $('region-cut').checked = options.cut;
    $('region-summary').textContent = view.summary;
    $('region-summary').classList.toggle('refused', !!points && !!shown && !!(shown.error || shown.result?.[mode].error));
    const rows = document.createDocumentFragment();
    for (const row of view.rows) {
      const item = document.createElement('li'), mark = document.createElement('span'), text = document.createElement('span');
      const name = document.createElement('strong'), detail = document.createElement('small');
      item.className = `region-row ${row.outcome}`; item.dataset.object = row.id;
      mark.className = 'region-mark'; mark.setAttribute('aria-hidden', 'true');
      name.textContent = row.label; detail.textContent = row.detail;
      text.append(name, detail); item.append(mark, text); rows.append(item);
    }
    $('region-targets').replaceChildren(rows);
    $('region-targets').hidden = !view.rows.length;
    const apply = $('region-apply');
    apply.textContent = OPERATIONS[mode].name;
    apply.classList.toggle('danger', mode === 'delete');
    apply.disabled = !!reason;
    apply.title = reason || `${OPERATIONS[mode].name} as shown (Enter)`;
    $('region-cancel').disabled = !editor.region;
  }

  function wire() {
    const panel = $('region-panel');
    // The panel floats over the canvas: pressing or scrolling it is not the canvas's.
    for (const name of ['pointerdown', 'wheel', 'dblclick']) panel.addEventListener(name, event => event.stopPropagation());
    // Escape cancels the region from the panel too, as it does on the canvas.
    // The keys that press and change its controls stay theirs, not the
    // canvas's: Enter on Cancel cancels rather than applying.
    panel.addEventListener('keydown', event => {
      if (event.key === 'Escape') {
        event.preventDefault(); event.stopPropagation(); cancel(); stage.focus({preventScroll: true});
      } else if (['Enter', ' ', 'Backspace', 'Delete'].includes(event.key)) event.stopPropagation();
    });
    for (const button of document.querySelectorAll('[data-region-mode]')) button.onclick = () => setOption('mode', button.dataset.regionMode);
    for (const button of document.querySelectorAll('[data-region-shape]')) button.onclick = () => setOption('shape', button.dataset.regionShape);
    for (const input of document.querySelectorAll('input[name=region-scope]')) input.onchange = () => setOption('scope', input.value);
    $('region-cut').onchange = () => setOption('cut', $('region-cut').checked);
    $('region-apply').onclick = () => later(() => runCommand(OPERATIONS[options.mode].command));
    $('region-cancel').onclick = cancel;
    return api;
  }
  const api = {options, refusal, command, apply, rectangle, corner, close, enter, back, cancel, drawRegion, renderPanel, wire};
  return api;
}
