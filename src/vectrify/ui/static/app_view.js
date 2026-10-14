import {clippingFrames} from './clipping.js';
import {MAX_ZOOM, hints, names} from './app_config.js';
import {pointOwners, splitKey, switchTool} from './selection.js';
import {SHAPE_TOOLS} from './shapes.js';

// View owns its controls; editor contains the live state shared with other controllers.
export function createViewController(editor, {
  $,
  DOMPoint = globalThis.DOMPoint,
  action,
  document = globalThis.document,
  drawOverlay,
  drawing,
  getComputedStyle = globalThis.getComputedStyle,
  level,
  loadGeometries,
  localToOverlay,
  overlay,
  renderInspector,
  reportView,
  screenMatrix,
  selectedPoints,
  selectionTargets,
  setBusy,
  stage,
  toast,
} = {}) {

  function focusSelection() {
    if (!editor.state?.selection.objects.length) return;
    const targets = new Set(editor.state.selection.objects.flatMap(selectionTargets));
    const points = [];
    for (const element of targets) {
      const clip = element.closest('clipPath');
      const matrices = clip ? clippingFrames(element, drawing, overlay.getScreenCTM()).map(frame => frame.matrix) : [localToOverlay(element)];
      if (!element.getBBox) continue;
      try {
        const b = element.getBBox();
        const style = getComputedStyle(element);
        const stroke = style.stroke === 'none' ? 0 : (parseFloat(style.strokeWidth) || 0) / 2;
        const x1 = b.x-stroke, y1 = b.y-stroke, x2 = b.x+b.width+stroke, y2 = b.y+b.height+stroke;
        for (const matrix of matrices.filter(Boolean)) points.push(...[[x1,y1],[x2,y1],[x2,y2],[x1,y2]].map(p => new DOMPoint(...p).matrixTransform(matrix)));
      } catch { /* Unrendered resources may have no display bounds. */ }
    }
    if (!points.length || points.some(p => !Number.isFinite(p.x) || !Number.isFinite(p.y))) return;
    const xs = points.map(p=>p.x), ys = points.map(p=>p.y);
    const left = Math.min(...xs), right = Math.max(...xs), top = Math.min(...ys), bottom = Math.max(...ys);
    const w = stage.clientWidth, h = stage.clientHeight;
    // Keep context around small details and frame the whole additive selection.
    editor.zoom = Math.max(.005, Math.min(MAX_ZOOM, Math.max(1,w-80)/Math.max(60,(right-left)*1.2), Math.max(1,h-80)/Math.max(60,(bottom-top)*1.2)));
    editor.pan = {x:w/2-((left+right)/2-editor.state.bounds[0])*editor.zoom,
           y:h/2-((top+bottom)/2-editor.state.bounds[1])*editor.zoom};
    updateView();
  }

  function updateView() {
    reportView();
    editor.clickCycle = null;
    $('artboard').style.transform = `translate(${editor.pan.x}px,${editor.pan.y}px) scale(${editor.zoom})`;
    // Transparency shows as a checkerboard of 8px squares at any zoom.
    $('artboard').style.setProperty('--checker', `${16/editor.zoom}px`);
    $('zoom-percent').textContent = `${Math.round(editor.zoom*100)}%`;
    $('canvas-label').style.left = `${editor.pan.x}px`; $('canvas-label').style.top = `${editor.pan.y-22}px`;
    drawOverlay();
  }

  function fit() {
    if (!editor.state) return;
    const w = stage.clientWidth, h = stage.clientHeight;
    editor.zoom = Math.max(.01, Math.min((w-90)/editor.state.bounds[2], (h-100)/editor.state.bounds[3]));
    editor.pan = {x:(w-editor.state.bounds[2]*editor.zoom)/2,y:(h-editor.state.bounds[3]*editor.zoom)/2}; updateView();
  }

  function zoomAt(factor, x=stage.clientWidth/2, y=stage.clientHeight/2) {
    const next = Math.max(.005, Math.min(MAX_ZOOM, editor.zoom*factor)), ratio = next/editor.zoom;
    editor.pan = {x:x-(x-editor.pan.x)*ratio, y:y-(y-editor.pan.y)*ratio}; editor.zoom = next; updateView();
  }

  function point(event, element = overlay) {
    const matrix = screenMatrix(element);
    if (!matrix) throw new Error('This object has no editable canvas position');
    return new DOMPoint(event.clientX,event.clientY).matrixTransform(matrix.inverse());
  }

  // Switching tools keeps the selected objects. Leaving a point tool hides the
  // points and remembers them; coming back shows them again if the objects are
  // the same.
  async function setTool(value) {
    if (!editor.state) return;
    const from = editor.tool;
    const switched = switchTool({objects: editor.state.selection.objects, points: selectedPoints(), memory: editor.pointMemory}, from, value);
    editor.activeHandle = null;
    editor.clickCycle = null; editor.lastPick = null; editor.pathDraft=[]; editor.pathHover=null; editor.redrawHover=null; editor.hoverPath = null; editor.tool=value; editor.pointMemory = switched.memory;
    reportView();
    document.querySelectorAll('.tool[data-tool]').forEach(button => {
      const active = button.dataset.tool === editor.tool;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    $('active-tool').dataset.tool = editor.tool;
    $('tool-strip').dataset.tool = editor.tool;
    $('tool-shortcut').textContent = {select: 'A', move: 'V', nodes: 'S', path: 'D', rect: 'M', ellipse: 'E', line: 'L', knife: 'C', redraw: 'R', hand: 'Space'}[editor.tool];
    $('tool-name').textContent = editor.tool === 'select' ? 'Select objects' : editor.tool === 'nodes' ? 'Edit points' : names[editor.tool];
    $('canvas-hint').textContent=hints[editor.tool];
    stage.style.cursor = editor.tool === 'hand' ? 'grab' : ['path','knife','redraw',...SHAPE_TOOLS].includes(editor.tool) ? 'crosshair' : 'default';
    renderInspector();
    const nodes = [...new Set(switched.points.map(key => splitKey(key)[1]))];
    if (level() === 'points') {
      setBusy('Loading path nodes…',1);
      try { await loadGeometries(); } catch(error) {toast(error.message,true);} finally {setBusy('',-1);}
      // Restored points show in the paths they were picked in.
      if (switched.points.length) editor.owners = pointOwners(switched.points, id => editor.geometries.get(id)?.id);
    }
    if (nodes.length || editor.state.selection.nodes.length) {
      editor.focusPoint = switched.points.includes(editor.focusPoint) ? editor.focusPoint : switched.points.at(-1) ?? null;
      await action('select', {objects: editor.state.selection.objects, nodes}, 'Selecting…');
    }
    renderInspector(); drawOverlay();
  }

  function wire() {
      return api;
  }
  const api = {focusSelection, updateView, fit, zoomAt, point, setTool, wire};
  return api;
}
