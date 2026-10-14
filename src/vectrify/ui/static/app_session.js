import {NS} from './app_config.js';

// Session owns its controls; editor contains the live state shared with other controllers.
export function createSessionController(editor, {
  $,
  document = globalThis.document,
  drawOverlay,
  files,
  fit,
  level,
  loadGeometries,
  loadReference,
  missingGeometries,
  overlay,
  placeReference,
  renderDrawing,
  renderInspector,
  renderObjects,
  selectedHandle,
  updateObjectSelection,
  updateView,
} = {}) {
  let objectsById = new Map();

  let savedVersion = null;

  function svgElement(id) { return editor.svgElements.get(id); }

  function object(id) { return objectsById.get(id); }

  function oneObject() { return editor.state?.selection.objects.length === 1 ? object(editor.state.selection.objects[0]) : null; }

  function xmlElement(name, attrs = {}) {
    const element = document.createElementNS(NS, name);
    for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
    return element;
  }

  async function applyState(next) {
    const changed = !editor.state || next.epoch !== editor.state.epoch || next.revision !== editor.state.revision;
    const replaced = !editor.state || next.epoch !== editor.state.epoch;
    const referenceChanged = replaced || (next.reference_revision !== undefined && next.reference_revision !== editor.state.reference_revision);
    const before = editor.state?.bounds;
    if (replaced) savedVersion = {epoch: next.epoch, project_version: next.project_version};
    editor.state = {...editor.state, ...next};
    editor.dirty = editor.state.epoch !== savedVersion.epoch || editor.state.project_version !== savedVersion.project_version;
    if (changed) objectsById = new Map(editor.state.objects.map(item => [item.id, item]));
    $('filename').textContent = files.currentName(); $('dirty').textContent = editor.dirty ? '•' : '';
    if (next.svg) renderDrawing();
    const [x, y, w, h] = editor.state.bounds;
    $('artboard').style.width = `${w}px`; $('artboard').style.height = `${h}px`;
    overlay.setAttribute('viewBox', `${x} ${y} ${w} ${h}`);
    overlay.setAttribute('preserveAspectRatio', 'none');
    $('dimensions').textContent = `${Math.round(w)} × ${Math.round(h)}`;
    if (!replaced && before && before.join() !== editor.state.bounds.join()) {
      // The artwork stays where it is on screen when the artboard's corner
      // moves, and the reference keeps its place over the artwork.
      if (before[0] !== x || before[1] !== y) {
        editor.pan = {x: editor.pan.x + (x - before[0]) * editor.zoom, y: editor.pan.y + (y - before[1]) * editor.zoom};
        updateView();
      }
      placeReference();
    }
    $('undo').disabled = !editor.state.undo.length; $('redo').disabled = !editor.state.redo.length;
    $('undo').title = editor.state.undo.length ? `Undo: ${editor.state.undo.at(-1)}` : 'Nothing to undo';
    $('redo').title = editor.state.redo.length ? `Redo: ${editor.state.redo[0]}` : 'Nothing to redo';
    if (replaced) {
      editor.activeHandle = null; editor.focusPoint = null; editor.pointMemory = null; editor.scope = null;
      editor.collapsedGroups.clear();
      for (const item of editor.state.objects) if (item.tag === 'defs') editor.collapsedGroups.add(item.id);
      editor.clipConsumers.clear(); fit();
    }
    if (changed) { editor.geometries = new Map(); editor.pathHoles.clear(); editor.clickCycle = null; editor.lastPick = null; }
    if (editor.scope && object(editor.scope)?.tag !== 'g') editor.scope = null;
    // Selection does not change the drawing or the tree's labels and swatches.
    // Keep its rows: rebuilding them reads every path's paint and forces style
    // work on large drawings for every click.
    if (changed || next.svg) renderObjects();
    else updateObjectSelection();
    if (level() === 'points') {
      if (missingGeometries().length) renderInspector();
      await loadGeometries();
    }
    if (editor.activeHandle && !selectedHandle()) editor.activeHandle = null;
    renderInspector(); drawOverlay();
    if (referenceChanged) await loadReference();
  }

  function markSaved(version = editor.state) {
    if (version.epoch !== editor.state.epoch) return;
    savedVersion = {epoch: version.epoch, project_version: version.project_version};
    editor.dirty = editor.state.project_version !== savedVersion.project_version;
    $('dirty').textContent = editor.dirty ? '•' : '';
  }

  function wire() {
      return api;
  }
  const api = {svgElement, object, oneObject, xmlElement, applyState, markSaved, wire};
  return api;
}
