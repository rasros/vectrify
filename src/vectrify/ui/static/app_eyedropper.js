// The Eyedropper samples a disc this many screen pixels in radius, never
// less than half a document unit as MCP's pick (the session's floor).
import {plural} from './util.js';

export const SAMPLE_PIXELS = 2;
export const MIN_SAMPLE_RADIUS = 0.5;

const SOURCES = {artwork: 'Artwork', reference: 'Reference'};
const TARGETS = {fill: 'Fill', stroke: 'Stroke'};

// The radius, in document units, sampled at *zoom* screen pixels per unit.
export const sampleRadius = zoom => Math.max(SAMPLE_PIXELS / zoom, MIN_SAMPLE_RADIUS);

// Eyedropper owns the Eyedropper tool's choices, its strip status and the
// mark at the pointer; editor contains the live state shared with other
// controllers. A click samples the colour under the pointer from the
// artwork or the reference, as the session renders them (see
// Session.sample), and makes it the selected objects' fill or stroke in one
// paint edit; with nothing selected, the drawing paint new shapes take.
export function createEyedropperController(editor, {
  $,
  action,
  drawOverlay,
  object,
  request,
  setDrawingColour,
  setTool,
  stage,
  toast,
  xmlElement,
} = {}) {
  // The choices, kept while the tool is put down and picked up again.
  const options = {source: 'artwork', target: 'fill'};
  let shownStrip = '';

  // Why *source* cannot be sampled now, or ''.
  function sourceRefusal(source = options.source) {
    return source === 'reference' && !editor.state?.reference ? 'No reference is loaded: load an image under Reference to sample it' : '';
  }

  // Why a sample cannot become the selection's *target* paint now, or ''.
  // With nothing selected it becomes the drawing paint, which always can.
  function targetRefusal(target = options.target) {
    for (const id of editor.state?.selection.objects || []) {
      const item = object(id); if (!item) continue;
      let inside = item;
      while (inside && inside.tag !== 'clipPath') inside = object(inside.parent);
      if (item.tag === 'defs' || inside) return `${item.label}: definitions and clipping contours are not painted`;
      if (item.inherited_locks.includes('paint') || item.inherited_locks.includes(target)) return `${item.label}: paint is locked; unlock it to sample a colour into its ${target}`;
    }
    return '';
  }

  const refusal = () => sourceRefusal() || targetRefusal();

  // Sample from *source* or into *target*, picking up the tool. A command
  // that cannot run is refused before this, with its reason.
  async function choose({source, target} = {}) {
    if (source) options.source = source;
    if (target) options.target = target;
    if (editor.tool !== 'eyedropper') await setTool('eyedropper');
    shownStrip = ''; drawOverlay();
  }

  // Where the sample goes, in words: the strip's status says it before a
  // click, briefly, beside the selection's count.
  function destination() {
    const count = editor.state?.selection.objects.length || 0;
    return count ? `${options.target} of ${plural(count, 'object')}` : `drawing ${options.target}`;
  }

  function renderStrip() {
    const reason = refusal(), status = reason || `${SOURCES[options.source]} → ${destination()}`;
    const signature = JSON.stringify([options, status]);
    if (signature === shownStrip) return;
    shownStrip = signature;
    for (const source of Object.keys(SOURCES)) $(`eyedropper-${source}`).setAttribute('aria-pressed', String(options.source === source));
    for (const target of Object.keys(TARGETS)) $(`eyedropper-${target}`).setAttribute('aria-pressed', String(options.target === target));
    $('eyedropper-status').textContent = status;
    $('eyedropper-status').classList.toggle('refused', !!reason);
  }

  // At the pointer: the disc that is sampled, and a swatch showing the
  // target before the click, solid for a fill and an outline for a stroke,
  // labelled with the source and target or with what stops a sample.
  function drawEyedropper() {
    if (editor.tool !== 'eyedropper' || !editor.state) return;
    renderStrip();
    const at = editor.drag?.kind === 'eyedropper' ? editor.drag.at : editor.eyedropperHover;
    if (!at) return;
    const px = 1 / editor.zoom, refused = !!refusal();
    const group = xmlElement('g', {'pointer-events': 'none', 'aria-hidden': 'true', class: `eyedropper-mark${refused ? ' refused' : ''}`});
    const ring = {cx: at.x, cy: at.y, r: Math.max(sampleRadius(editor.zoom), 3 * px)};
    group.append(xmlElement('circle', {...ring, class: 'eyedropper-halo', 'stroke-width': 3 * px}),
      xmlElement('circle', {...ring, class: 'eyedropper-ring', 'stroke-width': 1.5 * px}));
    const x = at.x + 10 * px, y = at.y + 10 * px, side = 12 * px;
    group.append(xmlElement('rect', {x, y, width: side, height: side, class: `eyedropper-swatch ${options.target}`, 'stroke-width': (options.target === 'fill' ? 1 : 3) * px}));
    const label = xmlElement('text', {x: x + side + 5 * px, y: y + side - 2 * px, 'font-size': 11 * px, 'stroke-width': 3 * px, class: 'eyedropper-label'});
    label.textContent = !refused ? `${SOURCES[options.source]} → ${TARGETS[options.target]}` : sourceRefusal() ? 'No reference' : targetRefusal().includes('locked') ? 'Paint locked' : 'Not painted';
    group.append(label);
    editor.overlayFrame.content.append(group);
  }

  // A click: sample at *at*, in document units, and apply the colour.
  // Nothing changes when the sample is refused or fails.
  async function sample(at) {
    const reason = refusal();
    if (reason) { toast(reason, true); return false; }
    const {source, target} = options, selected = [...editor.state.selection.objects];
    let colour;
    try {
      ({colour} = await request('/api/sample', {epoch: editor.state.epoch, revision: editor.state.revision, source, at: [at.x, at.y], radius: sampleRadius(editor.zoom)}));
    } catch (error) { toast(error.message, true); return false; }
    const from = source === 'reference' ? 'the reference' : 'the artwork';
    if (!selected.length) {
      setDrawingColour(target, colour);
      toast(`Sampled ${colour} from ${from}: new shapes and paths take it as their ${target}.`);
      return true;
    }
    if (!await action('paint', {changes: {[target]: colour}}, 'Applying the sampled colour…')) return false;
    toast(`Set the ${target} of ${plural(selected.length, 'object')} to ${colour}, sampled from ${from}. Undo puts it back.`);
    return true;
  }

  function wire() {
    // The mark follows the pointer over the canvas and leaves with it.
    stage.addEventListener('pointerleave', () => {
      if (editor.eyedropperHover && !editor.drag) { editor.eyedropperHover = null; drawOverlay(); }
    });
    return api;
  }
  const api = {options, sourceRefusal, targetRefusal, refusal, choose, destination, renderStrip, drawEyedropper, sample, wire};
  return api;
}
