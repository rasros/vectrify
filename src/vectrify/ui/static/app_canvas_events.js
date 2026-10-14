import {canvasGestures} from './gestures.js';
import {HeldGesture} from './input.js';

// Canvas_events owns its controls; editor contains the live state shared with other controllers.
export function createCanvasEventsController(editor, {
  $,
  ResizeObserver = globalThis.ResizeObserver,
  action,
  cancelShape,
  cancelStage,
  cutWithKnife,
  document = globalThis.document,
  drawOverlay,
  enterObject,
  finishBox,
  finishPath,
  finishPointDrag,
  finishGradient,
  finishPencil,
  finishShape,
  fit,
  gradientDrag,
  hitStack,
  input,
  knifeEnd,
  later,
  level,
  moveDrag,
  moveStage,
  object,
  point,
  pressStage,
  previewPointDrag,
  previewGradient,
  previewResize,
  previewRotate,
  previewPencil,
  previewShape,
  placePivot,
  redrawOutline,
  regionCorner,
  regionRectangle,
  releaseStage,
  renderDrawing,
  renderInspector,
  renderNodeInspector,
  resizeDrag,
  restoreGradient,
  restorePivot,
  restoreValues,
  rotateDrag,
  rotateSelection,
  sampleColour,
  selectAtPoint,
  selectPoints,
  setTool,
  snappedDrag,
  stage,
  toast,
  updateView,
  zoomAt,
} = {}) {

  // Each gesture owns its press, preview, completion and cancellation. The
  // canvas retains hit testing, pointer capture and the shared input queue.
  const gestures = canvasGestures({
    point, drawOverlay,
    view: {
      pan: () => editor.pan, zoom: () => editor.zoom,
      setPan: value => { editor.pan = value; }, update: updateView,
      setPanning: value => stage.classList.toggle('panning', value),
    },
    path: {draft: () => editor.pathDraft, setHover: value => { editor.pathHover = value; }, finish: finishPath},
    shapes: {preview: previewShape, finish: finishShape, cancel: cancelShape},
    pencil: {preview: previewPencil, finish: finishPencil},
    selection: {
      pick: selectAtPoint, box: finishBox, renderDrawing,
      move: offsets => action('move', {dx: 0, dy: 0, offsets}, 'Moving selection…'),
    },
    nodes: {
      preview: previewPointDrag, snap: snappedDrag, finish: finishPointDrag,
      select: selectPoints, renderInspector, renderNodeInspector,
      restore: gesture => {
        for (const [id, saved] of gesture.saved) if (editor.geometries.has(id)) restoreValues(editor.geometries.get(id), saved);
      },
    },
    moving: {drag: moveDrag},
    resize: {
      result: resizeDrag, preview: previewResize,
      refuse: message => toast(message, true),
      finish: result => action('resize', {anchor: result.anchor, scale: [result.sx, result.sy]}, 'Resizing…'),
    },
    rotate: {result: rotateDrag, preview: previewRotate, finish: rotateSelection, placePivot, restorePivot},
    gradient: {result: gradientDrag, preview: previewGradient, finish: finishGradient, restore: restoreGradient, refuse: message => toast(message, true)},
    knife: {end: knifeEnd, finish: cutWithKnife},
    region: {rectangle: regionRectangle, corner: regionCorner},
    eyedropper: {sample: sampleColour},
    redraw: {setHover: value => { editor.redrawHover = value; }, finish: redrawOutline},
  });

  function startGesture(kind, event, common, details = {}) {
    const controller = gestures[kind];
    editor.drag = {...common, kind, ...details};
    controller.press?.(editor.drag, event);
  }

  function holdGesture(event) {
    const gesture = new HeldGesture(event);
    editor.holding = gesture;
    try { stage.setPointerCapture(event.pointerId); } catch { /* The pointer may be gone already. */ }
    later(async () => {
      let released;
      const again = e => ({...e, target: document.elementFromPoint(e.clientX, e.clientY) || stage, preventDefault() {}});
      const live = gesture.replay({down: e => pressStage(again(e)), move: e => moveStage(again(e)), up: e => { released = releaseStage(again(e)); }, cancel: cancelStage});
      // Still pressed: the rest of the gesture goes on live.
      if (live && editor.holding === gesture) editor.holding = null;
      await released;
    });
  }

  function wire() {
    stage.addEventListener('pointerdown', event => {
      if (!editor.state || ![0,1].includes(event.button)) return;
      if (event.button === 1) event.preventDefault();
      if (editor.drag || (editor.holding && !editor.holding.released)) return;
      if (editor.pending || input.waiting) { holdGesture(event); return; }
      pressStage(event);
    });

    stage.addEventListener('pointerleave', () => {
      if (!editor.nearPoint && !editor.nearAnchor) return;
      editor.nearPoint = editor.nearAnchor = null; if (!editor.drag) drawOverlay();
    });

    stage.addEventListener('pointermove', event => {
      if (editor.holding && !editor.holding.released) { editor.holding.record(event); return; }
      moveStage(event);
    });

    stage.addEventListener('pointerup', event => {
      if (editor.holding && !editor.holding.released) { editor.holding.record(event); editor.holding = null; return; }
      releaseStage(event);
    });

    // Double-click a group to pick within it, or a path to edit its points.
    stage.addEventListener('dblclick', event => {
      if (!editor.state || editor.space) return;
      const x = event.clientX, y = event.clientY;
      later(async () => {
        await editor.queue;
        if (level() !== 'objects') return;
        const id = editor.lastPick?.first, item = object(id);
        if (!item || item.resource) return;
        await enterObject(id, hitStack(x, y));
      });
    });

    stage.addEventListener('pointercancel', event => {
      if (editor.holding && !editor.holding.released) { editor.holding.record(event); editor.holding = null; return; }
      cancelStage();
    });

    stage.addEventListener('auxclick',event=>{if(event.button===1)event.preventDefault();});

    stage.addEventListener('lostpointercapture', () => {
      if (editor.drag?.kind === 'pan') { gestures.pan.cancel(editor.drag); editor.drag = null; }
    });

    stage.addEventListener('wheel',event=>{event.preventDefault();if(!editor.state)return;const b=stage.getBoundingClientRect();zoomAt(Math.exp(-event.deltaY*.0015),event.clientX-b.left,event.clientY-b.top);},{passive:false});

    new ResizeObserver(()=>{if(editor.state)updateView();}).observe(stage);

    $('fit').onclick=fit;
     $('zoom-in').onclick=()=>zoomAt(1.25);
     $('zoom-out').onclick=()=>zoomAt(.8);

    document.querySelectorAll('.tool[data-tool]').forEach(button=>button.onclick=()=>later(()=>setTool(button.dataset.tool)));
      return api;
  }
  const api = {gestures, startGesture, holdGesture, wire};
  return api;
}
