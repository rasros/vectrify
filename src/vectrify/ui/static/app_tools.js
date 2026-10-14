import {pathEndpoints, snapIndex, snapPoint} from './snap.js';
import {pointKey} from './selection.js';
import {SNAP_RADIUS} from './app_config.js';
import {attach, contourLines, stretch} from './redraw.js';

// Tools owns its controls; editor contains the live state shared with other controllers.
export function createToolsController(editor, {
  $,
  DOMPoint = globalThis.DOMPoint,
  action,
  drawOverlay,
  drawing,
  drawingContainer,
  drawingPaint,
  filledPath,
  geometryNodes,
  localToOverlay,
  point,
  pointPaths,
  setTool,
  svgElement,
  toast,
  xmlElement,
} = {}) {

  function draftPathData(points, closed=false) {
    if (!points.length) return '';
    let d=`M${points[0].x} ${points[0].y}`;
    function edge(a,b) {
      if (a.out || b.in) {const u=a.out||a,v=b.in||b;return ` C${u.x} ${u.y} ${v.x} ${v.y} ${b.x} ${b.y}`;}
      return ` L${b.x} ${b.y}`;
    }
    for(let i=1;i<points.length;i++)d+=edge(points[i-1],points[i]);
    if(closed){if(points.at(-1).out||points[0].in)d+=edge(points.at(-1),points[0]);d+=' Z';}
    return d;
  }

  function drawPathDraft() {
    $('path-controls').hidden=editor.tool!=='path';
    $('path-finish').disabled=editor.pathDraft.length<2;
    $('path-close').disabled=editor.pathDraft.length<3;
    $('path-cancel').disabled=!editor.pathDraft.length;
    if(editor.tool!=='path'||!editor.pathDraft.length)return;
    const group=xmlElement('g',{'pointer-events':'none','aria-hidden':'true'});
    const points=editor.pathHover?[...editor.pathDraft,editor.pathHover]:editor.pathDraft;
    group.append(xmlElement('path',{d:draftPathData(points),fill:'none',stroke:'#052b3a','stroke-width':4/editor.zoom}),
      xmlElement('path',{d:draftPathData(points),fill:'none',stroke:'#5cdeff','stroke-width':2/editor.zoom}));
    editor.pathDraft.forEach((p,i)=>{
      for(const h of [p.in,p.out].filter(Boolean)){
        group.append(xmlElement('line',{x1:p.x,y1:p.y,x2:h.x,y2:h.y,stroke:'#5cdeff','stroke-width':1/editor.zoom}),
          xmlElement('circle',{cx:h.x,cy:h.y,r:3/editor.zoom,fill:'#5cdeff'}));
      }
      group.append(xmlElement('circle',{cx:p.x,cy:p.y,r:(i===0?5:3.5)/editor.zoom,fill:i===0?'#fff':'#5cdeff',stroke:'#052b3a','stroke-width':1.5/editor.zoom}));
    });
    editor.overlayFrame.content.append(group);
  }

  function drawKnife() {
    if(editor.drag?.kind!=='knife'||!editor.drag.moved)return;
    const {start:a,end:b}=editor.drag, line={x1:a.x,y1:a.y,x2:b.x,y2:b.y,'pointer-events':'none','aria-hidden':'true'};
    editor.overlayFrame.content.append(xmlElement('line',{...line,stroke:'#052b3a','stroke-width':4/editor.zoom}),
      xmlElement('line',{...line,stroke:'#ff8a5c','stroke-width':2/editor.zoom,'stroke-dasharray':`${6/editor.zoom} ${4/editor.zoom}`}));
  }

  // Snapping while dragging points or a handle: onto the on-curve points of
  // every visible path and the artboard's edges and corners, within a fixed
  // screen distance. Alt or Ctrl/⌘ drags freely.
  function snapTargets(moving, start) {
    const points = [];
    for (const path of drawing.querySelectorAll('path')) {
      if (path.closest('defs, clipPath, mask, pattern, symbol, marker')) continue;
      if (path.checkVisibility && !path.checkVisibility({visibilityProperty: true})) continue;
      const matrix = localToOverlay(path); if (!matrix) continue;
      // The points being dragged are no targets; their paths' others are.
      const id = path.dataset.objectId, shown = editor.drag.saved.has(id) && editor.geometries.get(id);
      const local = shown ? geometryNodes(id).filter(n => !moving.has(pointKey(id, n.id))).map(n => editor.drag.saved.get(id).get(n.id).slice(-2)) : pathEndpoints(path.getAttribute('d') || '');
      for (const [x, y] of local) {
        const p = new DOMPoint(x, y).matrixTransform(matrix);
        // Where the dragged point started is no target: it would hold it there.
        if (Math.hypot(p.x - start.x, p.y - start.y) > 1e-9) points.push([p.x, p.y]);
      }
    }
    return snapIndex(points, SNAP_RADIUS / editor.zoom);
  }

  // Where the dragged point or handle goes, in the overlay's frame.
  function snappedDrag(event) {
    let p = point(event);
    const lockedAxis = event.shiftKey ? (Math.abs(p.x - editor.drag.start.x) >= Math.abs(p.y - editor.drag.start.y) ? 'y' : 'x') : null;
    if (lockedAxis) p[lockedAxis] = editor.drag.start[lockedAxis];
    editor.drag.snap = null;
    if (!event.altKey && !event.ctrlKey && !event.metaKey) {
      editor.drag.snaps ??= snapTargets(new Set(editor.drag.part === 'endpoint' ? editor.drag.moving : []), editor.drag.start);
      editor.drag.snap = snapPoint(p.x, p.y, editor.drag.snaps, editor.state.bounds, SNAP_RADIUS / editor.zoom, lockedAxis);
      if (editor.drag.snap) p = new DOMPoint(editor.drag.snap.x, editor.drag.snap.y);
    }
    return p;
  }

  function drawSnap() {
    if (editor.drag?.kind === 'resize' && editor.drag.moved) {
      const [x, y, w, h] = editor.state.bounds;
      if (editor.drag.snapX !== null) editor.overlayFrame.content.append(xmlElement('line', {x1: editor.drag.snapX, y1: y, x2: editor.drag.snapX, y2: y + h, class: 'snap-edge'}));
      if (editor.drag.snapY !== null) editor.overlayFrame.content.append(xmlElement('line', {x1: x, y1: editor.drag.snapY, x2: x + w, y2: editor.drag.snapY, class: 'snap-edge'}));
      return;
    }
    const snap = editor.drag?.kind === 'node' && editor.drag.moved ? editor.drag.snap : null;
    if (!snap) return;
    const [x, y, w, h] = editor.state.bounds, r = 6 / editor.zoom;
    if (snap.target.x !== undefined) editor.overlayFrame.content.append(xmlElement('line', {x1: snap.target.x, y1: y, x2: snap.target.x, y2: y + h, class: 'snap-edge'}));
    if (snap.target.y !== undefined) editor.overlayFrame.content.append(xmlElement('line', {x1: x, y1: snap.target.y, x2: x + w, y2: snap.target.y, class: 'snap-edge'}));
    editor.overlayFrame.content.append(xmlElement('rect', {x: snap.x - r, y: snap.y - r, width: 2 * r, height: 2 * r, transform: `rotate(45 ${snap.x} ${snap.y})`, class: 'snap-target'}));
  }

  function knifeEnd(event) {
    const p=point(event), a=editor.drag.start;
    if(!event.shiftKey)return {x:p.x,y:p.y};
    const step=Math.PI/12, angle=Math.round(Math.atan2(p.y-a.y,p.x-a.x)/step)*step, length=Math.hypot(p.x-a.x,p.y-a.y);
    return {x:a.x+Math.cos(angle)*length,y:a.y+Math.sin(angle)*length};
  }

  async function cutWithKnife({start, end}) {
    // With nothing selected the knife cuts what it crosses, in the entered group.
    if(await action('knife',{start:[start.x,start.y],end:[end.x,end.y],within:editor.scope},'Cutting…')) {
      const pieces = editor.state.selection.objects.length, lines = editor.state.selection.objects.every(id => !filledPath(id));
      toast(lines ? `Cut the line${pieces > 1 ? ` into ${pieces} paths` : ' open'} where the knife crosses it; Join line ends joins it again.` : `Cut into ${pieces} pieces. They meet exactly along the cut; Merge shapes merges them again.`);
    }
  }

  // Redraw outline: the stroke, where its ends attach to a path's outline and
  // the stretch it will replace, all in the overlay's frame. The path is a
  // selected one, or with nothing selected the one under the pointer.
  function redrawLines() {
    const lines = [];
    for (const id of editor.state.selection.objects.length ? pointPaths() : editor.hoverPath ? [editor.hoverPath] : []) {
      const matrix = editor.geometries.has(id) && localToOverlay(svgElement(id));
      if (!matrix) continue;
      for (const line of contourLines(editor.geometries.get(id), ([x, y]) => { const p = new DOMPoint(x, y).matrixTransform(matrix); return [p.x, p.y]; }))
        lines.push({...line, object: id});
    }
    return lines.length ? lines : null;
  }

  // The stroke starts on whichever selected path it touches and ends on the
  // same contour.
  function redrawPlan(points, longWay) {
    const lines = points.length ? redrawLines() : null;
    if (!lines) return null;
    const start = attach(lines, ...points[0], 1/editor.zoom);
    const line = start && lines.find(l => l.id === start.contour);
    const own = line && lines.filter(l => l.object === line.object);
    const end = start && points.length > 1 ? attach(own, ...points.at(-1), 1/editor.zoom, start.contour) : null;
    return {object: line?.object, start, end, replaced: end ? stretch(line, start, end, longWay) : null};
  }

  function drawRedraw() {
    if (editor.tool !== 'redraw') return;
    const stroke = editor.drag?.kind === 'redraw' ? editor.drag.points : null;
    const plan = redrawPlan(stroke || (editor.redrawHover ? [editor.redrawHover] : []), editor.drag?.longWay);
    const group = xmlElement('g', {'pointer-events': 'none', 'aria-hidden': 'true'});
    if (plan?.replaced) {
      group.append(xmlElement('polyline', {points: plan.replaced.join(' '), fill: 'none', stroke: '#ff8a5c', 'stroke-width': 3/editor.zoom, 'stroke-dasharray': `${6/editor.zoom} ${4/editor.zoom}`}));
    }
    if (stroke && editor.drag.moved) {
      group.append(xmlElement('polyline', {points: stroke.join(' '), fill: 'none', stroke: '#052b3a', 'stroke-width': 4/editor.zoom}),
        xmlElement('polyline', {points: stroke.join(' '), fill: 'none', stroke: '#5cdeff', 'stroke-width': 2/editor.zoom}));
    }
    for (const end of [plan?.start, plan?.end]) {
      if (end) group.append(xmlElement('circle', {cx: end.x, cy: end.y, r: 5/editor.zoom, fill: '#fff', stroke: '#052b3a', 'stroke-width': 1.5/editor.zoom}));
    }
    editor.overlayFrame.content.append(group);
  }

  async function redrawOutline({points, longWay}) {
    const plan = redrawPlan(points, longWay);
    if (!redrawLines()) {toast(editor.state.selection.objects.length ? 'The selection has no path to redraw.' : 'Start on a path\'s outline, then draw along the edge it should follow.', true);return;}
    if (!plan?.start || !plan.end) {toast(`Start and end the stroke on the same outline of ${editor.state.selection.objects.length ? 'a selected' : 'one'} path.`, true);return;}
    await action('redraw_outline', {object: plan.object, points, pixel: 1/editor.zoom, long_way: Boolean(longWay)}, editor.state.reference ? 'Fitting to the reference…' : 'Redrawing outline…');
  }

  async function finishPath(closed) {
    if(editor.pathDraft.length<(closed?3:2))return;
    const draft=editor.pathDraft; editor.pathDraft=[];editor.pathHover=null;
    // The drawing paint and container, as the shape tools use them.
    if(await action('add_path',{d:draftPathData(draft,closed),...drawingPaint(closed),...drawingContainer()},'Drawing path…'))await setTool('select');
    else {editor.pathDraft=draft;drawOverlay();}
  }

  function wire() {
      return api;
  }
  const api = {draftPathData, drawPathDraft, drawKnife, snapTargets, snappedDrag, drawSnap, knifeEnd, cutWithKnife, redrawLines, redrawPlan, drawRedraw, redrawOutline, finishPath, wire};
  return api;
}
