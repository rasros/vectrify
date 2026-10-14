import {boxSelect, clickPoint, clickPointPath, dragBox, escapeStep, pickTarget, pointInside, pointKey, pointerTarget, rectInside, selectionContains, splitKey} from './selection.js';
import {SHAPE_TOOLS} from './shapes.js';

// Canvas owns its controls; editor contains the live state shared with other controllers.
export function createCanvasController(editor, {
  $,
  DOMPoint = globalThis.DOMPoint,
  action,
  clickEmpty,
  clickTargets,
  drawOverlay,
  drawing,
  finishPath,
  frameAt,
  geometryNodes,
  gestures,
  hitStack,
  hoverResize,
  later,
  level,
  loadGeometries,
  localToOverlay,
  nearKey,
  nearestPoint,
  nodeAt,
  nodeHandles,
  object,
  overlay,
  parents,
  point,
  pointPaths,
  pressFrame,
  redrawLines,
  regionShape,
  renderDrawing,
  renderInspector,
  renderNodeInspector,
  renderStatus,
  requestAnimationFrame = globalThis.requestAnimationFrame,
  sameClickSpot,
  screenMatrix,
  selectPoints,
  selectedHandle,
  selectedPoints,
  setTool,
  sharingPaths,
  stage,
  startGesture,
  svgElement,
  toast,
} = {}) {
  let hoverFrame = 0;

  function targetId(target) {
    const boundary = overlay.contains(target) && target.closest('[data-clip-object]');
    if (boundary) return boundary.dataset.clipObject;
    if (!drawing.contains(target)) return null;
    for(let element=target; element && element !== drawing; element=element.parentElement) if (element.dataset.objectId && object(element.dataset.objectId)) return element.dataset.objectId;
    return null;
  }

  function topSelection() {
    return editor.state.selection.objects.filter(id => {
      let parent = object(id)?.parent;
      while (parent) { if (editor.state.selection.objects.includes(parent)) return false; parent = object(parent)?.parent; }
      return true;
    });
  }

  function pathData(g) { return g.subpaths.map(s => s.nodes.map(n => n.command+n.values.join(' ')).join(' ') + (s.closed ? ' Z' : '')).join(' '); }

  // A point drag previews what the server will do: the points' handles ride along.
  function valuesById(g) { return new Map(g.subpaths.flatMap(s => s.nodes.map(n => [n.id, [...n.values]]))); }

  function restoreValues(g, saved) { for (const s of g.subpaths) for (const n of s.nodes) n.values = [...saved.get(n.id)]; }

  function carryHandles(g, saved) {
    // Mirrors the server: a moved point takes along the handles the edit left
    // alone, and a closed contour ending on its moveto moves as one point.
    for (let changed = true; changed;) {
      changed = false;
      for (const s of g.subpaths) {
        const nodes = s.nodes, olds = nodes.map(n => saved.get(n.id)), last = nodes.length - 1;
        const twins = s.closed && last > 1 && olds[last].at(-2) === olds[0].at(-2) && olds[last].at(-1) === olds[0].at(-1);
        nodes.forEach((node, i) => {
          const dx = node.values.at(-2) - olds[i].at(-2), dy = node.values.at(-1) - olds[i].at(-1);
          if (!dx && !dy) return;
          const slots = node.command === 'C' ? [[i, 2]] : [];
          if (i < last && nodes[i+1].command === 'C') slots.push([i+1, 0]);
          if (twins && (i === 0 || i === last)) slots.push([last-i, nodes[last-i].values.length-2]);
          for (const [j, k] of slots) {
            const v = nodes[j].values;
            if (v[k] !== olds[j][k] || v[k+1] !== olds[j][k+1]) continue;
            v[k] += dx; v[k+1] += dy; changed = true;
          }
        });
      }
    }
  }

  // Move the dragged points by the pointer's offset in the overlay, each in its
  // own path's frame, or the dragged handle to the pointer.
  function previewPointDrag(target) {
    const [grabbedPath, grabbed] = splitKey(editor.drag.key);
    for (const [id, saved] of editor.drag.saved) {
      const g = editor.geometries.get(id), element = svgElement(id), matrix = localToOverlay(element);
      if (!g || !matrix) continue;
      restoreValues(g, saved);
      const inverse = matrix.inverse();
      if (editor.drag.part === 'endpoint') {
        const from = editor.drag.start.matrixTransform(inverse), to = target.matrixTransform(inverse), dx = to.x - from.x, dy = to.y - from.y;
        for (const node of geometryNodes(id)) {
          if (!editor.drag.moving.includes(pointKey(id, node.id))) continue;
          const v = node.values; v[v.length-2] += dx; v[v.length-1] += dy;
        }
      } else if (id === grabbedPath) {
        const node = geometryNodes(id).find(n => n.id === grabbed), local = target.matrixTransform(inverse), offset = Number(editor.drag.part);
        node.values[offset] = local.x; node.values[offset+1] = local.y;
      }
      carryHandles(g, saved);
      if (editor.drag.part !== 'endpoint' && id === grabbedPath) {
        const anchor = nodeAt(editor.drag.anchor);
        if (anchor?.handles_aligned) {
          const slots = nodeHandles(editor.drag.anchor), offset = Number(editor.drag.part);
          const chosen = slots.find(h => h.node.id === grabbed && h.offset === offset);
          const opposite = slots.find(h => h !== chosen);
          if (chosen && opposite) {
            const [ax, ay] = anchor.values.slice(-2);
            const [hx, hy] = chosen.node.values.slice(offset, offset + 2);
            const [ox, oy] = opposite.node.values.slice(opposite.offset, opposite.offset + 2);
            const length = Math.hypot(ox - ax, oy - ay), norm = Math.hypot(hx - ax, hy - ay);
            if (length && norm) opposite.node.values.splice(opposite.offset, 2, ax - (hx - ax) * length / norm, ay - (hy - ay) * length / norm);
          }
        }
      }
      // Inset clips are derived by the server when this drag is committed.
      // Keep their inset d during the preview rather than replacing it with
      // the editable perimeter and temporarily painting over the outline.
      if (!element.hasAttribute('data-vectrify-clip-inset')) element.setAttribute('d', pathData(g));
      // The other paths drawing this geometry follow it.
      for (const other of sharingPaths(id)) if (!editor.drag.saved.has(other)) {
        const shared = svgElement(other);
        if (!shared?.hasAttribute('data-vectrify-clip-inset')) shared?.setAttribute('d', pathData(g));
      }
    }
  }

  // Press on a point: select it (with Shift, add or remove it), selecting its
  // path when it is not (with Shift, adding it), and start dragging the selected
  // points.
  function pressPoint(event, common) {
    const id = event.target.dataset.object, nodeId = event.target.dataset.node, part = event.target.dataset.part;
    const key = pointKey(id, nodeId), current = selectedPoints();
    const objects = clickPointPath(editor.state.selection.objects, pointPaths(), id, common.shift);
    let points = current;
    const toggle = part === 'endpoint' && common.shift && current.includes(key) ? clickPoint(current, key, true) : null;
    if (part === 'endpoint') {
      editor.activeHandle = null;
      // Defer removing a selected point until release so Shift can also constrain a drag.
      points = current.includes(key) ? current : clickPoint(current, key, common.shift);
      if (points !== current || objects !== editor.state.selection.objects) selectPoints(objects, points, key);
      else editor.focusPoint = key;
    } else {
      const anchor = event.target.dataset.anchor;
      if (!anchor) return;
      if (!current.includes(anchor) || objects !== editor.state.selection.objects) selectPoints(objects, [anchor], anchor);
      editor.activeHandle = {object: id, node: nodeId, offset: Number(part), anchor};
      renderNodeInspector(); drawOverlay();
    }
    // A handle moves alone; a point moves with the other selected points, and a
    // point of shared geometry picked in two paths moves once, as grabbed.
    const once = new Set();
    const moving = part === 'endpoint' ? [key, ...points.filter(k => k !== key)].filter(k => {
      const node = nodeAt(k), twin = `${editor.geometries.get(splitKey(k)[0])?.id} ${splitKey(k)[1]}`;
      if (!node || node.pinned || once.has(twin)) return false;
      once.add(twin); return true;
    }) : [key];
    if (part === 'endpoint' && (!points.includes(key) || nodeAt(key)?.pinned)) {
      if (toggle) selectPoints(objects, toggle, key);
      startGesture('point-click', event, common); return;
    }
    const paths = new Set(moving.map(k => splitKey(k)[0]));
    const saved = new Map([...paths].filter(p => editor.geometries.has(p)).map(p => [p, valuesById(editor.geometries.get(p))]));
    const node = nodeAt(key), offset = part === 'endpoint' ? node.values.length - 2 : Number(part);
    const start = new DOMPoint(node.values[offset], node.values[offset+1]).matrixTransform(localToOverlay(svgElement(id)));
    // A click on one of several selected points, without dragging, selects it alone.
    const collapse = part === 'endpoint' && !common.shift && current.includes(key) && current.length > 1;
    startGesture('node', event, common, {key, part, anchor: part === 'endpoint' ? key : event.target.dataset.anchor, moving, saved, start, collapse, toggle, objects});
  }

  // Box select: objects wholly inside the box at the entered group's level, or
  // in point tools the points of the selected paths.
  async function finishBox(finished) {
    const box = dragBox({x: finished.x, y: finished.y}, finished.end);
    if (level() === 'points') {
      const found = [];
      for (const id of pointPaths()) {
        const screen = screenMatrix(svgElement(id)); if (!screen) continue;
        for (const node of geometryNodes(id)) {
          const p = new DOMPoint(...node.values.slice(-2)).matrixTransform(screen);
          if (pointInside(p.x, p.y, box)) found.push(pointKey(id, node.id));
        }
      }
      return selectPoints(editor.state.selection.objects, boxSelect(selectedPoints(), found, finished.shift));
    }
    const container = editor.scope || editor.state.root;
    const found = editor.state.objects.filter(item => item.parent === container && !item.resource && !['defs', 'clipPath'].includes(item.tag)).filter(item => {
      const rect = svgElement(item.id)?.getBoundingClientRect();
      return rect && rect.width + rect.height > 0 && rectInside(rect, box);
    }).map(item => item.id);
    for (const guide of overlay.querySelectorAll('[data-clip-object]')) {
      if (rectInside(guide.getBoundingClientRect(), box)) found.push(guide.dataset.clipObject);
    }
    editor.focusPoint = null;
    return action('select', {objects: boxSelect(editor.state.selection.objects, found, finished.shift)}, 'Selecting…');
  }

  function pressStage(event) {
    if (!editor.state || editor.drag) return;
    const middle = event.button === 1;
    stage.focus({preventScroll:true});
    try { stage.setPointerCapture(event.pointerId); } catch { /* A replayed press may be released already. */ }
    const common = {x:event.clientX,y:event.clientY,time:event.timeStamp ?? performance.now(),shift:event.shiftKey || event.ctrlKey || event.metaKey,moved:false};
    const bounds = $('artboard').getBoundingClientRect();
    common.deselectOutside = !middle && !editor.space && (event.clientX < bounds.left || event.clientX > bounds.right ||
      event.clientY < bounds.top || event.clientY > bounds.bottom);
    if (middle || editor.space || editor.tool === 'hand') {
      startGesture('pan', event, common); return;
    }
    // A shape starts wherever it is pressed, the artboard's edge being no
    // limit; a click places one at the tool strip's size.
    if (SHAPE_TOOLS.includes(editor.tool)) {
      common.deselectOutside = false;
      startGesture('shape', event, common, {tool: editor.tool}); return;
    }
    if (editor.tool === 'path' && !common.deselectOutside) {
      const p=point(event), first=editor.pathDraft[0];
      if (first && editor.pathDraft.length>=3 && Math.hypot(p.x-first.x,p.y-first.y)*editor.zoom<8) {
        startGesture('closePath', event, common); return;
      }
      startGesture('drawPath', event, common); return;
    }
    const near = editor.tool === 'nodes' && !middle && !editor.space ? nearestPoint(event.clientX, event.clientY) : null;
    if (near) {
      // Points and handles remain editable outside the artboard, including the
      // reach around an edge point. Release must keep their click selection.
      common.deselectOutside = false;
      pressPoint({target: near}, common); return;
    }
    const hits = hitStack(event.clientX, event.clientY);
    const boundary = event.target?.closest?.('[data-clip-object]');
    if (boundary) {
      const clip = svgElement(boundary.dataset.clipObject)?.closest('clipPath');
      if (clip) editor.clipConsumers.set(clip.dataset.objectId, boundary.dataset.clipConsumer);
    }
    common.hits = hits;
    const handle = editor.tool === 'move' ? frameAt(event.clientX, event.clientY) : null;
    if (handle && handle !== 'inside') { pressFrame(event, common, handle); return; }
    const id = hits[0] || null;
    if (editor.tool === 'knife') {
      startGesture('knife', event, common, {id}); return;
    }
    if (editor.tool === 'redraw') {
      startGesture('redraw', event, common, {id}); return;
    }
    if (editor.tool === 'region') {
      // A polygon's corners may lie off the artboard.
      if (regionShape() === 'polygon') { common.deselectOutside = false; startGesture('regionCorner', event, common, {id}); return; }
      startGesture('region', event, common, {id}); return;
    }
    // Move drags a selected object or the frame interior. Select always starts
    // a selection gesture, even on selected objects and their frame.
    if (editor.tool === 'move' && (id || handle === 'inside')) {
      const targets = id ? clickTargets(hits) : [], picked = targets[0];
      const selectedHit = !id || selectionContains(editor.state.selection.objects, picked, parents(), editor.state.root) ||
        (sameClickSpot(event.clientX, event.clientY, targets) && editor.state.selection.objects.includes(targets[editor.clickCycle.index]));
      if (selectedHit && !common.shift) {
        // Each object moves in its parent's frame (a clipping contour in its
        // consumer's); the selection frame, as pressed, is what snaps.
        const members=topSelection().map(oid => {
          const element = svgElement(oid);
          return {id:oid, element, before:object(oid).attributes.transform || '', frame: element && localToOverlay(element.parentElement)};
        }).filter(m=>m.element && m.frame);
        startGesture('move', event, common, {id, members, box: editor.selectionBox && {...editor.selectionBox}}); return;
      }
    }
    startGesture('box', event, common, {id});
  }

  function hoverPoints(event) {
    if (hoverFrame) return;
    const x = event.clientX, y = event.clientY;
    hoverFrame = requestAnimationFrame(async () => {
      hoverFrame = 0;
      if (!['nodes', 'redraw'].includes(editor.tool) || editor.drag || editor.pending || editor.holding) return;
      let next;
      if (editor.tool === 'redraw') {
        // With nothing selected, the stroke redraws the path it starts on: the
        // last one the pointer was over, since the outline's edge is easy to miss.
        if (editor.state.selection.objects.length) return;
        next = pointerTarget(hitStack(x, y), editor.state.objects, [], editor.scope) ?? editor.hoverPath;
      } else {
        // Near one of the hovered path's points, it stays hovered off its body.
        const near = nearestPoint(x, y), key = nearKey(near) ?? null;
        const hit = near?.dataset.object || hitStack(x, y).find(id => object(id)?.tag === 'path');
        next = hit && !pointPaths().includes(hit) ? hit : null;
        // The point under the pointer shows its handles, kept on its handles.
        const anchor = near?.dataset.part === 'endpoint' ? pointKey(near.dataset.object, near.dataset.node) : near ? editor.nearAnchor : null;
        if (key !== editor.nearPoint || anchor !== editor.nearAnchor) {
          editor.nearPoint = key; editor.nearAnchor = anchor;
          if (next === editor.hoverPath) { drawOverlay(); return; }
        }
      }
      if (next === editor.hoverPath) return;
      editor.hoverPath = next;
      if (next && !editor.geometries.has(next)) { try { await loadGeometries(); } catch { return; } }
      drawOverlay();
    });
  }

  function moveStage(event) {
    if (!editor.drag) {
      if (editor.tool==='path' && editor.pathDraft.length) {editor.pathHover=point(event);drawOverlay();}
      if (editor.tool==='redraw' && redrawLines()) {const p=point(event);editor.redrawHover=[p.x,p.y];drawOverlay();}
      if (editor.tool==='region' && editor.region && !editor.region.closed) {const p=point(event);editor.regionHover={x:p.x,y:p.y};drawOverlay();}
      if (['nodes', 'redraw'].includes(editor.tool) && editor.state) hoverPoints(event);
      if (editor.tool === 'move' && editor.state) hoverResize(event);
      return;
    }
    editor.drag.moved ||= Math.hypot(event.clientX-editor.drag.x,event.clientY-editor.drag.y)>3;
    gestures[editor.drag.kind].move?.(editor.drag, event);
  }

  // A point drag sends the moved values: one point or handle as a node edit the
  // server carries the handles of, several points with their handles moved.
  async function finishPointDrag(finished) {
    const changes = {};
    for (const [id, saved] of finished.saved) {
      const g = editor.geometries.get(id); if (!g) continue;
      for (const node of geometryNodes(id)) {
        const before = saved.get(node.id);
        if (node.values.some((v, i) => v !== before[i])) (changes[id] ||= {})[node.id] = [...node.values];
      }
      restoreValues(g, saved);
    }
    if (!Object.keys(changes).length) { renderDrawing(); drawOverlay(); return; }
    const [id, nodeId] = splitKey(finished.key);
    if (finished.part !== 'endpoint' || finished.moving.length === 1) {
      const values = changes[id]?.[nodeId];
      if (values) await action('node', {object: id, node: nodeId, values, ...(finished.part !== 'endpoint' ? {handle_offset: Number(finished.part)} : {})}, 'Updating contour…');
      else { renderDrawing(); drawOverlay(); }
    } else await action('move_nodes', {changes}, 'Moving points…');
  }

  async function releaseStage(event) {
    if (!editor.drag) return;
    const finished = editor.drag, controller = gestures[finished.kind];
    editor.drag = null; stage.classList.remove('panning');
    if (stage.hasPointerCapture(event.pointerId)) stage.releasePointerCapture(event.pointerId);
    // A click on empty canvas outside the artboard clears the selection.
    if (finished.deselectOutside && !finished.moved) { await clickEmpty(false); return; }
    if (finished.moved && !controller.keepClickCycle) editor.clickCycle = null;
    await controller.release?.(finished);
  }

  async function enterObject(id, hits = []) {
    const item = object(id);
    if (item?.tag === 'g') {
      editor.scope = id; editor.clickCycle = null; editor.lastPick = null;
      const inner = hits.length ? pickTarget(hits[0], editor.scope, parents(), editor.state.root) : null;
      await action('select', {objects: [inner?.scope === id ? inner.id : id]}, 'Selecting…');
    } else if (item?.tag === 'path') {
      if (editor.state.selection.objects.length !== 1 || editor.state.selection.objects[0] !== id) await action('select', {objects: [id]}, 'Selecting…');
      await setTool('nodes');
    } else if (item?.tag === 'use') toast('Detach this instance to edit its points (Detach to editable path, under Path).');
    else if (['rect', 'circle', 'ellipse', 'line'].includes(item?.tag)) toast('Convert this shape to a path to edit its points (Convert to path, under Path).');
  }

  // Escape steps up one level: points to their paths, objects to their group,
  // then to nothing, and out of an entered group.
  async function stepUp() {
    if (selectedHandle()) { editor.activeHandle = null; renderInspector(); drawOverlay(); return; }
    const hadPoints = selectedPoints().length > 0;
    const next = escapeStep({objects: editor.state.selection.objects, points: selectedPoints(), scope: editor.scope}, parents(), editor.state.root);
    editor.scope = next.scope; editor.clickCycle = null; editor.focusPoint = null;
    if (next.objects.join() !== editor.state.selection.objects.join() || hadPoints) await action('select', {objects: next.objects}, 'Selecting…');
    renderStatus();
  }

  function cancelStage() {
    if (editor.drag) gestures[editor.drag.kind].cancel?.(editor.drag);
    stage.classList.remove('panning'); editor.clickCycle = null; editor.drag = null; editor.geometries = new Map();
    if (editor.state) { renderDrawing(); loadGeometries().then(drawOverlay).catch(() => {}); }
    drawOverlay();
  }

  function wire() {
    $('path-finish').onclick=()=>later(()=>finishPath(false));

    $('path-close').onclick=()=>later(()=>finishPath(true));

    $('path-cancel').onclick=()=>{editor.pathDraft=[];editor.pathHover=null;drawOverlay();};

      return api;
  }
  const api = {targetId, topSelection, pathData, valuesById, restoreValues, carryHandles, previewPointDrag, pressPoint, finishBox, pressStage, hoverPoints, moveStage, finishPointDrag, releaseStage, enterObject, stepUp, cancelStage, wire};
  return api;
}
