import {pointKey} from './selection.js';

// Markers owns its controls; editor contains the live state shared with other controllers.
export function createMarkersController(editor, {
  $,
  drawHandles,
  geometryNodes,
  localToOverlay,
  pointPaths,
  screenMatrix,
  shownPoints,
  stage,
  svgElement,
  xmlElement,
} = {}) {

  // The points of every selected path and of the paths in selected groups, and
  // of the unselected path under the pointer in Nodes, faintly, so a click on
  // one adds its path. A selected point of shared geometry is marked faintly in
  // the other paths drawing it.
  function drawPoints() {
    const stageBox = stage.getBoundingClientRect(), {strong, twins} = shownPoints(), chosen = new Set(strong), twin = new Set(twins);
    const selected = new Set(pointPaths()), editable = editor.tool === 'nodes';
    const paths = [...pointPaths(), ...(editable && editor.hoverPath && !selected.has(editor.hoverPath) ? [editor.hoverPath] : [])];
    // Limit handles in dense drawings by screen-space spacing, without dropping
    // geometry. Zooming in exposes the original nodes at their full resolution.
    const occupied = new Set(); let count = 0, total = 0;
    const unpicked = !chosen.size && !twin.size;
    for (const id of paths) {
      const element = svgElement(id), matrix = localToOverlay(element), screen = screenMatrix(element);
      if (!matrix || !screen || !editor.geometries.has(id)) continue;
      const ghost = !selected.has(id);
      const nodes = geometryNodes(id); total += nodes.length;
      for (const node of nodes) {
        // Nothing after the marker cap can be drawn unless it is selected or
        // shared with a selected point. Still count every original node above.
        if (unpicked && count >= 1200) break;
        const key = pointKey(id, node.id), picked = chosen.has(key), mirrored = twin.has(key);
        const x = node.values.at(-2), y = node.values.at(-1);
        const sx = screen.a*x + screen.c*y + screen.e, sy = screen.b*x + screen.d*y + screen.f;
        if (sx < stageBox.left || sx > stageBox.right || sy < stageBox.top || sy > stageBox.bottom) continue;
        const cell = `${Math.floor(sx/10)},${Math.floor(sy/10)}`;
        if (!picked && !mirrored && (occupied.has(cell) || count >= 1200)) continue;
        occupied.add(cell); count++;
        const p = {x: matrix.a*x + matrix.c*y + matrix.e, y: matrix.b*x + matrix.d*y + matrix.f};
        const near = editable && editor.nearPoint === `${id} ${node.id} endpoint`;
        const circle = xmlElement('circle', {cx:p.x, cy:p.y, r: (picked ? 4.8 : mirrored ? 4.2 : 3.3)*(near ? 1.5 : 1)/editor.zoom, class:`node${picked ? ' selected' : ''}${mirrored ? ' twin' : ''}${node.pinned ? ' pinned' : ''}${ghost ? ' ghost' : ''}${near ? ' near' : ''}${editable ? '' : ' passive'}`});
        circle.dataset.object = id; circle.dataset.node = node.id; circle.dataset.part = 'endpoint';
        editor.overlayFrame.content.append(circle);
      }
    }
    // The handles of the selected points, up to a few hundred of them, and of
    // the point under the pointer, to show what it has before it is picked.
    if (editable) for (const key of new Set([...[...chosen].slice(0, 300), ...(editor.nearAnchor ? [editor.nearAnchor] : [])])) drawHandles(key);
    $('node-count').textContent = `${count.toLocaleString()} / ${total.toLocaleString()}`;
  }

  function wire() {
      return api;
  }
  const api = {drawPoints, wire};
  return api;
}
