import {TOOL_LEVEL, instancePoints, pointOwners, pointTargets, splitKey} from './selection.js';

// Geometry owns its controls; editor contains the live state shared with other controllers.
export function createGeometryController(editor, {
  action,
  request,
  schedulePoll,
} = {}) {
  let shownCache = {};

  let targetCache = {state: null};

  const level = () => TOOL_LEVEL[editor.tool];

  const parents = () => new Map(editor.state.objects.map(item => [item.id, item.parent]));

  function pointTargetsNow() {
    if (targetCache.state !== editor.state) targetCache = {state: editor.state, ...pointTargets(editor.state.selection.objects, editor.state.objects)};
    return targetCache;
  }

  const pointPaths = () => pointTargetsNow().paths;

  const geometryIndexes = new WeakMap();

  function geometryIndex(id) {
    const geometry = editor.geometries.get(id);
    if (!geometry) return null;
    let index = geometryIndexes.get(geometry);
    if (!index) {
      const nodes = [], byId = new Map(), contours = new Map();
      for (const contour of geometry.subpaths) for (const node of contour.nodes) {
        nodes.push(node); byId.set(node.id, node); contours.set(node.id, contour);
      }
      index = {nodes, byId, contours};
      geometryIndexes.set(geometry, index);
    }
    return index;
  }

  function geometryNodes(id) { return geometryIndex(id)?.nodes || []; }

  function nodeAt(key) { const [id, node] = splitKey(key); return geometryIndex(id)?.byId.get(node); }

  function contourAt(key) { const [id, node] = splitKey(key); return geometryIndex(id)?.contours.get(node); }

  function shownPoints() {
    if (level() !== 'points') return {strong: [], twins: []};
    const nodes = editor.state.selection.nodes, cache = shownCache;
    if (cache.state !== editor.state || cache.geometries !== editor.geometries || cache.size !== editor.geometries.size || cache.owners !== editor.owners || cache.tool !== editor.tool) {
      const paths = pointPaths().filter(id => editor.geometries.has(id)).map(id => ({id, geometry: editor.geometries.get(id).id, nodes: geometryNodes(id).map(n => n.id)}));
      shownCache = {state: editor.state, geometries: editor.geometries, size: editor.geometries.size, owners: editor.owners, tool: editor.tool, ...(nodes.length ? instancePoints(paths, new Set(nodes), editor.owners) : {strong: [], twins: []})};
    }
    return shownCache;
  }

  const selectedPoints = () => shownPoints().strong;

  // The other paths drawing the geometry of a shown path.
  function sharingPaths(id) { return (editor.geometries.get(id)?.users || []).filter(user => user !== id); }

  function selectPoints(objects, keys, focus = keys.at(-1)) {
    editor.activeHandle = null;
    editor.focusPoint = focus ?? null;
    editor.owners = pointOwners(keys, id => editor.geometries.get(id)?.id);
    return action('select', {objects: [...new Set(objects)], nodes: [...new Set(keys.map(key => splitKey(key)[1]))]}, 'Selecting…');
  }

  function missingGeometries() {
    const wanted = level() === 'points' ? [...pointPaths(), ...(editor.hoverPath ? [editor.hoverPath] : [])] : [];
    return [...new Set(wanted)].filter(id => !editor.geometries.has(id));
  }

  function wire() {
      return api;
  }
  const api = {level, parents, pointTargetsNow, pointPaths, geometryIndexes, geometryIndex, geometryNodes, nodeAt, contourAt, shownPoints, selectedPoints, sharingPaths, selectPoints, missingGeometries, wire};
  return api;
}

// Cache only geometry read from the epoch and revision currently displayed.
export function createGeometryLoader(editor, {missingGeometries, request, schedulePoll}) {
  async function loadGeometries() {
    const missing = missingGeometries();
    if (!missing.length) return;
    const {epoch, revision} = editor.state;
    const result = await request('/api/nodes', {objects: missing, epoch, revision});
    if (editor.state.epoch !== epoch || editor.state.revision !== revision) return;
    if (result.epoch !== epoch || result.revision !== revision) { schedulePoll(0); return; }
    for (const [id, geometry] of Object.entries(result.geometries)) editor.geometries.set(id, geometry);
  }

    return {loadGeometries};
}
