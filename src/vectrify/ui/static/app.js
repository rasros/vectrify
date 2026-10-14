import {operationDialogs} from './app_jobs.js';
import {projectFiles} from './app_files.js';
import {createEditorState} from './app_state.js';
import {createTransportController} from './app_transport.js';
import {createActionsController} from './app_actions.js';
import {createSessionController} from './app_session.js';
import {createDrawingController} from './app_drawing.js';
import {createGeometryController, createGeometryLoader} from './app_geometry.js';
import {createPaintController} from './app_paint.js';
import {createColorsController} from './app_colors.js';
import {createTreeController} from './app_tree.js';
import {createTreeDragController} from './app_tree_drag.js';
import {createCommandsController} from './app_commands.js';
import {createSelectionController} from './app_selection.js';
import {createAgentsController} from './app_agents.js';
import {createAgentControlsController} from './app_agent_controls.js';
import {createCommandRegistryController} from './app_command_registry.js';
import {createArrowsController} from './app_arrows.js';
import {createStrokeController} from './app_stroke.js';
import {createGradientsController} from './app_gradients.js';
import {createViewController} from './app_view.js';
import {createHitsController} from './app_hits.js';
import {createFramesController} from './app_frames.js';
import {createOverlayController} from './app_overlay.js';
import {createResizingController} from './app_resizing.js';
import {createRotatingController} from './app_rotating.js';
import {createMarkersController} from './app_markers.js';
import {createLayoutController} from './app_layout.js';
import {createToolsController} from './app_tools.js';
import {createShapesController} from './app_shapes.js';
import {createPencilController} from './app_pencil.js';
import {createCanvasController} from './app_canvas.js';
import {createCanvasEventsController} from './app_canvas_events.js';
import {createJoinController} from './app_join.js';
import {createReferenceController} from './app_reference.js';
import {createReferenceDataController} from './app_reference_data.js';
import {createKeysController} from './app_keys.js';
import {createTransformController} from './app_transform.js';
import {createClippingController} from './app_clipping.js';
import {createRegionController} from './app_region.js';
import {createEyedropperController} from './app_eyedropper.js';
import {createAlignController} from './app_align.js';
import {createBooleansController} from './app_booleans.js';
import {createDocumentController} from './app_document.js';

// Each controller, under its key in the application, in the order they are
// created: one that takes another's VALUES at creation comes after it.
export const CONTROLLERS = {
  transport: createTransportController,
  actions: createActionsController,
  files: createFiles,
  session: createSessionController,
  drawing: createDrawingController,
  geometry: createGeometryController,
  geometry_loader: createGeometryLoader,
  paint: createPaintController,
  colors: createColorsController,
  tree: createTreeController,
  tree_drag: createTreeDragController,
  command_registry: createCommandRegistryController,
  commands: createCommandsController,
  selection: createSelectionController,
  agents: createAgentsController,
  agent_controls: createAgentControlsController,
  arrows: createArrowsController,
  stroke: createStrokeController,
  transform: createTransformController,
  gradients: createGradientsController,
  view: createViewController,
  hits: createHitsController,
  frames: createFramesController,
  overlay: createOverlayController,
  resizing: createResizingController,
  rotating: createRotatingController,
  markers: createMarkersController,
  layout: createLayoutController,
  tools: createToolsController,
  shapes: createShapesController,
  pencil: createPencilController,
  canvas_events: createCanvasEventsController,
  canvas: createCanvasController,
  clipping: createClippingController,
  region: createRegionController,
  eyedropper: createEyedropperController,
  align: createAlignController,
  booleans: createBooleansController,
  join: createJoinController,
  reference: createReferenceController,
  reference_data: createReferenceDataController,
  keys: createKeysController,
  document: createDocumentController,
  dialogs: createDialogs,
};

// The order controllers wire their events in.
const WIRED = ['transport', 'actions', 'session', 'drawing', 'geometry', 'paint', 'colors', 'tree', 'tree_drag',
  'commands', 'selection', 'agents', 'agent_controls', 'command_registry', 'arrows', 'stroke', 'transform',
  'gradients', 'view', 'hits', 'frames', 'overlay', 'resizing', 'rotating', 'markers', 'layout', 'tools', 'shapes',
  'pencil', 'canvas', 'canvas_events', 'clipping', 'region', 'eyedropper', 'align', 'booleans', 'join', 'reference',
  'reference_data', 'keys', 'document'];

// Which controller answers each collaborator a controller names among its
// options: a listed name calls the method of that name, {name: 'method'} a
// method named otherwise. Each resolves when it is called, so a controller can
// name one created after it. A controller is not given its own methods.
export const COLLABORATORS = {
  actions: ['action'],
  agent_controls: ['closeAgentPopover', 'showAgent', 'toggleAgents'],
  agents: ['drawFlash', 'pollAgent', 'pulse', 'reportView', 'schedulePoll', 'viewReport'],
  align: ['align', 'alignDescription', 'alignRefusal', 'distribute', 'distributeDescription', 'distributeRefusal', 'drawKeyObject', 'renderAlign'],
  booleans: [{booleanDescription: 'description', booleanRefusal: 'refusal', drawBooleanRoles: 'drawRoles', runBoolean: 'run'}],
  canvas: ['cancelStage', 'carryHandles', 'enterObject', 'finishBox', 'finishPointDrag', 'moveStage', 'pressStage', 'previewPointDrag',
    'releaseStage', 'restoreValues', 'stepUp', 'targetId', 'topSelection', 'valuesById'],
  canvas_events: ['startGesture'],
  clipping: ['openSetClip', 'removeClip', {removeClipRefusal: 'removeRefusal', setClipRefusal: 'setRefusal'}],
  colors: ['colorHex'],
  command_registry: ['commandName', 'commandRefusal', 'commandTitle'],
  commands: ['breakPoints', 'cutHole', 'deleteSelection', 'fillPaths', 'filledPath', 'joinCandidates', 'linePaths', 'noPoints', 'noReference',
    'noSelection', 'openJoin', 'openPalette', 'pointHandles', 'renameObject', 'renderCommands', 'runCommand', 'segmentPicked', 'splitParts',
    'twoEnds', 'visiblePaths'],
  document: ['fitArtboard', 'openArtboard', 'openNewDrawing'],
  drawing: ['renderDrawing'],
  eyedropper: ['drawEyedropper', {eyedropperChoose: 'choose', eyedropperSourceRefusal: 'sourceRefusal', eyedropperTargetRefusal: 'targetRefusal', sampleColour: 'sample'}],
  frames: ['drawOverlay', 'localToOverlay', 'screenMatrix'],
  geometry: ['contourAt', 'geometryNodes', 'level', 'missingGeometries', 'nodeAt', 'parents', 'pointPaths', 'pointTargetsNow', 'selectPoints',
    'selectedPoints', 'sharingPaths', 'shownPoints'],
  geometry_loader: ['loadGeometries'],
  gradients: ['createFillGradient', 'gradientRefusal', 'renderFillGradient', {drawGradient: 'drawHandles', finishGradient: 'finishHandle',
    gradientCursor: 'handleCursor', gradientDrag: 'handleDrag', gradientHandleAt: 'handleAt', gradientStatus: 'handlesStatus',
    pressGradient: 'pressHandle', previewGradient: 'previewHandle', restoreGradient: 'restoreHandle'}],
  hits: ['hitStack'],
  join: ['joinEnds', 'joinOpensDialog', 'mergeShapes'],
  layout: ['renderStatus', 'scheduleStrip'],
  markers: ['drawPoints'],
  overlay: ['drawHandles', 'drawOverlayContent', 'objectBounds', 'selectionBounds', 'selectionContour', 'selectionTargets', 'toggleClipping'],
  paint: ['cssColour', 'enable', 'gradientCss', 'objectContext', 'paint', 'paintColour', 'paintGradient', 'paintSwatch', 'paintValue',
    'renderInspector', 'resolvedPaint'],
  pencil: ['drawPencil', {finishPencil: 'finish', previewPencil: 'preview'}],
  reference: ['cycleReference', 'placeReference', 'refreshErrorView', 'removeReference', 'setView', 'showReference'],
  reference_data: ['loadReference'],
  region: ['drawRegion', {regionBack: 'back', regionCommand: 'command', regionCorner: 'corner', regionEnter: 'enter', regionRectangle: 'rectangle',
    regionRefusal: 'refusal'}],
  resizing: ['frameAt', 'frameSize', 'hoverResize', 'moveDrag', 'pressFrame', 'previewResize', 'resizeDrag', 'resizeRefusal'],
  rotating: ['drawRotation', 'flip', 'placePivot', 'pressRotation', 'previewRotate', 'restorePivot', 'rotateBy', 'rotateDrag', 'rotationAt',
    'rotationCursor', 'rotationStatus', 'selectionAngle', 'turnRefusal', {rotateRefusal: 'turnRefusal', rotateSelection: 'finishRotate'}],
  selection: ['clickEmpty', 'clickTargets', 'fillPointHoles', 'holeContours', 'nearKey', 'nearestPoint', 'nodeHandles', 'nudge',
    'pointContours', 'pointHolesToShapes', 'pointPairs', 'renderNodeInspector', 'sameClickSpot', 'selectAll', 'selectAllRefusal', 'selectAtPoint',
    'selectObject', 'selectedHandle', 'straightenHandles'],
  session: ['applyState', 'object', 'oneObject', 'svgElement', 'xmlElement'],
  shapes: ['drawShape', {cancelShape: 'cancel', drawingContainer: 'container', drawingPaint: 'paint', finishShape: 'finish', previewShape: 'preview',
    setDrawingColour: 'setColour'}],
  stroke: ['renderStrokeControls'],
  tools: ['cutWithKnife', 'drawKnife', 'drawPathDraft', 'drawRedraw', 'drawSnap', 'finishPath', 'knifeEnd', 'redrawLines', 'redrawOutline', 'snappedDrag'],
  transform: ['renderTransform', {transformRefusal: 'fieldRefusal'}],
  transport: ['later', 'laterMerged', 'request', 'setBusy', 'toast'],
  tree: ['renderObjects', 'revealObject', 'showTreeDrop', 'treeDropAt', 'updateObjectSelection'],
  tree_drag: ['endTreeDrag', 'pressTreeRow'],
  view: ['fit', 'focusSelection', 'point', 'setTool', 'updateView', 'zoomAt'],
};

// Collaborators taken as they are when a controller is created, not called.
export const VALUES = {
  canvas_events: ['gestures'],
  command_registry: ['COMMANDS', 'commandById'],
  transport: ['desktop', 'input'],
  tree: ['treeDropLine'],
};

// Opening and saving files, beside the controllers.
function createFiles(editor, {app}) {
  return projectFiles({
    getState: () => editor.state, getSession: () => editor.session, getBridge: () => app.transport.bridge,
    waitForEdits: () => editor.queue, isDirty: () => editor.dirty,
    markSaved: (...args) => app.session.markSaved(...args),
    request: app.transport.request, action: app.actions.action, setBusy: app.transport.setBusy, toast: app.transport.toast,
    refreshDrawing: async () => { await app.reference_data.loadReference(); app.view.fit(); },
  });
}

// The operation dialogs (Tidy, Snap edges, Generate, Fit colours, Clean up).
function createDialogs(editor, {app}) {
  function simplifyBounds() {
    const points = [];
    for (const id of editor.state.selection.objects) {
      const element = app.session.svgElement(id), matrix = element && app.frames.localToOverlay(element);
      if (!matrix) continue;
      const b = element.getBBox();
      for (const [x,y] of [[b.x,b.y],[b.x+b.width,b.y],[b.x+b.width,b.y+b.height],[b.x,b.y+b.height]]) points.push(new DOMPoint(x,y).matrixTransform(matrix));
    }
    if (!points.length) return editor.state.bounds;
    const left=Math.min(...points.map(p=>p.x)),right=Math.max(...points.map(p=>p.x)),top=Math.min(...points.map(p=>p.y)),bottom=Math.max(...points.map(p=>p.y));
    const padding=Math.max(8,Math.max(right-left,bottom-top)*.06);
    return [left-padding,top-padding,Math.max(1,right-left)+2*padding,Math.max(1,bottom-top)+2*padding];
  }
  return operationDialogs({
    getState: () => editor.state, getObject: app.session.object, waitForEdits: () => editor.queue,
    selectionBounds: simplifyBounds, viewReport: app.agents.viewReport,
    operation: (command, body) => app.transport.request('/api/operation', {command, ...body}),
    applyResult: app.session.applyState, toast: app.transport.toast,
  });
}

const entries = names => names.flatMap(name => typeof name === 'string' ? [[name, name]] : Object.entries(name));

// Compose controllers over one live editor state; each finds its collaborators
// on the application when it calls them.
export function createApplication({document = globalThis.document, window = globalThis.window, location = globalThis.location} = {}) {
  const editor = createEditorState();
  const $ = id => document.getElementById(id);
  const app = {editor};
  const options = {
    $, app, document, location, window,
    drawing: $('drawing'), overlay: $('overlay'), stage: $('stage'),
    moveLockRefusal: ids => app.align.lockRefusal(Object.fromEntries(ids.map(id => [id, true]))),
    regionShape: () => app.region.options.shape,
  };
  const provided = {};
  for (const [key, names] of Object.entries(COLLABORATORS)) for (const [name, method] of entries(names)) {
    options[name] = (...args) => app[key][method](...args);
    (provided[key] ||= []).push(name);
  }
  for (const [key, names] of Object.entries(VALUES)) for (const name of names) {
    Object.defineProperty(options, name, {get: () => app[key][name], enumerable: true});
    (provided[key] ||= []).push(name);
  }
  Object.defineProperty(options, 'files', {get: () => app.files, enumerable: true});
  const optionsFor = key => Object.create(options, Object.fromEntries((provided[key] || []).map(name => [name, {value: undefined}])));

  for (const [key, create] of Object.entries(CONTROLLERS)) app[key] = create(editor, optionsFor(key));

  async function start(){
    app.transport.setBusy('Opening editor…',1);
    try {
      const previous = sessionStorage.getItem('vectrify-session');
      let result = await app.transport.request('/api/session',{session:previous}); editor.session = result.session;
      result = await app.files.recover(previous, result);
      sessionStorage.setItem('vectrify-session',editor.session);
      await app.session.applyState(result);app.view.fit();
      app.agents.pollAgent();
    }
    catch(error){app.transport.toast(error.message,true);}finally{app.transport.setBusy('',-1);}
  }
  let wired = false;
  function wire() {
    if (wired) return app;
    wired = true;
    for (const key of WIRED) app[key].wire();
    return app;
  }
  return Object.assign(app, {start, wire});
}
