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
import {createCanvasController} from './app_canvas.js';
import {createCanvasEventsController} from './app_canvas_events.js';
import {createJoinController} from './app_join.js';
import {createReferenceController} from './app_reference.js';
import {createReferenceDataController} from './app_reference_data.js';
import {createKeysController} from './app_keys.js';
import {createTransformController} from './app_transform.js';
import {createClippingController} from './app_clipping.js';
import {createRegionController} from './app_region.js';
import {createAlignController} from './app_align.js';

// Compose controllers with explicit live state and named collaborators.
export function createApplication({document = globalThis.document, window = globalThis.window, location = globalThis.location} = {}) {
  const editor = createEditorState();
  const $ = id => document.getElementById(id);
  const drawing = $('drawing'), overlay = $('overlay'), stage = $('stage');
  const transportController = createTransportController(editor, {
    $,
    document,
    location,
    window,
  });
  const actionsController = createActionsController(editor, {
    $,
    applyState: (...args) => sessionController.applyState(...args),
    drawOverlay: (...args) => framesController.drawOverlay(...args),
    renderDrawing: (...args) => drawingController.renderDrawing(...args),
    renderInspector: (...args) => paintController.renderInspector(...args),
    request: (...args) => transportController.request(...args),
    setBusy: (...args) => transportController.setBusy(...args),
    toast: (...args) => transportController.toast(...args),
    window,
  });
  const files = projectFiles({
    getState: () => editor.state, getSession: () => editor.session, getBridge: () => transportController.bridge,
    waitForEdits: () => editor.queue, isDirty: () => editor.dirty,
    markSaved: (...args) => sessionController.markSaved(...args),
    request: transportController.request, action: actionsController.action, setBusy: transportController.setBusy, toast: transportController.toast,
    refreshDrawing: async () => { await reference_dataController.loadReference(); viewController.fit(); },
  });
  const sessionController = createSessionController(editor, {
    $,
    document,
    drawOverlay: (...args) => framesController.drawOverlay(...args),
    files,
    fit: (...args) => viewController.fit(...args),
    level: (...args) => geometryController.level(...args),
    loadGeometries: (...args) => geometryLoader.loadGeometries(...args),
    loadReference: (...args) => reference_dataController.loadReference(...args),
    missingGeometries: (...args) => geometryController.missingGeometries(...args),
    overlay,
    renderDrawing: (...args) => drawingController.renderDrawing(...args),
    renderInspector: (...args) => paintController.renderInspector(...args),
    renderObjects: (...args) => treeController.renderObjects(...args),
    selectedHandle: (...args) => selectionController.selectedHandle(...args),
    updateObjectSelection: (...args) => treeController.updateObjectSelection(...args),
  });
  const drawingController = createDrawingController(editor, {
    document,
    drawing,
    refreshErrorView: (...args) => referenceController.refreshErrorView(...args),
  });
  const geometryController = createGeometryController(editor, {
    action: (...args) => actionsController.action(...args),
    request: (...args) => transportController.request(...args),
    schedulePoll: (...args) => agentsController.schedulePoll(...args),
  });
  const geometryLoader = createGeometryLoader(editor, {
    missingGeometries: geometryController.missingGeometries,
    request: transportController.request,
    schedulePoll: (...args) => agentsController.schedulePoll(...args),
  });
  const paintController = createPaintController(editor, {
    $,
    action: (...args) => actionsController.action(...args),
    colorHex: (...args) => colorsController.colorHex(...args),
    document,
    drawing,
    later: (...args) => transportController.later(...args),
    object: (...args) => sessionController.object(...args),
    oneObject: (...args) => sessionController.oneObject(...args),
    renderCommands: (...args) => commandsController.renderCommands(...args),
    renderFillGradient: (...args) => gradientsController.renderFillGradient(...args),
    renderNodeInspector: (...args) => selectionController.renderNodeInspector(...args),
    renderAlign: (...args) => alignController.renderAlign(...args),
    renderStrokeControls: (...args) => strokeController.renderStrokeControls(...args),
    renderTransform: (...args) => transformController.renderTransform(...args),
    scheduleStrip: (...args) => layoutController.scheduleStrip(...args),
    selectObject: (...args) => selectionController.selectObject(...args),
    svgElement: (...args) => sessionController.svgElement(...args),
  });
  const colorsController = createColorsController(editor, {
  });
  const treeController = createTreeController(editor, {
    $,
    document,
    enterObject: (...args) => canvasController.enterObject(...args),
    later: (...args) => transportController.later(...args),
    object: (...args) => sessionController.object(...args),
    objectContext: (...args) => paintController.objectContext(...args),
    paintSwatch: (...args) => paintController.paintSwatch(...args),
    pressTreeRow: (...args) => tree_dragController.pressTreeRow(...args),
    selectObject: (...args) => selectionController.selectObject(...args),
  });
  const tree_dragController = createTreeDragController(editor, {
    $,
    action: (...args) => actionsController.action(...args),
    later: (...args) => transportController.later(...args),
    object: (...args) => sessionController.object(...args),
    showTreeDrop: (...args) => treeController.showTreeDrop(...args),
    toast: (...args) => transportController.toast(...args),
    treeDropAt: (...args) => treeController.treeDropAt(...args),
    treeDropLine: treeController.treeDropLine,
    window,
  });
  const command_registryController = createCommandRegistryController(editor, {
    $,
    action: (...args) => actionsController.action(...args),
    align: (...args) => alignController.align(...args),
    alignDescription: (...args) => alignController.alignDescription(...args),
    alignRefusal: (...args) => alignController.alignRefusal(...args),
    breakPoints: (...args) => commandsController.breakPoints(...args),
    closeAgentPopover: (...args) => agent_controlsController.closeAgentPopover(...args),
    contourAt: (...args) => geometryController.contourAt(...args),
    createFillGradient: (...args) => gradientsController.createFillGradient(...args),
    cutHole: (...args) => commandsController.cutHole(...args),
    cycleReference: (...args) => referenceController.cycleReference(...args),
    deleteSelection: (...args) => commandsController.deleteSelection(...args),
    distribute: (...args) => alignController.distribute(...args),
    distributeDescription: (...args) => alignController.distributeDescription(...args),
    distributeRefusal: (...args) => alignController.distributeRefusal(...args),
    document,
    download,
    enterObject: (...args) => canvasController.enterObject(...args),
    files,
    fillPaths: (...args) => commandsController.fillPaths(...args),
    fillPointHoles: (...args) => selectionController.fillPointHoles(...args),
    finishPath: (...args) => toolsController.finishPath(...args),
    fit: (...args) => viewController.fit(...args),
    flip: (...args) => rotatingController.flip(...args),
    focusSelection: (...args) => viewController.focusSelection(...args),
    gradientRefusal: (...args) => gradientsController.gradientRefusal(...args),
    holeContours: (...args) => selectionController.holeContours(...args),
    joinCandidates: (...args) => commandsController.joinCandidates(...args),
    joinEnds: (...args) => joinController.joinEnds(...args),
    joinOpensDialog: (...args) => joinController.joinOpensDialog(...args),
    level: (...args) => geometryController.level(...args),
    linePaths: (...args) => commandsController.linePaths(...args),
    mergeShapes: (...args) => joinController.mergeShapes(...args),
    noPoints: (...args) => commandsController.noPoints(...args),
    noReference: (...args) => commandsController.noReference(...args),
    noSelection: (...args) => commandsController.noSelection(...args),
    nodeAt: (...args) => geometryController.nodeAt(...args),
    object: (...args) => sessionController.object(...args),
    oneObject: (...args) => sessionController.oneObject(...args),
    openCleanup,
    openColours,
    openGenerate,
    openPalette: (...args) => commandsController.openPalette(...args),
    openSetClip: (...args) => clippingController.openSetClip(...args),
    openSnapEdges,
    openTidy,
    pointHandles: (...args) => commandsController.pointHandles(...args),
    pointHolesToShapes: (...args) => selectionController.pointHolesToShapes(...args),
    pointPairs: (...args) => selectionController.pointPairs(...args),
    regionCommand: (...args) => regionController.command(...args),
    regionRefusal: (...args) => regionController.refusal(...args),
    removeClip: (...args) => clippingController.removeClip(...args),
    removeClipRefusal: (...args) => clippingController.removeRefusal(...args),
    removeReference: (...args) => referenceController.removeReference(...args),
    renameObject: (...args) => commandsController.renameObject(...args),
    segmentPicked: (...args) => commandsController.segmentPicked(...args),
    selectedHandle: (...args) => selectionController.selectedHandle(...args),
    selectedPoints: (...args) => geometryController.selectedPoints(...args),
    setClipRefusal: (...args) => clippingController.setRefusal(...args),
    setTool: (...args) => viewController.setTool(...args),
    setView: (...args) => referenceController.setView(...args),
    splitParts: (...args) => commandsController.splitParts(...args),
    stepUp: (...args) => canvasController.stepUp(...args),
    straightenHandles: (...args) => selectionController.straightenHandles(...args),
    tidyTargets,
    toast: (...args) => transportController.toast(...args),
    toggleAgents: (...args) => agent_controlsController.toggleAgents(...args),
    toggleClipping: (...args) => overlayController.toggleClipping(...args),
    transformRefusal: (...args) => transformController.fieldRefusal(...args),
    turnRefusal: (...args) => rotatingController.turnRefusal(...args),
    twoEnds: (...args) => commandsController.twoEnds(...args),
    visiblePaths: (...args) => commandsController.visiblePaths(...args),
    zoomAt: (...args) => viewController.zoomAt(...args),
  });
  const commandsController = createCommandsController(editor, {
    $,
    COMMANDS: command_registryController.COMMANDS,
    action: (...args) => actionsController.action(...args),
    clickTargets: (...args) => selectionController.clickTargets(...args),
    colorHex: (...args) => colorsController.colorHex(...args),
    commandById: command_registryController.commandById,
    commandName: (...args) => command_registryController.commandName(...args),
    commandRefusal: (...args) => command_registryController.commandRefusal(...args),
    commandTitle: (...args) => command_registryController.commandTitle(...args),
    document,
    drawOverlay: (...args) => framesController.drawOverlay(...args),
    hitStack: (...args) => hitsController.hitStack(...args),
    later: (...args) => transportController.later(...args),
    level: (...args) => geometryController.level(...args),
    object: (...args) => sessionController.object(...args),
    paintSwatch: (...args) => paintController.paintSwatch(...args),
    pointContours: (...args) => selectionController.pointContours(...args),
    pointPairs: (...args) => selectionController.pointPairs(...args),
    pointPaths: (...args) => geometryController.pointPaths(...args),
    renderInspector: (...args) => paintController.renderInspector(...args),
    resolvedPaint: (...args) => paintController.resolvedPaint(...args),
    selectObject: (...args) => selectionController.selectObject(...args),
    selectPoints: (...args) => geometryController.selectPoints(...args),
    selectedHandle: (...args) => selectionController.selectedHandle(...args),
    selectedPoints: (...args) => geometryController.selectedPoints(...args),
    stage,
    toast: (...args) => transportController.toast(...args),
    window,
  });
  const selectionController = createSelectionController(editor, {
    $,
    action: (...args) => actionsController.action(...args),
    contourAt: (...args) => geometryController.contourAt(...args),
    document,
    focusSelection: (...args) => viewController.focusSelection(...args),
    geometryNodes: (...args) => geometryController.geometryNodes(...args),
    later: (...args) => transportController.later(...args),
    level: (...args) => geometryController.level(...args),
    nodeAt: (...args) => geometryController.nodeAt(...args),
    object: (...args) => sessionController.object(...args),
    overlay,
    parents: (...args) => geometryController.parents(...args),
    pointPaths: (...args) => geometryController.pointPaths(...args),
    pointTargetsNow: (...args) => geometryController.pointTargetsNow(...args),
    renderCommands: (...args) => commandsController.renderCommands(...args),
    request: (...args) => transportController.request(...args),
    revealObject: (...args) => treeController.revealObject(...args),
    runCommand: (...args) => commandsController.runCommand(...args),
    scheduleStrip: (...args) => layoutController.scheduleStrip(...args),
    selectPoints: (...args) => geometryController.selectPoints(...args),
    selectedPoints: (...args) => geometryController.selectedPoints(...args),
    sharingPaths: (...args) => geometryController.sharingPaths(...args),
    svgElement: (...args) => sessionController.svgElement(...args),
    toast: (...args) => transportController.toast(...args),
  });
  const agentsController = createAgentsController(editor, {
    applyState: (...args) => sessionController.applyState(...args),
    drawOverlay: (...args) => framesController.drawOverlay(...args),
    localToOverlay: (...args) => framesController.localToOverlay(...args),
    request: (...args) => transportController.request(...args),
    showAgent: (...args) => agent_controlsController.showAgent(...args),
    stage,
    svgElement: (...args) => sessionController.svgElement(...args),
    toast: (...args) => transportController.toast(...args),
    window,
    xmlElement: (...args) => sessionController.xmlElement(...args),
  });
  const agent_controlsController = createAgentControlsController(editor, {
    $,
    desktop: transportController.desktop,
    document,
    pollAgent: (...args) => agentsController.pollAgent(...args),
    pulse: (...args) => agentsController.pulse(...args),
    request: (...args) => transportController.request(...args),
    toast: (...args) => transportController.toast(...args),
    window,
  });
  const arrowsController = createArrowsController(editor, {
    drawing,
    svgElement: (...args) => sessionController.svgElement(...args),
  });
  const strokeController = createStrokeController(editor, {
    $,
    enable: (...args) => paintController.enable(...args),
    object: (...args) => sessionController.object(...args),
    paint: (...args) => paintController.paint(...args),
    resolvedPaint: (...args) => paintController.resolvedPaint(...args),
    toast: (...args) => transportController.toast(...args),
  });
  const transformController = createTransformController(editor, {
    $,
    action: (...args) => actionsController.action(...args),
    later: (...args) => transportController.later(...args),
    object: (...args) => sessionController.object(...args),
    resizeRefusal: (...args) => resizingController.resizeRefusal(...args),
    rotateBy: (...args) => rotatingController.rotateBy(...args),
    rotateRefusal: (...args) => rotatingController.turnRefusal(...args),
    selectionAngle: (...args) => rotatingController.selectionAngle(...args),
    selectionBounds: (...args) => overlayController.selectionBounds(...args),
    toast: (...args) => transportController.toast(...args),
    topSelection: (...args) => canvasController.topSelection(...args),
  });
  const gradientsController = createGradientsController(editor, {
    $,
    action: (...args) => actionsController.action(...args),
    colorHex: (...args) => colorsController.colorHex(...args),
    cssColour: (...args) => paintController.cssColour(...args),
    document,
    gradientCss: (...args) => paintController.gradientCss(...args),
    drawOverlay: (...args) => framesController.drawOverlay(...args),
    later: (...args) => transportController.later(...args),
    localToOverlay: (...args) => framesController.localToOverlay(...args),
    oneObject: (...args) => sessionController.oneObject(...args),
    overlay,
    paintColour: (...args) => paintController.paintColour(...args),
    paintGradient: (...args) => paintController.paintGradient(...args),
    paintValue: (...args) => paintController.paintValue(...args),
    point: (...args) => viewController.point(...args),
    renderInspector: (...args) => paintController.renderInspector(...args),
    resolvedPaint: (...args) => paintController.resolvedPaint(...args),
    setTool: (...args) => viewController.setTool(...args),
    startGesture: (...args) => canvas_eventsController.startGesture(...args),
    svgElement: (...args) => sessionController.svgElement(...args),
    toast: (...args) => transportController.toast(...args),
    xmlElement: (...args) => sessionController.xmlElement(...args),
  });
  const viewController = createViewController(editor, {
    $,
    action: (...args) => actionsController.action(...args),
    document,
    drawOverlay: (...args) => framesController.drawOverlay(...args),
    drawing,
    level: (...args) => geometryController.level(...args),
    loadGeometries: (...args) => geometryLoader.loadGeometries(...args),
    localToOverlay: (...args) => framesController.localToOverlay(...args),
    overlay,
    renderInspector: (...args) => paintController.renderInspector(...args),
    reportView: (...args) => agentsController.reportView(...args),
    screenMatrix: (...args) => framesController.screenMatrix(...args),
    selectedPoints: (...args) => geometryController.selectedPoints(...args),
    selectionTargets: (...args) => overlayController.selectionTargets(...args),
    setBusy: (...args) => transportController.setBusy(...args),
    stage,
    toast: (...args) => transportController.toast(...args),
  });
  const hitsController = createHitsController(editor, {
    document,
    object: (...args) => sessionController.object(...args),
    svgElement: (...args) => sessionController.svgElement(...args),
    targetId: (...args) => canvasController.targetId(...args),
  });
  const framesController = createFramesController(editor, {
    document,
    drawOverlayContent: (...args) => overlayController.drawOverlayContent(...args),
    drawing,
    overlay,
  });
  const overlayController = createOverlayController(editor, {
    $,
    drawFlash: (...args) => agentsController.drawFlash(...args),
    drawKeyObject: (...args) => alignController.drawKeyObject(...args),
    drawKnife: (...args) => toolsController.drawKnife(...args),
    drawOverlay: (...args) => framesController.drawOverlay(...args),
    drawPathDraft: (...args) => toolsController.drawPathDraft(...args),
    drawPoints: (...args) => markersController.drawPoints(...args),
    drawRedraw: (...args) => toolsController.drawRedraw(...args),
    drawRegion: (...args) => regionController.drawRegion(...args),
    drawRotation: (...args) => rotatingController.drawRotation(...args),
    drawGradient: (...args) => gradientsController.drawHandles(...args),
    drawShape: (...args) => shapesController.drawShape(...args),
    drawSnap: (...args) => toolsController.drawSnap(...args),
    drawing,
    later: (...args) => transportController.later(...args),
    level: (...args) => geometryController.level(...args),
    localToOverlay: (...args) => framesController.localToOverlay(...args),
    nodeAt: (...args) => geometryController.nodeAt(...args),
    nodeHandles: (...args) => selectionController.nodeHandles(...args),
    object: (...args) => sessionController.object(...args),
    overlay,
    renderStatus: (...args) => layoutController.renderStatus(...args),
    resizeRefusal: (...args) => resizingController.resizeRefusal(...args),
    svgElement: (...args) => sessionController.svgElement(...args),
    topSelection: (...args) => canvasController.topSelection(...args),
    xmlElement: (...args) => sessionController.xmlElement(...args),
  });
  const resizingController = createResizingController(editor, {
    hitStack: (...args) => hitsController.hitStack(...args),
    localToOverlay: (...args) => framesController.localToOverlay(...args),
    object: (...args) => sessionController.object(...args),
    objectBounds: (...args) => overlayController.objectBounds(...args),
    overlay,
    parents: (...args) => geometryController.parents(...args),
    point: (...args) => viewController.point(...args),
    rotationCursor: (...args) => rotatingController.rotationCursor(...args),
    stage,
    startGesture: (...args) => canvas_eventsController.startGesture(...args),
    svgElement: (...args) => sessionController.svgElement(...args),
    topSelection: (...args) => canvasController.topSelection(...args),
  });
  const rotatingController = createRotatingController(editor, {
    action: (...args) => actionsController.action(...args),
    drawOverlay: (...args) => framesController.drawOverlay(...args),
    localToOverlay: (...args) => framesController.localToOverlay(...args),
    object: (...args) => sessionController.object(...args),
    overlay,
    point: (...args) => viewController.point(...args),
    selectionBounds: (...args) => overlayController.selectionBounds(...args),
    startGesture: (...args) => canvas_eventsController.startGesture(...args),
    svgElement: (...args) => sessionController.svgElement(...args),
    topSelection: (...args) => canvasController.topSelection(...args),
    xmlElement: (...args) => sessionController.xmlElement(...args),
  });
  const markersController = createMarkersController(editor, {
    $,
    drawHandles: (...args) => overlayController.drawHandles(...args),
    geometryNodes: (...args) => geometryController.geometryNodes(...args),
    localToOverlay: (...args) => framesController.localToOverlay(...args),
    pointPaths: (...args) => geometryController.pointPaths(...args),
    screenMatrix: (...args) => framesController.screenMatrix(...args),
    shownPoints: (...args) => geometryController.shownPoints(...args),
    stage,
    svgElement: (...args) => sessionController.svgElement(...args),
    xmlElement: (...args) => sessionController.xmlElement(...args),
  });
  const layoutController = createLayoutController(editor, {
    $,
    document,
    frameSize: (...args) => resizingController.frameSize(...args),
    rotationStatus: (...args) => rotatingController.rotationStatus(...args),
    gradientStatus: (...args) => gradientsController.handlesStatus(...args),
    level: (...args) => geometryController.level(...args),
    object: (...args) => sessionController.object(...args),
    parents: (...args) => geometryController.parents(...args),
    pointPaths: (...args) => geometryController.pointPaths(...args),
    reportView: (...args) => agentsController.reportView(...args),
    selectedPoints: (...args) => geometryController.selectedPoints(...args),
    window,
  });
  const toolsController = createToolsController(editor, {
    $,
    action: (...args) => actionsController.action(...args),
    drawOverlay: (...args) => framesController.drawOverlay(...args),
    drawing,
    drawingContainer: (...args) => shapesController.container(...args),
    drawingPaint: (...args) => shapesController.paint(...args),
    filledPath: (...args) => commandsController.filledPath(...args),
    geometryNodes: (...args) => geometryController.geometryNodes(...args),
    localToOverlay: (...args) => framesController.localToOverlay(...args),
    point: (...args) => viewController.point(...args),
    pointPaths: (...args) => geometryController.pointPaths(...args),
    setTool: (...args) => viewController.setTool(...args),
    svgElement: (...args) => sessionController.svgElement(...args),
    toast: (...args) => transportController.toast(...args),
    xmlElement: (...args) => sessionController.xmlElement(...args),
  });
  const shapesController = createShapesController(editor, {
    $,
    action: (...args) => actionsController.action(...args),
    document,
    toast: (...args) => transportController.toast(...args),
    xmlElement: (...args) => sessionController.xmlElement(...args),
  });
  const canvas_eventsController = createCanvasEventsController(editor, {
    $,
    action: (...args) => actionsController.action(...args),
    cancelShape: (...args) => shapesController.cancel(...args),
    cancelStage: (...args) => canvasController.cancelStage(...args),
    cutWithKnife: (...args) => toolsController.cutWithKnife(...args),
    document,
    drawOverlay: (...args) => framesController.drawOverlay(...args),
    enterObject: (...args) => canvasController.enterObject(...args),
    finishBox: (...args) => canvasController.finishBox(...args),
    finishPath: (...args) => toolsController.finishPath(...args),
    finishPointDrag: (...args) => canvasController.finishPointDrag(...args),
    finishGradient: (...args) => gradientsController.finishHandle(...args),
    finishShape: (...args) => shapesController.finish(...args),
    fit: (...args) => viewController.fit(...args),
    gradientDrag: (...args) => gradientsController.handleDrag(...args),
    hitStack: (...args) => hitsController.hitStack(...args),
    input: transportController.input,
    knifeEnd: (...args) => toolsController.knifeEnd(...args),
    later: (...args) => transportController.later(...args),
    level: (...args) => geometryController.level(...args),
    moveDrag: (...args) => resizingController.moveDrag(...args),
    moveStage: (...args) => canvasController.moveStage(...args),
    object: (...args) => sessionController.object(...args),
    point: (...args) => viewController.point(...args),
    pressStage: (...args) => canvasController.pressStage(...args),
    previewPointDrag: (...args) => canvasController.previewPointDrag(...args),
    placePivot: (...args) => rotatingController.placePivot(...args),
    previewGradient: (...args) => gradientsController.previewHandle(...args),
    previewResize: (...args) => resizingController.previewResize(...args),
    previewRotate: (...args) => rotatingController.previewRotate(...args),
    previewShape: (...args) => shapesController.preview(...args),
    redrawOutline: (...args) => toolsController.redrawOutline(...args),
    regionCorner: (...args) => regionController.corner(...args),
    regionRectangle: (...args) => regionController.rectangle(...args),
    releaseStage: (...args) => canvasController.releaseStage(...args),
    renderDrawing: (...args) => drawingController.renderDrawing(...args),
    renderInspector: (...args) => paintController.renderInspector(...args),
    renderNodeInspector: (...args) => selectionController.renderNodeInspector(...args),
    resizeDrag: (...args) => resizingController.resizeDrag(...args),
    restoreGradient: (...args) => gradientsController.restoreHandle(...args),
    restorePivot: (...args) => rotatingController.restorePivot(...args),
    restoreValues: (...args) => canvasController.restoreValues(...args),
    rotateDrag: (...args) => rotatingController.rotateDrag(...args),
    rotateSelection: (...args) => rotatingController.finishRotate(...args),
    selectAtPoint: (...args) => selectionController.selectAtPoint(...args),
    selectPoints: (...args) => geometryController.selectPoints(...args),
    setTool: (...args) => viewController.setTool(...args),
    snappedDrag: (...args) => toolsController.snappedDrag(...args),
    stage,
    toast: (...args) => transportController.toast(...args),
    updateView: (...args) => viewController.updateView(...args),
    zoomAt: (...args) => viewController.zoomAt(...args),
  });
  const canvasController = createCanvasController(editor, {
    $,
    action: (...args) => actionsController.action(...args),
    clickEmpty: (...args) => selectionController.clickEmpty(...args),
    clickTargets: (...args) => selectionController.clickTargets(...args),
    drawOverlay: (...args) => framesController.drawOverlay(...args),
    drawing,
    finishPath: (...args) => toolsController.finishPath(...args),
    frameAt: (...args) => resizingController.frameAt(...args),
    geometryNodes: (...args) => geometryController.geometryNodes(...args),
    gestures: canvas_eventsController.gestures,
    gradientCursor: (...args) => gradientsController.handleCursor(...args),
    gradientHandleAt: (...args) => gradientsController.handleAt(...args),
    hitStack: (...args) => hitsController.hitStack(...args),
    hoverResize: (...args) => resizingController.hoverResize(...args),
    later: (...args) => transportController.later(...args),
    level: (...args) => geometryController.level(...args),
    loadGeometries: (...args) => geometryLoader.loadGeometries(...args),
    localToOverlay: (...args) => framesController.localToOverlay(...args),
    nearKey: (...args) => selectionController.nearKey(...args),
    nearestPoint: (...args) => selectionController.nearestPoint(...args),
    nodeAt: (...args) => geometryController.nodeAt(...args),
    nodeHandles: (...args) => selectionController.nodeHandles(...args),
    object: (...args) => sessionController.object(...args),
    overlay,
    parents: (...args) => geometryController.parents(...args),
    point: (...args) => viewController.point(...args),
    pointPaths: (...args) => geometryController.pointPaths(...args),
    pressFrame: (...args) => resizingController.pressFrame(...args),
    pressGradient: (...args) => gradientsController.pressHandle(...args),
    pressRotation: (...args) => rotatingController.pressRotation(...args),
    redrawLines: (...args) => toolsController.redrawLines(...args),
    regionShape: () => regionController.options.shape,
    renderDrawing: (...args) => drawingController.renderDrawing(...args),
    renderInspector: (...args) => paintController.renderInspector(...args),
    renderNodeInspector: (...args) => selectionController.renderNodeInspector(...args),
    renderStatus: (...args) => layoutController.renderStatus(...args),
    rotationAt: (...args) => rotatingController.rotationAt(...args),
    sameClickSpot: (...args) => selectionController.sameClickSpot(...args),
    screenMatrix: (...args) => framesController.screenMatrix(...args),
    selectPoints: (...args) => geometryController.selectPoints(...args),
    selectedHandle: (...args) => selectionController.selectedHandle(...args),
    selectedPoints: (...args) => geometryController.selectedPoints(...args),
    setTool: (...args) => viewController.setTool(...args),
    sharingPaths: (...args) => geometryController.sharingPaths(...args),
    stage,
    startGesture: (...args) => canvas_eventsController.startGesture(...args),
    svgElement: (...args) => sessionController.svgElement(...args),
    toast: (...args) => transportController.toast(...args),
  });

  const clippingController = createClippingController(editor, {
    $,
    action: (...args) => actionsController.action(...args),
    document,
    noSelection: (...args) => commandsController.noSelection(...args),
    object: (...args) => sessionController.object(...args),
    paintSwatch: (...args) => paintController.paintSwatch(...args),
    plural: (...args) => selectionController.plural(...args),
    toast: (...args) => transportController.toast(...args),
  });

  const regionController = createRegionController(editor, {
    $,
    action: (...args) => actionsController.action(...args),
    document,
    drawOverlay: (...args) => framesController.drawOverlay(...args),
    later: (...args) => transportController.later(...args),
    localToOverlay: (...args) => framesController.localToOverlay(...args),
    object: (...args) => sessionController.object(...args),
    plural: (...args) => selectionController.plural(...args),
    renderCommands: (...args) => commandsController.renderCommands(...args),
    request: (...args) => transportController.request(...args),
    runCommand: (...args) => commandsController.runCommand(...args),
    selectionContour: (...args) => overlayController.selectionContour(...args),
    setTool: (...args) => viewController.setTool(...args),
    stage,
    svgElement: (...args) => sessionController.svgElement(...args),
    toast: (...args) => transportController.toast(...args),
    xmlElement: (...args) => sessionController.xmlElement(...args),
  });

  const alignController = createAlignController(editor, {
    $,
    action: (...args) => actionsController.action(...args),
    document,
    drawOverlay: (...args) => framesController.drawOverlay(...args),
    later: (...args) => transportController.later(...args),
    level: (...args) => geometryController.level(...args),
    object: (...args) => sessionController.object(...args),
    objectBounds: (...args) => overlayController.objectBounds(...args),
    renderCommands: (...args) => commandsController.renderCommands(...args),
    toast: (...args) => transportController.toast(...args),
    topSelection: (...args) => canvasController.topSelection(...args),
    xmlElement: (...args) => sessionController.xmlElement(...args),
  });

  const joinController = createJoinController(editor, {
    action: (...args) => actionsController.action(...args),
    fillPaths: (...args) => commandsController.fillPaths(...args),
    joinCandidates: (...args) => commandsController.joinCandidates(...args),
    linePaths: (...args) => commandsController.linePaths(...args),
    openJoin: (...args) => commandsController.openJoin(...args),
    plural: (...args) => selectionController.plural(...args),
    toast: (...args) => transportController.toast(...args),
  });
  const referenceController = createReferenceController(editor, {
    $,
    action: (...args) => actionsController.action(...args),
    document,
    fit: (...args) => viewController.fit(...args),
    loadReference: (...args) => reference_dataController.loadReference(...args),
    renderInspector: (...args) => paintController.renderInspector(...args),
    reportView: (...args) => agentsController.reportView(...args),
    request: (...args) => transportController.request(...args),
    scheduleStrip: (...args) => layoutController.scheduleStrip(...args),
    stage,
    toast: (...args) => transportController.toast(...args),
    viewReport: (...args) => agentsController.viewReport(...args),
  });
  const reference_dataController = createReferenceDataController(editor, {
    request: (...args) => transportController.request(...args),
    showReference: (...args) => referenceController.showReference(...args),
  });
  const keysController = createKeysController(editor, {
    $,
    action: (...args) => actionsController.action(...args),
    cycleReference: (...args) => referenceController.cycleReference(...args),
    document,
    drawOverlay: (...args) => framesController.drawOverlay(...args),
    endTreeDrag: (...args) => tree_dragController.endTreeDrag(...args),
    files,
    finishPath: (...args) => toolsController.finishPath(...args),
    fit: (...args) => viewController.fit(...args),
    focusSelection: (...args) => viewController.focusSelection(...args),
    gestures: canvas_eventsController.gestures,
    input: transportController.input,
    later: (...args) => transportController.later(...args),
    level: (...args) => geometryController.level(...args),
    oneObject: (...args) => sessionController.oneObject(...args),
    openPalette: (...args) => commandsController.openPalette(...args),
    regionBack: (...args) => regionController.back(...args),
    regionEnter: (...args) => regionController.enter(...args),
    renameObject: (...args) => commandsController.renameObject(...args),
    renderDrawing: (...args) => drawingController.renderDrawing(...args),
    renderInspector: (...args) => paintController.renderInspector(...args),
    runCommand: (...args) => commandsController.runCommand(...args),
    selectedPoints: (...args) => geometryController.selectedPoints(...args),
    setTool: (...args) => viewController.setTool(...args),
    stage,
    stepUp: (...args) => canvasController.stepUp(...args),
    window,
  });
  function download(project) { return files.download(project); }
  async function start(){
    transportController.setBusy('Opening editor…',1);
    try {
      const previous = sessionStorage.getItem('vectrify-session');
      let result = await transportController.request('/api/session',{session:previous}); editor.session = result.session;
      result = await files.recover(previous, result);
      sessionStorage.setItem('vectrify-session',editor.session);
      await sessionController.applyState(result);viewController.fit();
      agentsController.pollAgent();
    }
    catch(error){transportController.toast(error.message,true);}finally{transportController.setBusy('',-1);}
  }
  function simplifyBounds() {
    const points = [];
    for (const id of editor.state.selection.objects) {
      const element = sessionController.svgElement(id), matrix = element && framesController.localToOverlay(element);
      if (!matrix) continue;
      const b = element.getBBox();
      for (const [x,y] of [[b.x,b.y],[b.x+b.width,b.y],[b.x+b.width,b.y+b.height],[b.x,b.y+b.height]]) points.push(new DOMPoint(x,y).matrixTransform(matrix));
    }
    if (!points.length) return editor.state.bounds;
    const left=Math.min(...points.map(p=>p.x)),right=Math.max(...points.map(p=>p.x)),top=Math.min(...points.map(p=>p.y)),bottom=Math.max(...points.map(p=>p.y));
    const padding=Math.max(8,Math.max(right-left,bottom-top)*.06);
    return [left-padding,top-padding,Math.max(1,right-left)+2*padding,Math.max(1,bottom-top)+2*padding];
  }
  const dialogs = operationDialogs({
    getState: () => editor.state, getObject: sessionController.object, waitForEdits: () => editor.queue,
    selectionBounds: simplifyBounds, viewReport: agentsController.viewReport,
    operation: (command, body) => transportController.request('/api/operation', {command, ...body}),
    applyResult: sessionController.applyState, toast: transportController.toast,
  });
  function tidyTargets() { return dialogs.tidyTargets(); }
  function openTidy() { return dialogs.openTidy(); }
  function openSnapEdges() { return dialogs.openSnapEdges(); }
  function openGenerate() { return dialogs.openGenerate(); }
  function openColours(fill) { return dialogs.openColours(fill); }
  function openCleanup() { return dialogs.openCleanup(); }

  let wired = false;
  function wire() {
      if (wired) return application;
      wired = true;
      transportController.wire();
      actionsController.wire();
      sessionController.wire();
      drawingController.wire();
      geometryController.wire();
      paintController.wire();
      colorsController.wire();
      treeController.wire();
      tree_dragController.wire();
      commandsController.wire();
      selectionController.wire();
      agentsController.wire();
      agent_controlsController.wire();
      command_registryController.wire();
      arrowsController.wire();
      strokeController.wire();
      transformController.wire();
      gradientsController.wire();
      viewController.wire();
      hitsController.wire();
      framesController.wire();
      overlayController.wire();
      resizingController.wire();
      rotatingController.wire();
      markersController.wire();
      layoutController.wire();
      toolsController.wire();
      shapesController.wire();
      canvasController.wire();
      canvas_eventsController.wire();
      clippingController.wire();
      regionController.wire();
      alignController.wire();
      joinController.wire();
      referenceController.wire();
      reference_dataController.wire();
      keysController.wire();
      return application;
  }
  const application = {
    editor,
    transport: transportController,
    actions: actionsController,
    session: sessionController,
    drawing: drawingController,
    geometry: geometryController,
    paint: paintController,
    colors: colorsController,
    tree: treeController,
    tree_drag: tree_dragController,
    commands: commandsController,
    selection: selectionController,
    agents: agentsController,
    agent_controls: agent_controlsController,
    command_registry: command_registryController,
    arrows: arrowsController,
    stroke: strokeController,
    transform: transformController,
    gradients: gradientsController,
    view: viewController,
    hits: hitsController,
    frames: framesController,
    overlay: overlayController,
    resizing: resizingController,
    rotating: rotatingController,
    markers: markersController,
    layout: layoutController,
    tools: toolsController,
    shapes: shapesController,
    canvas: canvasController,
    canvas_events: canvas_eventsController,
    clipping: clippingController,
    region: regionController,
    align: alignController,
    join: joinController,
    reference: referenceController,
    reference_data: reference_dataController,
    keys: keysController,
    files,
    dialogs,
    start,
    wire,
  };
  return application;
}
