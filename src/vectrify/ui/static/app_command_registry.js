import {splitKey} from './selection.js';
import {breakable} from './lines.js';
import {JOIN_REACH} from './app_config.js';
import {SHAPES} from './clipping.js';
import {ALIGNMENTS, distributeName} from './align.js';
import {BOOLEANS} from './booleans.js';
import {EXPORTED, hiddenRefusal, visibilityRefusal} from './visibility.js';

// Command_registry owns its controls; editor contains the live state shared with other controllers.
// *app* is the application: commands find the controllers that run them on it
// when they run, so the registry can be created before them.
export function createCommandRegistryController(editor, {
  $,
  app,
  document = globalThis.document,
} = {}) {

  // Every command the editor offers, once. The inspector, the tool strip, the
  // context menu, the command palette and the shortcuts all run these and show
  // the same name and the same reason a command cannot run now. *name* is what
  // it is called (and what the palette finds); *label* names it for the
  // current selection; *group* is where it is listed (Paint, Transform,
  // Arrange, Path, Clipping, Repair and Reference for objects, Nodes for
  // points); *description* is the tooltip of a control that runs it;
  // *disabled* gives the reason it cannot run now, or '' when it can.
  const COMMANDS = [
    {id: 'tool-select', name: 'Select tool', group: 'Tools', keys: 'A', keywords: 'pick arrow objects box', run: () => app.view.setTool('select')},
    {id: 'tool-move', name: 'Move tool', group: 'Tools', keys: 'V', keywords: 'translate position line objects', run: () => app.view.setTool('move')},
    {id: 'tool-nodes', name: 'Nodes tool', group: 'Tools', keys: 'S', keywords: 'edit points handles', run: () => app.view.setTool('nodes')},
    {id: 'tool-path', name: 'Draw path tool', group: 'Tools', keys: 'D', keywords: 'pen draw path shape', run: () => app.view.setTool('path')},
    {id: 'tool-rect', name: 'Rectangle tool', group: 'Tools', keys: 'M', keywords: 'draw rectangle square rounded corners box shape', run: () => app.view.setTool('rect')},
    {id: 'tool-ellipse', name: 'Ellipse tool', group: 'Tools', keys: 'E', keywords: 'draw ellipse circle oval shape', run: () => app.view.setTool('ellipse')},
    {id: 'tool-line', name: 'Line tool', group: 'Tools', keys: 'L', keywords: 'draw straight line segment shape', run: () => app.view.setTool('line')},
    {id: 'tool-pencil', name: 'Pencil tool', group: 'Tools', keys: 'P', keywords: 'freehand draw sketch brush stroke pen smooth curve path', run: () => app.view.setTool('pencil')},
    {id: 'tool-eyedropper', name: 'Eyedropper tool', group: 'Tools', keys: 'I', keywords: 'sample pick colour color picker dropper fill stroke reference', run: () => app.view.setTool('eyedropper')},
    {id: 'tool-knife', name: 'Knife tool', group: 'Tools', keys: 'C', keywords: 'cut slice', run: () => app.view.setTool('knife')},
    {id: 'tool-region', name: 'Region tool', group: 'Tools', keys: 'X', keywords: 'extract delete erase area rectangle polygon lasso crop', run: () => app.view.setTool('region')},
    {id: 'tool-gradient', name: 'Gradient tool', group: 'Tools', keys: 'G', keywords: 'linear gradient endpoints handles fill direction', run: () => app.view.setTool('gradient')},
    {id: 'tool-redraw', name: 'Redraw outline tool', group: 'Tools', keys: 'R', keywords: 'lasso outline fix', run: () => app.view.setTool('redraw')},
    {id: 'tool-hand', name: 'Pan tool', group: 'Tools', keys: 'Hold Space', keywords: 'hand scroll', run: () => app.view.setTool('hand')},
    {id: 'new-drawing', name: 'New drawing…', group: 'File', keys: 'Ctrl/⌘ Alt N', keywords: 'new blank empty document create start size width height', run: () => app.document.openNewDrawing()},
    {id: 'open', name: 'Open…', group: 'File', keywords: 'svg project load', run: () => $('open-file').click()},
    {id: 'restore', name: 'Restore saved…', group: 'File', keywords: 'recovery browser copy', run: () => $('restore-saved').click()},
    {id: 'save', name: 'Save', group: 'File', keys: 'Ctrl/⌘ S', keywords: 'file write', run: () => app.files.save()},
    {id: 'save-as', name: 'Save As…', group: 'File', keys: 'Ctrl/⌘ Shift S', keywords: 'file destination menu', run: () => app.files.save(true)},
    {id: 'save-project', name: 'Download project', group: 'File', keywords: 'download vectrify reference locks pins', run: () => app.files.download(true)},
    {id: 'artboard', name: 'Artboard size…', group: 'File', keywords: 'page canvas document bounds dimensions width height resize viewbox origin', run: () => app.document.openArtboard()},
    {id: 'fit-artboard', name: 'Fit artboard to artwork', group: 'File', keywords: 'page canvas document bounds trim crop resize viewbox', run: () => app.document.fitArtboard()},
    {id: 'export', name: 'Export SVG', group: 'File', keywords: 'download save', run: () => app.files.download(false)},
    {id: 'agents', name: 'Allow agents to edit', label: () => editor.agent.status.enabled ? 'Stop agents editing' : 'Allow agents to edit', group: 'File', keywords: 'mcp claude ai assistant agent live', run: () => app.agent_controls.toggleAgents()},
    {id: 'help', name: 'Keyboard shortcuts', group: 'Help', keys: '?', keywords: 'help keys', run: () => $('help-dialog').showModal()},
    {id: 'command-palette', name: 'Command palette', group: 'Help', keys: 'Ctrl/⌘ K', hidden: () => true, run: () => app.commands.openPalette()},
    {id: 'undo', name: 'Undo', label: () => editor.state?.undo.length ? `Undo ${editor.state.undo.at(-1).toLowerCase()}` : 'Undo', group: 'Edit', keys: 'Ctrl/⌘ Z', run: () => app.actions.action('undo', {}, 'Undoing…'), disabled: () => !editor.state.undo.length && 'Nothing to undo'},
    {id: 'redo', name: 'Redo', label: () => editor.state?.redo.length ? `Redo ${editor.state.redo[0].toLowerCase()}` : 'Redo', group: 'Edit', keys: 'Ctrl/⌘ Shift Z', run: () => app.actions.action('redo', {}, 'Redoing…'), disabled: () => !editor.state.redo.length && 'Nothing to redo'},
    {id: 'copy', name: 'Copy objects', group: 'Object', keys: 'Ctrl/⌘ C', description: 'Copy the selected objects to paste them as independent copies',
      run: async () => { if (await app.actions.action('copy', {}, 'Copying objects…')) app.transport.toast('Objects copied. Ctrl/⌘ V pastes them.'); },
      disabled: () => app.commands.noSelection() || (editor.state.selection.objects.some(id => app.session.object(id)?.resource) && 'Copy drawing objects, not definitions')},
    {id: 'paste', name: 'Paste objects', group: 'Object', keys: 'Ctrl/⌘ V', description: 'Paste independent copies of the copied objects at the same position',
      run: () => app.actions.action('paste', {}, 'Pasting objects…'), disabled: () => !editor.state.clipboard && 'Copy objects first'},
    {id: 'rename', name: 'Rename…', group: 'Object', keys: 'F2', run: () => app.commands.renameObject(), disabled: () => !app.session.oneObject() && 'Select one object to rename'},
    {id: 'delete', name: 'Delete objects', group: 'Object', keys: 'Delete', keywords: 'remove erase', description: 'Delete the selected objects',
      danger: true, run: () => app.actions.action('delete'), disabled: () => app.commands.noSelection()},
    // Hiding is saved with the drawing (display="none"): the eye on each row
    // of the Objects tree does the same for its object.
    {id: 'hide-objects', name: 'Hide objects', group: 'Object', keywords: 'visibility visible invisible eye conceal display none',
      description: `Hide the selected objects; the Objects tree still selects them. ${EXPORTED}`,
      run: () => app.actions.action('visibility', {hidden: true}, 'Hiding…'), disabled: () => visibilityRefusal(editor.state.selection.objects, app.session.object, true)},
    {id: 'show-objects', name: 'Show objects', group: 'Object', keywords: 'visibility visible unhide reveal eye display',
      description: 'Show the selected hidden objects again', run: () => app.actions.action('visibility', {hidden: false}, 'Showing…'),
      disabled: () => visibilityRefusal(editor.state.selection.objects, app.session.object, false)},
    {id: 'show-all', name: 'Show all hidden objects', group: 'Object', keywords: 'visibility visible unhide reveal everything eye display',
      description: 'Show every hidden object in the drawing', run: () => app.actions.action('visibility', {all: true, hidden: false}, 'Showing…'),
      disabled: () => !editor.state.objects.some(item => item.hidden) && 'Nothing is hidden'},
    {id: 'fit', name: 'Fit the drawing', group: 'View', keys: 'F', keywords: 'zoom', run: () => app.view.fit()},
    {id: 'zoom-selection', name: 'Zoom to selection', group: 'View', keys: 'Z', run: () => app.view.focusSelection(), disabled: () => app.commands.noSelection()},
    {id: 'zoom-in', name: 'Zoom in', group: 'View', keys: 'Scroll', run: () => app.view.zoomAt(1.25)},
    {id: 'zoom-out', name: 'Zoom out', group: 'View', keys: 'Scroll', run: () => app.view.zoomAt(.8)},
    {id: 'finish-path', name: 'Finish path', group: 'Pen', keys: 'Enter', run: () => app.tools.finishPath(false), disabled: () => (editor.tool !== 'path' || editor.pathDraft.length < 2) && 'Draw two or more points with the pen first'},
    {id: 'close-path', name: 'Close shape', group: 'Pen', run: () => app.tools.finishPath(true), disabled: () => (editor.tool !== 'path' || editor.pathDraft.length < 3) && 'Draw three or more points with the pen first'},
    {id: 'select-all', name: 'Select all', label: () => app.geometry.level() === 'points' ? 'Select all points' : 'Select all', group: 'Select', keys: 'Ctrl/⌘ A', keywords: 'everything every objects points nodes',
      description: 'Select every shown object at this level, inside the entered group if any; in Nodes, every point of the selected paths', run: () => app.selection.selectAll(), disabled: () => app.selection.selectAllRefusal()},
    {id: 'enter', name: 'Enter', label: () => app.session.oneObject()?.tag === 'g' ? 'Enter group' : 'Edit points', group: 'Select', keys: 'Double-click', keywords: 'group points nodes into',
      run: () => app.canvas.enterObject(app.session.oneObject().id), disabled: () => (!(app.session.oneObject()?.tag === 'g' || (app.session.oneObject()?.tag === 'path' && !app.session.oneObject().resource)) && 'Select one group or visible path')
        || (app.session.oneObject() && hiddenRefusal([app.session.oneObject().id], app.session.object, 'edit'))},
    {id: 'step-up', name: 'Select one level up', group: 'Select', keys: 'Escape', keywords: 'leave exit group', run: () => app.canvas.stepUp(), disabled: () => !editor.state.selection.objects.length && !editor.scope && 'Nothing to step up from'},
    // Paint
    {id: 'fill', name: 'Edit fill', group: 'Paint', keywords: 'colour color', run: () => $('fill-value').focus(), disabled: () => app.commands.noSelection()},
    // The Eyedropper's choices: what it samples and what the sample sets.
    {id: 'sample-fill', name: 'Sample fill colour', group: 'Paint', keywords: 'eyedropper colour color picker dropper',
      description: 'Pick up the Eyedropper (I) to set the fill to a colour clicked in the artwork or the reference',
      run: () => app.eyedropper.choose({target: 'fill'}), disabled: () => app.eyedropper.targetRefusal('fill')},
    {id: 'sample-stroke', name: 'Sample stroke colour', group: 'Paint', keywords: 'eyedropper colour color picker dropper outline',
      description: 'Pick up the Eyedropper (I) to set the stroke to a colour clicked in the artwork or the reference',
      run: () => app.eyedropper.choose({target: 'stroke'}), disabled: () => app.eyedropper.targetRefusal('stroke')},
    {id: 'sample-artwork', name: 'Sample from the artwork', group: 'Paint', keywords: 'eyedropper colour color picker dropper drawing source',
      description: 'Sample the drawing as it shows, with the Eyedropper (I)', run: () => app.eyedropper.choose({source: 'artwork'})},
    {id: 'sample-reference', name: 'Sample from the reference', group: 'Paint', keywords: 'eyedropper colour color picker dropper image source',
      description: 'Sample the reference image, with the Eyedropper (I)', run: () => app.eyedropper.choose({source: 'reference'}), disabled: () => app.eyedropper.sourceRefusal('reference')},
    {id: 'create-gradient', name: 'Create linear gradient', group: 'Paint', keywords: 'fill colour color stops shading', run: () => app.gradients.createFillGradient(), disabled: () => app.gradients.gradientRefusal()},
    {id: 'stroke', name: 'Edit stroke', group: 'Paint', keywords: 'colour color outline', run: () => $('stroke-value').focus(), disabled: () => app.commands.noSelection()},
    // Transform
    {id: 'position-size', name: 'Position and size…', group: 'Transform', keywords: 'x y width height dimensions bounds coordinates numeric resize scale',
      description: 'Type where the selection frame is and how large', run: () => $('transform-x').select(), disabled: () => app.transform.fieldRefusal('x')},
    {id: 'move-by', name: 'Move by…', group: 'Transform', keywords: 'offset position translate', run: () => $('move-x').select(), disabled: () => app.commands.noSelection()},
    {id: 'rotate', name: 'Rotate…', group: 'Transform', keywords: 'rotation angle turn degrees spin pivot', description: 'Type the selection\'s angle; it turns about the pivot shown in Move',
      run: () => $('transform-angle').select(), disabled: () => app.transform.fieldRefusal('angle')},
    {id: 'flip-horizontal', name: 'Flip horizontal', group: 'Transform', keys: 'Shift H', keywords: 'mirror reflect left right', description: 'Mirror the selection left to right about the centre of its frame',
      run: () => app.rotating.flip('horizontal'), disabled: () => app.rotating.turnRefusal('flip')},
    {id: 'flip-vertical', name: 'Flip vertical', group: 'Transform', keys: 'Shift V', keywords: 'mirror reflect upside down top bottom', description: 'Mirror the selection top to bottom about the centre of its frame',
      run: () => app.rotating.flip('vertical'), disabled: () => app.rotating.turnRefusal('flip')},
    // Arrange: stacking and grouping.
    {id: 'to-back', name: 'Send to back', group: 'Arrange', keys: 'Ctrl/⌘ Shift [', keywords: 'stacking order', description: 'Move the selection behind everything else in its group',
      run: () => app.actions.action('reorder', {to: 'back'}), disabled: () => app.commands.noSelection() || (editor.state.selection.objects.some(id => app.session.object(id)?.resource) && 'Definitions and clipping boundaries keep their place')},
    {id: 'backward', name: 'Send backward', group: 'Arrange', keys: 'Ctrl/⌘ [', keywords: 'stacking order', description: 'Move the object one step back',
      run: () => app.actions.action('reorder', {step: -1}), disabled: () => !app.session.oneObject() && 'Select one object to restack'},
    {id: 'forward', name: 'Bring forward', group: 'Arrange', keys: 'Ctrl/⌘ ]', keywords: 'stacking order', description: 'Move the object one step forward',
      run: () => app.actions.action('reorder', {step: 1}), disabled: () => !app.session.oneObject() && 'Select one object to restack'},
    {id: 'to-front', name: 'Bring to front', group: 'Arrange', keys: 'Ctrl/⌘ Shift ]', keywords: 'stacking order', description: 'Move the selection in front of everything else in its group',
      run: () => app.actions.action('reorder', {to: 'front'}), disabled: () => app.commands.noSelection() || (editor.state.selection.objects.some(id => app.session.object(id)?.resource) && 'Definitions and clipping boundaries keep their place')},
    {id: 'group', name: 'Group', group: 'Arrange', keys: 'Ctrl/⌘ G', keywords: 'group objects', description: 'Gather the selected objects into a new group', run: () => app.actions.action('group'), disabled: () => app.commands.noSelection()},
    {id: 'ungroup', name: 'Ungroup', group: 'Arrange', keys: 'Ctrl/⌘ Shift G', description: 'Release the selected groups\' children, keeping their look', run: () => app.actions.action('ungroup'),
      disabled: () => app.commands.noSelection() || (!editor.state.selection.objects.every(id => app.session.object(id)?.tag === 'g') && 'Select one or more groups')},
    // Align lines the selection up to the reference chosen beside it: the
    // selection frame, the artboard or the key object. Distribute spaces the
    // objects between the outer two. Both move without resizing.
    ...['left', 'top', 'centre', 'middle', 'right', 'bottom'].map(alignment => ({
      id: `align-${alignment}`, name: ALIGNMENTS[alignment].name, group: 'Arrange',
      keywords: `align line up ${ALIGNMENTS[alignment].edge} selection artboard key object reference`,
      get description() { return app.align.alignDescription(alignment); }, run: () => app.align.align(alignment), disabled: () => app.align.alignRefusal(alignment)})),
    ...[['x', 'gaps'], ['y', 'gaps'], ['x', 'centres'], ['y', 'centres']].map(([axis, spacing]) => ({
      id: `distribute-${spacing}-${axis === 'x' ? 'horizontal' : 'vertical'}`, name: distributeName(axis, spacing), group: 'Arrange',
      keywords: 'distribute space spacing even evenly equal gaps centres centers',
      get description() { return app.align.distributeDescription(axis, spacing); }, run: () => app.align.distribute(axis, spacing), disabled: () => app.align.distributeRefusal(axis, spacing)})),
    // Path: operations on whole paths. Point operations are under Nodes.
    {id: 'merge-shapes', name: 'Merge shapes', label: () => app.join.joinOpensDialog() ? 'Merge shapes…' : 'Merge shapes', group: 'Path', keys: 'Ctrl/⌘ J',
      keywords: 'join union combine area outlines fills overlapping contours', description: 'Merge the selected paths by area into one shape, choosing its colours; a single filled path unions its overlapping contours',
      run: () => app.join.mergeShapes(), disabled: mergeRefusal},
    {id: 'join-line-ends', name: 'Join line ends', group: 'Path', keys: 'Ctrl/⌘ J', keywords: 'join connect dashed broken lines strokes gaps close',
      description: `Join the selected lines' ends that continue each other within ${JOIN_REACH} screen pixels; zoom out to reach wider gaps`,
      run: () => app.join.joinEnds(), disabled: () => !app.commands.linePaths().length && 'Select stroked lines without a fill, or groups that contain them'},
    {id: 'combine-paths', name: 'Combine paths', group: 'Path', keywords: 'compound collect contours preserve geometry combine',
      description: 'Collect the selected paths into one path without changing their contours, using the frontmost path\'s paint',
      run: () => app.actions.action('combine_paths', {}, 'Combining paths…'), disabled: () => app.commands.joinCandidates().length < 2 && 'Select two or more paths, or groups containing them'},
    {id: 'stroke-to-path', name: 'Stroke to path', group: 'Path', keywords: 'line to fill outline expand convert',
      description: 'Turn each selected stroked line into a filled shape covering what its stroke paints',
      run: () => app.actions.action('line_to_fill', {}, 'Converting strokes…'), disabled: () => !app.commands.linePaths().length && 'Select stroked lines without a fill'},
    {id: 'fill-to-centerline', name: 'Fill to centerline', group: 'Path', keywords: 'fill to line centreline skeleton thin stroke convert',
      description: 'Turn each selected thin filled shape into a stroked line down its middle',
      run: () => app.actions.action('fill_to_line', {}, 'Converting fills…'), disabled: () => !app.commands.fillPaths().length && 'Select thin filled shapes'},
    {id: 'split-parts', name: 'Split parts', group: 'Path', keywords: 'separate disconnected break apart', description: 'Separate disconnected contours while keeping holes together', run: () => app.commands.splitParts(),
      disabled: () => app.commands.noSelection() || (app.session.oneObject()?.tag === 'use' ? 'Detach this instance to an editable path first' : !app.commands.visiblePaths() && 'Only visible paths can be split')},
    {id: 'cut-hole', name: 'Cut out as hole', group: 'Path', keywords: 'subtract difference punch', description: 'Cut the inner (or front) of two selected paths out of the other as a hole, keeping the other\'s paint',
      run: () => app.commands.cutHole(), disabled: () => (editor.state.selection.objects.length !== 2 || !app.commands.visiblePaths()) && 'Select two visible paths, one inside or overlapping the other'},
    // Booleans: the backmost selected path is the target and keeps its paint;
    // the description names it and the cutters before anything changes.
    ...Object.entries(BOOLEANS).map(([operation, {name, keywords}]) => ({
      id: operation, name, group: 'Path', keywords,
      get description() { return app.booleans.description(operation); }, run: () => app.booleans.run(operation), disabled: () => app.booleans.refusal()})),
    {id: 'convert-to-path', name: 'Convert to path', group: 'Path', keywords: 'rectangle rect circle ellipse line basic shape editable nodes points',
      description: 'Turn the selected rectangles, circles, ellipses and lines into paths with points to edit, keeping their look',
      run: () => app.actions.action('convert_to_path', {}, 'Converting to paths…'), disabled: convertRefusal},
    // The part of the paths inside a region drawn with the Region tool, which
    // shows what changes before anything does.
    {id: 'extract-region', name: 'Extract region', group: 'Path', keywords: 'region area rectangle polygon cut take separate piece crop lasso',
      description: 'Take what lies inside a region drawn with the Region tool (X) into paths of their own, cutting objects that cross its edge if chosen',
      run: () => app.region.command('extract'), disabled: () => app.region.refusal('extract')},
    {id: 'delete-region', name: 'Delete region', group: 'Path', keywords: 'region area rectangle polygon erase remove cut lasso', danger: true,
      description: 'Delete what lies inside a region drawn with the Region tool (X), cutting objects that cross its edge if chosen',
      run: () => app.region.command('delete'), disabled: () => app.region.refusal('delete')},
    {id: 'detach', name: 'Detach', label: () => app.session.oneObject()?.tag === 'use' ? 'Detach to editable path' : 'Detach shared geometry', group: 'Path', keywords: 'instance use shared unlink',
      description: 'Give this object its own copy of geometry it shares with others', run: () => app.actions.action('detach'),
      disabled: () => !(app.session.oneObject()?.tag === 'use' || app.session.oneObject()?.shared)
        && (SHAPES.has(app.session.oneObject()?.tag) ? 'A basic shape shares nothing: use Convert to path' : 'Select one instance, or a path that shares its geometry')},
    // Clipping: the boundary is a closed path; what lies outside it is hidden.
    {id: 'set-clip', name: 'Set clip…', group: 'Clipping', keywords: 'clip clipping mask crop boundary hide outside path',
      description: 'Clip the selected objects to a selected closed path, choosing which is clipped and which is the boundary', run: () => app.clipping.openSetClip(), disabled: () => app.clipping.setRefusal()},
    {id: 'remove-clip', name: 'Remove clip', group: 'Clipping', keywords: 'unclip clipping mask crop release',
      description: 'Stop clipping the selected objects; their boundary paths stay in the drawing', run: () => app.clipping.removeClip(), disabled: () => app.clipping.removeRefusal()},
    {id: 'show-clips', name: 'Show clipping boundaries', label: () => editor.showClipping ? 'Hide clipping boundaries' : 'Show clipping boundaries', group: 'Clipping', keywords: 'clip contour outline mask view',
      description: 'Show clipping boundaries as selectable outlines', run: () => app.overlay.toggleClipping()},
    // Repair
    {id: 'tidy', name: 'Tidy…', group: 'Repair', keywords: 'simplify snap fit shape optimize nodes points reference', description: 'Tidy the selected paths, or what is in view: fit them to the reference, or simplify them without one',
      run: () => app.dialogs.openTidy(), disabled: () => !app.dialogs.tidyTargets() && !editor.state.reference && 'Select one or more paths, or groups that contain them, or add a reference to tidy what is in view'},
    {id: 'cleanup', name: 'Clean up…', group: 'Repair', keywords: 'duplicate vertices merge tidy redundant', description: 'Remove redundant points and duplicate paths, and merge compatible neighbours', run: () => app.dialogs.openCleanup(), disabled: () => app.commands.noSelection()},
    {id: 'snap-edges', name: 'Snap edges…', group: 'Repair', keywords: 'gaps touching shared seams', description: 'Snap the touching edges of the selected paths together', run: () => app.dialogs.openSnapEdges(),
      disabled: () => editor.state.selection.objects.length < 2 ? 'Select two or more paths' : !app.commands.visiblePaths() && 'Every object must be a visible path'},
    // Reference
    {id: 'load-reference', name: 'Load reference…', label: () => editor.state?.reference ? 'Replace reference…' : 'Load reference…', group: 'Reference', run: () => $('reference-file').click()},
    {id: 'remove-reference', name: 'Remove reference', group: 'Reference', run: () => app.reference.removeReference(), disabled: () => app.commands.noReference()},
    {id: 'toggle-overlay', name: 'Cycle the view: drawing, overlay, reference only, error', group: 'Reference', keys: 'W', run: () => app.reference.cycleReference(), disabled: () => app.commands.noReference()},
    {id: 'cycle-view-backward', name: 'Cycle the view backward', group: 'Reference', keys: 'Shift W', run: () => app.reference.cycleReference(-1), disabled: () => app.commands.noReference()},
    {id: 'drawing-only', name: 'Show the drawing only', group: 'Reference', run: () => app.reference.setView('drawing'), disabled: () => app.commands.noReference()},
    {id: 'generate', name: 'Generate from reference…', group: 'Reference', run: () => app.dialogs.openGenerate(), disabled: () => app.commands.noReference()},
    {id: 'fit-colours', name: 'Fit colours…', group: 'Reference', run: () => app.dialogs.openColours('flat'), disabled: () => app.commands.noReference() || app.commands.noSelection()},
    {id: 'fit-gradient', name: 'Fit gradient…', group: 'Reference', keywords: 'gradient ramp shading linear fill colour color', run: () => app.dialogs.openColours('linear'), disabled: () => app.commands.noReference() || app.commands.noSelection()},
    // Nodes: operations on the selected points.
    {id: 'handles-0', name: 'No handles', group: 'Nodes', keys: '1', keywords: 'corner sharp', run: () => app.commands.pointHandles(0), disabled: () => app.commands.noPoints()},
    {id: 'handles-1', name: 'One handle', group: 'Nodes', keys: '2', run: () => app.commands.pointHandles(1), disabled: () => app.commands.noPoints()},
    {id: 'handles-2', name: 'Two handles', group: 'Nodes', keys: '3', keywords: 'smooth', run: () => app.commands.pointHandles(2), disabled: () => app.commands.noPoints()},
    {id: 'straighten-handles', name: 'Straighten handles', group: 'Nodes', keys: '7', keywords: 'align smooth tangent line', run: () => app.selection.straightenHandles(),
      description: 'Toggle aligned handles: keep both in line while dragging, preserving their lengths', disabled: () => app.commands.noPoints()},
    {id: 'pin', name: 'Pin points', label: () => app.geometry.selectedPoints().length && app.geometry.selectedPoints().every(key => app.geometry.nodeAt(key)?.pinned) ? 'Unpin points' : 'Pin points', group: 'Nodes',
      run: () => app.actions.action('pin', {points: app.selection.pointPairs(), pinned: !app.geometry.selectedPoints().every(key => app.geometry.nodeAt(key)?.pinned)}), disabled: () => app.commands.noPoints()},
    // Break cuts at the points, or with the two ends of a segment picked,
    // deletes that segment.
    {id: 'break-points', name: 'Break', group: 'Nodes', keys: '4', keywords: 'cut split disconnect open loop delete segment remove edge gap',
      description: 'Cut the line at the selected points, each piece ending on its own copy of the point; with the two points at a segment\'s ends, delete that segment instead', run: () => app.commands.breakPoints(),
      disabled: () => app.commands.noPoints() || (!app.commands.segmentPicked() && !app.geometry.selectedPoints().some(key => breakable(app.geometry.contourAt(key), splitKey(key)[1]))
        && 'Pick a point between a line\'s ends or on a closed contour, or the two points at a segment\'s ends')},
    {id: 'join-points', name: 'Join points', group: 'Nodes', keys: '5', keywords: 'join connect ends close line',
      description: 'Join the two selected points into one line, however far apart', run: () => app.actions.action('join_two_ends', {points: app.selection.pointPairs()}, 'Joining points…'),
      disabled: () => app.commands.noPoints() || (!app.commands.twoEnds() && 'Select exactly two points to join')},
    {id: 'split-edge', name: 'Add node', group: 'Nodes', keys: '6', keywords: 'split edge insert point vertex',
      description: 'Add a node halfway along the edge leading into each selected node, preserving the curve', run: () => app.actions.action('split', {points: app.selection.pointPairs()}),
      disabled: () => app.commands.noPoints() || (app.geometry.selectedPoints().some(key => app.geometry.contourAt(key)?.nodes.length < 2) && 'This contour has no edge to add a node on')},
    {id: 'delete-points', name: 'Delete points', label: () => app.selection.selectedHandle() ? 'Delete handle' : 'Delete points', group: 'Nodes', keys: 'Delete', keywords: 'remove erase handle',
      description: 'Delete the selected handle or points, reconnecting the line past them', danger: true, run: () => app.commands.deleteSelection(),
      disabled: () => app.geometry.level() !== 'points' ? 'Select points in Nodes (S) first' : app.selection.selectedHandle() ? '' : app.commands.noPoints() || (app.geometry.selectedPoints().some(key => app.geometry.nodeAt(key)?.pinned) && 'Unpin the points to delete them')},
    {id: 'delete-contour', name: 'Delete contour', group: 'Nodes', keys: 'Shift Delete', description: 'Delete the contours the selected points are on', run: () => app.actions.action('delete_contour', {points: app.selection.pointPairs()}, 'Deleting contours…'),
      disabled: () => app.commands.noPoints() || (app.geometry.selectedPoints().some(key => app.geometry.contourAt(key)?.nodes.some(n => n.pinned)) && 'Unpin the contour\'s points to delete it')},
    {id: 'fill-hole', name: 'Fill hole', group: 'Nodes', keywords: 'holes remove', description: 'Remove the holes the points are on, in however many paths, so each path fills their area',
      run: () => app.selection.fillPointHoles(), disabled: () => app.commands.noPoints() || (!app.selection.holeContours(app.geometry.selectedPoints()) && 'Select points on holes')},
    {id: 'hole-to-shape', name: 'Hole to shape', group: 'Nodes', keywords: 'holes shapes extract', description: 'Move the holes the points are on out into new shapes, each with its path\'s paint, just above it',
      run: () => app.selection.pointHolesToShapes(), disabled: () => app.commands.noPoints() || (!app.selection.holeContours(app.geometry.selectedPoints()) && 'Select points on holes')},
    // Ctrl/⌘ J runs whichever join fits the selection: the two selected points
    // in a point tool, else merging shapes or joining line ends.
    {id: 'join', name: 'Join', label: () => commandName(joinTarget()), group: 'Path', keys: 'Ctrl/⌘ J', hidden: () => true,
      run: () => joinTarget().run(), disabled: () => app.geometry.level() === 'points' ? commandRefusal(joinTarget())
        : commandRefusal(joinTarget()) && 'Select paths to merge, or lines whose ends to join'},
  ];

  const commandById = new Map(COMMANDS.map(command => [command.id, command]));

  const commandName = command => command.label?.() || command.name;

  function commandRefusal(command) {
    if (!editor.state) return 'The drawing is still opening';
    return command.disabled?.() || '';
  }

  // A control's tooltip: what the command does, or why it cannot run now.
  function commandTitle(command) {
    const reason = commandRefusal(command);
    if (reason) return `${commandName(command)}: ${reason}`;
    return `${command.description || commandName(command)}${command.keys ? ` (${command.keys})` : ''}`;
  }

  // Merging needs paths, not only lines: a single filled path unions its own
  // contours, and two or more merge by area.
  function mergeRefusal() {
    const candidates = app.commands.joinCandidates().length, fills = app.commands.fillPaths().length, lines = app.commands.linePaths().length;
    if (candidates === 1 && fills === 1) return '';
    if (candidates > 1 && lines < candidates) return '';
    if (candidates > 1) return 'These are lines without a fill: use Join line ends';
    return 'Select two or more paths, or groups that contain them, or one filled path';
  }

  // Convert to path is for basic shapes only; instances and shared geometry
  // are Detach's, which keeps the two edits apart.
  function convertRefusal() {
    const selected = editor.state.selection.objects.map(app.session.object);
    if (!selected.length) return app.commands.noSelection();
    if (selected.every(item => SHAPES.has(item?.tag))) return '';
    if (selected.some(item => item?.tag === 'use' || item?.shared)) return 'Instances and shared geometry are detached: use Detach';
    return 'Select rectangles, circles, ellipses or lines';
  }

  function joinTarget() {
    if (app.geometry.level() === 'points') return commandById.get('join-points');
    return commandById.get(!mergeRefusal() || commandRefusal(commandById.get('join-line-ends')) ? 'merge-shapes' : 'join-line-ends');
  }

  function wire() {
    document.addEventListener('pointerdown', event => { if (!$('agent-popover').hidden && !event.target.closest('#agent-popover, #agent-toggle')) app.agent_controls.closeAgentPopover(); });
      return api;
  }
  const api = {COMMANDS, commandById, commandName, commandRefusal, commandTitle, joinTarget, wire};
  return api;
}
