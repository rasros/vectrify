import {splitKey} from './selection.js';
import {breakable} from './lines.js';
import {JOIN_REACH} from './app_config.js';

// Command_registry owns its controls; editor contains the live state shared with other controllers.
export function createCommandRegistryController(editor, {
  $,
  action,
  breakPoints,
  closeAgentPopover,
  contourAt,
  createFillGradient,
  cutHole,
  cycleReference,
  deleteSelection,
  document = globalThis.document,
  download,
  enterObject,
  files,
  fillPaths,
  fillPointHoles,
  finishPath,
  fit,
  focusSelection,
  gradientRefusal,
  holeContours,
  joinCandidates,
  joinEnds,
  joinOpensDialog,
  level,
  linePaths,
  mergeShapes,
  noPoints,
  noReference,
  noSelection,
  nodeAt,
  object,
  oneObject,
  openCleanup,
  openColours,
  openGenerate,
  openPalette,
  openSnapEdges,
  openTidy,
  pointHandles,
  pointHolesToShapes,
  pointPairs,
  removeReference,
  renameObject,
  segmentPicked,
  selectedHandle,
  selectedPoints,
  setTool,
  setView,
  splitParts,
  stepUp,
  straightenHandles,
  tidyTargets,
  toast,
  toggleAgents,
  toggleClipping,
  twoEnds,
  visiblePaths,
  zoomAt,
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
    {id: 'tool-select', name: 'Select tool', group: 'Tools', keys: 'A', keywords: 'pick arrow objects box', run: () => setTool('select')},
    {id: 'tool-move', name: 'Move tool', group: 'Tools', keys: 'V', keywords: 'translate position line objects', run: () => setTool('move')},
    {id: 'tool-nodes', name: 'Nodes tool', group: 'Tools', keys: 'S', keywords: 'edit points handles', run: () => setTool('nodes')},
    {id: 'tool-path', name: 'Draw path tool', group: 'Tools', keys: 'D', keywords: 'pen draw path shape', run: () => setTool('path')},
    {id: 'tool-knife', name: 'Knife tool', group: 'Tools', keys: 'C', keywords: 'cut slice', run: () => setTool('knife')},
    {id: 'tool-redraw', name: 'Redraw outline tool', group: 'Tools', keys: 'R', keywords: 'lasso outline fix', run: () => setTool('redraw')},
    {id: 'tool-hand', name: 'Pan tool', group: 'Tools', keys: 'Hold Space', keywords: 'hand scroll', run: () => setTool('hand')},
    {id: 'open', name: 'Open…', group: 'File', keywords: 'svg project load', run: () => $('open-file').click()},
    {id: 'restore', name: 'Restore saved…', group: 'File', keywords: 'recovery browser copy', run: () => $('restore-saved').click()},
    {id: 'save', name: 'Save', group: 'File', keys: 'Ctrl/⌘ S', keywords: 'file write', run: () => files.save()},
    {id: 'save-as', name: 'Save As…', group: 'File', keys: 'Ctrl/⌘ Shift S', keywords: 'file destination menu', run: () => files.save(true)},
    {id: 'save-project', name: 'Download project', group: 'File', keywords: 'download vectrify reference locks pins', run: () => download(true)},
    {id: 'export', name: 'Export SVG', group: 'File', keywords: 'download save', run: () => download(false)},
    {id: 'agents', name: 'Allow agents to edit', label: () => editor.agentStatus.enabled ? 'Stop agents editing' : 'Allow agents to edit', group: 'File', keywords: 'mcp claude ai assistant agent live', run: toggleAgents},
    {id: 'help', name: 'Keyboard shortcuts', group: 'Help', keys: '?', keywords: 'help keys', run: () => $('help-dialog').showModal()},
    {id: 'command-palette', name: 'Command palette', group: 'Help', keys: 'Ctrl/⌘ K', hidden: () => true, run: openPalette},
    {id: 'undo', name: 'Undo', label: () => editor.state?.undo.length ? `Undo ${editor.state.undo.at(-1).toLowerCase()}` : 'Undo', group: 'Edit', keys: 'Ctrl/⌘ Z', run: () => action('undo', {}, 'Undoing…'), disabled: () => !editor.state.undo.length && 'Nothing to undo'},
    {id: 'redo', name: 'Redo', label: () => editor.state?.redo.length ? `Redo ${editor.state.redo[0].toLowerCase()}` : 'Redo', group: 'Edit', keys: 'Ctrl/⌘ Shift Z', run: () => action('redo', {}, 'Redoing…'), disabled: () => !editor.state.redo.length && 'Nothing to redo'},
    {id: 'copy', name: 'Copy objects', group: 'Object', keys: 'Ctrl/⌘ C', description: 'Copy the selected objects to paste them as independent copies',
      run: async () => { if (await action('copy', {}, 'Copying objects…')) toast('Objects copied. Ctrl/⌘ V pastes them.'); },
      disabled: () => noSelection() || (editor.state.selection.objects.some(id => object(id)?.resource) && 'Copy drawing objects, not definitions')},
    {id: 'paste', name: 'Paste objects', group: 'Object', keys: 'Ctrl/⌘ V', description: 'Paste independent copies of the copied objects at the same position',
      run: () => action('paste', {}, 'Pasting objects…'), disabled: () => !editor.state.clipboard && 'Copy objects first'},
    {id: 'rename', name: 'Rename…', group: 'Object', keys: 'F2', run: renameObject, disabled: () => !oneObject() && 'Select one object to rename'},
    {id: 'delete', name: 'Delete objects', group: 'Object', keys: 'Delete', keywords: 'remove erase', description: 'Delete the selected objects',
      danger: true, run: () => action('delete'), disabled: noSelection},
    {id: 'fit', name: 'Fit the drawing', group: 'View', keys: 'F', keywords: 'zoom', run: fit},
    {id: 'zoom-selection', name: 'Zoom to selection', group: 'View', keys: 'Z', run: focusSelection, disabled: noSelection},
    {id: 'zoom-in', name: 'Zoom in', group: 'View', keys: 'Scroll', run: () => zoomAt(1.25)},
    {id: 'zoom-out', name: 'Zoom out', group: 'View', keys: 'Scroll', run: () => zoomAt(.8)},
    {id: 'finish-path', name: 'Finish path', group: 'Pen', keys: 'Enter', run: () => finishPath(false), disabled: () => (editor.tool !== 'path' || editor.pathDraft.length < 2) && 'Draw two or more points with the pen first'},
    {id: 'close-path', name: 'Close shape', group: 'Pen', run: () => finishPath(true), disabled: () => (editor.tool !== 'path' || editor.pathDraft.length < 3) && 'Draw three or more points with the pen first'},
    {id: 'enter', name: 'Enter', label: () => oneObject()?.tag === 'g' ? 'Enter group' : 'Edit points', group: 'Select', keys: 'Double-click', keywords: 'group points nodes into',
      run: () => enterObject(oneObject().id), disabled: () => !(oneObject()?.tag === 'g' || (oneObject()?.tag === 'path' && !oneObject().resource)) && 'Select one group or visible path'},
    {id: 'step-up', name: 'Select one level up', group: 'Select', keys: 'Escape', keywords: 'leave exit group', run: stepUp, disabled: () => !editor.state.selection.objects.length && !editor.scope && 'Nothing to step up from'},
    // Paint
    {id: 'fill', name: 'Edit fill', group: 'Paint', keywords: 'colour color', run: () => $('fill-value').focus(), disabled: noSelection},
    {id: 'create-gradient', name: 'Create linear gradient', group: 'Paint', keywords: 'fill colour color stops shading', run: createFillGradient, disabled: gradientRefusal},
    {id: 'stroke', name: 'Edit stroke', group: 'Paint', keywords: 'colour color outline', run: () => $('stroke-value').focus(), disabled: noSelection},
    // Transform
    {id: 'move-by', name: 'Move by…', group: 'Transform', keywords: 'offset position translate', run: () => $('move-x').select(), disabled: noSelection},
    // Arrange: stacking and grouping.
    {id: 'to-back', name: 'Send to back', group: 'Arrange', keys: 'Ctrl/⌘ Shift [', keywords: 'stacking order', description: 'Move the selection behind everything else in its group',
      run: () => action('reorder', {to: 'back'}), disabled: () => noSelection() || (editor.state.selection.objects.some(id => object(id)?.resource) && 'Definitions and clipping boundaries keep their place')},
    {id: 'backward', name: 'Send backward', group: 'Arrange', keys: 'Ctrl/⌘ [', keywords: 'stacking order', description: 'Move the object one step back',
      run: () => action('reorder', {step: -1}), disabled: () => !oneObject() && 'Select one object to restack'},
    {id: 'forward', name: 'Bring forward', group: 'Arrange', keys: 'Ctrl/⌘ ]', keywords: 'stacking order', description: 'Move the object one step forward',
      run: () => action('reorder', {step: 1}), disabled: () => !oneObject() && 'Select one object to restack'},
    {id: 'to-front', name: 'Bring to front', group: 'Arrange', keys: 'Ctrl/⌘ Shift ]', keywords: 'stacking order', description: 'Move the selection in front of everything else in its group',
      run: () => action('reorder', {to: 'front'}), disabled: () => noSelection() || (editor.state.selection.objects.some(id => object(id)?.resource) && 'Definitions and clipping boundaries keep their place')},
    {id: 'group', name: 'Group', group: 'Arrange', keys: 'Ctrl/⌘ G', keywords: 'group objects', description: 'Gather the selected objects into a new group', run: () => action('group'), disabled: noSelection},
    {id: 'ungroup', name: 'Ungroup', group: 'Arrange', keys: 'Ctrl/⌘ Shift G', description: 'Release the selected groups\' children, keeping their look', run: () => action('ungroup'),
      disabled: () => noSelection() || (!editor.state.selection.objects.every(id => object(id)?.tag === 'g') && 'Select one or more groups')},
    // Path: operations on whole paths. Point operations are under Nodes.
    {id: 'merge-shapes', name: 'Merge shapes', label: () => joinOpensDialog() ? 'Merge shapes…' : 'Merge shapes', group: 'Path', keys: 'Ctrl/⌘ J',
      keywords: 'join union combine area outlines fills overlapping contours', description: 'Merge the selected paths by area into one shape, choosing its colours; a single filled path unions its overlapping contours',
      run: mergeShapes, disabled: mergeRefusal},
    {id: 'join-line-ends', name: 'Join line ends', group: 'Path', keys: 'Ctrl/⌘ J', keywords: 'join connect dashed broken lines strokes gaps close',
      description: `Join the selected lines' ends that continue each other within ${JOIN_REACH} screen pixels; zoom out to reach wider gaps`,
      run: joinEnds, disabled: () => !linePaths().length && 'Select stroked lines without a fill, or groups that contain them'},
    {id: 'combine-paths', name: 'Combine paths', group: 'Path', keywords: 'compound collect contours preserve geometry combine',
      description: 'Collect the selected paths into one path without changing their contours, using the frontmost path\'s paint',
      run: () => action('combine_paths', {}, 'Combining paths…'), disabled: () => joinCandidates().length < 2 && 'Select two or more paths, or groups containing them'},
    {id: 'stroke-to-path', name: 'Stroke to path', group: 'Path', keywords: 'line to fill outline expand convert',
      description: 'Turn each selected stroked line into a filled shape covering what its stroke paints',
      run: () => action('line_to_fill', {}, 'Converting strokes…'), disabled: () => !linePaths().length && 'Select stroked lines without a fill'},
    {id: 'fill-to-centerline', name: 'Fill to centerline', group: 'Path', keywords: 'fill to line centreline skeleton thin stroke convert',
      description: 'Turn each selected thin filled shape into a stroked line down its middle',
      run: () => action('fill_to_line', {}, 'Converting fills…'), disabled: () => !fillPaths().length && 'Select thin filled shapes'},
    {id: 'split-parts', name: 'Split parts', group: 'Path', keywords: 'separate disconnected break apart', description: 'Separate disconnected contours while keeping holes together', run: splitParts,
      disabled: () => noSelection() || (oneObject()?.tag === 'use' ? 'Detach this instance to an editable path first' : !visiblePaths() && 'Only visible paths can be split')},
    {id: 'cut-hole', name: 'Cut out as hole', group: 'Path', keywords: 'subtract difference punch', description: 'Cut the inner (or front) of two selected paths out of the other as a hole, keeping the other\'s paint',
      run: cutHole, disabled: () => (editor.state.selection.objects.length !== 2 || !visiblePaths()) && 'Select two visible paths, one inside or overlapping the other'},
    {id: 'detach', name: 'Detach', label: () => oneObject()?.tag === 'use' ? 'Detach to editable path' : 'Detach shared geometry', group: 'Path', keywords: 'instance use shared unlink',
      description: 'Give this object its own copy of geometry it shares with others', run: () => action('detach'),
      disabled: () => !(oneObject()?.tag === 'use' || oneObject()?.shared) && 'Select one instance, or a path that shares its geometry'},
    // Clipping
    {id: 'show-clips', name: 'Show clipping boundaries', label: () => editor.showClipping ? 'Hide clipping boundaries' : 'Show clipping boundaries', group: 'Clipping', keywords: 'clip contour outline mask view',
      description: 'Show clipping boundaries as selectable outlines', run: toggleClipping},
    // Repair
    {id: 'tidy', name: 'Tidy…', group: 'Repair', keywords: 'simplify snap fit shape optimize nodes points reference', description: 'Tidy the selected paths, or what is in view: fit them to the reference, or simplify them without one',
      run: openTidy, disabled: () => !tidyTargets() && !editor.state.reference && 'Select one or more paths, or groups that contain them, or add a reference to tidy what is in view'},
    {id: 'cleanup', name: 'Clean up…', group: 'Repair', keywords: 'duplicate vertices merge tidy redundant', description: 'Remove redundant points and duplicate paths, and merge compatible neighbours', run: openCleanup, disabled: noSelection},
    {id: 'snap-edges', name: 'Snap edges…', group: 'Repair', keywords: 'gaps touching shared seams', description: 'Snap the touching edges of the selected paths together', run: openSnapEdges,
      disabled: () => editor.state.selection.objects.length < 2 ? 'Select two or more paths' : !visiblePaths() && 'Every object must be a visible path'},
    // Reference
    {id: 'load-reference', name: 'Load reference…', label: () => editor.state?.reference ? 'Replace reference…' : 'Load reference…', group: 'Reference', run: () => $('reference-file').click()},
    {id: 'remove-reference', name: 'Remove reference', group: 'Reference', run: removeReference, disabled: noReference},
    {id: 'toggle-overlay', name: 'Cycle the view: drawing, overlay, reference only, error', group: 'Reference', keys: 'W', run: () => cycleReference(), disabled: noReference},
    {id: 'cycle-view-backward', name: 'Cycle the view backward', group: 'Reference', keys: 'Shift W', run: () => cycleReference(-1), disabled: noReference},
    {id: 'drawing-only', name: 'Show the drawing only', group: 'Reference', run: () => setView('drawing'), disabled: noReference},
    {id: 'generate', name: 'Generate from reference…', group: 'Reference', run: openGenerate, disabled: noReference},
    {id: 'fit-colours', name: 'Fit colours…', group: 'Reference', run: () => openColours('flat'), disabled: () => noReference() || noSelection()},
    {id: 'fit-gradient', name: 'Fit gradient…', group: 'Reference', keywords: 'gradient ramp shading linear fill colour color', run: () => openColours('linear'), disabled: () => noReference() || noSelection()},
    // Nodes: operations on the selected points.
    {id: 'handles-0', name: 'No handles', group: 'Nodes', keys: '1', keywords: 'corner sharp', run: () => pointHandles(0), disabled: noPoints},
    {id: 'handles-1', name: 'One handle', group: 'Nodes', keys: '2', run: () => pointHandles(1), disabled: noPoints},
    {id: 'handles-2', name: 'Two handles', group: 'Nodes', keys: '3', keywords: 'smooth', run: () => pointHandles(2), disabled: noPoints},
    {id: 'straighten-handles', name: 'Straighten handles', group: 'Nodes', keys: '7', keywords: 'align smooth tangent line', run: straightenHandles,
      description: 'Toggle aligned handles: keep both in line while dragging, preserving their lengths', disabled: noPoints},
    {id: 'pin', name: 'Pin points', label: () => selectedPoints().length && selectedPoints().every(key => nodeAt(key)?.pinned) ? 'Unpin points' : 'Pin points', group: 'Nodes',
      run: () => action('pin', {points: pointPairs(), pinned: !selectedPoints().every(key => nodeAt(key)?.pinned)}), disabled: noPoints},
    // Break cuts at the points, or with the two ends of a segment picked,
    // deletes that segment.
    {id: 'break-points', name: 'Break', group: 'Nodes', keys: '4', keywords: 'cut split disconnect open loop delete segment remove edge gap',
      description: 'Cut the line at the selected points, each piece ending on its own copy of the point; with the two points at a segment\'s ends, delete that segment instead', run: breakPoints,
      disabled: () => noPoints() || (!segmentPicked() && !selectedPoints().some(key => breakable(contourAt(key), splitKey(key)[1]))
        && 'Pick a point between a line\'s ends or on a closed contour, or the two points at a segment\'s ends')},
    {id: 'join-points', name: 'Join points', group: 'Nodes', keys: '5', keywords: 'join connect ends close line',
      description: 'Join the two selected points into one line, however far apart', run: () => action('join_two_ends', {points: pointPairs()}, 'Joining points…'),
      disabled: () => noPoints() || (!twoEnds() && 'Select exactly two points to join')},
    {id: 'split-edge', name: 'Add node', group: 'Nodes', keys: '6', keywords: 'split edge insert point vertex',
      description: 'Add a node halfway along the edge leading into each selected node, preserving the curve', run: () => action('split', {points: pointPairs()}),
      disabled: () => noPoints() || (selectedPoints().some(key => contourAt(key)?.nodes.length < 2) && 'This contour has no edge to add a node on')},
    {id: 'delete-points', name: 'Delete points', label: () => selectedHandle() ? 'Delete handle' : 'Delete points', group: 'Nodes', keys: 'Delete', keywords: 'remove erase handle',
      description: 'Delete the selected handle or points, reconnecting the line past them', danger: true, run: deleteSelection,
      disabled: () => level() !== 'points' ? 'Select points in Nodes (S) first' : selectedHandle() ? '' : noPoints() || (selectedPoints().some(key => nodeAt(key)?.pinned) && 'Unpin the points to delete them')},
    {id: 'delete-contour', name: 'Delete contour', group: 'Nodes', keys: 'Shift Delete', description: 'Delete the contours the selected points are on', run: () => action('delete_contour', {points: pointPairs()}, 'Deleting contours…'),
      disabled: () => noPoints() || (selectedPoints().some(key => contourAt(key)?.nodes.some(n => n.pinned)) && 'Unpin the contour\'s points to delete it')},
    {id: 'fill-hole', name: 'Fill hole', group: 'Nodes', keywords: 'holes remove', description: 'Remove the holes the points are on, in however many paths, so each path fills their area',
      run: fillPointHoles, disabled: () => noPoints() || (!holeContours(selectedPoints()) && 'Select points on holes')},
    {id: 'hole-to-shape', name: 'Hole to shape', group: 'Nodes', keywords: 'holes shapes extract', description: 'Move the holes the points are on out into new shapes, each with its path\'s paint, just above it',
      run: pointHolesToShapes, disabled: () => noPoints() || (!holeContours(selectedPoints()) && 'Select points on holes')},
    // Ctrl/⌘ J runs whichever join fits the selection: the two selected points
    // in a point tool, else merging shapes or joining line ends.
    {id: 'join', name: 'Join', label: () => commandName(joinTarget()), group: 'Path', keys: 'Ctrl/⌘ J', hidden: () => true,
      run: () => joinTarget().run(), disabled: () => level() === 'points' ? commandRefusal(joinTarget())
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
    const candidates = joinCandidates().length, fills = fillPaths().length, lines = linePaths().length;
    if (candidates === 1 && fills === 1) return '';
    if (candidates > 1 && lines < candidates) return '';
    if (candidates > 1) return 'These are lines without a fill: use Join line ends';
    return 'Select two or more paths, or groups that contain them, or one filled path';
  }

  function joinTarget() {
    if (level() === 'points') return commandById.get('join-points');
    return commandById.get(!mergeRefusal() || commandRefusal(commandById.get('join-line-ends')) ? 'merge-shapes' : 'join-line-ends');
  }

  function wire() {
    document.addEventListener('pointerdown', event => { if (!$('agent-popover').hidden && !event.target.closest('#agent-popover, #agent-toggle')) closeAgentPopover(); });
      return api;
  }
  const api = {COMMANDS, commandById, commandName, commandRefusal, commandTitle, joinTarget, wire};
  return api;
}
