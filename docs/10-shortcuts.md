# Shortcuts and limits

The keyboard shortcuts, the clipboard, and what the editor does not
support. Press ? in the editor for the shortcuts list.

## Tools

Tool keys switch at once, even while an edit runs.

- A Select, V Move, S Nodes, R Redraw outline, I Eyedropper
- D Draw path, M Rectangle, E Ellipse (Shift for a circle), L Line, P Pencil
- C Knife, X Region, G Gradient
- Space (hold) Pan, or drag with the middle mouse button in any tool

## View

- F fit the drawing, Z zoom to the selection, scroll wheel to zoom
- W cycle the view forward (drawing, overlay, reference, error), Shift W
  backward

## Tool strip digits

The digits 1-9 press the active tool's numbered controls, also when they
are in the strip's ⋯ menu.

- Nodes: 1-3 handles (none, one, two), 4 Break, 5 Join points, 6 Add node,
  7 Straighten
- Select and Move: 1-4 the stacking commands
- Draw path: 1 Finish, 2 Close shape, 3 Cancel
- Pencil: 1 Close shape
- Eyedropper: 1 Artwork, 2 Reference, 3 Fill, 4 Stroke

## Commands

- Ctrl/⌘ K command palette
- Ctrl/⌘ Z undo, Ctrl/⌘ Shift Z redo
- Ctrl/⌘ S save, Ctrl/⌘ Shift S save as, Ctrl/⌘ Alt N new drawing
- Ctrl/⌘ A select all
- Ctrl/⌘ G group, Ctrl/⌘ Shift G ungroup
- Ctrl/⌘ J join (Join points, Merge shapes or Join line ends)
- Ctrl/⌘ ] and [ bring forward and send backward; add Shift for front and
  back
- Shift H and Shift V flip horizontal and vertical
- Arrow keys nudge by 1 unit, Shift arrows by 10
- Delete delete; Shift Delete delete the contours of the selected points
- F2 rename
- Escape cancel, or step up one selection level

In text fields these keys keep their usual text behaviour.

## Clipboard

Ctrl/⌘ C copies the selected objects, with their groups, styling and the
shapes they reference. Ctrl/⌘ V pastes independent copies at the same
position and selects them. The copy is a snapshot: editing or deleting the
originals afterwards does not change it. The clipboard belongs to this
editor session.

## Limits

- Only a static subset of SVG imports; anything else is reported instead
  of dropped. See
  [Import, export and project files](12-document-model.md#import-export-and-project-files).
- Point tools edit paths. Basic shapes need Convert to path, and instances
  need Detach.
- Editing shared geometry needs every object that uses it selected.
- A group whose compositing or references cannot be split without
  changing its look refuses to ungroup.
- The local server refuses requests from other origins, unknown sessions
  and stale document revisions.

Next: [Operations](11-operations.md)
