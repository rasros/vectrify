# Transforming

How to move, resize, rotate, flip, align and stack objects.

All of these measure the selection by its geometric bounds: each object's
outline box, without stroke width or clipping, through its transforms. The
selection frame in Move shows these bounds. Every change is one undoable
edit, and objects inside rotated or scaled groups move, turn and resize on
the page exactly as the frame shows. A locked position or geometry refuses
the change and says why.

## Moving

Move (V) selects, moves and resizes. Click an object to select it, then
drag it, or anywhere inside its frame, to move the selection. Dragging on
empty canvas outside the frame box-selects.

While moving, the frame's edges and centre snap to the edges and centres
of other objects and of the artboard, and dashed guides show what it lines
up with. Once the drag is under way:

- Shift keeps the move horizontal or vertical.
- Alt or Ctrl/⌘ moves freely, without snapping.
- Escape cancels.

The arrow keys nudge the selection by 1 document unit, or 10 with Shift,
whatever the zoom: selected points in a point tool, otherwise the selected
objects. Nudges do not snap, and nudges less than a second apart are one
undo step.

## Resizing

In Move, drag an edge of the frame to resize along one axis, or a corner to
resize both; the cursor changes near them. Shift keeps the aspect ratio,
Alt resizes from the centre, and Ctrl/⌘ turns off snapping. The tool strip
shows the new size while you drag.

Resizing keeps stroke widths: a resized object's strokes are scaled back
by the resize's mean factor. A non-uniform resize still makes a stroke a
little wider along the stretched axis, as SVG strokes follow their
transform.

## Position and size fields

The Transform section of the Object tab types the frame directly. X and Y
are its top-left corner; W and H resize from that corner, so changing a
size keeps X and Y. The lock beside W and H keeps the proportions. Enter or
leaving a field applies it, Escape restores it. Move by offsets the
selection by a distance. Several selected objects move and scale together
and keep their relative placement.

## Rotating and flipping

In Move, a round handle sits above the selection frame, and the pivot, a
ringed cross, at its centre.

- Drag the handle to rotate about the pivot. Shift turns in 15° steps.
- Drag the pivot to rotate about another point. It snaps to the frame's
  corners, edge middles and centre; Alt places it freely. A moved pivot
  stays for the same selection until the drawing changes otherwise.
- Rotation, under Transform, shows the selection's angle on the page in
  degrees clockwise, and typing an angle turns it to that angle.
- Flip horizontal (Shift H) and Flip vertical (Shift V) mirror the
  selection about the centre of its frame.

Strokes, arrowheads, clipping and gradients turn and mirror with their
objects.

## Aligning and distributing

Align and Distribute are under Actions → Arrange. They move objects and
never resize them.

- Align left, centre, right, top, middle or bottom lines up the selected
  objects' edges or centres.
- To chooses what they line up to: the Selection's frame (two or more
  objects), the Artboard (one object is enough), or a Key object, which
  stays put while the others line up to it.
- The key object is the last object you added to the selection on its own,
  by a click, Shift-click or tree row; a box select chooses none. Key
  beside To picks another, and the canvas marks it with an amber frame.
- Distribute horizontally or vertically spaces three or more objects by
  their centres. The outer two stay put. Equal gaps puts the same space
  between neighbours, and equal centres the same distance between centres.

Pinned points do not stop a whole object from moving.

## Grouping, stacking and deleting

- Group (Ctrl/⌘ G) gathers the selected objects into a new group, even
  across groups, at the frontmost selected object's place. Ungroup
  (Ctrl/⌘ Shift G) gives the group's transform, paint, clipping and opacity
  to its children.
- Bring forward and Send backward (Ctrl/⌘ ] and [) move one step; Bring to
  front and Send to back (add Shift) move to the end of the group, keeping
  the selection's order. In Select and Move the keys 1-4 press them too.
- Delete removes the selected objects. If they hold pinned points, it asks
  first.

Next: [Drawing and points](05-drawing.md)
