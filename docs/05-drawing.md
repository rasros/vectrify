# Drawing and points

How to draw new shapes and paths, edit their points, and redraw part of an
outline along the reference.

## Drawing paint and where shapes go

All drawing tools share the Fill, Stroke and Width at the end of the tool
strip, remembered in this browser. Closed shapes take the fill and the
stroke; lines and open paths take the stroke colour. Width is in document
units.

New shapes go into the entered group, or the top level, in front of what
is there. Each shape is one undoable edit and is selected afterwards, and
the tool stays on for the next one.

## Pen

Draw path (D) draws a path point by point:

1. Click to add a corner, or drag to add a smooth point with mirrored
   handles.
2. Backspace removes the last point.
3. Click the first point or Close shape for a filled shape, or press Enter
   or Finish for an open path. Escape or Cancel discards it.

## Rectangles, ellipses and lines

Rectangle (M), Ellipse (E) and Line (L) draw by dragging. Shift makes a
square, a circle or a line at 45° steps, and Alt draws from the centre.
Both work mid-drag.

A click places a shape at the size in the tool strip: W and H, or a line's
Length and Angle. The fields keep the size of the last shape, so a click
repeats it. Corners rounds new rectangles.

These are native `rect`, `ellipse`, `circle` and `line` elements, and stay
so in saved files. They have no points to edit until Convert to path
(Actions → Path) turns them into paths.

## Pencil

Pencil (P) draws a freehand path. The stroke is fitted to a few curves as
you draw, and the path you release is the one previewed. Smoothing (1-20,
2 by default) is how far, in screen pixels, the curves may stray from the
stroke; more gives fewer points. Close shape closes each stroke back to its
start as a filled shape.

## Editing points

Nodes (S) shows and edits the points of every selected path, including
paths inside selected groups.

- Click a point to select it; Shift-click adds or removes; box-select
  several. Hover an unselected path to see its points faintly, and click
  one to switch to it.
- Drag a point to move all selected points, with their handles. Drag a
  handle to change its curve. Shift constrains the drag horizontally or
  vertically.
- Dragged points snap to the points of visible paths and to the artboard's
  edges and corners; an orange diamond marks the target. Alt or Ctrl/⌘
  drags without snapping.
- Type exact coordinates for one point in the Points tab.

The tool strip and the keys 1-7 do the rest:

- None, One and Two (1, 2, 3) make the selected points corners, curved on
  one side, or smooth with handles in line. An open contour's end points
  can have only one handle.
- Break (4) cuts a line at the selected points. With both points of a
  segment selected, it removes that segment instead.
- Join points (5) joins any two selected points into one line, bridging a
  gap with a curve. Two ends of one line close it.
- Add node (6) adds a point halfway along the edge into each selected
  point, keeping the curve.
- Straighten (7) keeps a point's two handles aligned through it, also
  while you drag one of them.

Delete removes the selected points and reconnects the line past them; a
contour left too short is removed. Shift Delete removes the whole contours
the points are on. With a handle selected, Delete retracts only that handle.
Without selected points, Delete removes the selected objects.

When the selected points are all on holes, Fill hole removes those holes,
and Hole to shape moves them out into new paths with the original's paint.
Both are in the Points tab under Contours & holes.

## Redraw outline

Redraw outline (R) fixes a stretch of an outline in one stroke, like a
magnetic lasso: a missing spike, a notch, a blade of grass.

1. Press on a path's outline.
2. Draw roughly along the reference's edge.
3. Release on the same contour.

The stretch between the two ends is replaced by curves fitted to the
nearest strong edge in the reference, close to your stroke. White dots show
where the ends attach and the stretch to be replaced is dashed. On a closed
contour it takes the shorter way round; hold Shift for the longer. Without
a reference it fits your stroke itself.

Points outside the stretch keep their place and identity. With paths
selected only those are redrawn; otherwise the stroke redraws the path it
starts on. It cannot extend an open contour past its ends, and a spike only
a few reference pixels wide may come out thinner than drawn.

Next: [Paint](06-paint.md)
