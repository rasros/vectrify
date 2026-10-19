# Path operations

How to join, combine, cut, convert and clip paths.

The commands in this chapter are under Actions → Path and Actions →
Clipping in the right panel, and in the palette; the common ones are in
the right-click menu too. Each is one undoable edit, and each refuses with
a reason when locks, pinned points or shared geometry are in the way.
Basic shapes (rectangles, ellipses, circles and lines) need Convert to path
first.

## Joining

There is one command for each kind of join, and Ctrl/⌘ J runs the one that
fits the selection:

- Join points joins two selected points; see
  [Editing points](05-drawing.md#editing-points).
- Join line ends joins stroked lines where one carries on from another:
  ends close together on screen and roughly in line. The nearest and
  straightest pairs join first. A dashed outline becomes one line, and a
  ring of dashes closes. Filled paths are left alone.
- Merge shapes merges filled paths by area into one, through a dialog. Its
  colour is mixed by area by default, or taken from any of the paths. The
  result takes the frontmost path's place. Pieces that touch, such as the
  halves of a knife cut, merge back into one region.
- Merge shapes on a single path unions its overlapping contours and
  removes the seams between them, without a dialog.

Combine paths collects the selected paths into one compound path without
changing any contour. The result takes the frontmost path's paint, so
overlapping contours can form holes under its fill rule.

## Booleans

Union, Subtract, Intersect and Exclude combine the filled areas of two or
more paths. The backmost selected path is the target and keeps its name,
paint and place; the paths in front of it are the cutters and are deleted.
Hover a command to see the target outlined solid and the cutters dashed
before anything changes.

- Union keeps what is inside any of them.
- Subtract keeps the target's area outside every cutter.
- Intersect keeps what is inside all of them.
- Exclude keeps what is inside an odd number of them.

Holes stay holes and curves stay curves. A result that would be empty is
refused.

Cut out as hole takes two filled paths and picks the cutter by containment
rather than stacking: when one lies inside the other it becomes a hole in
it, and when they only partly overlap the front one cuts the back one. The
outer path keeps its own points where it can.

## Converting and splitting

- Convert to path turns rectangles, ellipses, circles and lines into paths
  with the same look, so their points can be edited.
- Fill to centerline turns a thin filled shape, such as a traced piece of
  outline, into a stroked line down its middle, as wide as the shape.
- Stroke to path does the reverse: a stroked line becomes the filled shape
  its stroke paints.
- Split disconnected parts makes each separate piece of a path its own
  path. Holes and touching contours stay together.
- Detach to editable path gives an instance its own path to edit.

## Knife

The Knife (C) cuts along a straight line: drag across the paths and
release. Shift snaps the line to 15° steps. With nothing selected it cuts
every unlocked path it crosses; with a selection, only the selected paths.

- A stroked line comes apart where the knife crosses it. The pieces on
  the side with less line become a new path, so cutting across a loop
  takes the loop away whole, ready to delete.
- A filled shape is cut only where the knife runs through it from outside
  to outside, and only between the drag's two ends, so cutting across one
  arm of a U leaves the other arm attached.

The pieces keep the original's paint and place, meet exactly along the
cut, and are selected afterwards. Join line ends or Merge shapes puts them
back together.

## Extracting and deleting regions

Extract region moves the part of some paths inside an area into new paths
of their own, just above the originals. Delete region deletes that part.
Both use the Region tool (X):

1. Choose Extract or Delete in the Region panel over the canvas.
2. Drag a rectangle, or choose Polygon and click its corners, then click
   the first corner or press Enter.
3. Choose what it acts on: the selected objects, or every unlocked path
   or shape in the area.
4. Choose whether to cut objects crossing the edge (on by default).

Nothing changes yet. The canvas colours each object by what will happen
to it: amber for contours taken whole, orange for one cut along the edge,
red for one deleted whole, grey dashed for one left as it is. The panel
says the same in words. Press the panel's button or Enter to apply, or
Escape to cancel.

With cutting off, only contours wholly inside the area are taken. A
shape's holes stay with the shape around them.

## Clipping

A clip hides whatever lies outside a closed path, the boundary. The
boundary itself stays in the drawing as it was.

1. Select the objects to clip and the closed path to clip them to, for
   example a group of shading and the body shape it should stay inside.
2. Run Set clip. Its dialog marks each selected object as Clipped or
   Boundary, starting with the frontmost closed path as the boundary.
3. Choose the boundary geometry. Linked (the default) shares the
   boundary's points, so reshaping the boundary reshapes the clip. Copied
   takes a copy of the outline as it is now.
4. Press Set clip.

The clip is placed where the boundary is when you set it; moving the
boundary later does not move the clip, so set it again to follow. Several
boundaries clip to their combined area, holes in a boundary stay hidden,
and setting a clip again replaces the old one. Remove clip stops clipping
the selected objects and leaves the boundary in place. Exported SVG
contains an ordinary `clipPath`, and vectrify keeps a linked clip linked
when it reopens the file.

Show clips, in the footer, outlines every clipping boundary in amber
dashes. Click an outline to select it, and use Nodes to edit its points.

Next: [Tracing a reference](08-tracing.md)
