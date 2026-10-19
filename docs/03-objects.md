# Objects and selection

How to find, select, name, hide and protect the objects in a drawing.

## The Objects tree

The Objects tree lists the drawing back to front: a row paints over the
rows above it. Groups fold with the arrow beside them, or Left and Right on
a focused row; folding changes only the tree. Swatches show each object's
resolved fill and stroke, inherited paint included. Definitions, such as
shared geometry, gradients and clipping boundaries, are listed in their own
folded branch with role labels, and instances name their source.

Drag rows to restack them. Dragging a selected row carries the whole
selection. A line shows where the objects will land; near the end of a
group, move the pointer left or right to drop inside it or after it.
Dropping on the middle of a group row moves the objects into that group, at
its front. Objects moved into another group keep their place and paint on
the page, and take on the new group's clipping and opacity. Escape cancels
a drag; each drop is one undoable edit.

To rename an object, select it and edit Name in the Object tab, or press
F2. Clearing the field restores the automatic label. Names survive project
saves and SVG export.

## Selecting

The selection is always a set of objects, and optionally points inside
them. Points may span several paths; point commands act on all of them in
one edit.

- Click picks the outermost group under the pointer. Double-click a group
  to enter it: clicks then pick inside it, and the tool strip shows a
  breadcrumb such as "Drawing › Trees". Click outside the group to leave
  it. Double-click a path to edit its points in Nodes.
- Click again at the same spot to cycle through the overlapping objects
  there, front to back.
- Shift-click adds or removes an object on the canvas; Ctrl/⌘-click does
  the same in the tree.
- Drag on empty canvas to box-select what lies wholly inside the box:
  objects in an object tool, points of the selected paths in a point tool.
  Shift toggles what the box finds. In Select every drag box-selects; in
  Move a drag on the selection moves it instead.
- Ctrl/⌘ A selects every object at the current level, or in a point tool
  every point of the selected paths.
- Escape cancels a gesture under way, otherwise steps up one level: points
  to their paths, a path to its group, the top level to nothing.

Selecting in the tree centres the canvas on the selection, and selecting on
the canvas reveals the row in the tree. Switching tools keeps the
selection. Undo, redo and saved projects restore it, points included.

## Hiding objects

The eye at the end of a row hides or shows that object. Hide objects, Show
objects and Show all hidden objects do the same for the selection or the
whole drawing. Hiding is part of the drawing, not a view setting: a hidden
object is saved as `display="none"`, so it stays hidden in exported SVG and
to agents.

Hidden objects are skipped on the canvas: clicks, box select and Select all
pass over them, and nothing snaps to them. Their rows stay in the tree,
dimmed, and can still be selected there to change paint, name, locks or
stacking. Moving, resizing and editing points need them shown first. Tidy
and Fit colours leave hidden paths out.

## Locks

The Locks section of the Object tab locks an object's geometry, paint,
position or structure. A lock on a group covers everything in it. Locks
hold for every edit, from a tool, a dialog, an automated operation or an
agent: a refused edit names the locked object and changes nothing, and
controls that would be refused are dimmed with the reason. A paint lock
also protects an object's visibility.

## Pinned points

Pin points in the Points tab (or Pin points in the palette) to keep them in
place. Pinned points keep their position and identity through drags, Tidy
and the path commands; an edit that would move or remove one is refused.
Pins do not stop the whole object from being moved. Deleting an object
that has pinned points asks first.

## Shared geometry and instances

An imported SVG may draw one geometry from several places: through `use`
instances, or as paths sharing points with a clipping boundary. Editing a
shared point moves every path that draws it; the Points tab says how many.
Selected instances have no points of their own. Detach (in the right panel or the palette)
gives an instance or a path its own copy of the geometry to edit alone.

Next: [Transforming](04-transforming.md)
