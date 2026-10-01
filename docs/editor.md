# Local SVG editor

Start the editor with:

```sh
uv run vectrify
```

Or open a specific SVG and its reference image:

```sh
uv run vectrify drawing.svg --reference original.png
```

With the `desktop` extra the editor opens in a native window (pywebview: the
platform's web view, or Qt WebEngine on Linux), and the page calls the Python
backend directly; saving asks for a file with a native dialog. Without the
extra, or with `--serve` (and optionally `--port`), vectrify serves the editor
on loopback and prints the address to open in a browser. The UI is included in
the Python package and needs no Node build step.
For source checkouts, `PYTHONPATH=src python -m vectrify.ui` works too.

## Layout

- **Top bar**: the name, the drawing's title and the file actions
  (**Commands**, **Open…**, **Restore saved…**, **Save project**, **Export
  SVG**). It keeps to one row at any width down to a phone's: the badge, the
  Ctrl K hint and then the title make room first, and the actions that still
  do not fit move into a **⋯** menu at its end, the least important first
  (**Restore saved…**, then **Open…**, then **Save project**), so
  **Commands** and **Export SVG** stay in view.
- **Left**: the tool rail, the **Objects** tree and, below it, the
  **Reference** panel. The rail holds the object tools **Select** (`V`) and
  **Knife** (`K`), the point tools **Nodes** (`N`) and **Redraw outline**
  (`R`), **Draw path** (`P`), which creates and ignores the selection, and
  **Pan** (`H`). The Reference panel holds the reference image and the tools
  that match the drawing to it (see Operations); its heading folds it away,
  and loading a reference opens it again.
- **Tool strip**, above the canvas: undo and redo, then only the active tool's
  controls, and on the right the selection's level and count, such as
  "Objects · 3 selected" or "Points · 5 points in 3 paths". Select has the
  stacking buttons and, inside an entered group, a breadcrumb such as
  "Drawing › Trees" (click a part to pick at that level). Nodes has the handle
  buttons (None / One / Two), **Pin points**, **Split edge**, **Delete**,
  **Delete contour**, X / Y for a single point, and **Fill hole** / **Hole to
  shape** when the points are on holes. Draw path has Finish, Close shape and
  Cancel. Knife, Redraw outline, Draw path and Pan show a one-line hint. The
  strip keeps to one row: when the window is too narrow for the
  tool's controls, the least important of them (Delete contour first, then
  Split edge and the coordinates, and so on) move into a **⋯** menu at the
  end of the controls, which holds them until there is room again; the status
  stays in view. A command run from the menu closes it.
- **Right**: only the selection's properties, in the same order every time.
  With points selected, the points come first: a single point's coordinates
  (in its path's own frame), its handles and whether it is pinned, and a note
  when the points are of shared geometry or the selection holds instances.
  Then the
  objects' name, role and connections, paint (fill, stroke, stroke width,
  opacity), **Move by** an offset, locks, and the **Actions** on the
  selection: Group, Ungroup, Join, Convert line/fill, Split parts, Cut out as
  hole, Snap edges…, Clean up…, Detach and Delete. An action that cannot run
  is dimmed, with the reason as its tooltip. Detach shows only for an instance
  or a path that shares its geometry, and Delete deletes the selected points
  in a point tool, else the selected objects.
- **Right-click** on the canvas or in the tree for a menu of the actions that
  apply to what is there: a right-click on something unselected selects it
  first. In a point tool the menu offers the point commands.
- **Command palette** (Ctrl/⌘ K, or **Commands** in the top bar): type to find
  any command by name, group or a keyword (tools, actions, points, reference
  tools, open, save, export, view), with its shortcut. ↑ / ↓ choose,
  Enter runs, Escape closes. A command that cannot run now shows why instead.
- The footer shows the artboard size, the reference overlay toggle (`O`),
  with its opacity and an amber canvas border while it is visible, and zoom.
  `O` cycles the view: the drawing, the reference over it, and the reference
  alone at full strength in place of the drawing, to compare by flipping
  between them; `Shift+O` always goes straight back to the drawing. The
  Reference panel picks the same three views.

Dialogs remain where a preview or confirmation is needed: Generate from
reference, Tidy, Fit colours (and Fit gradient), Join (for filled paths), Clean up, Snap edges,
Restore saved project and Keyboard shortcuts, each titled as the command that
opens it. They open from the Reference panel, the Actions, the context menu or
the palette.

Input never gets lost while an edit is running. Keys, clicks, commands, tree
clicks and drops, and canvas drags made meanwhile wait for it, then apply in
the order they were made, each once, on the drawing the edit left: a click or
drag is hit-tested again when it runs, so a drag started during an edit acts
on its result. A drag still held when the edit finishes carries on live.
Escape cancels a waiting drag at once. Viewing needs no wait: panning,
zooming, fit and the overlay toggle act immediately. Edits made during a
canvas drag also wait for the drag to finish.

## Selecting

The selection always has objects, and optionally points inside them. Points
may span several paths, say five points in three paths; point commands act on
all of them as one undoable edit.

- In an object tool (Select, Knife) a click picks the outermost group
  under the pointer, as in most editors. Only Select moves and resizes
  objects by dragging; in Knife a drag cuts. **Double-click** a group to enter it:
  clicks then pick within it, and the breadcrumb shows where you are. A click
  outside the entered group leaves it. Double-click a path to switch to Nodes
  on it. Picking an object in the tree enters the group it is in.
- Repeated canvas clicks at the same spot cycle through the overlapping
  objects at that level, front to back, then wrap around. Moving or zooming
  the view, editing, or selecting in the tree resets the cycle. A double-click
  acts on what was picked before it: its second click does not cycle on.
- **Shift-click** adds or removes an object; in the tree, Ctrl-click
  (Command-click on Mac) does too. Tree selection centres and zooms the canvas
  on the whole selection; selecting on the canvas reveals the row in the tree,
  clearing a search filter if needed.
- **Box select**: drag on the canvas, from empty space or from an unselected
  object, to select what lies wholly inside the box: objects at the entered
  group's level in an object tool, points of the selected paths in a point
  tool. Shift toggles what the box finds. In Select, dragging a selected
  object, or empty canvas inside the selection's frame, moves the selection
  instead.
- **Resize** in Select works like resizing a window. The selection's frame is
  the bounding box of the selected objects on the page, with small ticks at
  its corners; the tool strip gives its size. Within 6 screen pixels of an
  edge the cursor shows ↔ or ↕, and dragging resizes along that axis only; at
  a corner it shows a diagonal arrow and resizes both. Inside the frame it
  shows the move cursor (over an unselected object, which a press picks
  instead, the plain one). Shift keeps the aspect ratio (at an edge, about the
  middle of the other axis), Alt resizes from the centre, and the dragged
  edge or corner snaps to other objects' bounds and the artboard's edges
  within 8 screen pixels; hold Ctrl/⌘ to drag without snapping. The canvas
  previews the new size, the tool strip shows it as W × H, and releasing
  applies it as one undoable **Resize**; Escape cancels. Each object keeps its
  geometry: the scale is composed onto its own `transform`, in its parent
  group's frame, so an object inside a rotated or scaled group resizes on the
  page exactly as the frame shows, and a selected group takes its selected
  children along once. Strokes keep their width: the stroke widths in a
  resized object are divided by the scale's mean (√(sx·sy)), written on the
  object where it inherited one, except where paint is locked. SVG strokes
  follow their transform, so a non-uniform resize still makes a stroke a
  little wider along the stretched axis, and resizing an object in a rotated
  frame along one page axis skews it; the editor does not use
  `vector-effect`, which exporters and renderers do not all support. Locked
  position or geometry refuses the resize, and the cursor over the frame says
  so before you drag.
- In a point tool, the points of every selected path show, and those of
  every path inside a selected group, however deep; box select and the point
  commands span all of them. Selected instances (`use`) have no points of
  their own: the right panel asks to Detach them. Hovering an unselected path
  in Nodes shows its points faintly; clicking one selects that point and its
  path in place of the selection (Shift adds the path). Shift-click adds or
  removes a point; a click on one of several selected points, without
  dragging, selects it alone.
- Paths drawing one shared geometry share its points. A point is selected in
  the path it was clicked in, drawn strong there; the other selected paths
  drawing that geometry show it faintly, since editing it moves them all, and
  the right panel says how many other paths it changes. Detach a path to edit
  it alone. A point picked in two such paths is still one point: it moves,
  splits or changes handles once.
- Switching tools never discards the selection. Going from a point tool to an
  object tool keeps the objects and hides the points; going back restores the
  same points if the objects have not changed meanwhile, and clears them if
  they have.
- **Escape** first cancels what is under way (a drag, a path being drawn, a
  hole inspection, a redraw stroke). Otherwise it steps up one level: points
  to their paths, a path to the group it is in (a point tool stays, showing
  the group's points), top-level objects to nothing, and, with nothing
  selected, out of an entered group.
- A click outside the artboard clears the selection in every tool; in a point
  tool a click on empty canvas drops the points first.

Selected geometry has a translucent blue interior and a blue contour on the
canvas; holes stay clear and stroke-only paths remain unfilled. Highlights
preserve inherited clipping, including visible instances of a selected
definition. The server keeps the selection, points included, so undo, redo
and saved projects restore it.

## What works

- Open SVGs and Vectrify project files; export SVG or download an editable project.
- Pan/zoom/fit, reference image overlay and opacity adjustment (Reference
  panel). Press `O` to cycle the view, or click its toggle in the footer.
- The object list shows definition and clipping containers in their actual
  hierarchy. Shared geometry and clipping contours have neutral symbols and
  role labels; instances name their source. The right panel explains these
  roles and links to source geometry, clipping boundaries, and their users.
- Tree swatches show resolved fill/stroke colors, including inherited paint;
  group swatches preview their contents. The right panel shows effective hex
  colors and names the parent they come from. No-paint and mixed selections
  have distinct indicators. Editing inherited paint sets an object override.
  A gradient fill shows as its ramp, in the tree's swatch and the right
  panel's picker, with "Linear gradient" in place of a colour value; picking
  a colour or typing one makes the fill flat again and removes the object's
  own gradient. Gradients and their stops are listed under Definitions.
- Fill/stroke/opacity, dragging and resizing the selected objects, and
  numeric offsets (**Move by**).
- **Nodes (N)** edits the points of every selected path. Delete/Backspace
  deletes the selected points, a contour's start point included: the next
  point then starts the contour. A contour left with fewer than two points
  (three when closed) is deleted, and a path left without contours is deleted
  too, so a stray speck can be removed point by point or at once with **Delete
  contour**. A press takes the nearest point or handle within 10 screen
  pixels, so a point need not be hit exactly; the one under the pointer is
  drawn larger, with its handles. Drag blue handles for curves; a handle
  shorter than 16 screen pixels is drawn that far out along its direction on a
  dashed line, so it shows clear of its point, and a handle on its point
  counts as none. Zoom in (up to 25600%) to reveal dense points. **None**,
  **One** and **Two** handles turn the selected points into corners, points
  curved on one side (again to switch sides), or smooth points with their
  handles in line. The keys 1, 2 and 3 do the same. **Pin points** keeps them
  in place (again to unpin); **Split edge** adds a point on the edge leading
  into each selected point. For drawn lines the strip also has **Break**,
  **Delete segment** and **Join ends**. **Break** cuts a line at the selected
  points: an open line comes apart there, each piece ending on its own copy of
  the point (both copies stay selected, so a drag moves them together; click
  one to move it alone), and a closed contour opens at the point. A line's
  ends are free already. **Delete segment** takes out the segment between two
  selected neighbouring points, splitting the line there (a closed contour
  opens); with more points selected, every segment between two of them goes,
  and a piece left as a lone point is dropped. **Join** joins any two selected
  points, in one path or two, filled or not, into one line however far apart
  they are; a point that is not a free end yet is broken there first: ends
  that meet become one point, others are bridged by a curve leaving each end
  along its line. Two ends of one line close it. A small loop in a line comes
  out by selecting the points on it and pressing Delete, which reconnects the
  line past them, or by cutting it off with the knife. Dragging a point moves
  every selected point by the same offset, each in its own path's frame, and
  each takes both of its handles along, so the curve keeps its shape around
  it; dragging a handle moves only that handle. Typing a single point's
  coordinates does the same for it. Pinned points stay put, and no other path
  moves. While dragging, the point or handle under the pointer snaps to the
  points of every visible path (the other points of the dragged paths
  included) and to the artboard's edges and corners, within 8 screen pixels at
  any zoom; an orange diamond marks the target and a dashed line the edge.
  Hold Alt or Ctrl/⌘ to drag without snapping. Where a point started is never
  a target. When the selected points are all on holes, in one path or several,
  **Fill hole** removes those holes (and islands inside them) and **Hole to
  shape** moves them out into new paths, each with its path's paint, stacked
  just above it and selected afterwards, as one undoable edit. Islands inside
  a hole go along as holes of the new shape.
- **Draw path (P)**: click to add corners, drag to set mirrored Bézier handles.
  Click the first point or **Close shape** for a filled shape, or press Enter /
  **Finish** for an open stroked path. Backspace removes the last point; Escape
  or **Cancel** discards the draft. Creation is one undoable edit and selects the
  new path for paint and point editing. Middle mouse panning works while drawing.
- Group/ungroup, stacking order, delete and detach shared geometry.
  **Send backward**/**Bring forward** (Ctrl/⌘ `[` / `]`, or the ↓ / ↑ buttons
  of the Select strip) move one object a step; **Send to back**/**Bring to
  front** (add Shift, or ⇊ / ⇈) move the selection to the back or front of its
  group, keeping the selected objects' order.
- Drag rows in the object tree to restack them. The tree lists objects back to
  front: a row paints over the rows above it. Dragging a selected row carries
  the whole selection, which keeps its order. A line shows where the objects
  land; where a group ends, move the pointer left or right to drop inside it
  or after it. Dropping onto the middle of a group row moves the objects into
  that group, at its front. Escape cancels. Objects moved to another group keep
  their look: the transform and paint they inherited are written onto them.
  Moves that cannot keep it are refused with a reason: leaving or entering a
  group with opacity or clipping, a changed transform on a path with shared
  edges, and objects with instances that would change too. Locked objects and
  groups, definitions, clipping contours and a group into itself are refused
  as well. Each drop is one undoable edit. A drag can start while an edit is
  running: dropped before the edit is done, the objects land next to the rows
  they were dropped by, on the tree the edit leaves, and the drop is refused
  with a reason if those rows are gone or have moved to another group. A drag
  still held when the edit finishes follows the new rows under the pointer.
- **Join** (Actions, and the Nodes strip) does what fits the selection: two
  selected points join each other, as above; lines join at their ends, below;
  and filled paths merge by area, through a dialog (**Join…**). In a point
  tool it needs the two points.
- Joining filled paths combines selected paths and groups, recursively
  including paths in nested groups and counting overlapping selections only
  once, even if other objects sit between them. The result always occupies the
  frontmost selected position; intervening objects stay in their existing
  order. Fill/stroke colours and numeric paint properties are averaged by
  clipped painted area (including strokes, excluding holes), so larger shapes
  contribute more. Colour alpha is included; fully unpainted selections fall
  back to equal weights. Clicking Join opens an options dialog: keep the
  default area-weighted color mix or choose any participating path's
  fill/stroke colors (including inherited colors and fill/stroke alpha). Width
  and overall opacity stay area-weighted. Cancel leaves the document
  untouched; stacking remains fixed at frontmost. Filled overlaps use a curved
  boolean union, preventing cancellation holes and internal seams. Disjoint
  contours keep exact nodes; boolean intersections create new nodes and
  require affected pins to be removed first. Two pieces that abut, such as the
  halves of a knife cut or parts split apart, merge back into one region this
  way. Empty selected group wrappers are removed after joining; groups
  containing non-path shapes must be converted or selected more narrowly
  first. Undo restores the original geometry, paint and stacking. Transforms
  and per-path clipping are resolved into the common parent when joining
  across groups. The frontmost ancestry is split as needed so unselected
  objects retain their stacking, styling and clipping. Area weighting uses the
  original clipped painted areas. Filled regions and stroke-only outlines join
  separately. Cross-group joins reject object-bounding-box clips, clipped
  stroke-only paths, non-uniformly scaled strokes, and group-opacity cases
  where splitting would change unselected artwork. Degenerate clip
  intersections use a winding-aware fallback; its curve tolerance is 0.01 SVG
  units (straight edges stay exact).
- Joining lines joins the open ends of the selected stroked lines (paths with
  no fill, or the paths in selected groups) where one line carries on from
  another: ends at most 12 screen pixels apart, so zooming out reaches wider
  gaps, and in line with each other, the line turning at most 60° across the
  gap and at each end. The nearest and straightest pairs go first, each end
  once; ends at one spot pair however sharp the corner when they are the only
  two there. A dashed outline becomes one line, and a ring of dashes closes.
  Ends that meet become one point; a gap is bridged by a curve that leaves
  each end along its line. The joined line lands in the frontmost of its paths
  and keeps that path's paint; a path left without lines is deleted, and the
  paths holding joined lines are selected. Side-by-side ends of parallel lines
  do not join. Lines join at their ends rather than by area.
- **Convert line/fill** (Actions) flips each selected path, as one edit; it
  reads **Fill to line** or **Line to fill** when the selection holds only one
  kind. Fill to line turns each selected thin filled shape, such as a part of
  an outline that a trace drew as a fill, into a stroked line down its middle:
  the shape is rasterised about 12 pixels across its thickness, thinned to a
  centreline as the Cel art tracer does, and fitted with curves. Lines meeting
  at a junction run on through it the straightest way. The stroke width is the
  shape's thickness measured along the centreline, its colour and opacity the
  fill's, with round caps and joins, so the stroke covers about what the fill
  did. The path keeps its ID, place in the stack and group; its points are
  new, so pinned points are refused. A shape about as wide as it is long is
  not a line and is refused. Line to fill does the reverse for stroked paths
  without a fill: each becomes the filled shape its stroke paints, with its
  width, caps and joins, in the stroke's colour.
- **Split disconnected parts** separates selected drawing paths into independent
  geometries. Holes and touching/overlapping contours stay together; original
  curves and pins are preserved. A group retains styling,
  transforms and clipping. Undo restores the compound path in one step.
  Only geometric contact or overlap connects contours; gaps are not bridged,
  regardless of stroke width, caps or joins. Cubic curves are approximated at
  0.01 document units for classification; output curves remain exact. Shared assets must be
  detached first. **Detach to editable path** converts a path instance into a
  selected independent path ready to split, retaining its paint and transforms
  in wrapper groups. Other instances keep their shared definition.
- **Cut out as hole** (Actions) takes two selected filled paths. When one
  lies inside the other it becomes a hole in it; when they only partly overlap
  the front one cuts the back one. The outer path keeps its paint and
  stacking place, and the cutting path is deleted, as one undoable edit. The
  cutter's contours are added to the outer path, reversed when needed so they
  cut under its fill rule (nonzero or evenodd), and its own holes become
  islands again; transforms of either path's groups are taken into account.
  When added contours cannot reproduce the difference (a partial overlap), the
  outer outline is recomputed as a curved boolean difference, which renumbers
  its points and so is refused for pinned points.
  Locks, shared geometry and instances of the outer path are refused.
- The **Knife** tool (`K`) cuts the paths it crosses along a straight line:
  drag across them and release. With nothing selected it cuts every drawing
  path the line crosses whose geometry and structure are unlocked, inside the
  entered group if one is entered; with a selection, only the selected paths
  and the paths inside selected groups. Shift snaps the line to 15° steps. A stroked line without a fill comes apart
  wherever the dragged line crosses it, each piece ending on its own copy of
  the crossing, and a closed one opens up. The pieces on the side of the knife
  with less of the cut lines become a new path just above, so cutting across
  a loop in a line takes the loop away whole to be deleted; two cuts across
  its neck do the same. Lines the knife misses stay where they are, and
  **Join** joins pieces again. A filled path is cut only when the line
  runs through it from outside to outside; a line that ends inside a shape or
  misses it leaves it alone. The cut runs along the
  whole line through that path, so a line across one arm of a U also cuts the
  other arm if the line's extension reaches it. Each cut path becomes one path
  per side of the line (compound when that side has several parts, holes
  kept), with the original's paint, transform, locks and stacking place; the
  first piece keeps its ID. Curves stay curves. The two pieces meet exactly
  on the same seam points but stay independent: dragging a seam point moves
  only that piece, and **Join** merges them back into one. Pinned
  points, shared geometry, instances and locks are refused. The pieces are
  selected afterwards, and the cut is one undoable edit. A line that crosses
  nothing the knife can cut says so; a plain click selects.
- The **Redraw outline** tool (`R`) fixes a stretch of one path's outline in
  one gesture, like a magnetic lasso: a missing spike, a notch or a grass
  blade. Press on a path's outline (or on one of its points), draw roughly
  along the reference's edge and release on the same contour: the stroke
  redraws the path it starts on, the last unlocked path the pointer was over
  (inside the entered group, if one is), which is then selected. With paths
  selected (or a group: every path the point tools show), only those can be
  redrawn, and the selection stays as it was. Each
  end attaches to the nearest point within 6 screen pixels, else to the
  nearest place on the outline within 10; white dots show where, and the
  stretch that will be replaced is dashed. On a closed contour that is the
  shorter way round between the two ends; hold Shift while drawing for the
  longer. The stretch is replaced by curves fitted to the reference's edge
  near the stroke: the cheapest path through a band 8 screen pixels (at
  least 3 reference pixels) either side of the stroke, cheap along strong
  colour edges and following the stroke where there is none, moved onto the
  edge's peak within a pixel, fitted with short cubics that keep sharp turns
  as corners and simplified to 0.6 reference pixels. Without a reference the
  stroke itself is fitted, to about a screen pixel. Every point outside the
  stretch keeps its ID, place and handles; the ends are split exactly where
  they attach. Pinned points inside the stretch and geometry or structure
  locks are refused. The redraw is one undoable edit; a
  plain click selects, and Escape cancels a stroke.
- Geometry/paint/position/structure locks and backend-enforced constraints.
- Undo/redo; a drag is one transaction, not one undo entry per pointer move.

Keyboard shortcuts are available from the `?` button. `V`, `N`, `P`, `K`, `R`
and `H` switch tools, and Ctrl/Command-K opens the command palette. Drag with the middle mouse button (in any tool), or hold Space
and drag, to pan. Use the scroll wheel to zoom, and press `F` to
fit. Ctrl/Command-Z undoes; add Shift to redo. Ctrl/Command-S saves a project.

Projects preserve object/node identities, locks, pins,
selection and the reference image. SVG exports contain the drawing. Downloads
use the browser's download location and do not overwrite the original input.
The server keeps sessions in memory; save a project before stopping it. Browser
reloads reconnect to the current session, but undo history is not stored in
project files. Saving also keeps a browser recovery copy keyed to this tab's
session; after a server restart, the tab restores that saved project. Edits made
after the last save are not included in recovery. **Restore saved** also lets
you choose a saved browser recovery copy from a new tab.

## Operations

The **Reference** panel, below the objects on the left, is where the
reference lives, whichever tool is active: **Load…** (or **Replace…**) takes
a PNG, JPEG or WebP image, shown as a thumbnail with its name, and × removes
it. With a reference loaded the panel picks the view (**Drawing**,
**Overlay** or **Reference** alone, as `O` cycles them) and the overlay's
opacity; its heading gives the state. Below are the tools that compare the
drawing with it: **Generate…**, **Tidy…**, **Fit colours…** and **Fit
gradient…**, acting on the selection made with any tool. A tool that cannot
run is dimmed, with the reason as its tooltip. Each is also in the command
palette.

**Tidy…** is a quick clean-up of the selected paths, or the paths
inside selected groups, against the reference around them, never the whole
image. By default it snaps their points onto the reference's edges and removes
the points they do not need, in a few seconds; to reshape a path, use Redraw
outline (R). Tick the steps it may use:

- **Snap to reference** (on by default) moves the points onto the reference's
  nearest edges. With **Add detail** it also adds points where the path
  misses a piece of the shape or covers too much: one point, two, or a spike
  of three whose base stays on the outline, reaching bit by bit along a strand
  that curls away. **Pixels per added point** is how many reference pixels
  each new point has to fix to be kept, and **Search beyond the path** how far
  past the selection, as a share of its size, the reference is read: a point
  can only reach that far. Add detail tries a limited number of new points per
  round, so on a large path it adds the ones that fix most first.
- **Simplify** (on by default) removes the points the outline does not need,
  moving it no more than the tolerance in reference pixels, and turns curves
  whose handles lie on their line within the tolerance into straight
  segments.
- **Fit shape** (off by default: slow on large paths) moves points and curve
  handles by gradient descent, with a number of fitting steps and a maximum
  movement per fit in local SVG units. Straight segments are fitted as curves,
  so it can give them handles where the reference curves; those it leaves
  straight stay lines. It runs on the GPU when PyTorch CUDA and the Vectrify
  CUDA extension are there, and on the CPU otherwise; outlined (stroked) fills
  need the GPU.

Each round tries every ticked step on the paths as they stand and keeps the one
that brings them closest to the reference, if it fixes at least the **Minimum
improvement** (1% by default) of the difference where it acted: over the
pixels it changed and a thin band around them, so a small fix on a large
selection counts as much as on a small one. When none helps, Simplify gets its
turn, and the run ends once nothing changes, the rounds (4 by default) run out
or the **Time limit** (10 s by default) passes. Each round gives each step a
share of the time left, and a step that runs out hands back how far it got; a
run that runs out of time keeps the best result so far, as Stop does, and says
so. No step may leave an outline folded over itself: a result where a path
crosses itself more than it did, a twist or a curve looped round, is not kept,
and the shape fit pulls such folds back as it goes. Concave outlines are fine,
and a path that already crossed itself may keep those crossings. With several
parallel workers a round's steps run side by side, with one shape fit at a
time. Without a reference only Simplify runs, keeping the paths' look. Pinned
endpoints stay fixed, and surviving points keep their identity. Colour is left
to Fit colours. The job runs in the background with progress, Stop & keep
best, and reference/before/after previews; the result lists the steps it
took. Apply is one undoable edit.

**Fit colours…** solves the flat fill colour of every selected object that
best matches the reference, with geometry locked. Each object is rendered with
its fill black and white, which measures its exact coverage (including
antialiasing, opacity, clipping and objects in front), so the best colour has
a closed form; outlines painted in the fill colour follow it. Objects are
fitted back to front; more passes help where fitted objects overlap. No GPU is
needed. Its **Fill** setting chooses a flat colour or a linear gradient. **Fit
gradient…** opens the same dialog with Linear gradient chosen: each selected
shape gets the two-stop gradient, along the direction the reference changes
most, that best matches it, starting and ending where the reference's ramp
does, inside the shape when it is flat beyond them. A shape the reference
paints evenly keeps a flat colour. Apply keeps it as one undoable edit.

**Clean up…** removes duplicate and collinear vertices and empty or
duplicate paths, and merges compatible neighbouring paths into compound paths,
within the selection only. Paths referenced by instances or clips are kept.
Coordinates are never rounded.

Generate, Improve, Simplify and Snap edges run through the shared operation
contract in `vectrify.operations` (see `docs/operations.md`) via
`POST /api/operation`.

**Generate from reference…** (Reference panel) traces the reference into
new shapes. The SAMVG method, a general-purpose tracer for photos and painterly
images, segments the image with SAM and traces each region into filled paths.
Choose the model (ViT-H is best, ViT-B is faster), the resolution SAM segments
at (by default the reference's own size; a fixed size shrinks a larger
reference to it, which is faster and uses less GPU memory, and enlarges a
smaller one first, so outlines do not follow its pixels) and the maximum
shapes. The rest is fixed: small holes in a region are filled, regions
narrower than 3 reference pixels everywhere (outlines and hairlines) and those
hidden entirely by the ones above are left out, each region's edge is moved
onto the reference's own edges nearby (SAM's masks are coarser than the image
and smooth away thin spikes and notches), neighbouring regions whose
difference is not worth a shape of their own are merged and recoloured, and a
backdrop rectangle beneath them all, in the colour of what no region claims
(usually the drawn outlines), fills the gaps between regions. Each outline is
smoothed over about one SAM pixel, traced densely and simplified to within
half a reference pixel, so it gets as many curves as its shape needs. The result is placed over the
artboard exactly where the reference is shown, as one new group at the front of
the whole drawing or of a selected group. With a group selected, only the
reference around what it already paints is traced. Text is traced as shapes, since the editor
does not support SVG text. Preview shows the reference, before and after, with
the change in reference error; Apply adds the group as one undoable edit.
SAMVG needs the `samvg` extra and holds the GPU while it runs.

The Cel art method is for flat, outlined illustrations such as cel and anime
art, and is the dialog's default. It follows their drawn lines: it finds the
lines (marks narrower than the surface either side and darker than it in
their brightest channel, so a black line still counts against a navy fill of
the same brightness, or only a little darker than a dark fill beside it; bold
outlines as dark as ink up to twice as wide; faint narrow shading, and dark
notches as dark as the surface they open into, are left to the fills, as are
dark shapes much wider than the lines). In a grainy or JPEG-compressed
picture the lines are found after a small median filter smooths the grain
away. It fills the space between the lines with a shrinking ball so a small
gap in a line does not join the regions either side, splits each region where
its colour changes with no line, and merges neighbours down to the chosen
number of **Regions** (0, the default, keeps one per 10,000 pixels of the
traced area, at least 50 and at most 2,000), those of a similar colour first,
and regions of
different colours a drawn line separates last (the same colour either side of
a line merges freely, since the line is drawn over it). The line pixels go to
the regions either side, so neighbours
meet at the line's middle; each edge between two regions is traced once and
used by both, so they meet exactly with no gap or overlap. Before an edge
is fitted, its pixel staircase is smoothed away between its corners, which
stay sharp, so the outlines come out smooth and with few points. Each region is
coloured from its own pixels, not the lines'. The lines are thinned to
centrelines and drawn over the regions as round-capped strokes in their ink:
a thin line's antialiased middle mixes its ink with the surface, so it is
drawn in the darker ink at the width of ink it holds, not in grey. There is
one path per line colour and width (up to four widths a colour, each path's
lines within about 40% of its width), each at its lines' measured width;
a line whose width changes a lot along it is cut
where it changes, so each part gets the width it has, and the parts still
meet end to end. Thinning's whiskers and the tiny loops it leaves round
a pinhole where lines meet are dropped, the lines of one path run on through
the junctions where they meet the straightest way rather than stopping
there, and a gap of up to one and a half line widths between two lines that
carry on from each other is bridged, so an outline comes out as a few long
lines rather than many short ones; **Line width** fixes the width instead. Lines stay strokes
however much they taper; only with **Trace lines as strokes** off are they
filled shapes. **Outline tolerance** is how far a traced edge or line may
stray from the reference, in reference pixels. It runs on the CPU and needs
no extra.

The Colour regions method fits a palette on the GPU and traces each colour
region. Dark outlines can be treated as ordinary regions, kept as separate
linework, or kept as linework with the regions beneath cleaned up, which
defines each region once and reuses it for fill and clip. **Merge and clean up
geometry** merges compatible paths and drops redundant vertices afterwards. It
needs CUDA and PyTorch (the `vision` or `samvg` extra).

## Limits

Point tools edit `path` elements, selected or inside a selected group. Local
`use` instances can be selected, styled, moved and detached; editing a referenced source still
requires selecting every affected consumer, as enforced by the backend. Groups
with compositing or reference relationships that cannot be ungrouped without
changing appearance return a clear error. The knife cuts only along straight
lines, across whole filled shapes; there are no tools for cutting a stroke-only
path or welding endpoints. Redraw outline replaces a stretch between two places
on one contour; it cannot extend an open contour past its ends, and a spike
only a few reference pixels wide may come out thinner than drawn.

Only the documented static SVG subset is accepted. Unsupported imports are
reported instead of silently dropping content. The editor namespaces SVG IDs
inside the page so artwork cannot collide with editor controls. The local server
rejects external origins, unknown sessions and stale document revisions.

## Names and snapped edges

Select one object and edit **Name** in the right panel (or press **F2**).
Enter or leaving the field applies the name; Escape cancels. Clearing the field
restores its automatic label. Names support undo/redo and survive project saves
and SVG export/reimport, without changing object IDs or shared references.

**Snap edges…** matches touching edges of two or more selected, closed
paths: every pair of them that comes within the contact distance. Adjust the
distance, preview the snapped spans highlighted in cyan, then apply. Of each
pair the front contour is the reference: the edges of the region behind are
split where the front's points project onto them, and its matched points and
curve handles move onto the front's. Curves are subdivided to accommodate
different node spacing without flattening. The result is ordinary geometry
and one undoable edit; nothing links the paths afterwards, so editing,
moving or deleting one never changes the other, and SVG export keeps the
coincident contours as they are.

The paths must be closed and unclipped and may not share geometry. Pins and
locks are enforced. Within one run, an edge snapped for one pair is not
matched or moved again for another, so a region can meet two neighbours at
once. It matches line-to-line and cubic-to-cubic spans, rather than
rebuilding mismatched contour types. Contact distance is measured in canvas
SVG units, including paths in differently transformed groups; transforms,
paint and stacking remain intact.

Projects saved while the editor linked shared boundaries (version 2) still
open; the links are dropped and the contours they joined are kept as they
are.
