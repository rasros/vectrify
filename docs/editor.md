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

## What works

- Open SVGs and Vectrify project files; export SVG or download an editable project.
- Click objects on the canvas or in the searchable object list; Shift-click for
  multiple selection. Ctrl-click (Command-click on Mac) also toggles selection
  in the object tree. Tree selection centers and zooms the canvas to the whole
  selection with padding, including when adding or removing objects. Selected geometry has a translucent blue interior and a blue contour on the canvas;
  holes stay clear and stroke-only paths remain unfilled. Highlights preserve inherited clipping,
  including visible instances of a selected definition.
- Repeated canvas clicks at the same spot cycle through overlapping painted
  shapes from front to back, then wrap around. Selecting on the canvas reveals
  the item in the tree (clearing a search filter if needed). Moving or zooming
  the view, editing, or selecting in the tree resets the cycle. Dragging a
  selected shape still moves it instead of cycling.
- Pan/zoom/fit, reference image overlay and opacity adjustment. Press `O` to
  hide/show the reference, or click its canvas-toolbar toggle. An amber status
  shows its opacity, with an amber canvas border while the overlay is visible.
- The object list shows definition and clipping containers in their actual
  hierarchy. Shared geometry and clipping contours have neutral symbols and
  role labels; instances name their source. The inspector explains these roles
  and links to source geometry, clipping boundaries, and their users.
- Tree swatches show resolved fill/stroke colors, including inherited paint;
  group swatches preview their contents. The inspector shows effective hex
  colors and names the parent they come from. No-paint and mixed selections
  have distinct indicators. Editing inherited paint sets an object override.
- Fill/stroke/opacity, selected-object dragging and numeric offsets.
- **Edit nodes (N)** has its own right-hand panel: path/point counts, selected
  point coordinates, pinning, edge subdivision and point deletion. Paint,
  stacking and other element controls return with **Element properties (V)**.
  Delete/Backspace in Nodes affects only the selected point, a contour's start
  point included: the next point then starts the contour. A contour left with
  fewer than two points (three when closed) is deleted, and a path left without
  contours is deleted too, so a stray speck can be removed point by point or
  at once with **Delete contour**. Select one path,
  then click a point; drag blue handles for curves. Zoom in to reveal dense nodes.
  **No handles**, **One handle** and **Two handles** turn the selected point
  into a corner, a point curved on one side (click again to switch sides), or
  a smooth point with its handles in line. The keys 1, 2 and 3 do the same.
  Dragging a point (or typing its coordinates) moves both of its handles by
  the same offset, so the curve keeps its shape around it; dragging a handle
  moves only that handle. Pinned points stay put, and no other path moves.
  While dragging, a point or handle snaps to the points of every visible path
  (other points of the same path included) and to the artboard's edges and
  corners, within 8 screen pixels at any zoom; an orange diamond marks the
  target and a dashed line the edge. Hold Alt or Ctrl/⌘ to drag without
  snapping. Where a point started is never a target.
  When the selected point is on a hole of the path, **Fill hole** removes that
  hole (and islands inside it) and **Hole to shape** moves it out into a new
  path with the path's paint, stacked just above it and selected afterwards.
  Islands inside the hole go along as holes of the new shape.
- **Draw path (P)**: click to add corners, drag to set mirrored Bézier handles.
  Click the first point or **Close shape** for a filled shape, or press Enter /
  **Finish** for an open stroked path. Backspace removes the last point; Escape
  or **Cancel** discards the draft. Creation is one undoable edit and selects the
  new path for paint and node editing. Middle mouse panning works while drawing.
- Group/ungroup, stacking order, delete and detach shared geometry.
  **Backward**/**Forward** (Ctrl/⌘ `[` / `]`) move one object a step;
  **To back**/**To front** (add Shift) move the selection to the back or front
  of its group, keeping the selected objects' order.
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
  as well. Each drop is one undoable edit.
- **Join paths…** combines selected paths and groups, recursively including
  paths in nested groups and counting overlapping selections only once, even if other objects
  sit between them. The result always occupies the frontmost selected position;
  intervening objects stay in their existing order. Fill/stroke colours and
  numeric paint properties are averaged by clipped painted area (including
  strokes, excluding holes), so larger shapes contribute more. Colour alpha is
  included; fully unpainted selections fall back to equal weights. Clicking
  Join paths opens an options dialog: keep the default area-weighted color mix or
  choose any participating path's fill/stroke colors (including inherited
  colors and fill/stroke alpha). Width and overall opacity stay area-weighted.
  Cancel leaves the document untouched; stacking remains fixed at frontmost.
  Filled overlaps use a curved boolean union, preventing cancellation holes and
  internal seams. Disjoint contours keep exact nodes; boolean intersections
  create new nodes and require affected pins to be removed first. Two pieces
  that abut, such as the halves of a knife cut or parts split apart, merge
  back into one region this way.
  Empty selected group wrappers are removed after joining; groups containing
  non-path shapes must be converted or selected more narrowly first. Undo restores
  the original geometry, paint and stacking. Transforms and per-path clipping
  are resolved into the common parent when joining across groups. The frontmost
  ancestry is split as needed so unselected objects retain their stacking,
  styling and clipping. Area weighting uses the original clipped painted areas.
  Filled regions and stroke-only outlines join separately. Cross-group joins
  reject object-bounding-box clips, clipped stroke-only paths,
  non-uniformly scaled strokes, and group-opacity cases where splitting would
  change unselected artwork. Degenerate clip intersections use a winding-aware
  fallback; its curve tolerance is 0.01 SVG units (straight edges stay exact).
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
- **Cut out as hole** (Structure) takes two selected filled paths. When one
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
- The **Knife** tool (`K`) cuts selected filled paths, or the paths inside
  selected groups, along a straight line: drag across them and release. Shift
  snaps the line to 15° steps. A path is cut only when the line runs through
  it from outside to outside; a line that ends inside a shape or misses it
  leaves it alone, and stroke-only paths are skipped. The cut runs along the
  whole line through that path, so a line across one arm of a U also cuts the
  other arm if the line's extension reaches it. Each cut path becomes one path
  per side of the line (compound when that side has several parts, holes
  kept), with the original's paint, transform, locks and stacking place; the
  first piece keeps its ID. Curves stay curves. The two pieces meet exactly
  on the same seam points but stay independent: dragging a seam point moves
  only that piece, and **Join paths** merges them back into one. Pinned
  points, shared geometry, instances and locks are refused. The pieces are
  selected afterwards, and the cut is one undoable edit. Without a selection
  the knife asks you to select shapes first; a plain click selects.
- The **Redraw outline** tool (`R`) fixes a stretch of one path's outline in
  one gesture, like a magnetic lasso: a missing spike, a notch or a grass
  blade. Select a path, press on its outline (or on one of its points), draw
  roughly along the reference's edge and release on the same contour. Each
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
- **Inspect holes** on a selected drawing path previews its holes. Pick holes
  on the canvas or in the list, zoom to an individual hole with **View**, or
  select all holes up to a maximum area (SVG document units squared).
  **Find shapes inside chosen holes** lists fully enclosed painted objects;
  explicitly check any to delete. Partially overlapping objects stay out of
  this list. Filling and optional deletion are a single undoable action.
  Filling removes the chosen interior contours and their nested islands while
  retaining all other curves exactly. **Turn selected holes into shapes**
  instead moves each chosen hole out into its own path with the path's paint,
  just above it (also available per hole from the node tool). Pins and locks
  are enforced. Ambiguous crossing/coincident contours are
  not offered as holes.
- Geometry/paint/position/structure locks and backend-enforced constraints.
- Undo/redo; a drag is one transaction, not one undo entry per pointer move.

Keyboard shortcuts are available from the `?` button. `V`, `N`, `P`, `K`, and
`H` switch tools. Drag with the middle mouse button (in any tool), or hold Space
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

**Retrace shape** (Shift+R) replaces the outline of each selected
path with a fresh trace of its object in the reference, keeping the path's ID,
paint and place in the stacking order. Draw or keep a rough shape over the
object and press it: the new outline lands as one undoable edit, with a toast
giving the change in reference error. **With SAM** prompts SAM with the path's
box, points inside it and points just outside it that look different, and
takes the mask that agrees with the path, keeps to one colour and has its
outline on the reference's edges; holes in the object stay holes. It needs the
`samvg` extra and a CUDA GPU; without one it retraces **By colour**, which
grows the region from the path's inside over pixels of its colour, within a
margin around it. Either way the region's edges are moved onto the reference's
own and the outline is traced as SAMVG traces its regions. The first retrace
loads SAM (ViT-H) and encodes the reference, a few seconds; later ones on the
same reference reuse both and take under a second for a typical shape, more
for a very large one. The model is released after three idle minutes, when
the reference changes, and before another GPU tool runs. Only visible filled
paths can be retraced; locked geometry, pinned points and shared geometry are
refused (unpin or detach first). It replaces Optimize nodes for fixing a whole
shape; Optimize nodes remains for tidying one.

**Optimize nodes…** is a quick clean-up of the selected paths, or the paths
inside selected groups, against the reference around them, never the whole
image. By default it snaps their points onto the reference's edges and removes
the points they do not need, in a few seconds; to reshape a path, use Retrace
shape (Shift+R) or Redraw outline (R). Tick the steps it may use:

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

**Edit with LLM…** sends the drawing, a render of it, the reference and your
instruction to a multimodal model. Choose which objects and kinds of change are
allowed (shape and position, paint, adding/removing/restacking). The reply is
replayed as ordinary edits: anything outside the chosen objects or permissions
is left out and reported, and locks and pins are enforced. With several
replies, pick one by preview and reference error. The Generate dialog's LLM
method draws the reference from scratch instead. Both need an API key or a local
server, set up under **Settings** in the top bar. A local server is any
OpenAI-compatible endpoint, such as `http://localhost:11434/v1` for Ollama,
with a model that accepts images. Settings also holds each provider's
model and reasoning effort. The editor shows only the last four
characters of a saved key.

**Fit colours…** solves the flat fill colour of every selected object that
best matches the reference, with geometry locked. Each object is rendered with
its fill black and white, which measures its exact coverage (including
antialiasing, opacity, clipping and objects in front), so the best colour has
a closed form; outlines painted in the fill colour follow it. Objects are
fitted back to front; more passes help where fitted objects overlap. No GPU is
needed.

**Clean up geometry…** removes duplicate and collinear vertices and empty or
duplicate paths, and merges compatible neighbouring paths into compound paths,
within the selection only. Paths referenced by instances or clips are kept.
Coordinates are never rounded.

Generate, Improve, Simplify and Snap edges run through the shared operation
contract in `vectrify.operations` (see `docs/operations.md`) via
`POST /api/operation`.

**Generate from reference…** (in the Reference panel) traces the reference into
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
lines (marks narrower and darker than the surface either side; faint narrow
shading, and dark notches as dark as the surface they open into, are left to
the fills), fills the space between them with a shrinking ball so a small gap in
a line does not join the regions either side, splits each region where its
colour changes with no line, and merges neighbours down to the chosen number
of **Regions**, those of a similar colour first and those a drawn line
separates last. The line pixels go to the regions either side, so neighbours
meet at the line's middle; each edge between two regions is traced once and
used by both, so they meet exactly with no gap or overlap. Each region is
coloured from its own pixels, not the lines'. The lines are thinned to
centrelines and drawn over the regions as round-capped strokes, one path per
line colour (two when some lines of a colour are much bolder), each at its
lines' measured width; **Line width** fixes the width instead. With **Trace
lines as strokes** off, or when most lines taper along their length, they are
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

Node handles edit direct `path` elements. Local `use` instances can
be selected, styled, moved and detached; editing a referenced source still
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

Select one object and edit **Object name** in the right panel (or press **F2**).
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
