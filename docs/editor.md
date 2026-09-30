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
  Delete/Backspace in Nodes affects only the selected point. Select one path,
  then click a point; drag blue handles for curves. Zoom in to reveal dense nodes.
  **No handles**, **One handle** and **Two handles** turn the selected point
  into a corner, a point curved on one side (click again to switch sides), or
  a smooth point with its handles in line.
- **Draw path (P)**: click to add corners, drag to set mirrored Bézier handles.
  Click the first point or **Close shape** for a filled shape, or press Enter /
  **Finish** for an open stroked path. Backspace removes the last point; Escape
  or **Cancel** discards the draft. Creation is one undoable edit and selects the
  new path for paint and node editing. Middle mouse panning works while drawing.
- Group/ungroup, stacking order, delete and detach shared geometry.
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
  create new nodes and require affected pins/boundary links to be removed first.
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
  curves, pins and boundary links are preserved. A group retains styling,
  transforms and clipping. Undo restores the compound path in one step.
  Only geometric contact or overlap connects contours; gaps are not bridged,
  regardless of stroke width, caps or joins. Cubic curves are approximated at
  0.01 document units for classification; output curves remain exact. Shared assets must be
  detached first. **Detach to editable path** converts a path instance into a
  selected independent path ready to split, retaining its paint and transforms
  in wrapper groups. Other instances keep their shared definition.
- **Inspect holes** on a selected drawing path previews its holes. Pick holes
  on the canvas or in the list, zoom to an individual hole with **View**, or
  select all holes up to a maximum area (SVG document units squared).
  **Find shapes inside chosen holes** lists fully enclosed painted objects;
  explicitly check any to delete. Partially overlapping objects stay out of
  this list. Filling and optional deletion are a single undoable action.
  Filling removes the chosen interior contours and their nested islands while
  retaining all other curves exactly. Pins, locks and linked boundaries are
  enforced. Ambiguous crossing/coincident contours are not offered as holes.
- Geometry/paint/position/structure locks and backend-enforced constraints.
- Undo/redo; a drag is one transaction, not one undo entry per pointer move.

Keyboard shortcuts are available from the `?` button. `V`, `N`, `P`, and `H`
switch tools. Drag with the middle mouse button (in any tool), or hold Space
and drag, to pan. Use the scroll wheel to zoom, and press `F` to
fit. Ctrl/Command-Z undoes; add Shift to redo. Ctrl/Command-S saves a project.

Projects preserve object/node identities, shared boundaries, locks, pins,
selection and the reference image. SVG exports contain the drawing. Downloads
use the browser's download location and do not overwrite the original input.
The server keeps sessions in memory; save a project before stopping it. Browser
reloads reconnect to the current session, but undo history is not stored in
project files. Saving also keeps a browser recovery copy keyed to this tab's
session; after a server restart, the tab restores that saved project. Edits made
after the last save are not included in recovery. **Restore saved** also lets
you choose a saved browser recovery copy from a new tab.

## Operations

**Optimize nodes…** fits the selected paths, or the paths inside selected
groups, to the reference around them, never the whole image. Tick the steps it
may use:

- **Fit shape** moves points and curve handles by gradient descent, with a
  number of fitting steps and a maximum movement per fit in local SVG units.
  It runs on the GPU when PyTorch CUDA and the Vectrify CUDA extension are
  there, and on the CPU otherwise; outlined (stroked) fills need the GPU.
- **Snap to reference** moves the points onto the reference's nearest edges.
  With **Add detail** it also adds points where the path misses a piece of the
  shape or covers too much: one point, two, or a spike of three whose base
  stays on the outline, reaching bit by bit along a strand that curls away.
- **Simplify** removes the points the outline does not need, moving it no more
  than the tolerance in reference pixels.

Each round tries every ticked step on the paths as they stand and keeps the one
that brings them closest to the reference; when none helps, Simplify gets its
turn, and the run ends once nothing changes or the rounds run out. So a rough
shape can be snapped, fitted, thinned and fitted again in whatever order works.
With several parallel workers a round's steps run side by side, with one shape
fit at a time. Without a reference only Simplify runs, keeping the paths'
look. Pinned endpoints and linked boundary edges stay fixed, and surviving
points keep their identity. Colour is left to Fit colours. The job runs in the
background with progress, Stop & keep best, and reference/before/after
previews; the result lists the steps it took. Apply is one undoable edit.

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

Generate, Improve, Simplify and Share boundary run through the shared operation
contract in `vectrify.operations` (see `docs/operations.md`) via
`POST /api/operation`.

**Generate from reference…** (in the Reference panel) traces the reference into
new shapes. The SAMVG method segments the image with SAM and traces each region
into filled paths. Choose the model (ViT-H is best, ViT-B is faster), the
resolution SAM segments at (higher traces a large image with smoother edges;
a smaller reference is enlarged to it first, so outlines do not follow its
pixels),
the maximum shapes and curves per outline, whether to fill small holes, and the
thinnest region to keep: regions narrower than that everywhere, such as
outlines and hairlines, are left out (0 keeps everything). Regions hidden entirely by
those above are left out, and a backdrop rectangle beneath them all, in the
colour of what no region claims (usually the drawn outlines), fills the gaps
between regions. **Merge small patches** (on by default) joins neighbouring regions whose
difference is not worth a shape of its own, such as the small patches SAM
leaves along edges or one region cut in pieces, and recolours the result.
**Flatten overlaps** cuts every region down to its visible
part, so none overlap; the thin strips that cutting leaves along edges go to
a neighbouring region instead of becoming shapes of their own. Outlines are
smoothed over about one SAM pixel before curves are fitted, so they do not
follow the masks' raster steps. The result is placed over the
artboard exactly where the reference is shown, as one new group at the front of
the whole drawing or of a selected group. With a group selected, only the
reference around what it already paints is traced. Text is traced as shapes, since the editor
does not support SVG text. Preview shows the reference, before and after, with
the change in reference error; Apply adds the group as one undoable edit.
SAMVG needs the `samvg` extra and holds the GPU while it runs.

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
changing appearance return a clear error. There are no tools for cutting a
continuous path or welding endpoints.

Only the documented static SVG subset is accepted. Unsupported imports are
reported instead of silently dropping content. The editor namespaces SVG IDs
inside the page so artwork cannot collide with editor controls. The local server
rejects external origins, unknown sessions and stale document revisions.

## Names and shared boundaries

Select one object and edit **Object name** in the right panel (or press **F2**).
Enter or leaving the field applies the name; Escape cancels. Clearing the field
restores its automatic label. Names support undo/redo and survive project saves
and SVG export/reimport, without changing object IDs or shared references.

**Share boundary…** matches touching edges of two selected, closed paths.
Adjust the contact distance, preview the cyan highlighted shared spans, then
apply. The frontmost contour is the reference; the other region snaps to it.
Curves are subdivided to accommodate different node spacing without flattening.
Shared endpoints and curve handles propagate direct node edits to the linked
region, subject to every region's pins and locks. **Unlink boundaries** removes
these constraints without changing the geometry; undo restores them.

Both paths must be closed and unclipped, and neither may share geometry or
already have linked boundaries. It matches line-to-line and
cubic-to-cubic spans, rather than rebuilding mismatched contour types. Moving a
linked region separately requires unlinking first. Project files retain the
editing links; exported SVG retains the coincident contours but not the links.

Boundary matching measures contact distance in canvas SVG units, including
paths in differently transformed groups. Linked edits convert between each
path’s local coordinates; transforms, paint and stacking remain intact.
