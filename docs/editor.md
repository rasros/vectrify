# Local SVG editor

Start the editor with:

```sh
uv run vectrify-ui
```

Or open a specific SVG and its reference image:

```sh
uv run vectrify-ui drawing.svg --reference original.png --port 8765
```

Open the printed localhost address in a browser. The server listens on loopback
only. The UI is included in the Python package and needs no Node build step.
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
- **Draw path (P)**: click to add corners, drag to set mirrored Bézier handles.
  Click the first point or **Close shape** for a filled shape, or press Enter /
  **Finish** for an open stroked path. Backspace removes the last point; Escape
  or **Cancel** discards the draft. Creation is one undoable edit and selects the
  new path for paint and node editing. Middle mouse panning works while drawing.
- Group/ungroup, stacking order, delete and detach shared geometry.
- **Join outlines** combines selected paths and groups, recursively including
  paths in nested groups and counting overlapping selections only once, even if other objects
  sit between them. The result always occupies the frontmost selected position;
  intervening objects stay in their existing order. Fill/stroke colours and
  numeric paint properties are averaged by clipped painted area (including
  strokes, excluding holes), so larger shapes contribute more. Colour alpha is
  included; fully unpainted selections fall back to equal weights. Clicking
  Join opens an options dialog: keep the default area-weighted color mix or
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
  currently reject object-bounding-box clips, clipped stroke-only paths,
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

## Current scope

This is the first manual editor, brought forward from step 5 of the UI plan.
Improve now exposes **Optimize path**: select a path, open the popup, and run the
existing `Mutation: path fit` filled-path GPU optimizer. Choose whether nodes,
handles and color may change, the step budget (8 by default), maximum movement
in local SVG units, and the working crop resolution. Pinned endpoints, node
selection and property locks are enforced. Lines, holes and node identities are
preserved. Matching fill/stroke colors are fitted together on closed paths with
round or miter joins; stroke width and join style stay fixed. Miter joins
respect the inherited miter limit, falling back to bevels at over-limit corners.
Temporary GPU stroke outlines preserve joins across processing chunks; the
original SVG curves remain editable, and candidates still pass the actual SVG
rendering check. Separate-color outlines, open stroked
paths, shared geometry and object-bounds clips are not supported by this first
adapter. Detach an instance before fitting it independently.

The operation runs on a snapshot in a background job, with progress,
Stop & keep best, and reference/before/after previews. Apply is one undoable edit;
Discard leaves the drawing untouched. Results cannot overwrite changes to the
drawing or reference. Reference error is RGB MSE measured with Cairo in the
shown crop at the chosen resolution; an unchanged candidate is retained if no
checked candidate improves it. This is a resolution-specific comparison, not a
full-resolution quality guarantee. Surrounding artwork, clipping, group opacity
and objects in front are included in the frozen compositing context. Fitting
requires PyTorch CUDA and the optional native Vectrify CUDA extension.

**Search improvements…** runs NSGA-II local search against the reference on
the selection or the whole drawing. Choose what may change (shape and
position, paint, stacking order), the number of candidates, workers and pool
size. Only the chosen objects are mutated; locks, pins and unselected objects
stay fixed because every result is replayed as an ordinary edit. The results
are ranked by pixel error against the reference: pick the recommendation or an
alternative, compare previews, and Apply it as one undoable edit. Stop keeps
the best candidates found so far. No LLM call or GPU fitting is involved.

**Edit with LLM…** sends the drawing, a render of it, the reference and your
instruction to a multimodal model. Choose which objects and kinds of change are
allowed (shape and position, paint, adding/removing/restacking). The reply is
replayed as ordinary edits: anything outside the chosen objects or permissions
is left out and reported, and locks and pins are enforced. With several
replies, pick one by preview and reference error. The Generate dialog's LLM
method draws the reference from scratch instead. Both need an API key in the
environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY` or `GEMINI_API_KEY`).

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

**Smooth / simplify…** reduces selected paths, path instances or groups of paths
without a reference image. It fits shorter runs of lines/cubic curves to the
current contour, using an adjustable approximation tolerance in local SVG
units. Keep sharp corners is enabled by default; pinned endpoints always remain
exact. Preview shows the drawing before and after, node/coordinate counts and
serialized path-data reduction. Changing settings requires a new preview.
Apply is a single undoable transaction; cancelled or stale previews never edit
the drawing. Paint, stacking, contour identities, hole containment and contact
relationships remain intact. Explicit self-touching junctions are held in place; ambiguous edge crossings
are kept unchanged. Fits that increase coordinate count, node count or path-data size are
not accepted. This operation requires geometry and structure permissions;
shared assets require selecting all affected consumers or detaching first.
Linked boundaries must be detached explicitly before simplifying.

Improve, Simplify and Share boundary run through the shared operation
contract in `vectrify.operations` (see `docs/operations.md`) via
`POST /api/operation`.

**Generate from reference…** (in the Reference panel) traces the reference into
new shapes. The SAMVG method segments the image with SAM and traces each region
into filled paths and thin strokes. Choose the model (ViT-H is best, ViT-B is
faster), the maximum layers and curve segments. The result is placed over the
artboard exactly where the reference is shown, as one new group at the front of
the whole drawing or of a selected group. With a focus region set, only that
part of the reference is traced. Text is traced as shapes, since the editor
does not support SVG text. Preview shows the reference, before and after, with
the change in reference error; Apply adds the group as one undoable edit.
SAMVG needs the `samvg` extra and holds the GPU while it runs.

The Colour regions method fits a palette on the GPU and traces each colour
region, with an optional dark-outline layer ("Preserve dark linework") or clean
mode ("Clean shared regions and ink"), which defines each region once and
reuses it for fill and clip. Geometry cleanup merges compatible paths and drops
redundant vertices afterwards. It needs CUDA and the `vision` extra.

Node handles currently edit direct `path` elements. Local `use` instances can
be selected, styled, moved and detached; editing a referenced source still
requires selecting every affected consumer, as enforced by the backend. Groups
with compositing or reference relationships that cannot be ungrouped without
changing appearance return a clear error. Cutting a continuous path, welding endpoints and additional
additional shape tools remain future manual-editing work.

Only the documented static SVG subset is accepted. Unsupported imports are
reported instead of silently dropping content. The editor namespaces SVG IDs
inside the page so artwork cannot collide with editor controls. The local HTTP
adapter rejects external origins, unknown sessions and stale document revisions.

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

The first version supports direct paths,
without clipping or existing shared boundaries. It matches line-to-line and
cubic-to-cubic spans, rather than rebuilding mismatched contour types. Moving a
linked region separately requires unlinking first. Project files retain the
editing links; exported SVG retains the coincident contours but not the links.

Boundary matching measures contact distance in canvas SVG units, including
paths in differently transformed groups. Linked edits convert between each
path’s local coordinates; transforms, paint and stacking remain intact.
