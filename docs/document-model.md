# Document model

`vectrify.document` is the editor's document model. It is independent of the
search, LLM clients and Torch. Automated operations reach it through
`vectrify.operations` (see `docs/operations.md`).

## Editing state

`Document`, `Element`, `Geometry`, `Subpath`, and `PathNode` are immutable values.
An element has a stable object ID; paths reference geometry assets by ID.
Geometry contains independently identified subpaths and segment endpoints. A
cubic node stores both control points and its endpoint. Changing coordinates
preserves all those identities. Immutable snapshots share unchanged data.

Local SVG `use` and `clip-path` references are retained. Dependency traversal
includes groups, references, and clipping, so modifying a definition cannot
silently change an unselected or locked consumer. The affected objects can be
queried with `Document.geometry_users`.

`Selection` carries explicit object IDs and optional node IDs. Selecting a
group includes its children. An empty selection permits no changes.
`Selection.all()` is the explicit whole-document choice. Node IDs must belong
to selected objects.

Locks can cover `geometry`, `paint`, `transform`, or `structure`, or individual
attributes such as `fill` and `stroke-width`. Ancestor locks apply to children,
including referenced definitions. Pinned nodes keep their endpoints fixed in
local geometry coordinates; their cubic handles remain editable. Pins do not
prevent moving an entire object through a permitted transform.

## Transactions

```python
from vectrify.document import Editor, Selection, import_svg, export_svg

editor = Editor(import_svg(source_svg))
editor.select(Selection(object_ids=frozenset({"mountain"})))

with editor.transaction("Adjust colour", allowed=frozenset({"fill"})) as edit:
    edit.set_attributes("mountain", {"fill": "#aabbcc"})

geometry = editor.snapshot.document.geometry_for("mountain")
node = geometry.subpaths[0].nodes[1]
editor.pin_node("mountain", geometry.subpaths[0].nodes[0].id)

with editor.transaction("Refine contour") as edit:
    # For a cubic, supply all six coordinates (two handles and the endpoint).
    moved = (*node.values[:-2], node.endpoint[0] + 0.5, node.endpoint[1])
    edit.update_node("mountain", node.id, moved)
    preview_svg = export_svg(edit.preview)

editor.undo()
editor.redo()
```

Transactions capture document revision, selection, and edit permissions when
created. They edit a private snapshot and apply all changes as one undo entry.
An exception aborts the transaction. A rejected command poisons the transaction
so catching the exception cannot accidentally commit an earlier partial edit.
Explicit `abort()` discards a preview; no-op transactions create no history.

A transaction cannot commit after any document edit, lock/pin change, undo, or
redo. Revisions increase monotonically even when undo restores identical
content. Changing UI selection does not change the document revision or
retarget an existing transaction. A new edit after undo discards the redo
branch. Undo/redo applies within one editor session.

Operations cannot unlock objects: `Editor.set_locks` and `Editor.pin_node` are
separate explicit user commands with their own undo history. Properties,
coordinates, or references cannot be mutated through the read-only snapshot.

## Shared geometry

`share_geometry(target_id, source_id)` explicitly links a path to another
geometry asset. Later geometry edits require permission for every visible
consumer. Locks on definitions and consumers are both respected. Sharing cannot
replace geometry that contains pinned endpoints.

`detach_geometry(object_id)` copies an asset, including pins, with fresh
geometry/subpath/node IDs while retaining the selected object ID. For a `use`
referencing a path in `defs`, it creates a new definition alongside the source
and retargets only that instance. This preserves inherited styles, transforms,
and clipping. Detaching group instances or references outside `defs` is not
implemented and returns a clear error.

`EdgeRef(geometry_id, node_id, reversed=False)` identifies the incoming edge of
an endpoint. Referencing a moveto identifies the implicit closing line of a
closed subpath. `link_boundary(members)` records an explicit shared boundary
between coincident lines or cubics, including reversed traversals. Existing
coordinates must match exactly after orientation; linking never snaps geometry
or guesses adjacency. An edge belongs to at most one boundary.

```python
from vectrify.document import EdgeRef

fill = editor.snapshot.document.geometry_for("fill")
outline = editor.snapshot.document.geometry_for("outline")
editor.select(Selection(object_ids=frozenset({"fill", "outline"})))
with editor.transaction("Link contour") as edit:
    edit.link_boundary((
        EdgeRef(fill.id, fill.subpaths[0].nodes[1].id),
        EdgeRef(outline.id, outline.subpaths[0].nodes[1].id, reversed=True),
    ))
```

Editing an endpoint or handle propagates to equivalent coordinates, including
junctions connected through several boundaries. Every changed node must pass
selection and pin checks, and all consumers must pass geometry permissions and
inherited locks. A rejected propagated edit rejects the whole transaction.
`detach_boundary(member)` explicitly unlinks one traversal without changing its
coordinates or IDs. Remaining members stay linked when at least two remain.
Detaching a whole geometry asset also drops its boundary membership if the old
asset no longer has any consumers.

Boundary members are compared in a common frame: each `EdgeRef` carries a
`matrix` that maps its local geometry coordinates into it, so paths under
different transforms can share a boundary. `link_boundary` itself never finds
or snaps edges; `Transaction.share_boundaries(tolerance)` (the `link/boundaries`
method, in `vectrify.document.contact`) finds the touching spans of every
pair of two or more selected paths whose bounds come within the tolerance,
subdivides them, snaps the rear contour of each pair to the front one and links
them. Edges already in a boundary are never matched again, and a span whose
snapping would move a slot of a linked edge is skipped, so existing links stay
valid. It checks everything before changing the transaction, so a caller can
catch a refusal and carry on; SAMVG's Generate does that to link the seams of a
flattened trace (`object_ids` names the inserted paths, and geometry the
transaction added itself needs no geometry permission to snap).

## Node topology and selection remapping

`split_edge(object_id, node_id, t=0.5)` inserts a node on a line, cubic, or closing
edge. Cubics use De Casteljau subdivision, preserving the curve to floating-point
precision. Every linked traversal splits at the corresponding point, including
opposite directions. Existing endpoints, pins, subpaths, and geometry IDs remain
stable. The method returns the inserted node IDs. Splitting needs both geometry
and structure permission; with a node filter, every affected incoming edge's
endpoint must be selected. Existing node selections stay on their original IDs.

`delete_node(object_id, node_id)` removes a non-moveto node and joins the remaining
sequence, retaining the next segment's command and controls. This intentionally
changes the curve; it is a manual deletion command, not an approximation or
simplification algorithm. Pinned nodes cannot be deleted. Adjacent shared edges
must first be detached explicitly.

`Transaction.preview_selection` previews remapping of the captured selection.
`node_remapping` exposes old IDs and their replacement IDs (empty for deletion).
Detaching geometry maps nodes onto the clone, retaining old IDs too when another
selected consumer still uses them. Chained edits compose the mapping. At commit,
remapping applies to the current UI selection rather than overwriting a selection
changed while the preview was open. If deletion removes the last selected node,
object scope is cleared: a node filter can never silently become permission to
edit an entire object. Undo/redo restores the corresponding selections.

## Objects and stacking

Structural commands require whole-object selection and `structure` permission.
Ancestor locks and dependent `use`/clip consumers are checked, just as for
geometry edits. They participate in the same preview/apply/abort transaction.

- `insert_object(parent_id, element, index=None, geometries=())` inserts an
  identified subtree into an explicitly selected container. New IDs must be
  unique; path assets and references are validated. New objects can be edited
  further within the same transaction.
- `delete_objects(object_ids)` removes explicit subtrees, rejecting pinned
  content and dangling references. It does not silently delete other consumers.
- `reorder_object(object_id, index)` moves one selected object to a final sibling
  index; zero is the back. Other siblings retain their relative ordering. This
  can change occlusion.
- `group_objects(object_ids)` wraps consecutive selected siblings in a neutral
  group, preserving IDs, paint order and appearance. Nonconsecutive objects
  require an explicit reorder first.
- `ungroup_object(object_id)` transfers inherited paint and composes transforms
  onto children. Groups with nontrivial opacity, clipping, locks or references
  that cannot be transferred safely return an explicit error. Group opacity
  cannot generally be distributed without changing overlapping children.

`object_remapping` records deletion and group-to-child replacement. Like node
remapping, it composes through batches and applies to the current UI selection
at commit. Grouping leaves existing child selections intact; ungrouping a
selected group selects its surviving children. Undo restores the previous
objects and selection. Object commands cannot bypass a node-only scope.

## Painted areas

`HitIndex(document, tolerance=0.1)` computes where each object of an immutable
snapshot paints. `area(object_id)` is that shape (None if it paints nothing),
a group's being the union of its children's; `bounds(object_ids)` is the box
around several, which operations use to crop the reference to the selection.
Join and hole inspection use the same areas.

Areas follow painted geometry, including holes, strokes, transforms, nested
clips, and `objectBoundingBox` clipping. Transparent objects paint nothing.
Occlusion by other objects is ignored. Coordinates are root SVG user-space/viewBox coordinates,
not viewport or screen pixels. Curves and round geometry are approximated with
the supplied document-unit tolerance; changing zoom does not change scope unless
the caller deliberately changes that tolerance. Boolean geometry uses Shapely 2.
An index always describes the snapshot it was built from.

## Import, export, and project files

The importer accepts SVG roots, groups, definitions, clipping, paths,
rectangles, circles, ellipses, lines, polygons, polylines, and local instances.
Paths normalize relative coordinates, horizontal/vertical lines, and quadratic
and shorthand Beziers to absolute M/L/C/Z without coordinate rounding.
Polygons and polylines become paths. Compound paths retain holes and closure.

Solid paint, opacity, stroke width/caps/joins, affine transforms, and local
clipping are supported. Lengths are unitless document coordinates.
Arcs, text, gradients, filters, patterns, arbitrary CSS, nested viewports,
external references, and other unsupported input produce import errors listing
the unsupported features. Content is never silently removed. XML declarations
of entities or document types are rejected.

`export_svg` creates ordinary SVG containing object IDs. SVG alone does not
encode editor node identities, locks, pins, or explicit asset-sharing metadata.
Use `save_project(document, selection)` and `load_project(json_text)` for a
versioned project snapshot retaining those identities, constraints, and current
selection. Version 2 adds explicit boundary links and still reads version 1
projects. Plain SVG export retains the appearance but not boundary metadata.
Project loading validates references and the supported subset.
