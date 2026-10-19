# Document model

For developers: `vectrify.document`, the editor's document model, its
transactions and its file formats. It does not depend on Torch.

## Values and identities

`Document`, `Element`, `Geometry`, `Subpath` and `PathNode` are immutable
values that share unchanged data between snapshots. An element has a
stable object id, and a path references a geometry asset by id. Geometry
holds independently identified subpaths and nodes; a cubic node stores
both control points and its endpoint. Moving coordinates keeps every
identity.

Local `use` and `clip-path` references are kept. Dependency traversal
follows groups, references and clipping, so editing a definition cannot
silently change an unselected or locked consumer.
`Document.geometry_users` lists the objects a geometry affects.

`Selection` carries explicit object ids and optional node ids, which must
belong to selected objects. Selecting a group includes its children. An
empty selection permits no changes; `Selection.all()` is the explicit
whole-document choice.

Locks cover `geometry`, `paint`, `transform` or `structure`, or single
attributes such as `fill`. Ancestor locks apply to children and to
referenced definitions. Pinned nodes keep their endpoints fixed in local
coordinates; their handles stay editable, and the whole object can still
be transformed.

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

A transaction captures the revision, selection and permissions when it is
created, edits a private snapshot, and commits all its changes as one undo
entry. An exception aborts it, and a refused command poisons it so an
earlier partial edit cannot be committed by catching the error. `abort()`
discards a preview; a transaction that changes nothing adds no history.

Transactions are strict by default and cannot commit after another
revision. With `rebase=True`, as operation proposals and UI and MCP
commands from older revisions use, commit merges the transaction's field
changes into the current document. Edits to independent objects,
properties or nodes complete together; overlaps, and changes to the locks,
pins, shared consumers or coordinate frames they depend on, reject the
edit atomically.

Revisions only increase, even when undo restores identical content.
Changing the selection is not a revision. Each history entry has a stable
`id` and an `author`, `person` or `agent`. The UI undoes only `person`
entries; MCP names exact ids and can restore either. Restoring applies the
entry's field changes to the current document, keeping unrelated later
edits, and a conflict or missing id rejects the whole batch. A new edit
discards only its own author's redo branch.

Operations cannot unlock: `Editor.set_locks` and `Editor.pin_node` are
separate user commands.

## Shared geometry

`share_geometry(target_id, source_id)` links a path to another geometry
asset; later edits need permission for every visible consumer.
`detach_geometry(object_id)` copies the asset, pins included, with fresh
geometry, subpath and node ids. For a `use` of a path in `defs` it adds a
new definition and retargets only that instance; detaching group instances
is not supported.

## Snapping touching edges

Paths are never linked to one another. `Transaction.snap_edges(tolerance,
object_ids=None)`, in `vectrify.document.contact`, makes neighbouring
paths meet as plain geometry. For each pair of selected paths within the
tolerance (in root user space), the front path is the reference: the rear
path's edges are split where the front's nodes project onto them, and its
matched nodes and handles move onto the front's. An edge snapped for one
pair is not moved again for another. It checks everything before changing
the transaction, so a caller can catch a refusal and carry on, as
`generated_result` does for a trace's seams. It returns an `EdgeRef` for
each matched span, which previews highlight.

## Nodes and remapping

- `split_edge(object_id, node_id, t=0.5)` inserts a node on an edge by De
  Casteljau subdivision, keeping the curve and every existing id.
- `delete_node(object_id, node_id)` removes a node and joins the sequence,
  keeping the next segment's command and controls. Deleting a contour's
  start makes the next node the start. A subpath left too short is removed,
  as by `delete_contour`, and a path left empty is deleted. Pinned nodes
  cannot be deleted.

`node_remapping` and `object_remapping` record old ids and their
replacements (empty for deletion), compose through chained edits, and are
applied to the current selection at commit. If deletion removes the last
selected node, object scope is cleared, so a node filter never becomes
permission to edit a whole object.

## Objects and stacking

Structural commands need whole-object selection and `structure`
permission. Ancestor locks and dependent consumers are checked as for
geometry.

- `insert_object(parent_id, element, index=None, geometries=())` inserts
  a subtree with unique, validated ids.
- `delete_objects(object_ids, allow_pinned=False)` refuses pinned content
  unless confirmed, and never leaves dangling references.
- `reorder_object(object_id, index)` moves an object among its siblings;
  zero is the back.
- `move_objects(object_ids, parent_id, index)` moves objects into a
  container, writing inherited transform and paint onto them.
- `group_objects(object_ids)` and `ungroup_object(object_id)` group at the
  frontmost selected position, and ungroup by distributing transform,
  paint, opacity and clipping to the children.
- `cut_paths(start, end)` cuts filled paths the segment crosses from
  outside to outside, with Skia's curved booleans, restoring exact
  coordinates at the crossings. The first piece keeps the ids, and pinned
  endpoints keep theirs.

## Painted areas

`HitIndex(document, tolerance=0.1)` computes where each object of a
snapshot paints, with Shapely. `area(object_id)` is that shape, a group's
being the union of its children's; `bounds(object_ids)` is the box around
several. Areas follow holes, strokes, transforms and clipping, ignore
occlusion, and are in root user space, approximated to `tolerance`
document units.

## Gradient paint

A fill or stroke is a colour, `none`, or `url(#id)` naming a
`linearGradient` in `defs`. A gradient holds only `stop` children and takes
`x1`, `y1`, `x2`, `y2`, `gradientUnits`, `gradientTransform` and
`spreadMethod`. On import, a gradient inheriting from another by `href` is
resolved so each stands alone.

`Transaction.set_fill(object_id, fill)` sets a colour, or a private
`LinearGradient` from `vectrify.document.paint`. A private gradient names
its shape in `Element.paint_owner`, is shown through the shape's fill
rather than under Definitions, and is collected when the shape stops using
it. Imported gradients without an owner stay shared and are never
collected implicitly; replacing a shared fill creates a private one.
`solid_paint` gives a gradient's mean colour where one colour is needed.

## Import, export and project files

The importer accepts roots, groups, definitions, clipping, paths,
rectangles, circles, ellipses, lines, polygons, polylines and local
instances. Path data is normalized to absolute M, L, C and Z without
rounding, and polygons and polylines become paths. Solid paint, linear
gradients, opacity, stroke width, caps and joins, affine transforms and
local clipping are supported; lengths are unitless. Arcs, text, radial
gradients, filters, patterns, arbitrary CSS, nested viewports, external
references and XML entity or doctype declarations are refused with a list
of what is unsupported.

`export_svg` writes ordinary SVG with object ids. Private gradient
ownership is kept in `data-vectrify-paint-owner`. SVG does not carry node
identities, locks, pins or shared geometry; `save_project(document,
selection)` and `load_project(json_text)` do. Version 4 is written, and
versions 1 to 3 load. Shared boundary links in version 2 files are dropped
on load, keeping the contours.
