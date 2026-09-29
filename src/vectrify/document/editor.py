"""Selection-constrained transactions and monotonic revision history."""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, replace

from vectrify.document.components import disconnected_parts
from vectrify.document.holes import find_holes
from vectrify.document.join import (
    bake_group_path,
    join_paint,
    painted_weights,
    path_style,
    union_geometry,
)
from vectrify.document.model import (
    Document,
    DocumentError,
    EdgeRef,
    EditKind,
    Element,
    Geometry,
    Selection,
    SharedBoundary,
    new_id,
    references,
)
from vectrify.document.svg import GEOMETRY, PAINT, validate_attributes
from vectrify.document.topology import edge, propagate_node, split_edges


def _on_chord(
    h: tuple[float, ...], a: tuple[float, float], b: tuple[float, float]
) -> bool:
    """Whether handle *h* lies on the straight line from *a* to *b*, so that a
    curve with its other handle retracted is just that line."""
    ax, ay = b[0] - a[0], b[1] - a[1]
    hx, hy = h[0] - a[0], h[1] - a[1]
    length = math.hypot(ax, ay)
    if length == 0:
        return math.hypot(hx, hy) < 1e-9
    along = (hx * ax + hy * ay) / length
    return abs(hx * ay - hy * ax) / length < 1e-6 * max(1.0, length) and (
        -1e-9 <= along <= length + 1e-9
    )


def _toward(
    a: tuple[float, float], b: tuple[float, float], share: float
) -> tuple[float, float]:
    """The point *share* of the way from *a* to *b*."""
    return a[0] + (b[0] - a[0]) * share, a[1] + (b[1] - a[1]) * share


class EditRejectedError(DocumentError):
    """An edit violates scope, locks, or node constraints."""


class StaleRevisionError(EditRejectedError):
    """The editor changed after the transaction captured its snapshot."""


@dataclass(frozen=True)
class Snapshot:
    revision: int
    document: Document
    selection: Selection


@dataclass(frozen=True)
class HistoryEntry:
    label: str
    before: Document
    after: Document
    before_selection: Selection
    after_selection: Selection


class Editor:
    def __init__(self, document: Document, *, selection: Selection | None = None):
        selection = selection or Selection()
        document.validate()
        document.selection_ids(selection)
        self._document = document
        self._selection = selection
        self._revision = 0
        self._undo: list[HistoryEntry] = []
        self._redo: list[HistoryEntry] = []

    @property
    def snapshot(self) -> Snapshot:
        return Snapshot(self._revision, self._document, self._selection)

    @property
    def undo_labels(self) -> tuple[str, ...]:
        return tuple(entry.label for entry in self._undo)

    @property
    def redo_labels(self) -> tuple[str, ...]:
        return tuple(entry.label for entry in reversed(self._redo))

    def select(self, selection: Selection) -> None:
        self._document.selection_ids(selection)
        self._selection = selection

    def transaction(
        self,
        label: str,
        *,
        selection: Selection | None = None,
        allowed: frozenset[str] = frozenset(EditKind),
        expected_revision: int | None = None,
        base: Snapshot | None = None,
    ) -> Transaction:
        """Open an edit; with *base*, edit that earlier snapshot instead.

        A transaction over an earlier snapshot can be built off the editor's
        thread and commits only if no revision has happened since.
        """
        if expected_revision is not None and expected_revision != self._revision:
            raise StaleRevisionError("Document revision has changed")
        base = base or self.snapshot
        return Transaction(self, label, selection or base.selection, allowed, base)

    def _apply(
        self,
        document: Document,
        label: str,
        revision: int,
        selection: Selection | None = None,
    ) -> Snapshot:
        if revision != self._revision:
            raise StaleRevisionError("Document revision has changed")
        document.validate()
        selection = selection if selection is not None else self._selection
        document.selection_ids(selection)
        if document != self._document:
            self._undo.append(
                HistoryEntry(
                    label, self._document, document, self._selection, selection
                )
            )
            self._redo.clear()
            self._document = document
            self._selection = selection
            self._revision += 1
        return self.snapshot

    def rename_object(self, object_id: str, name: str) -> Snapshot:
        """User-facing metadata; identities, references and edit locks stay intact."""
        if object_id not in self._selection.object_ids:
            raise DocumentError("Select the object before renaming it")
        if not isinstance(name, str):
            raise DocumentError("Object name must be text")
        element = self._document.element(object_id)
        document = self._document.replace_element(replace(element, name=name.strip()))
        return self._apply(document, "Rename object", self._revision)

    def set_locks(self, object_id: str, locks: frozenset[str]) -> Snapshot:
        """Explicit user command; operation transactions cannot unlock objects."""
        element = self._document.element(object_id)
        known = set(EditKind) | PAINT | GEOMETRY[element.tag] | {"transform"}
        if not locks <= known:
            raise DocumentError("Unknown property lock")
        document = self._document.replace_element(
            replace(element, locks=frozenset(locks))
        )
        return self._apply(document, "Set property locks", self._revision)

    def pin_node(
        self, object_id: str, node_id: str, *, pinned: bool = True
    ) -> Snapshot:
        """Pin an endpoint in geometry coordinates, including shared instances."""
        geometry = self._document.geometry_for(object_id)
        node = geometry.node(node_id)
        document = self._document.replace_geometry(
            geometry.replace_node(replace(node, pinned=pinned))
        )
        return self._apply(
            document, "Pin node" if pinned else "Unpin node", self._revision
        )

    def undo(self) -> Snapshot:
        if self._undo:
            entry = self._undo.pop()
            self._redo.append(entry)
            self._document, self._selection = entry.before, entry.before_selection
            self._revision += 1
        return self.snapshot

    def redo(self) -> Snapshot:
        if self._redo:
            entry = self._redo.pop()
            self._undo.append(entry)
            self._document, self._selection = entry.after, entry.after_selection
            self._revision += 1
        return self.snapshot


class Transaction:
    def __init__(
        self,
        editor: Editor,
        label: str,
        selection: Selection,
        allowed: frozenset[str],
        base: Snapshot | None = None,
    ):
        self._editor = editor
        self._base = base or editor.snapshot
        self._working = self._base.document
        self._selection = selection
        self._ids = self._working.selection_ids(selection)
        self._allowed = frozenset(allowed)
        self._label = label
        self._closed = False
        self._failed = False
        self._node_remap: dict[str, set[str]] = {}
        self._object_remap: dict[str, set[str]] = {}

    @property
    def preview(self) -> Document:
        return self._working

    @property
    def allowed(self) -> frozenset[str]:
        """The edit kinds and attributes this transaction may change."""
        return self._allowed

    @property
    def scope(self) -> frozenset[str]:
        """Object IDs this transaction may edit (the selection and descendants)."""
        return frozenset(self._ids)

    @property
    def preview_selection(self) -> Selection:
        return self._remap_selection(self._selection)

    @property
    def node_remapping(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Old node to replacement IDs; empty means removed, including edit chains."""
        return tuple(
            (old, tuple(sorted(new))) for old, new in sorted(self._node_remap.items())
        )

    def _record_remap(self, mapping: dict[str, set[str]]) -> None:
        for old, targets in self._node_remap.items():
            self._node_remap[old] = set().union(*(mapping.get(n, {n}) for n in targets))
        for old, targets in mapping.items():
            self._node_remap.setdefault(old, set()).update(targets)

    @property
    def object_remapping(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return tuple(
            (old, tuple(sorted(new))) for old, new in sorted(self._object_remap.items())
        )

    def _record_object_remap(self, mapping: dict[str, set[str]]) -> None:
        for old, targets in self._object_remap.items():
            self._object_remap[old] = set().union(
                *(mapping.get(n, {n}) for n in targets)
            )
        for old, targets in mapping.items():
            self._object_remap.setdefault(old, set()).update(targets)

    def _remap_selection(self, selection: Selection) -> Selection:
        existing = {e.id for e in self._working.elements()}
        objects = set(selection.object_ids)
        for old in selection.object_ids:
            if old in self._object_remap:
                objects.discard(old)
            objects.update(self._object_remap.get(old, ()))
        selection = replace(selection, object_ids=frozenset(objects & existing))
        if not selection.node_ids:
            return selection
        object_scope = replace(selection, node_ids=frozenset())
        available = set()
        for object_id in self._working.selection_ids(object_scope):
            try:
                geometry = self._working.geometry_for(object_id)
            except DocumentError:
                continue
            available.update(n.id for s in geometry.subpaths for n in s.nodes)
        candidates = set(selection.node_ids)
        for old in selection.node_ids:
            candidates.update(self._node_remap.get(old, ()))
        selected = frozenset(candidates & available)
        # Losing the final selected node must never broaden the scope to an object.
        return replace(selection, node_ids=selected) if selected else Selection()

    @contextmanager
    def _change(self):
        if self._closed or self._failed:
            raise EditRejectedError("Transaction is closed or has a failed edit")
        self._failed = True
        yield
        self._working.validate()
        self._failed = False

    def _check_locks(
        self, object_id: str, kind: EditKind, attribute: str | None = None
    ):
        for element in self._working.ancestry(object_id):
            if kind in element.locks or (
                attribute is not None and attribute in element.locks
            ):
                raise EditRejectedError(
                    f"{object_id}: {attribute or kind.value} is locked"
                )

    def _authorize(
        self, affected: frozenset[str], kind: EditKind, attribute: str | None = None
    ):
        if kind not in self._allowed and attribute not in self._allowed:
            raise EditRejectedError(
                f"Changing {attribute or kind.value} is not permitted"
            )
        visible = {
            item
            for item in affected
            if not any(
                e.tag in {"defs", "clipPath"} for e in self._working.ancestry(item)
            )
        }
        required = visible or set(affected)
        if not required <= self._ids:
            raise EditRejectedError(
                "Edit affects unselected objects; select all dependents "
                "or detach shared geometry"
            )
        for item in affected:
            self._check_locks(item, kind, attribute)

    def set_attributes(self, object_id: str, changes: dict[str, str | None]) -> None:
        with self._change():
            element = self._working.element(object_id)
            attrs = dict(element.attributes)
            affected = self._working.dependents({object_id})
            for key, value in changes.items():
                if key in {"href", "clip-path", "id", "d"}:
                    raise EditRejectedError(
                        "References and path topology need explicit structural commands"
                    )
                if attrs.get(key) == value:
                    continue
                kind = (
                    EditKind.PAINT
                    if key in PAINT
                    else EditKind.TRANSFORM
                    if key == "transform"
                    else EditKind.GEOMETRY
                )
                if self._selection.node_ids and kind != EditKind.PAINT:
                    raise EditRejectedError(
                        "Whole-object edits require an object selection "
                        "without a node filter"
                    )
                self._authorize(affected, kind, key)
                if value is None:
                    attrs.pop(key, None)
                else:
                    attrs[key] = value
            validate_attributes(element.tag, attrs)
            self._working = self._working.replace_element(
                replace(element, attributes=tuple(attrs.items()))
            )

    def update_node(
        self, object_id: str, node_id: str, values: tuple[float, ...]
    ) -> None:
        with self._change():
            geometry = self._working.geometry_for(object_id)
            node = geometry.node(node_id)
            updated = replace(node, values=tuple(values))
            if updated == node:
                return
            candidate = propagate_node(self._working, geometry.id, updated)
            self._authorize_geometry_change(candidate)
            self._working = candidate

    def update_nodes(
        self, object_id: str, values: dict[str, tuple[float, ...]]
    ) -> None:
        """Apply a fixed-topology node proposal with one authorization pass."""
        with self._change():
            geometry = self._working.geometry_for(object_id)
            for node_id in values:
                geometry.node(node_id)
            if any(
                m.geometry_id == geometry.id
                for b in self._working.boundaries
                for m in b.members
            ):
                candidate = self._working
                for node_id, coordinates in values.items():
                    candidate = propagate_node(
                        candidate,
                        geometry.id,
                        replace(geometry.node(node_id), values=coordinates),
                    )
            else:
                candidate = self._working.replace_geometry(
                    replace(
                        geometry,
                        subpaths=tuple(
                            replace(
                                s,
                                nodes=tuple(
                                    replace(n, values=values.get(n.id, n.values))
                                    for n in s.nodes
                                ),
                            )
                            for s in geometry.subpaths
                        ),
                    )
                )
            self._authorize_geometry_change(candidate)
            self._working = candidate

    def _authorize_geometry_change(self, candidate: Document) -> None:
        """Check all propagated edits before exposing any changed geometry."""
        for geometry in candidate.geometries:
            original = self._working.geometry(geometry.id)
            if geometry == original:
                continue
            self._authorize(
                self._working.geometry_users(geometry.id), EditKind.GEOMETRY
            )
            old_nodes = {n.id: n for s in original.subpaths for n in s.nodes}
            for subpath in geometry.subpaths:
                for node in subpath.nodes:
                    old = old_nodes.get(node.id)
                    if old is None or node == old:
                        continue
                    if (
                        self._selection.node_ids
                        and node.id not in self._selection.node_ids
                    ):
                        raise EditRejectedError("Node is outside the selected nodes")
                    if old.pinned and old.endpoint != node.endpoint:
                        raise EditRejectedError("Endpoint is pinned")

    def share_boundaries(self, tolerance: float = 1.0) -> int:
        from vectrify.document.contact import match_boundaries

        with self._change():
            if self._selection.node_ids:
                raise EditRejectedError("Select whole paths to share boundaries")
            candidate, count = match_boundaries(
                self._working, self._selection.object_ids, tolerance
            )
            for oid in self._selection.object_ids:
                self._authorize(frozenset({oid}), EditKind.STRUCTURE)
            self._authorize_geometry_change(candidate)
            self._working = candidate
            return count

    def link_boundary(self, members: tuple[EdgeRef, ...]) -> str:
        """Link exactly coincident local edges; no inference or coordinate snapping."""
        with self._change():
            if self._selection.node_ids:
                raise EditRejectedError("Linking requires a whole object selection")
            boundary = SharedBoundary(new_id("boundary"), members)
            for member in members:
                edge(self._working, member)
                self._authorize(
                    self._working.geometry_users(member.geometry_id), EditKind.STRUCTURE
                )
            self._working = replace(
                self._working, boundaries=(*self._working.boundaries, boundary)
            )
            return boundary.id

    def detach_boundary(self, member: EdgeRef) -> None:
        """Unlink an edge without changing coordinates or any node identity."""
        with self._change():
            if self._selection.node_ids:
                raise EditRejectedError("Detaching requires a whole object selection")
            edge(self._working, member)
            self._authorize(
                self._working.geometry_users(member.geometry_id), EditKind.STRUCTURE
            )
            boundaries = []
            for boundary in self._working.boundaries:
                members = tuple(
                    m
                    for m in boundary.members
                    if (m.geometry_id, m.node_id)
                    != (member.geometry_id, member.node_id)
                )
                if len(members) >= 2:
                    boundaries.append(replace(boundary, members=members))
            self._working = replace(self._working, boundaries=tuple(boundaries))

    def split_edge(
        self, object_id: str, node_id: str, t: float = 0.5
    ) -> tuple[str, ...]:
        """Insert a node on a line/cubic/closing edge, subdividing linked edges too.

        t follows this object's SVG edge direction. All existing endpoints retain
        their identities and pins. Returns the newly inserted node IDs.
        """
        with self._change():
            if not math.isfinite(t) or not 0 < t < 1:
                raise EditRejectedError(
                    "Split parameter must lie strictly between 0 and 1"
                )
            geometry = self._working.geometry_for(object_id)
            ref = EdgeRef(geometry.id, node_id)
            boundary = next(
                (
                    b
                    for b in self._working.boundaries
                    if any(
                        m.geometry_id == geometry.id and m.node_id == node_id
                        for m in b.members
                    )
                ),
                None,
            )
            members = boundary.members if boundary else (ref,)
            for member in members:
                affected = self._working.geometry_users(member.geometry_id)
                self._authorize(affected, EditKind.STRUCTURE)
                self._authorize(affected, EditKind.GEOMETRY)
                if (
                    self._selection.node_ids
                    and member.node_id not in self._selection.node_ids
                ):
                    raise EditRejectedError("Edge is outside the selected nodes")
            candidate, added = split_edges(self._working, ref, t)
            self._authorize_geometry_change(candidate)
            self._working = candidate
            return added

    def delete_node(self, object_id: str, node_id: str) -> None:
        """Remove a non-moveto node; linked adjacent edges must first be detached."""
        with self._change():
            geometry = self._working.geometry_for(object_id)
            node = geometry.node(node_id)
            if node.command == "M":
                raise EditRejectedError("Delete the subpath to remove its moveto")
            if node.pinned:
                raise EditRejectedError("Endpoint is pinned")
            if self._selection.node_ids and node_id not in self._selection.node_ids:
                raise EditRejectedError("Node is outside the selected nodes")
            affected = self._working.geometry_users(geometry.id)
            self._authorize(affected, EditKind.STRUCTURE)
            self._authorize(affected, EditKind.GEOMETRY)
            for boundary in self._working.boundaries:
                for member in boundary.members:
                    linked = edge(self._working, member)
                    if member.geometry_id == geometry.id and node_id in {
                        linked.start.id,
                        linked.end.id,
                    }:
                        raise EditRejectedError(
                            "Detach adjacent shared edges before deleting a node"
                        )
            updated = replace(
                geometry,
                subpaths=tuple(
                    replace(s, nodes=tuple(n for n in s.nodes if n.id != node_id))
                    for s in geometry.subpaths
                ),
            )
            self._working = self._working.replace_geometry(updated)
            self._record_remap({node_id: set()})

    def _drop_unused_boundary_members(self) -> None:
        owned = {e.geometry_id for e in self._working.elements()}
        boundaries = []
        for boundary in self._working.boundaries:
            members = tuple(m for m in boundary.members if m.geometry_id in owned)
            if len(members) >= 2:
                boundaries.append(replace(boundary, members=members))
        self._working = replace(self._working, boundaries=tuple(boundaries))

    def share_geometry(self, object_id: str, source_id: str) -> None:
        """Explicitly replace a path's geometry with another path's asset."""
        with self._change():
            target = self._working.element(object_id)
            if target.tag != "path" or self._selection.node_ids:
                raise EditRejectedError("Sharing requires a whole path selection")
            original = self._working.geometry_for(object_id)
            if any(n.pinned for s in original.subpaths for n in s.nodes):
                raise EditRejectedError(
                    "Cannot replace geometry containing pinned endpoints"
                )
            affected = self._working.dependents({object_id})
            self._authorize(affected, EditKind.STRUCTURE)
            self._authorize(affected, EditKind.GEOMETRY)
            source = self._working.geometry_for(source_id)
            self._working = self._working.replace_element(
                replace(target, geometry_id=source.id)
            )
            self._drop_unused_boundary_members()

    def detach_geometry(self, object_id: str) -> str:
        """Copy an asset while preserving the selected object's identity.

        A use becomes an editable path inside wrappers that retain instance
        style, transforms and clipping without changing the shared definition.
        """
        with self._change():
            element = self._working.element(object_id)
            if self._selection.node_ids:
                raise EditRejectedError("Detaching requires a whole object selection")
            self._authorize(self._working.dependents({object_id}), EditKind.STRUCTURE)
            original = self._working.geometry_for(object_id)
            geometry = original.detached()
            if element.tag == "path":
                self._working = replace(
                    self._working, geometries=(*self._working.geometries, geometry)
                )
                self._working = self._working.replace_element(
                    replace(element, geometry_id=geometry.id)
                )
            elif element.tag == "use":
                source = self._working.element((element.get("href") or "")[1:])
                ancestors = self._working.ancestry(source.id)
                if source.tag != "path" or not any(e.tag == "defs" for e in ancestors):
                    raise EditRejectedError(
                        "Detach currently supports use references to paths in defs"
                    )
                if any(object_id in references(e) for e in self._working.elements()):
                    raise EditRejectedError(
                        "Detach references to this instance before making it a path"
                    )
                self._check_locks(source.id, EditKind.STRUCTURE)
                # Keep the instance ID on the editable path so selection and node
                # remapping still point to geometry, not its presentation wrapper.
                cloned = replace(source, id=object_id, geometry_id=geometry.id)
                child = cloned
                x, y = element.get("x", "0"), element.get("y", "0")
                if float(x or 0) or float(y or 0):
                    child = Element(
                        new_id("object"),
                        "g",
                        (("transform", f"translate({x} {y})"),),
                        children=(child,),
                    )
                wrapper = Element(
                    new_id("object"),
                    "g",
                    tuple(
                        (k, v)
                        for k, v in element.attributes
                        if k not in {"href", "x", "y"}
                    ),
                    children=(child,),
                    locks=element.locks,
                )
                parent = self._working.ancestry(object_id)[-2]
                self._working = replace(
                    self._working, geometries=(*self._working.geometries, geometry)
                )
                self._working = self._working.replace_element(
                    replace(
                        parent,
                        children=tuple(
                            wrapper if e.id == object_id else e for e in parent.children
                        ),
                    )
                )
            else:
                raise EditRejectedError("Object has no detachable path geometry")
            self._drop_unused_boundary_members()
            self._record_remap(
                {
                    old.id: {new.id}
                    for old_sub, new_sub in zip(
                        original.subpaths, geometry.subpaths, strict=True
                    )
                    for old, new in zip(old_sub.nodes, new_sub.nodes, strict=True)
                }
            )
            return geometry.id

    def replace_geometry(self, object_id: str, geometry: Geometry) -> None:
        """Give a path new contours, which may change its node structure.

        Needs geometry and structure permission for every user of the asset.
        Pinned endpoints and linked boundaries must be released first, since
        new contours cannot keep them. The asset keeps its ID; its nodes do not.
        """
        with self._change():
            self._whole_objects()
            original = self._working.geometry_for(object_id)
            affected = self._working.geometry_users(original.id)
            self._authorize(affected, EditKind.STRUCTURE)
            self._authorize(affected, EditKind.GEOMETRY)
            if any(n.pinned for s in original.subpaths for n in s.nodes):
                raise EditRejectedError("Unpin endpoints before replacing a contour")
            if any(
                m.geometry_id == original.id
                for b in self._working.boundaries
                for m in b.members
            ):
                raise EditRejectedError(
                    "Detach linked boundaries before replacing a contour"
                )
            updated = replace(geometry, id=original.id)
            self._working = self._working.replace_geometry(updated)
            retained = {n.id for sub in updated.subpaths for n in sub.nodes}
            self._record_remap(
                {
                    n.id: set()
                    for sub in original.subpaths
                    for n in sub.nodes
                    if n.id not in retained
                }
            )

    def set_node_handles(self, object_id: str, node_id: str, count: int) -> None:
        """Give a point no handles (a corner), one, or two (a smooth point).

        A point's incoming handle is the last control point of its own
        segment and its outgoing handle the first of the next one; a straight
        segment becomes a curve to hold one, and a curve whose handles are
        both retracted becomes straight again. Two handles are set along the
        line from the previous point to the next, a third of each segment
        long. One handle keeps the curve coming in and straightens the way out;
        asking again for one moves it to the other side.
        """
        if count not in {0, 1, 2}:
            raise EditRejectedError("A point has no handles, one or two")
        geometry = self._working.geometry_for(object_id)
        found = next(
            (
                (index, subpath, position)
                for index, subpath in enumerate(geometry.subpaths)
                for position, node in enumerate(subpath.nodes)
                if node.id == node_id
            ),
            None,
        )
        if found is None:
            raise DocumentError(f"Unknown path node: {node_id}")
        index, subpath, position = found
        nodes = list(subpath.nodes)
        point = nodes[position].endpoint
        # The closing line and a moveto hold no handles.
        incoming = position if position > 0 else None
        outgoing = position + 1 if position + 1 < len(nodes) else None

        def at(node_index: int) -> tuple[float, float]:
            return nodes[node_index].endpoint

        def handle_in() -> tuple[float, float] | None:
            if incoming is None or nodes[incoming].command != "C":
                return None
            h = nodes[incoming].values[2:4]
            return None if h == point else (h[0], h[1])

        def handle_out() -> tuple[float, float] | None:
            if outgoing is None or nodes[outgoing].command != "C":
                return None
            h = nodes[outgoing].values[0:2]
            return None if h == point else (h[0], h[1])

        def set_in(h: tuple[float, float] | None) -> None:
            if incoming is None:
                return
            node, start = nodes[incoming], at(incoming - 1)
            if h is None:
                if node.command == "C":
                    c1 = node.values[0:2]
                    nodes[incoming] = (
                        replace(node, command="L", values=point)
                        if _on_chord(c1, start, point)
                        else replace(node, values=(*c1, *point, *point))
                    )
                return
            c1 = (
                node.values[0:2]
                if node.command == "C"
                else _toward(start, point, 1 / 3)
            )
            nodes[incoming] = replace(node, command="C", values=(*c1, *h, *point))

        def set_out(h: tuple[float, float] | None) -> None:
            if outgoing is None:
                return
            node, end = nodes[outgoing], at(outgoing)
            if h is None:
                if node.command == "C":
                    c2 = node.values[2:4]
                    nodes[outgoing] = (
                        replace(node, command="L", values=end)
                        if _on_chord(c2, point, end)
                        else replace(node, values=(*point, *c2, *end))
                    )
                return
            c2 = node.values[2:4] if node.command == "C" else _toward(end, point, 1 / 3)
            nodes[outgoing] = replace(node, command="C", values=(*h, *c2, *end))

        # Smooth handle positions: along the neighbours' chord, a third of each
        # segment long, or toward the only neighbour at an open end.
        before = at(incoming - 1) if incoming is not None else None
        after = at(outgoing) if outgoing is not None else None
        ends = [p for p in (before, after) if p is not None]
        if not ends:
            raise EditRejectedError("This point has no segment to hold a handle")
        tail, head = (before or point), (after or point)
        dx, dy = head[0] - tail[0], head[1] - tail[1]
        norm = math.hypot(dx, dy) or 1.0
        ux, uy = dx / norm, dy / norm

        def smooth(neighbour: tuple[float, float] | None, sign: float):
            if neighbour is None:
                return None
            length = math.dist(point, neighbour) / 3
            return (point[0] + sign * ux * length, point[1] + sign * uy * length)

        if count == 0:
            set_in(None)
            set_out(None)
        elif count == 2:
            set_in(handle_in() or smooth(before, -1))
            set_out(handle_out() or smooth(after, 1))
        else:
            only_in = handle_in() is not None and handle_out() is None
            if (only_in and outgoing is not None) or incoming is None:
                set_in(None)
                set_out(handle_out() or smooth(after, 1))
            else:
                set_out(None)
                set_in(handle_in() or smooth(before, -1))
        subpaths = list(geometry.subpaths)
        subpaths[index] = replace(subpath, nodes=tuple(nodes))
        self.reshape_path(object_id, replace(geometry, subpaths=tuple(subpaths)))

    def reshape_path(self, object_id: str, geometry: Geometry) -> None:
        """Give a path new contours while its surviving nodes keep their identity.

        Nodes whose ID the new geometry repeats are the same nodes, moved or
        not; IDs it lacks are removed and new IDs are inserted. Pinned
        endpoints must survive where they are, and linked boundary edges must
        come through unchanged. Only moving nodes needs geometry permission;
        adding or removing them needs structure too.
        """
        with self._change():
            self._whole_objects()
            original = self._working.geometry_for(object_id)
            updated = replace(geometry, id=original.id)
            old = {n.id: n for s in original.subpaths for n in s.nodes}
            new = {n.id: n for s in updated.subpaths for n in s.nodes}
            affected = self._working.geometry_users(original.id)
            self._authorize(affected, EditKind.GEOMETRY)
            if old.keys() != new.keys():
                self._authorize(affected, EditKind.STRUCTURE)
            for node_id, node in old.items():
                kept = new.get(node_id)
                if node.pinned and (kept is None or kept.endpoint != node.endpoint):
                    raise EditRejectedError("Endpoint is pinned")
            candidate = self._working.replace_geometry(updated)
            for boundary in self._working.boundaries:
                for member in boundary.members:
                    if member.geometry_id != original.id:
                        continue
                    before = edge(self._working, member)
                    try:
                        after = edge(candidate, member)
                    except DocumentError:
                        after = None
                    if after is None or after.points != before.points:
                        raise EditRejectedError(
                            "Linked boundary edges must stay as they are"
                        )
            self._working = candidate
            self._record_remap({n: set() for n in old.keys() - new.keys()})

    def split_disconnected(self, object_id: str) -> tuple[str, ...]:
        """Partition a compound path into independently editable exact geometries.

        A wrapper retains compositing, clipping, transforms and inherited locks.
        Connected rings (including holes) remain in the same child path.
        """
        with self._change():
            self._whole_objects()
            document = self._working
            element = document.element(object_id)
            if element.tag != "path" or any(
                e.tag in {"defs", "clipPath"} for e in document.ancestry(object_id)
            ):
                raise EditRejectedError(
                    "Select a drawing path to split, not a definition or instance"
                )
            self._authorize(document.dependents({object_id}), EditKind.STRUCTURE)
            self._authorize(document.dependents({object_id}), EditKind.GEOMETRY)
            geometry = document.geometry_for(object_id)
            if any(object_id in references(e) for e in document.elements()):
                raise EditRejectedError(
                    "This path is referenced; detach its instances before splitting"
                )
            if sum(e.geometry_id == geometry.id for e in document.elements()) != 1:
                raise EditRejectedError("Detach shared geometry before splitting")
            paint = {}
            for ancestor in document.ancestry(object_id):
                paint.update(dict(ancestor.attributes))
            parts = disconnected_parts(
                geometry,
                filled=paint.get("fill", "black") != "none",
            )
            if len(parts) < 2:
                return (object_id,)
            geometries = tuple(Geometry(new_id("geometry"), part) for part in parts)
            child_attributes = tuple(a for a in element.attributes if a[0] == "opacity")
            children = tuple(
                Element(new_id("object"), "path", child_attributes, geometry_id=g.id)
                for g in geometries
            )
            group = Element(
                new_id("group"),
                "g",
                tuple(a for a in element.attributes if a[0] != "opacity"),
                children,
                locks=element.locks,
            )
            parent = document.ancestry(object_id)[-2]
            updated = document.replace_element(
                replace(
                    parent,
                    children=tuple(
                        group if child.id == object_id else child
                        for child in parent.children
                    ),
                )
            )
            node_geometry = {
                n.id: g.id for g in geometries for s in g.subpaths for n in s.nodes
            }
            self._working = replace(
                updated,
                geometries=tuple(g for g in document.geometries if g.id != geometry.id)
                + geometries,
                boundaries=tuple(
                    replace(
                        b,
                        members=tuple(
                            replace(m, geometry_id=node_geometry[m.node_id])
                            if m.geometry_id == geometry.id
                            else m
                            for m in b.members
                        ),
                    )
                    for b in document.boundaries
                ),
            )
            ids = tuple(child.id for child in children)
            self._ids = (self._ids - {object_id}) | {group.id, *ids}
            self._record_object_remap({object_id: set(ids)})
            return ids

    def join_paths(
        self, object_ids: frozenset[str], *, color_source: str | None = None
    ) -> str:
        """Join regions at the frontmost position with mixed or source colors."""
        with self._change():
            self._whole_objects()
            document = self._working
            selected = [document.element(oid) for oid in object_ids]
            members = {e.id: e for item in selected for e in Document(item).elements()}
            if any(e.tag not in {"g", "path"} for e in members.values()):
                raise EditRejectedError("Join paths or groups containing only paths")
            groups = {e.id for e in members.values() if e.tag == "g"}
            if not groups or (
                len(selected) == 1
                and selected[0].tag == "g"
                and all(c.tag == "path" for c in selected[0].children)
            ):
                return self._join_paths(object_ids, color_source)
            if any(set(references(e)) & groups for e in document.elements()):
                raise EditRejectedError(
                    "Detach references to these groups before joining"
                )
            self._authorize(document.dependents(groups), EditKind.STRUCTURE)
            self._authorize(document.dependents(groups), EditKind.GEOMETRY)
            joined = self._join_paths(
                frozenset(e.id for e in members.values() if e.tag == "path"),
                color_source,
            )

            def remove_empty(element: Element) -> Element:
                children = [remove_empty(c) for c in element.children]
                return replace(
                    element,
                    children=tuple(
                        c for c in children if c.id not in groups or c.children
                    ),
                )

            self._working = replace(
                self._working, root=remove_empty(self._working.root)
            )
            self._record_object_remap({oid: {joined} for oid in groups})
            return joined

    def _join_paths(
        self, object_ids: frozenset[str], color_source: str | None = None
    ) -> str:
        document = self._working
        selected = [document.element(oid) for oid in object_ids]
        group = selected[0] if len(selected) == 1 and selected[0].tag == "g" else None
        paths = list(group.children) if group else selected
        if len(paths) < 2 or any(p.tag != "path" for p in paths):
            raise EditRejectedError(
                "Select at least two paths, or a group containing only paths"
            )
        if color_source is not None and color_source not in {p.id for p in paths}:
            raise EditRejectedError("Choose a color source from the paths being joined")
        parents = {document.ancestry(p.id)[-2].id for p in paths}
        if len(parents) != 1:
            return self._join_across_groups(paths, color_source)
        parent = document.element(next(iter(parents)))
        path_ids = {p.id for p in paths}
        paths = [p for p in parent.children if p.id in path_ids]
        if any(e.tag in {"defs", "clipPath"} for e in document.ancestry(paths[0].id)):
            raise EditRejectedError("Join drawing paths, not definitions or instances")
        removed = path_ids | ({group.id} if group else set())
        self._authorize(document.dependents(removed), EditKind.STRUCTURE)
        self._authorize(document.dependents(removed), EditKind.GEOMETRY)
        if any(set(references(e)) & removed for e in document.elements()):
            raise EditRejectedError("Detach references to these objects before joining")
        attrs = dict(paths[-1].attributes)
        structural = {k: v for k, v in attrs.items() if k not in PAINT}
        if any(
            {k: v for k, v in p.attributes if k not in PAINT} != structural
            for p in paths
        ):
            raise EditRejectedError(
                "Paths must have matching transforms and clipping to join"
            )
        if attrs.get("clip-path", "none") != "none":
            raise EditRejectedError(
                "Move clipping to the containing group before joining"
            )
        styles = [path_style(document, p) for p in paths]
        if len({s["fill"] != "none" for s in styles}) > 1:
            raise EditRejectedError(
                "Join filled regions separately from stroke-only outlines"
            )
        if any(style != styles[-1] for style in styles):
            paint = join_paint(
                styles,
                painted_weights(document, paths),
                next((i for i, p in enumerate(paths) if p.id == color_source), None),
            )
            for path, style in zip(paths, styles, strict=True):
                for key, value in paint.items():
                    if value != style[key]:
                        self._authorize(
                            document.dependents({path.id}), EditKind.PAINT, key
                        )
            attrs.update(paint)
        else:
            paint = styles[-1]
        originals = [document.geometry_for(p.id) for p in paths]
        old_ids = {g.id for g in originals}
        if len(old_ids) != len(paths) or any(
            e.geometry_id in old_ids and e.id not in path_ids
            for e in document.elements()
        ):
            raise EditRejectedError("Detach shared geometry before joining")
        geometry = Geometry(
            new_id("geometry"), tuple(s for g in originals for s in g.subpaths)
        )
        filled = paint.get("fill", "black") != "none"
        if filled:
            owners = {s.id: g.id for g in originals for s in g.subpaths}
            contact = any(
                len({owners[s.id] for s in part}) > 1
                for part in disconnected_parts(geometry, filled=True)
            )
            if contact or len({s["fill-rule"] for s in styles}) > 1:
                if any(
                    n.pinned for g in originals for s in g.subpaths for n in s.nodes
                ):
                    raise EditRejectedError(
                        "Unpin selected regions before joining overlapping contours"
                    )
                if any(
                    m.geometry_id in old_ids
                    for b in document.boundaries
                    for m in b.members
                ):
                    raise EditRejectedError(
                        "Detach boundaries before joining overlapping regions"
                    )
                geometry = union_geometry(originals, styles)
                for path, style in zip(paths, styles, strict=True):
                    if style["fill-rule"] != "nonzero":
                        self._authorize(
                            document.dependents({path.id}),
                            EditKind.PAINT,
                            "fill-rule",
                        )
                attrs["fill-rule"] = "nonzero"
                self._record_remap(
                    {
                        n.id: set()
                        for g in originals
                        for s in g.subpaths
                        for n in s.nodes
                    }
                )
        locks = frozenset().union(*(p.locks for p in paths))
        if group:
            if float(group.get("opacity", "1") or "1") != 1:
                raise EditRejectedError(
                    "Select the group's paths to retain group opacity when joining"
                )
            if attrs.get("transform") and group.get("clip-path", "none") != "none":
                raise EditRejectedError(
                    "Select the group's paths to retain its clipping coordinate system"
                )
            merged = dict(group.attributes)
            # A child's clip-path:none does not cancel an ancestor's clip.
            merged.update({k: v for k, v in attrs.items() if k != "clip-path"})
            if group.get("transform") and attrs.get("transform"):
                merged["transform"] = f"{group.get('transform')} {attrs['transform']}"
            attrs = merged
            locks |= group.locks
        joined = Element(
            new_id("object"),
            "path",
            tuple(attrs.items()),
            geometry_id=geometry.id,
            locks=locks,
        )
        if group:
            container = document.ancestry(group.id)[-2]
            updated = document.replace_element(
                replace(
                    container,
                    children=tuple(
                        joined if c.id == group.id else c for c in container.children
                    ),
                )
            )
        else:
            front = paths[-1].id
            updated = document.replace_element(
                replace(
                    parent,
                    children=tuple(
                        joined if child.id == front else child
                        for child in parent.children
                        if child.id == front or child.id not in path_ids
                    ),
                )
            )
        self._working = replace(
            updated,
            geometries=(
                *(g for g in document.geometries if g.id not in old_ids),
                geometry,
            ),
            boundaries=tuple(
                replace(
                    b,
                    members=tuple(
                        replace(m, geometry_id=geometry.id)
                        if m.geometry_id in old_ids
                        else m
                        for m in b.members
                    ),
                )
                for b in document.boundaries
            ),
        )
        self._ids = (self._ids - removed) | {joined.id}
        self._record_object_remap({oid: {joined.id} for oid in removed})
        return joined.id

    def _join_across_groups(
        self, paths: list[Element], color_source: str | None
    ) -> str:
        document = self._working
        path_ids = {p.id for p in paths}
        paths = [p for p in document.elements() if p.id in path_ids]
        ancestries = [document.ancestry(p.id) for p in paths]
        common = document.root
        for ancestors in zip(*ancestries, strict=False):
            if len({a.id for a in ancestors}) != 1:
                break
            common = ancestors[0]
        if any(a.tag in {"defs", "clipPath"} for chain in ancestries for a in chain):
            raise EditRejectedError("Join drawing paths, not definitions or instances")
        affected = document.dependents(path_ids)
        self._authorize(affected, EditKind.STRUCTURE)
        self._authorize(affected, EditKind.GEOMETRY)
        if any(set(references(e)) & path_ids for e in document.elements()):
            raise EditRejectedError("Detach references to these objects before joining")
        originals = [document.geometry_for(p.id) for p in paths]
        old_ids = {g.id for g in originals}
        if len(old_ids) != len(paths) or any(
            e.geometry_id in old_ids and e.id not in path_ids
            for e in document.elements()
        ):
            raise EditRejectedError("Detach shared geometry before joining")
        baked = [bake_group_path(document, p, common.id) for p in paths]
        styles = [style for _, style, _ in baked]
        if len({s["fill"] != "none" for s in styles}) > 1:
            raise EditRejectedError(
                "Join filled regions separately from stroke-only outlines"
            )
        paint = join_paint(
            styles,
            painted_weights(document, paths),
            next((i for i, p in enumerate(paths) if p.id == color_source), None),
        )
        normalized = []
        updated = document
        for path, original, (geometry, style, changed) in zip(
            paths, originals, baked, strict=True
        ):
            if changed:
                if any(n.pinned for s in original.subpaths for n in s.nodes):
                    raise EditRejectedError(
                        "Unpin selected paths before resolving "
                        "group transforms or clipping"
                    )
                if any(
                    m.geometry_id == original.id
                    for b in document.boundaries
                    for m in b.members
                ):
                    raise EditRejectedError(
                        "Detach shared boundaries before resolving "
                        "group transforms or clipping"
                    )
                surviving = {n.id for s in geometry.subpaths for n in s.nodes}
                self._record_remap(
                    {
                        n.id: set()
                        for s in original.subpaths
                        for n in s.nodes
                        if n.id not in surviving
                    }
                )
            resolved = dict(paint, **{"fill-rule": style["fill-rule"]})
            before = path_style(document, path)
            for key, value in resolved.items():
                if value != before[key]:
                    self._authorize(document.dependents({path.id}), EditKind.PAINT, key)
            chain = document.ancestry(path.id)
            below = chain[
                next(i for i, a in enumerate(chain) if a.id == common.id) + 1 :
            ]
            locks = frozenset().union(*(a.locks for a in below))
            normalized.append(
                replace(path, attributes=tuple(resolved.items()), locks=locks)
            )
            updated = updated.replace_geometry(geometry)

        front = paths[-1].id
        front_chain = {a.id for a in document.ancestry(front)}

        def prune(element: Element) -> Element:
            return replace(
                element,
                children=tuple(
                    prune(c) for c in element.children if c.id not in path_ids
                ),
            )

        def split(element: Element) -> tuple[Element | None, Element | None]:
            # Split only the frontmost ancestry. Wrappers retain the paint,
            # transform and clip of unselected siblings on each side.
            before, after = [], []
            reached = False
            for child in element.children:
                if child.id == front:
                    reached = True
                elif child.id in front_chain:
                    left, right = split(child)
                    if left:
                        before.append(left)
                    if right:
                        after.append(right)
                    reached = True
                elif child.id not in path_ids:
                    (after if reached else before).append(prune(child))
            if before and after and float(element.get("opacity", "1") or 1) != 1:
                raise EditRejectedError(
                    "Resolve group opacity before joining "
                    "across both sides of that group"
                )
            left = replace(element, children=tuple(before)) if before else None
            right = (
                replace(
                    element,
                    id=new_id("object") if left else element.id,
                    children=tuple(after),
                )
                if after
                else None
            )
            return left, right

        children = []
        for child in common.children:
            if child.id == front:
                children.extend(normalized)
            elif child.id in front_chain:
                left, right = split(child)
                if left:
                    children.append(left)
                children.extend(normalized)
                if right:
                    children.append(right)
            elif child.id not in path_ids:
                children.append(prune(child))
        updated = updated.replace_element(replace(common, children=tuple(children)))
        self._working = updated
        return self._join_paths(frozenset(path_ids), color_source)

    def fill_holes(self, object_id: str, hole_ids: frozenset[str]) -> None:
        """Fill explicitly chosen holes, including their nested contour islands."""
        with self._change():
            self._whole_objects()
            document = self._working
            holes = {hole.id: hole for hole in find_holes(document, object_id)}
            if not hole_ids or not hole_ids <= holes.keys():
                raise EditRejectedError("Choose existing holes to fill")
            geometry = document.geometry_for(object_id)
            self._authorize(document.geometry_users(geometry.id), EditKind.GEOMETRY)
            self._authorize(document.geometry_users(geometry.id), EditKind.STRUCTURE)
            removed = set().union(*(holes[hid].subpath_ids for hid in hole_ids))
            nodes = {
                n.id for s in geometry.subpaths if s.id in removed for n in s.nodes
            }
            if any(
                n.pinned for s in geometry.subpaths if s.id in removed for n in s.nodes
            ):
                raise EditRejectedError(
                    "Unpin the selected hole contours before filling"
                )
            if any(
                m.geometry_id == geometry.id and m.node_id in nodes
                for b in document.boundaries
                for m in b.members
            ):
                raise EditRejectedError(
                    "Detach shared boundaries on the selected holes first"
                )
            self._working = document.replace_geometry(
                replace(
                    geometry,
                    subpaths=tuple(s for s in geometry.subpaths if s.id not in removed),
                )
            )
            self._record_remap({nid: set() for nid in nodes})

    def _whole_objects(self) -> None:
        if self._selection.node_ids:
            raise EditRejectedError("This command requires a whole object selection")

    def insert_object(
        self,
        parent_id: str,
        element: Element,
        *,
        index: int | None = None,
        geometries: tuple[Geometry, ...] = (),
    ) -> str:
        """Insert an identified subtree into an explicitly selected container."""
        with self._change():
            self._whole_objects()
            parent = self._working.element(parent_id)
            if parent.tag not in {"svg", "g", "defs", "clipPath"}:
                raise EditRejectedError("Insertion requires a container")
            self._authorize(self._working.dependents({parent_id}), EditKind.STRUCTURE)
            index = len(parent.children) if index is None else index
            if not 0 <= index <= len(parent.children):
                raise EditRejectedError("Insertion index is outside the container")
            self._working = replace(
                self._working, geometries=(*self._working.geometries, *geometries)
            )
            self._working = self._working.replace_element(
                replace(
                    parent,
                    children=(
                        *parent.children[:index],
                        element,
                        *parent.children[index:],
                    ),
                )
            )
            self._ids |= frozenset(e.id for e in Document(element).elements())
            return element.id

    def delete_objects(self, object_ids: frozenset[str]) -> None:
        """Delete explicit subtrees; surviving references must never dangle."""
        with self._change():
            self._whole_objects()
            removed = set()
            for object_id in object_ids:
                if object_id == self._working.root.id:
                    raise EditRejectedError("Cannot delete the document root")
                removed.update(
                    e.id for e in Document(self._working.element(object_id)).elements()
                )
            self._authorize(self._working.dependents(removed), EditKind.STRUCTURE)
            for element in self._working.elements():
                if element.id not in removed and set(references(element)) & removed:
                    raise EditRejectedError(
                        "Delete or retarget dependent references first"
                    )
                if element.id in removed:
                    # A pinned definition reached through a use is protected too.
                    pending = [element]
                    seen = set()
                    while pending:
                        item = pending.pop()
                        if item.id in seen:
                            continue
                        seen.add(item.id)
                        pending.extend(item.children)
                        if item.tag == "use":
                            pending.append(
                                self._working.element((item.get("href") or "")[1:])
                            )
                        if item.geometry_id and any(
                            n.pinned
                            for sub in self._working.geometry(item.geometry_id).subpaths
                            for n in sub.nodes
                        ):
                            raise EditRejectedError(
                                "Cannot delete an object with pinned endpoints"
                            )

            def prune(element: Element) -> Element:
                return replace(
                    element,
                    children=tuple(
                        prune(c) for c in element.children if c.id not in removed
                    ),
                )

            self._working = replace(self._working, root=prune(self._working.root))
            self._drop_unused_boundary_members()
            self._record_object_remap({old: set() for old in removed})
            self._ids -= removed

    def reorder_object(self, object_id: str, index: int) -> None:
        """Move one selected object to a final sibling index (zero is back)."""
        with self._change():
            self._whole_objects()
            ancestry = self._working.ancestry(object_id)
            if len(ancestry) < 2:
                raise EditRejectedError("Cannot reorder the document root")
            parent, element = ancestry[-2:]
            if not 0 <= index < len(parent.children):
                raise EditRejectedError("Stacking index is outside the container")
            self._authorize(self._working.dependents({object_id}), EditKind.STRUCTURE)
            children = [c for c in parent.children if c.id != object_id]
            children.insert(index, element)
            self._working = self._working.replace_element(
                replace(parent, children=tuple(children))
            )

    def group_objects(self, object_ids: frozenset[str]) -> str:
        """Wrap consecutive siblings without changing paint order or inheritance."""
        with self._change():
            self._whole_objects()
            if not object_ids or self._working.root.id in object_ids:
                raise EditRejectedError("Choose non-root siblings to group")
            parents = {self._working.ancestry(oid)[-2].id for oid in object_ids}
            if len(parents) != 1:
                raise EditRejectedError("Grouping requires siblings in one container")
            parent = self._working.element(next(iter(parents)))
            positions = [i for i, c in enumerate(parent.children) if c.id in object_ids]
            start, stop = min(positions), max(positions) + 1
            if len(positions) != stop - start:
                raise EditRejectedError(
                    "Reorder nonconsecutive objects before grouping"
                )
            self._authorize(
                self._working.dependents(set(object_ids)), EditKind.STRUCTURE
            )
            group = Element(new_id("group"), "g", children=parent.children[start:stop])
            self._working = self._working.replace_element(
                replace(
                    parent,
                    children=(*parent.children[:start], group, *parent.children[stop:]),
                )
            )
            self._ids |= {group.id}
            return group.id

    def ungroup_object(self, object_id: str) -> tuple[str, ...]:
        """Distribute inherited paint/transforms when group compositing permits it."""
        with self._change():
            self._whole_objects()
            group = self._working.element(object_id)
            if group.tag != "g":
                raise EditRejectedError("Ungroup requires a group")
            self._authorize(self._working.dependents({object_id}), EditKind.STRUCTURE)
            if group.locks:
                raise EditRejectedError("Unlock the group before removing its locks")
            if (
                float(group.get("opacity", "1") or "1") != 1
                or group.get("clip-path", "none") != "none"
            ):
                raise EditRejectedError(
                    "Group opacity or clipping must be resolved before ungrouping"
                )
            if any(object_id in references(e) for e in self._working.elements()):
                raise EditRejectedError("Retarget references before ungrouping")
            inherited = {
                k: v for k, v in group.attributes if k in PAINT and k != "opacity"
            }
            descendants = {e.id for e in Document(group).elements()} - {group.id}
            if (inherited or group.get("transform")) and any(
                set(references(e)) & descendants for e in self._working.elements()
            ):
                raise EditRejectedError(
                    "Detach child references before distributing group attributes"
                )
            children = []
            for child in group.children:
                attrs = inherited | dict(child.attributes)
                transforms = [
                    t for t in (group.get("transform"), child.get("transform")) if t
                ]
                if transforms:
                    attrs["transform"] = " ".join(transforms)
                children.append(replace(child, attributes=tuple(attrs.items())))
            parent = self._working.ancestry(object_id)[-2]
            index = next(i for i, c in enumerate(parent.children) if c.id == object_id)
            self._working = self._working.replace_element(
                replace(
                    parent,
                    children=(
                        *parent.children[:index],
                        *children,
                        *parent.children[index + 1 :],
                    ),
                )
            )
            ids = tuple(c.id for c in children)
            self._record_object_remap({object_id: set(ids)})
            self._ids -= {object_id}
            return ids

    def commit(self) -> Snapshot:
        if self._closed or self._failed:
            raise EditRejectedError("Transaction is closed or has a failed edit")
        self._closed = True
        return self._editor._apply(
            self._working,
            self._label,
            self._base.revision,
            self._remap_selection(self._editor.snapshot.selection),
        )

    def abort(self) -> None:
        self._closed = True

    def __enter__(self) -> Transaction:
        if self._closed:
            raise EditRejectedError("Transaction is closed")
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> None:
        if exc_type is not None:
            self.abort()
        elif not self._closed:
            self.commit()
