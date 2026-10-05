"""Selection-constrained transactions and monotonic revision history."""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace

import pathops
from shapely import make_valid
from shapely.geometry import Polygon

from vectrify.document.components import disconnected_parts
from vectrify.document.history import kept_selection, merge_edit, restore_change
from vectrify.document.hit_test import IDENTITY, mapped, multiply, transform
from vectrify.document.holes import (
    Hole,
    filled_region,
    find_holes,
    reversed_subpath,
    ring_points,
    signed_area,
)
from vectrify.document.join import (
    bake_group_path,
    curve_path,
    join_paint,
    painted_weights,
    path_geometry,
    path_style,
    transformed_geometry,
    union_geometry,
)
from vectrify.document.knife import cut_geometry, cut_strokes
from vectrify.document.lines import (
    break_at,
    contour_ends,
    delete_segment,
    end_pairs,
    joined,
    segments_among,
)
from vectrify.document.model import (
    Document,
    DocumentError,
    EditKind,
    Element,
    Geometry,
    PathNode,
    Selection,
    Subpath,
    new_id,
    paint_server,
    references,
)
from vectrify.document.paint import LinearGradient
from vectrify.document.redraw import redrawn
from vectrify.document.svg import GEOMETRY, GRADIENTS, PAINT, validate_attributes
from vectrify.document.topology import (
    EdgeRef,
    inverse_matrix,
    mapped_point,
    split_edges,
)
from vectrify.document.transforms import ancestry_matrix, object_matrix, root_matrix

# Initial values of inherited paint, written onto a moved object that would
# otherwise inherit something else from its new group.
INITIAL_PAINT = {
    "fill": "black",
    "stroke": "none",
    "fill-rule": "nonzero",
    "clip-rule": "nonzero",
    "fill-opacity": "1",
    "stroke-opacity": "1",
    "stroke-width": "1",
    "stroke-linecap": "butt",
    "stroke-linejoin": "miter",
    "stroke-miterlimit": "4",
}

# A stroke width in user units, as resizing can divide it.
LENGTH = re.compile(r"\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)(?:px)?\s*")


def _transform_text(matrix: tuple[float, ...]) -> str | None:
    """*matrix* as a translate and a scale where it is one, else as a matrix."""
    a, b, c, d, e, f = (round(v, 12) + 0.0 for v in matrix)
    if b or c:
        return f"matrix({' '.join(f'{v:.12g}' for v in (a, b, c, d, e, f))})"
    parts = [f"translate({e:.12g} {f:.12g})"] if e or f else []
    if (a, d) != (1, 1):
        parts.append(f"scale({a:.12g} {d:.12g})")
    return " ".join(parts) or None


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


def _carry_handles(before: Document, after: Document) -> Document:
    """Take the handles of every point an edit moved along with it.

    The curve keeps its shape around a moved point and a retracted handle
    stays a corner; a handle the edit already placed is left where it is.
    A closed contour ending on its moveto shows one point for two nodes, so
    either drags the other.
    """
    while True:
        changes: dict[tuple[str, str], list[float]] = {}
        for geometry in after.geometries:
            old = before.geometry(geometry.id)
            if geometry == old:
                continue
            for subpath, previous in zip(geometry.subpaths, old.subpaths, strict=True):
                nodes, olds = subpath.nodes, previous.nodes
                last = len(nodes) - 1
                twins = (
                    subpath.closed
                    and last > 1
                    and olds[last].endpoint == olds[0].endpoint
                )
                for i, (node, was) in enumerate(zip(nodes, olds, strict=True)):
                    if node.endpoint == was.endpoint:
                        continue
                    dx = node.endpoint[0] - was.endpoint[0]
                    dy = node.endpoint[1] - was.endpoint[1]
                    slots = [(i, 2)] if node.command == "C" else []
                    if i < last and nodes[i + 1].command == "C":
                        slots.append((i + 1, 0))
                    if twins and i in {0, last}:
                        slots.append((last - i, len(nodes[last - i].values) - 2))
                    for j, k in slots:
                        if nodes[j].values[k : k + 2] != olds[j].values[k : k + 2]:
                            continue
                        key = geometry.id, nodes[j].id
                        values = changes.setdefault(key, list(nodes[j].values))
                        values[k : k + 2] = (values[k] + dx, values[k + 1] + dy)
        if not changes:
            return after
        for (gid, nid), values in changes.items():
            geometry = after.geometry(gid)
            node = replace(geometry.node(nid), values=tuple(values))
            after = after.replace_geometry(geometry.replace_node(node))


# Elements that paint their own geometry, and so can take a gradient fill.
DRAWN = frozenset({"path", "rect", "circle", "ellipse", "line"})


def _add_definition(document: Document, definition: Element) -> Document:
    """*document* with *definition* last in the root's first ``defs``."""
    root = document.root
    index = next((i for i, c in enumerate(root.children) if c.tag == "defs"), None)
    if index is None:
        defs = Element(new_id("object"), "defs", (), (definition,))
        children = (defs, *root.children)
    else:
        defs = root.children[index]
        defs = replace(defs, children=(*defs.children, definition))
        children = (*root.children[:index], defs, *root.children[index + 1 :])
    return replace(document, root=replace(root, children=children))


def _without(document: Document, object_id: str) -> Document:
    """*document* without the element *object_id* and its subtree."""

    def prune(element: Element) -> Element:
        children = tuple(prune(c) for c in element.children if c.id != object_id)
        return (
            replace(element, children=children)
            if children != element.children
            else element
        )

    return replace(document, root=prune(document.root))


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
    # The editor's revision once this edit was made.
    revision: int = 0
    id: str = field(default_factory=lambda: new_id("edit"))
    author: str = "person"


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
        # Put before the label of each edit made meanwhile, as "Agent: " is
        # while an agent edits, so the history says who made it.
        self.label_prefix = ""
        self.author = "person"
        self._versions: OrderedDict[int, Document] = OrderedDict({0: document})

    @property
    def snapshot(self) -> Snapshot:
        return Snapshot(self._revision, self._document, self._selection)

    def snapshot_at(self, revision: int) -> Snapshot:
        """A recent document version, with the current UI selection."""
        document = self._versions.get(revision)
        if document is None:
            raise DocumentError(
                "This edit's starting state has expired. Try the edit again."
            )
        return Snapshot(revision, document, self._selection)

    def _remember(self) -> None:
        self._versions[self._revision] = self._document
        while len(self._versions) > 256:
            self._versions.popitem(last=False)

    def fork(self, revision: int, selection: Selection) -> Editor:
        """Plan commands on a client's version, without mutating live state."""
        base = self.snapshot_at(revision)
        fork = Editor(base.document, selection=selection)
        fork._revision = revision
        fork._versions = OrderedDict({revision: base.document})
        fork.author, fork.label_prefix = self.author, self.label_prefix
        return fork

    def merge(
        self, base: Snapshot, document: Document, label: str, selection: Selection
    ) -> Snapshot:
        merged = merge_edit(base.document, document, self._document)
        return self._apply(
            merged, label, self._revision, kept_selection(selection, merged)
        )

    @property
    def undo_labels(self) -> tuple[str, ...]:
        return tuple(entry.label for entry in self._undo)

    @property
    def redo_labels(self) -> tuple[str, ...]:
        return tuple(entry.label for entry in reversed(self._redo))

    @property
    def undo_entries(self) -> tuple[HistoryEntry, ...]:
        """The undo stack, oldest first, as *undo_labels* lists it."""
        return tuple(self._undo)

    @property
    def redo_entries(self) -> tuple[HistoryEntry, ...]:
        """The redo stack, next redo first, as *redo_labels* lists it."""
        return tuple(reversed(self._redo))

    def squash(self, since: int, label: str | None = None) -> None:
        """Make the edits after the first *since* on the undo stack one step.

        Undoing it then undoes them all; *label* names it, else the first's.
        """
        entries = self._undo[since:]
        if len(entries) < 2:
            return
        first, last = entries[0], entries[-1]
        del self._undo[since:]
        self._undo.append(
            replace(
                last,
                label=first.label if label is None else label,
                before=first.before,
                before_selection=first.before_selection,
            )
        )

    def settle(self, revision: int) -> None:
        """Count every edit since *revision* as one revision.

        For a call made of several edits under one lock, none of whose
        in-between revisions anyone saw: the drawing is then one revision
        past *revision*, as the one undo step they were squashed into says.
        """
        if self._revision <= revision + 1:
            return
        self._revision = revision + 1
        if self._undo and self._undo[-1].revision > self._revision:
            self._undo[-1] = replace(self._undo[-1], revision=self._revision)
        self._versions = OrderedDict(
            (r, d) for r, d in self._versions.items() if r <= self._revision
        )
        self._remember()

    def rollback(self, since: int) -> None:
        """Take back the edits after the first *since*, leaving no redo."""
        if len(self._undo) <= since:
            return
        first = self._undo[since]
        del self._undo[since:]
        self._document, self._selection = first.before, first.before_selection
        self._revision += 1
        self._remember()

    def reselect(self, since: int, before: Selection, after: Selection) -> None:
        """Give the edits after the first *since* the selections to show on
        undo (*before*) and redo (*after*), as when an agent edits on a
        selection of its own and gives the person theirs back."""
        last = len(self._undo) - 1
        for i in range(since, last + 1):
            entry = self._undo[i]
            self._undo[i] = replace(
                entry,
                before_selection=before if i == since else entry.before_selection,
                after_selection=after if i == last else entry.after_selection,
            )

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
        rebase: bool = False,
    ) -> Transaction:
        """Open an edit; with *base*, edit that earlier snapshot instead.

        A transaction over an earlier snapshot can be built off the editor's
        thread and commits only if no revision has happened since.
        """
        if expected_revision is not None and expected_revision != self._revision:
            raise StaleRevisionError("Document revision has changed")
        base = base or self.snapshot
        return Transaction(
            self, label, selection or base.selection, allowed, base, rebase=rebase
        )

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
                    label
                    if label.startswith(self.label_prefix)
                    else self.label_prefix + label,
                    self._document,
                    document,
                    self._selection,
                    selection,
                    self._revision + 1,
                    author=self.author,
                )
            )
            # Each author owns their redo branch. An edit by the other author
            # must not throw it away; restoring it still checks for conflicts.
            self._redo = [e for e in self._redo if e.author != self.author]
            self._document = document
            self._selection = selection
            self._revision += 1
            self._remember()
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
        return self.pin_nodes(((object_id, node_id),), pinned=pinned)

    def pin_nodes(
        self, points: Iterable[tuple[str, str]], *, pinned: bool = True
    ) -> Snapshot:
        """Pin or unpin (object ID, node ID) endpoints as one undoable edit."""
        points = tuple(points)
        document = self._document
        for object_id, node_id in points:
            geometry = document.geometry_for(object_id)
            node = geometry.node(node_id)
            document = document.replace_geometry(
                geometry.replace_node(replace(node, pinned=pinned))
            )
        label = "Pin node" if pinned else "Unpin node"
        if len(points) > 1:
            label += "s"
        return self._apply(document, label, self._revision)

    def undo(
        self,
        entry_ids: Sequence[str] | None = None,
        *,
        author: str | None = None,
        preserve_selection: bool = False,
    ) -> Snapshot:
        return self._restore("undo", entry_ids, author, preserve_selection)

    def redo(
        self,
        entry_ids: Sequence[str] | None = None,
        *,
        author: str | None = None,
        preserve_selection: bool = False,
    ) -> Snapshot:
        return self._restore("redo", entry_ids, author, preserve_selection)

    def _restore(
        self,
        command: str,
        entry_ids: Sequence[str] | None,
        author: str | None,
        preserve_selection: bool,
    ) -> Snapshot:
        source = self._undo if command == "undo" else self._redo
        destination = self._redo if command == "undo" else self._undo
        if entry_ids is None:
            latest = next(
                (e for e in reversed(source) if author is None or e.author == author),
                None,
            )
            if latest is None:
                return self.snapshot
            entry_ids = [latest.id]
        if (
            isinstance(entry_ids, str)
            or not entry_ids
            or any(not isinstance(eid, str) for eid in entry_ids)
            or len(set(entry_ids)) != len(entry_ids)
        ):
            raise DocumentError("Give distinct history entry ids")
        available = {e.id: e for e in source}
        entries = []
        for eid in entry_ids:
            entry = available.get(eid)
            if entry is None or (author is not None and entry.author != author):
                raise DocumentError(f"No such {command} entry: {eid}")
            entries.append(entry)
        document, selection = self._document, self._selection
        # Plan the entire batch first; a missing ID or conflict leaves both
        # the document and history stacks untouched.
        for entry in entries:
            expected, desired = (
                (entry.after, entry.before)
                if command == "undo"
                else (entry.before, entry.after)
            )
            document = restore_change(expected, desired, document)
            if not preserve_selection:
                selection = (
                    entry.before_selection
                    if command == "undo"
                    else entry.after_selection
                )
            selection = kept_selection(selection, document)
        self._document, self._selection = document, selection
        selected = set(entry_ids)
        source[:] = [e for e in source if e.id not in selected]
        destination.extend(entries)
        self._revision += 1
        self._remember()
        return self.snapshot


class Transaction:
    def __init__(
        self,
        editor: Editor,
        label: str,
        selection: Selection,
        allowed: frozenset[str],
        base: Snapshot | None = None,
        *,
        rebase: bool = False,
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
        self._rebase = rebase

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

    def _record_removed_nodes(
        self, before: Iterable[Subpath], after: Iterable[Subpath]
    ) -> None:
        """Record vanished node identities after an authorized geometry edit."""
        original = {n.id for subpath in before for n in subpath.nodes}
        surviving = {n.id for subpath in after for n in subpath.nodes}
        self._record_remap({node_id: set() for node_id in original - surviving})

    def _add_geometries(self, geometries: Iterable[Geometry]) -> None:
        """Register new geometry; the enclosing change validates IDs and users."""
        self._working = replace(
            self._working, geometries=(*self._working.geometries, *geometries)
        )

    def _insert_siblings(
        self,
        object_id: str,
        children: tuple[Element, ...],
        geometries: Iterable[Geometry],
    ) -> tuple[str, ...]:
        """Insert authorized pieces above their source and extend edit scope.

        Callers authorize the source and decide whether the pieces replace it
        in selection. This helper changes only document storage and edit scope.
        """
        parent = self._working.ancestry(object_id)[-2]
        index = next(
            i for i, child in enumerate(parent.children) if child.id == object_id
        )
        self._working = self._working.replace_element(
            replace(
                parent,
                children=(
                    *parent.children[: index + 1],
                    *children,
                    *parent.children[index + 1 :],
                ),
            )
        )
        self._add_geometries(geometries)
        ids = tuple(child.id for child in children)
        self._ids |= frozenset(ids)
        return ids

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
        self._private_paints()
        self._working.validate()
        self._failed = False

    def _private_paints(self) -> None:
        """Copy private paint with copied shapes, and collect abandoned assets.

        Structural commands can copy paint or remove its owner (knife, split,
        join, insertion). Normalize once for every command so all those paths
        retain appearance without accidentally sharing a private asset.
        """
        document = self._working
        private = {e.id: e for e in document.elements() if e.paint_owner is not None}
        if not private:
            return
        for element in document.elements():
            servers = set(references(element)) & private.keys()
            attrs = dict(element.attributes)
            for server in servers:
                gradient = private[server]
                if gradient.paint_owner == element.id:
                    continue
                clone = replace(
                    gradient,
                    id=new_id("object"),
                    paint_owner=element.id,
                    children=tuple(
                        replace(c, id=new_id("object")) for c in gradient.children
                    ),
                )
                document = _add_definition(document, clone)
                for kind in ("fill", "stroke"):
                    if paint_server(element.get(kind)) == server:
                        attrs[kind] = f"url(#{clone.id})"
            if attrs != dict(element.attributes):
                document = document.replace_element(
                    replace(element, attributes=tuple(attrs.items()))
                )
        used = {ref for e in document.elements() for ref in references(e)}
        for server in private.keys() - used:
            self._check_locks(server, EditKind.PAINT, "fill")
            document = _without(document, server)
        self._working = document

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

    def set_fill(self, object_id: str, fill: str | LinearGradient | None) -> None:
        """Give an object a solid fill, or a linear gradient of its own.

        Private gradients have explicit ownership. SVG definitions are storage
        details; imported/shared gradients remain shared until replaced with a
        private fill. Refitting reuses private gradient and stop identities.
        """
        with self._change():
            document = self._working
            element = document.element(object_id)
            if element.tag in GRADIENTS:
                raise EditRejectedError("Gradients and stops have no fill of their own")
            if isinstance(fill, LinearGradient) and element.tag not in DRAWN:
                raise EditRejectedError(
                    "Only paths and basic shapes can have a gradient fill"
                )
            self._authorize(document.dependents({object_id}), EditKind.PAINT, "fill")
            old = paint_server(element.get("fill"))
            own = old is not None and document.element(old).paint_owner == object_id
            attrs = dict(element.attributes)
            if isinstance(fill, LinearGradient):
                if own and old is not None:
                    current = document.element(old)
                    self._check_locks(old, EditKind.PAINT, "fill")
                    made = fill.element(old, tuple(c.id for c in current.children))
                    updated = replace(
                        current, attributes=made.attributes, children=made.children
                    )
                    if updated != current:
                        document = document.replace_element(updated)
                else:
                    gradient = replace(
                        fill.element(new_id("object")), paint_owner=object_id
                    )
                    document = _add_definition(document, gradient)
                    attrs["fill"] = f"url(#{gradient.id})"
            else:
                if fill is None:
                    attrs.pop("fill", None)
                else:
                    attrs["fill"] = fill
            validate_attributes(element.tag, attrs)
            document = document.replace_element(
                replace(document.element(object_id), attributes=tuple(attrs.items()))
            )
            if (
                own
                and old is not None
                and old not in references(document.element(object_id))
            ):
                self._check_locks(old, EditKind.PAINT, "fill")
                document = _without(document, old)
            self._working = document

    def update_node(
        self, object_id: str, node_id: str, values: tuple[float, ...]
    ) -> None:
        with self._change():
            geometry = self._working.geometry_for(object_id)
            node = geometry.node(node_id)
            updated = replace(node, values=tuple(values))
            if updated == node:
                return
            candidate = self._working.replace_geometry(geometry.replace_node(updated))
            candidate = _carry_handles(self._working, candidate)
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
        """Check every changed geometry before exposing any of it.

        Geometry this transaction added is its own to shape: a trace can be
        snapped together before it lands without leave to edit geometry.
        """
        added = {g.id for g in self._working.geometries} - {
            g.id for g in self._base.document.geometries
        }
        for geometry in candidate.geometries:
            original = self._working.geometry(geometry.id)
            if geometry == original or geometry.id in added:
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

    def snap_edges(
        self, tolerance: float = 1.0, object_ids: frozenset[str] | None = None
    ) -> tuple[EdgeRef, ...]:
        """Snap the touching edges of the selected paths, or of *object_ids*.

        The result is ordinary geometry: nothing links the paths afterwards.
        Returns the front edge of each matched span, in root user space.
        Nothing changes before every check has passed, so a refusal leaves the
        transaction usable for other edits.
        """
        from vectrify.document.contact import snap_edges

        if self._closed or self._failed:
            raise EditRejectedError("Transaction is closed or has a failed edit")
        if self._selection.node_ids:
            raise EditRejectedError("Select whole paths to snap their edges")
        object_ids = self._selection.object_ids if object_ids is None else object_ids
        candidate, spans = snap_edges(self._working, object_ids, tolerance)
        for oid in object_ids:
            self._authorize(frozenset({oid}), EditKind.STRUCTURE)
        self._authorize_geometry_change(candidate)
        with self._change():
            self._working = candidate
        return spans

    def split_edge(
        self, object_id: str, node_id: str, t: float = 0.5
    ) -> tuple[str, ...]:
        """Insert a node on a line/cubic/closing edge.

        t follows this object's SVG edge direction. All existing endpoints retain
        their identities and pins. Returns the newly inserted node IDs.
        """
        with self._change():
            if not math.isfinite(t) or not 0 < t < 1:
                raise EditRejectedError(
                    "Split parameter must lie strictly between 0 and 1"
                )
            geometry = self._working.geometry_for(object_id)
            affected = self._working.geometry_users(geometry.id)
            self._authorize(affected, EditKind.STRUCTURE)
            self._authorize(affected, EditKind.GEOMETRY)
            if self._selection.node_ids and node_id not in self._selection.node_ids:
                raise EditRejectedError("Edge is outside the selected nodes")
            candidate, added = split_edges(
                self._working, EdgeRef(geometry.id, node_id), t
            )
            self._authorize_geometry_change(candidate)
            self._working = candidate
            return added

    def redraw_outline(
        self,
        object_id: str,
        contour_id: str,
        start: tuple[str, float],
        end: tuple[str, float],
        points: Sequence[tuple[str, Sequence[float]]],
        *,
        long_way: bool = False,
    ) -> tuple[str, ...]:
        """Replace the stretch of a contour between two places with new segments.

        A place is a segment's end node and a t along it, 1 at the node;
        *points* are the new segments' commands and local values, drawn from
        *start* to *end*. On a closed contour the shorter way round, in root
        user space, is replaced, or the longer one with *long_way*. Nodes
        outside the stretch keep their IDs; pinned points inside it refuse
        the edit. Returns the IDs of the new nodes.
        """
        with self._change():
            self._whole_objects()
            geometry = self._working.geometry_for(object_id)
            affected = self._working.geometry_users(geometry.id)
            self._authorize(affected, EditKind.STRUCTURE)
            self._authorize(affected, EditKind.GEOMETRY)
            updated, removed = redrawn(
                geometry,
                contour_id,
                start,
                end,
                points,
                matrix=root_matrix(self._working, object_id),
                long_way=long_way,
            )
            if any(geometry.node(n).pinned for n in removed):
                raise EditRejectedError("Unpin the points in the stretch to redraw it")
            self._working = self._working.replace_geometry(updated)
            self._record_remap({n: set() for n in removed})
            old = {n.id for s in geometry.subpaths for n in s.nodes}
            return tuple(
                n.id for s in updated.subpaths for n in s.nodes if n.id not in old
            )

    def delete_node(self, object_id: str, node_id: str) -> None:
        """Remove a point, the contour's start point included.

        The next point then starts the contour; a closed one still closes
        through the old start's neighbours. A contour left with fewer than
        two points (three when closed) is deleted, and a path left without
        contours is deleted too. A closed contour ending on its moveto shows
        one point for the two nodes, so either deletes both.
        """
        with self._change():
            geometry = self._working.geometry_for(object_id)
            node = geometry.node(node_id)
            subpath = next(s for s in geometry.subpaths if node in s.nodes)
            nodes = subpath.nodes
            last = len(nodes) - 1
            twins = (
                subpath.closed
                and last > 1
                and nodes[last].endpoint == nodes[0].endpoint
            )
            removed = {node_id}
            if twins and node_id in {nodes[0].id, nodes[last].id}:
                removed = {nodes[0].id, nodes[last].id}
            if any(geometry.node(n).pinned for n in removed):
                raise EditRejectedError("Endpoint is pinned")
            if self._selection.node_ids and node_id not in self._selection.node_ids:
                raise EditRejectedError("Node is outside the selected nodes")
            if len(nodes) - int(twins) - 1 < (3 if subpath.closed else 2):
                self._delete_contour(object_id, subpath)
                return
            affected = self._working.geometry_users(geometry.id)
            self._authorize(affected, EditKind.STRUCTURE)
            self._authorize(affected, EditKind.GEOMETRY)
            kept = [n for n in nodes if n.id not in removed]
            if nodes[0].id in removed:
                start = kept[0]
                kept[0] = replace(start, command="M", values=start.endpoint)
                # The curve into the new start closes the contour: it keeps
                # its handles as an explicit segment back to the start.
                if subpath.closed and start.command == "C":
                    kept.append(PathNode(new_id("node"), "C", start.values))
            updated = replace(
                geometry,
                subpaths=tuple(
                    replace(s, nodes=tuple(kept)) if s.id == subpath.id else s
                    for s in geometry.subpaths
                ),
            )
            self._working = self._working.replace_geometry(updated)
            self._record_remap({n: set() for n in removed})

    def delete_contour(self, object_id: str, node_id: str) -> None:
        """Remove the contour holding a point; a path left without contours
        is deleted."""
        with self._change():
            geometry = self._working.geometry_for(object_id)
            node = geometry.node(node_id)
            if self._selection.node_ids and node_id not in self._selection.node_ids:
                raise EditRejectedError("Node is outside the selected nodes")
            subpath = next(s for s in geometry.subpaths if node in s.nodes)
            self._delete_contour(object_id, subpath)

    def _delete_contour(self, object_id: str, subpath: Subpath) -> None:
        if any(n.pinned for n in subpath.nodes):
            raise EditRejectedError("Unpin the contour's endpoints to delete it")
        geometry = self._working.geometry_for(object_id)
        affected = self._working.geometry_users(geometry.id)
        self._authorize(affected, EditKind.STRUCTURE)
        self._authorize(affected, EditKind.GEOMETRY)
        if len(geometry.subpaths) == 1:
            owners = {
                e.id for e in self._working.elements() if e.geometry_id == geometry.id
            }
            if owners != {object_id}:
                raise EditRejectedError(
                    "Detach shared geometry before deleting its last contour"
                )
            self._remove_objects(frozenset({object_id}))
        else:
            self._working = self._working.replace_geometry(
                replace(
                    geometry,
                    subpaths=tuple(s for s in geometry.subpaths if s.id != subpath.id),
                )
            )
        self._record_remap({n.id: set() for n in subpath.nodes})

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
            from vectrify.document.regions import SHAPES, as_path, shape_geometry

            if element.tag in SHAPES:
                # A basic shape: a path of its outline, its paint kept.
                affected = self._working.dependents({object_id})
                self._authorize(affected, EditKind.GEOMETRY)
                if any(object_id in references(e) for e in self._working.elements()):
                    raise EditRejectedError(
                        "Detach references to this shape before making it a path"
                    )
                geometry = shape_geometry(element)
                self._working = replace(
                    self._working, geometries=(*self._working.geometries, geometry)
                ).replace_element(as_path(element, geometry))
                return object_id
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
        Pinned endpoints must be released first, since new contours cannot
        keep them. The asset keeps its ID; its nodes do not.
        """
        with self._change():
            self._whole_objects()
            original = self._working.geometry_for(object_id)
            affected = self._working.geometry_users(original.id)
            self._authorize(affected, EditKind.STRUCTURE)
            self._authorize(affected, EditKind.GEOMETRY)
            if any(n.pinned for s in original.subpaths for n in s.nodes):
                raise EditRejectedError("Unpin endpoints before replacing a contour")
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
        asking again for one moves it to the other side. Closed contours wrap
        through their closing segment; its endpoint and the moveto are one point.
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
        # Materialize an implicit closing line when it needs a handle. Keep
        # existing IDs, including both representations of the closing point.
        last = len(nodes) - 1
        if (
            subpath.closed
            and last > 0
            and position in {0, last}
            and nodes[last].endpoint != nodes[0].endpoint
            and count > 0
        ):
            nodes.append(PathNode(new_id("node"), "L", nodes[0].endpoint))
            last += 1
        seam = (
            subpath.closed
            and last > 0
            and nodes[last].endpoint == nodes[0].endpoint
            and position in {0, last}
        )
        incoming = position if position > 0 else None
        outgoing = position + 1 if position + 1 < len(nodes) else None
        if seam:
            incoming, outgoing = last, 1

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
        if dx == dy == 0:
            dx, dy = head[0] - point[0], head[1] - point[1]
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
        if geometry.node(node_id).handles_aligned:
            self.set_node_handle_alignment(object_id, node_id, True)

    def delete_handle(self, object_id: str, node_id: str, offset: int) -> None:
        """Retract just one control point, preserving the other handle."""
        if offset not in {0, 2}:
            raise EditRejectedError("Choose an incoming or outgoing handle")
        geometry = self._working.geometry_for(object_id)
        subpaths = list(geometry.subpaths)
        for index, subpath in enumerate(subpaths):
            nodes = list(subpath.nodes)
            for position, node in enumerate(nodes):
                if node.id != node_id:
                    continue
                if node.command != "C" or position == 0:
                    raise EditRejectedError("This segment has no curve handle")
                start, end = nodes[position - 1].endpoint, node.endpoint
                values = list(node.values)
                values[offset : offset + 2] = start if offset == 0 else end
                nodes[position] = (
                    replace(node, command="L", values=end)
                    if tuple(values[:2]) == start and tuple(values[2:4]) == end
                    else replace(node, values=tuple(values))
                )
                subpaths[index] = replace(subpath, nodes=tuple(nodes))
                self.reshape_path(
                    object_id, replace(geometry, subpaths=tuple(subpaths))
                )
                return
        raise DocumentError(f"Unknown path node: {node_id}")

    def set_node_handle_alignment(
        self, object_id: str, node_id: str, aligned: bool, side: int | None = None
    ) -> None:
        """Toggle a point's aligned handles, straightening them when enabled."""
        if aligned:
            self.straighten_node_handles(object_id, node_id, side)
        geometry = self._working.geometry_for(object_id)
        geometry.node(node_id)
        subpaths = []
        for subpath in geometry.subpaths:
            ids = {node_id}
            first, last = subpath.nodes[0], subpath.nodes[-1]
            if (
                subpath.closed
                and first.endpoint == last.endpoint
                and node_id in {first.id, last.id}
            ):
                ids.update((first.id, last.id))
            subpaths.append(
                replace(
                    subpath,
                    nodes=tuple(
                        replace(node, handles_aligned=aligned)
                        if node.id in ids
                        else node
                        for node in subpath.nodes
                    ),
                )
            )
        self.reshape_path(object_id, replace(geometry, subpaths=tuple(subpaths)))

    def move_handle(
        self, object_id: str, node_id: str, offset: int, point: tuple[float, float]
    ) -> None:
        """Move one handle, keeping its opposite aligned when toggled on."""
        if offset not in {0, 2}:
            raise EditRejectedError("Choose an incoming or outgoing handle")
        geometry = self._working.geometry_for(object_id)
        node = geometry.node(node_id)
        if node.command != "C":
            raise EditRejectedError("This segment has no curve handle")
        subpath = next(s for s in geometry.subpaths if node in s.nodes)
        index = subpath.nodes.index(node)
        anchor = subpath.nodes[index - 1] if offset == 0 else node
        values = list(node.values)
        values[offset : offset + 2] = point
        self.update_node(object_id, node_id, tuple(values))
        if anchor.handles_aligned:
            self.straighten_node_handles(object_id, anchor.id, offset)

    def straighten_node_handles(
        self, object_id: str, node_id: str, side: int | None = None
    ) -> None:
        """Align existing handles through their point, keeping their lengths.

        With a side (0 outgoing, 2 incoming), keep that handle fixed and
        rotate the other. Otherwise use the two directions' bisector.
        Retracted handles stay retracted; no new handles are introduced.
        """
        if side not in {None, 0, 2}:
            raise EditRejectedError("Choose an incoming or outgoing handle")
        geometry = self._working.geometry_for(object_id)
        subpaths = list(geometry.subpaths)
        for index, subpath in enumerate(subpaths):
            nodes = list(subpath.nodes)
            position = next((i for i, n in enumerate(nodes) if n.id == node_id), None)
            if position is None:
                continue
            point = nodes[position].endpoint
            last = len(nodes) - 1
            seam = (
                subpath.closed
                and last > 0
                and nodes[last].endpoint == nodes[0].endpoint
                and position in {0, last}
            )
            incoming, outgoing = (last, 1) if seam else (position, position + 1)
            if (
                incoming == 0
                or outgoing >= len(nodes)
                or nodes[incoming].command != "C"
                or nodes[outgoing].command != "C"
            ):
                return
            hin, hout = nodes[incoming].values[2:4], nodes[outgoing].values[:2]
            lin, lout = math.dist(point, hin), math.dist(point, hout)
            if not lin or not lout:
                return
            vin = ((point[0] - hin[0]) / lin, (point[1] - hin[1]) / lin)
            vout = ((hout[0] - point[0]) / lout, (hout[1] - point[1]) / lout)
            direction = (
                vin
                if side == 2
                else vout
                if side == 0
                else (vin[0] + vout[0], vin[1] + vout[1])
            )
            norm = math.hypot(*direction)
            if norm < 1e-12:
                direction, norm = vout, 1.0
            ux, uy = direction[0] / norm, direction[1] / norm
            for segment, offset, length, sign in (
                (incoming, 2, lin, -1),
                (outgoing, 0, lout, 1),
            ):
                if side == offset:
                    continue
                node = nodes[segment]
                values = list(node.values)
                values[offset : offset + 2] = (
                    point[0] + sign * ux * length,
                    point[1] + sign * uy * length,
                )
                nodes[segment] = replace(node, values=tuple(values))
            subpaths[index] = replace(subpath, nodes=tuple(nodes))
            self.reshape_path(object_id, replace(geometry, subpaths=tuple(subpaths)))
            return
        raise DocumentError(f"Unknown path node: {node_id}")

    def reshape_path(self, object_id: str, geometry: Geometry) -> None:
        """Give a path new contours while its surviving nodes keep their identity.

        Nodes whose ID the new geometry repeats are the same nodes, moved or
        not; IDs it lacks are removed and new IDs are inserted. Pinned
        endpoints must survive where they are. Only moving nodes needs
        geometry permission; adding or removing them needs structure too.
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
            self._working = self._working.replace_geometry(updated)
            self._record_remap({n: set() for n in old.keys() - new.keys()})

    def break_points(
        self, object_id: str, node_ids: Iterable[str]
    ) -> dict[str, set[str]]:
        """Cut a path's contours at points: an open line comes apart there,
        each piece ending on its own copy of the point, and a closed contour
        opens. Both copies stay selected. Returns the points' IDs, old to new:
        a closed contour's start may come out under other IDs."""
        geometry = self._working.geometry_for(object_id)
        copies: dict[str, set[str]] = {}
        for node_id in node_ids:
            geometry, mapping = break_at(geometry, node_id)
            copies.update(mapping)
        if not copies:
            raise EditRejectedError(
                "Break a line at a point between its ends, or a closed contour "
                "at any point: a line's ends are free already"
            )
        self.reshape_path(object_id, geometry)
        self._record_remap(copies)
        return copies

    def delete_segments(self, object_id: str, node_ids: frozenset[str]) -> None:
        """Take out the segments between neighbouring points among
        *node_ids*: the contour splits there, or a closed one opens."""
        geometry = self._working.geometry_for(object_id)
        removed: dict[str, set[str]] = {}
        segments = segments_among(geometry, node_ids)
        if not segments:
            raise EditRejectedError(
                "Select the two points at the ends of the segment to delete"
            )
        for segment in segments:
            geometry, gone = delete_segment(geometry, segment)
            removed.update(gone)
        if not geometry.subpaths:
            self.delete_objects(frozenset({object_id}))
            return
        self.reshape_path(object_id, geometry)
        self._record_remap(removed)

    def join_ends(
        self,
        object_ids: frozenset[str],
        reach: float = 0.0,
        *,
        ends: Sequence[tuple[str, str]] = (),
        curve: bool = True,
    ) -> tuple[str, ...]:
        """Join the open ends of the selected lines that continue each other.

        Ends at most *reach* apart in root user space pair, nearest and
        straightest first, when the line runs on across the gap; or just the
        two *ends* given, as (path, point) pairs. Joined lines become one
        contour, bridged by a curve, or a straight line without *curve*, where
        their ends do not meet. Each lands in the frontmost of its paths and
        keeps that path's paint; a path left without contours is deleted.
        Returns the paths holding the joined lines.
        """
        with self._change():
            self._whole_objects()
            document = self._working
            members = {
                e.id
                for oid in object_ids
                for e in Document(document.element(oid)).elements()
            }
            paths = [
                e
                for e in document.elements()
                if e.tag == "path"
                and e.id in members
                and not any(
                    a.tag in {"defs", "clipPath"} for a in document.ancestry(e.id)
                )
            ]
            if ends:
                paths = [p for p in paths if p.id in {oid for oid, _ in ends}]
            # Picked by hand, any path's free ends join; found by distance,
            # only drawn lines', never a filled shape's.
            if not ends:
                paths = [p for p in paths if path_style(document, p)["fill"] == "none"]
            frames = {path.id: root_matrix(document, path.id) for path in paths}
            contours: list[tuple[PathNode, ...]] = []
            owners: list[str] = []
            for path in paths:
                geometry = transformed_geometry(
                    document.geometry_for(path.id), frames[path.id]
                )
                for subpath in geometry.subpaths:
                    if not subpath.closed and len(subpath.nodes) > 1:
                        contours.append(subpath.nodes)
                        owners.append(path.id)
            all_ends = [
                end
                for i, nodes in enumerate(contours)
                for end in contour_ends(i, nodes)
            ]
            if ends:
                chosen = []
                for oid, nid in ends:
                    found = [
                        e
                        for e in all_ends
                        if owners[e.contour] == oid
                        and contours[e.contour][0 if e.side == 0 else -1].id == nid
                    ]
                    if not found:
                        raise EditRejectedError("Choose two points of paths to join")
                    chosen.append(found[0])
                if len(chosen) != 2 or chosen[0] == chosen[1]:
                    raise EditRejectedError("Choose two ends to join")
                pairs = [(chosen[0], chosen[1])]
            else:
                pairs = end_pairs(all_ends, reach)
            if not pairs:
                raise EditRejectedError(
                    "No two line ends here are close enough and in line to join: "
                    "zoom out to reach wider gaps, or pick the two ends in Nodes"
                )
            chains, merged = joined(contours, pairs, curve=curve)
            order = {p.id: i for i, p in enumerate(paths)}
            used = {c for members, _ in chains for c in members}
            landing: dict[str, list[Subpath]] = {}
            for members, subpath in chains:
                target = max((owners[c] for c in members), key=order.__getitem__)
                inverse = inverse_matrix(frames[target])
                local = transformed_geometry(Geometry("chain", (subpath,)), inverse)
                landing.setdefault(target, []).append(local.subpaths[0])
            changed = {owners[c] for c in used}
            for oid in changed:
                geometry = document.geometry_for(oid)
                self._authorize(
                    self._working.geometry_users(geometry.id), EditKind.STRUCTURE
                )
                self._authorize(
                    self._working.geometry_users(geometry.id), EditKind.GEOMETRY
                )
            moved = {n.id for c in used for n in contours[c]}
            emptied = set()
            for oid in changed:
                geometry = document.geometry_for(oid)
                subpaths = tuple(
                    s
                    for s in geometry.subpaths
                    if not any(n.id in moved for n in s.nodes)
                ) + tuple(landing.get(oid, ()))
                if subpaths:
                    self._working = self._working.replace_geometry(
                        replace(geometry, subpaths=subpaths)
                    )
                else:
                    emptied.add(oid)
            self._record_remap(merged)
            if emptied:
                target = next(iter(landing))
                gone = {document.geometry_for(oid).id for oid in emptied}
                self._remove_objects(frozenset(emptied))
                # Their lines live on in other paths, so their geometry goes.
                self._working = replace(
                    self._working,
                    geometries=tuple(
                        g for g in self._working.geometries if g.id not in gone
                    ),
                )
                self._record_object_remap({oid: {target} for oid in emptied})
            return tuple(p.id for p in paths if p.id in landing)

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
            private_paint = {
                kind
                for kind in ("fill", "stroke")
                if (server := paint_server(element.get(kind))) is not None
                and document.element(server).paint_owner == object_id
            }
            # Private paint belongs to each resulting shape, rather than the
            # wrapper. Their user space is unchanged; normalization copies it.
            child_attributes = tuple(
                a for a in element.attributes if a[0] in {"opacity", *private_paint}
            )
            children = tuple(
                Element(new_id("object"), "path", child_attributes, geometry_id=g.id)
                for g in geometries
            )
            group = Element(
                new_id("group"),
                "g",
                tuple(
                    a
                    for a in element.attributes
                    if a[0] not in {"opacity", *private_paint}
                ),
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
            self._working = replace(
                updated,
                geometries=tuple(g for g in document.geometries if g.id != geometry.id)
                + geometries,
            )
            ids = tuple(child.id for child in children)
            self._ids = (self._ids - {object_id}) | {group.id, *ids}
            self._record_object_remap({object_id: set(ids)})
            return ids

    def cut_paths(
        self, start: tuple[float, float], end: tuple[float, float]
    ) -> tuple[str, ...]:
        """Cut the selected paths the line from *start* to *end* crosses.

        The points are in root SVG user space. Each crossed filled path
        becomes two paths, one per side of the line, each compound if that
        side has several parts; both pieces meet on the same seam nodes, so
        they fit exactly without being linked. A stroke-only path comes apart
        where the segment crosses its lines: the pieces on the side with less
        of them become the second path, so a loop cut across comes away.
        Both keep the original's attributes, locks and stacking place; the
        first keeps its ID and geometry ID. Paths the line only enters or
        misses are left alone. Returns the IDs of the pieces.
        """
        with self._change():
            self._whole_objects()
            document = self._working
            pieces: list[str] = []
            for element in document.elements():
                if element.id not in self._ids or element.tag != "path":
                    continue
                ancestry = document.ancestry(element.id)
                if any(e.tag in {"defs", "clipPath"} for e in ancestry):
                    continue
                style = path_style(document, element)
                if style["fill"] == "none" and style["stroke"] == "none":
                    continue
                matrix = ancestry_matrix(ancestry)
                inverse = inverse_matrix(matrix)
                geometry = document.geometry_for(element.id)
                local = mapped_point(start, inverse), mapped_point(end, inverse)
                if style["fill"] == "none":
                    cut = cut_strokes(geometry, *local, new_id("geometry"))
                else:
                    cut = cut_geometry(
                        geometry,
                        style["fill-rule"],
                        *local,
                        geometry.id,
                        new_id("geometry"),
                    )
                if cut is None:
                    continue
                self._authorize(document.dependents({element.id}), EditKind.STRUCTURE)
                self._authorize(document.dependents({element.id}), EditKind.GEOMETRY)
                if any(element.id in references(e) for e in document.elements()):
                    raise EditRejectedError(
                        "This path is referenced; detach its instances before cutting"
                    )
                if sum(e.geometry_id == geometry.id for e in document.elements()) != 1:
                    raise EditRejectedError("Detach shared geometry before cutting")
                if any(n.pinned for s in geometry.subpaths for n in s.nodes):
                    raise EditRejectedError("Unpin endpoints before cutting a path")
                if not cut.second.subpaths:
                    # A closed line cut once only opens up.
                    self._working = self._working.replace_geometry(cut.first)
                    self._record_removed_nodes(geometry.subpaths, cut.first.subpaths)
                    pieces.append(element.id)
                    continue
                piece = replace(element, id=new_id("object"), geometry_id=cut.second.id)
                self._working = self._working.replace_geometry(cut.first)
                self._insert_siblings(element.id, (piece,), (cut.second,))
                self._record_removed_nodes(
                    geometry.subpaths, (*cut.first.subpaths, *cut.second.subpaths)
                )
                self._record_object_remap({element.id: {element.id, piece.id}})
                pieces.extend((element.id, piece.id))
            if not pieces:
                raise EditRejectedError(
                    "Drag the knife across a line, or across a filled shape "
                    "from outside it to outside it"
                )
            return tuple(pieces)

    def join_paths(
        self, object_ids: frozenset[str], *, color_source: str | None = None
    ) -> str:
        """Join regions at the frontmost position with mixed or source colors."""
        return self._merge_paths(object_ids, color_source)

    def combine_paths(
        self, object_ids: frozenset[str], *, paint_source: str | None = None
    ) -> str:
        """Collect unchanged contours into one path, using the frontmost paint.

        *paint_source* chooses another participating path's entire style.
        Transforms are resolved when necessary; clipping is never cut into
        the geometry. Overlaps retain the source style's compound fill rule.
        """
        return self._merge_paths(object_ids, paint_source, combine=True)

    def _merge_paths(
        self,
        object_ids: frozenset[str],
        color_source: str | None,
        *,
        combine: bool = False,
    ) -> str:
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
                return self._join_paths(object_ids, color_source, combine=combine)
            if any(set(references(e)) & groups for e in document.elements()):
                raise EditRejectedError(
                    "Detach references to these groups before joining"
                )
            self._authorize(document.dependents(groups), EditKind.STRUCTURE)
            self._authorize(document.dependents(groups), EditKind.GEOMETRY)
            joined = self._join_paths(
                frozenset(e.id for e in members.values() if e.tag == "path"),
                color_source,
                combine=combine,
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
        self,
        object_ids: frozenset[str],
        color_source: str | None = None,
        *,
        combine: bool = False,
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
            return self._join_across_groups(paths, color_source, combine=combine)
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
            if combine:
                return self._join_across_groups(paths, color_source, combine=True)
            raise EditRejectedError(
                "Paths must have matching transforms and clipping to join"
            )
        if attrs.get("clip-path", "none") != "none":
            raise EditRejectedError(
                "Move clipping to the containing group before joining"
            )
        styles = [path_style(document, p) for p in paths]
        if not combine and len({s["fill"] != "none" for s in styles}) > 1:
            raise EditRejectedError(
                "Join filled regions separately from stroke-only outlines"
            )
        if combine:
            source_index = next(
                (i for i, p in enumerate(paths) if p.id == color_source),
                len(paths) - 1,
            )
            paint = dict(styles[source_index])
            for path, style in zip(paths, styles, strict=True):
                for key, value in paint.items():
                    if value != style.get(key):
                        self._authorize(
                            document.dependents({path.id}), EditKind.PAINT, key
                        )
            attrs.update(paint)
        elif any(style != styles[-1] for style in styles):
            paint = join_paint(
                styles,
                painted_weights(document, paths),
                next((i for i, p in enumerate(paths) if p.id == color_source), None),
                document,
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
        if filled and not combine:
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
        )
        self._ids = (self._ids - removed) | {joined.id}
        self._record_object_remap({oid: {joined.id} for oid in removed})
        return joined.id

    def _join_across_groups(
        self,
        paths: list[Element],
        color_source: str | None,
        *,
        combine: bool = False,
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
        if combine:
            for chain in ancestries:
                start = next(i for i, a in enumerate(chain) if a.id == common.id) + 1
                if any(a.get("clip-path", "none") != "none" for a in chain[start:]):
                    raise EditRejectedError(
                        "Move clipping to the common containing group before combining"
                    )
        baked = [bake_group_path(document, p, common.id) for p in paths]
        styles = [style for _, style, _ in baked]
        if not combine and len({s["fill"] != "none" for s in styles}) > 1:
            raise EditRejectedError(
                "Join filled regions separately from stroke-only outlines"
            )
        if combine:
            source_index = next(
                (i for i, p in enumerate(paths) if p.id == color_source),
                len(paths) - 1,
            )
            paint = dict(styles[source_index])
        else:
            paint = join_paint(
                styles,
                painted_weights(document, paths),
                next((i for i, p in enumerate(paths) if p.id == color_source), None),
                document,
            )
        normalized = []
        updated = document
        for path, original, (geometry, style, changed) in zip(
            paths, originals, baked, strict=True
        ):
            if changed:
                if not combine and any(
                    n.pinned for s in original.subpaths for n in s.nodes
                ):
                    raise EditRejectedError(
                        "Unpin selected paths before resolving "
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
            resolved = (
                dict(paint)
                if combine
                else dict(paint, **{"fill-rule": style["fill-rule"]})
            )
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
        return self._join_paths(frozenset(path_ids), color_source, combine=combine)

    def fill_holes(self, object_id: str, hole_ids: frozenset[str]) -> None:
        """Fill explicitly chosen holes, including their nested contour islands."""
        with self._change():
            self._whole_objects()
            self._remove_holes(object_id, hole_ids, "fill")

    def holes_to_shapes(
        self, object_id: str, hole_ids: frozenset[str]
    ) -> tuple[str, ...]:
        """Turn chosen holes into shapes of their own, just above the path.

        Each hole's contour leaves the path, which fills there, and becomes a
        new path with the same paint. Islands inside a hole go with it and
        stay holes of the new shape, their direction alternating inward so
        either fill rule leaves them open.
        """
        with self._change():
            self._whole_objects()
            document = self._working
            element = document.element(object_id)
            if any(object_id in references(e) for e in document.elements()):
                raise EditRejectedError(
                    "This path is referenced; detach its instances first"
                )
            geometry = document.geometry_for(object_id)
            if sum(e.geometry_id == geometry.id for e in document.elements()) != 1:
                raise EditRejectedError("Detach shared geometry first")
            holes = self._remove_holes(object_id, hole_ids, "turn into shapes")
            order = [s.id for s in geometry.subpaths]
            # A hole inside a chosen hole's island already goes with it.
            chosen = sorted(
                (
                    holes[hid]
                    for hid in hole_ids
                    if not any(
                        hid in holes[other].subpath_ids
                        for other in hole_ids
                        if other != hid
                    )
                ),
                key=lambda h: order.index(h.id),
            )
            shapes = []
            for hole in chosen:
                subs = [s for s in geometry.subpaths if s.id in hole.subpath_ids]
                rings = [ring_points(s) for s in subs]
                polygons = [make_valid(Polygon(r)) for r in rings]
                contours = []
                for sub, ring, polygon in zip(subs, rings, polygons, strict=True):
                    depth = sum(
                        other is not polygon
                        and other.area > polygon.area
                        and other.covers(polygon)
                        for other in polygons
                    )
                    contour = Subpath(
                        new_id("subpath"),
                        tuple(
                            replace(n, id=new_id("node"), pinned=False)
                            for n in sub.nodes
                        ),
                        True,
                    )
                    if (signed_area(ring) > 0) != (depth % 2 == 0):
                        contour = reversed_subpath(contour)
                    contours.append(contour)
                shapes.append(Geometry(new_id("geometry"), tuple(contours)))
            children = tuple(
                Element(
                    new_id("object"),
                    "path",
                    element.attributes,
                    geometry_id=g.id,
                    locks=element.locks,
                )
                for g in shapes
            )
            return self._insert_siblings(object_id, children, shapes)

    def _remove_holes(
        self, object_id: str, hole_ids: frozenset[str], verb: str
    ) -> dict[str, Hole]:
        """Drop chosen hole contours and their islands from a path."""
        document = self._working
        holes = {hole.id: hole for hole in find_holes(document, object_id)}
        if not hole_ids or not hole_ids <= holes.keys():
            raise EditRejectedError(f"Choose existing holes to {verb}")
        geometry = document.geometry_for(object_id)
        self._authorize(document.geometry_users(geometry.id), EditKind.GEOMETRY)
        self._authorize(document.geometry_users(geometry.id), EditKind.STRUCTURE)
        removed = set().union(*(holes[hid].subpath_ids for hid in hole_ids))
        if any(n.pinned for s in geometry.subpaths if s.id in removed for n in s.nodes):
            raise EditRejectedError(f"Unpin the selected hole contours to {verb}")
        updated = replace(
            geometry,
            subpaths=tuple(s for s in geometry.subpaths if s.id not in removed),
        )
        self._working = document.replace_geometry(updated)
        self._record_removed_nodes(geometry.subpaths, updated.subpaths)
        return holes

    def cut_out_hole(self, object_ids: frozenset[str]) -> str:
        """Cut one of two filled paths out of the other as a hole.

        A path lying inside the other is cut out of it; when neither contains
        the other, the front one cuts the back one. The cutter's contours are
        added to the outer path, turned so they cut under its fill rule, and
        the cutter is deleted. Where added contours cannot reproduce the
        difference (a partial overlap), the outer outline is recomputed as a
        curved boolean difference instead. Returns the outer path's ID.
        """
        self._whole_objects()
        document = self._working
        elements = [e for e in document.elements() if e.id in object_ids]
        if len(object_ids) != 2 or len(elements) != 2:
            raise EditRejectedError("Select two paths: a shape and one to cut out")
        if any(
            e.tag != "path"
            or any(a.tag in {"defs", "clipPath"} for a in document.ancestry(e.id))
            for e in elements
        ):
            raise EditRejectedError("Select two drawing paths to cut a hole")
        styles = [path_style(document, e) for e in elements]
        if any(style["fill"] == "none" for style in styles):
            raise EditRejectedError("Both paths need a fill to cut a hole")
        matrices = [root_matrix(document, e.id) for e in elements]
        geometries = [document.geometry_for(e.id) for e in elements]
        regions = [
            mapped(filled_region(g, style["fill-rule"]), matrix)
            for g, style, matrix in zip(geometries, styles, matrices, strict=True)
        ]
        tolerance = 1e-6 * max(r.area for r in regions) + 1e-9
        if regions[0].intersection(regions[1]).area <= tolerance:
            raise EditRejectedError("The two paths do not overlap")
        # The back path is the outer one unless the front one contains it.
        inside = [
            regions[i].difference(regions[1 - i]).area <= 1e-4 * regions[i].area
            for i in (0, 1)
        ]
        outer, inner = (1, 0) if inside[0] and not inside[1] else (0, 1)
        outer_element, inner_element = elements[outer], elements[inner]
        outer_geometry, inner_geometry = geometries[outer], geometries[inner]
        if any(outer_element.id in references(e) for e in document.elements()):
            raise EditRejectedError(
                "This path is referenced; detach its instances before cutting a hole"
            )
        if sum(e.geometry_id == outer_geometry.id for e in document.elements()) != 1:
            raise EditRejectedError("Detach shared geometry before cutting a hole")
        rule, cutter_rule = styles[outer]["fill-rule"], styles[inner]["fill-rule"]
        cutter = transformed_geometry(
            inner_geometry,
            multiply(inverse_matrix(matrices[outer]), matrices[inner]),
        ).detached()
        cutter = replace(
            cutter,
            subpaths=tuple(
                replace(
                    s,
                    closed=True,
                    nodes=tuple(replace(n, pinned=False) for n in s.nodes),
                )
                for s in cutter.subpaths
            ),
        )
        expected = filled_region(outer_geometry, rule).difference(
            filled_region(cutter, cutter_rule)
        )
        if expected.area <= tolerance:
            raise EditRejectedError("Cutting this hole would leave nothing")
        for contours in (
            cutter.subpaths,
            tuple(reversed_subpath(s) for s in cutter.subpaths),
        ):
            candidate = replace(
                outer_geometry, subpaths=outer_geometry.subpaths + contours
            )
            result = filled_region(candidate, rule)
            if result.symmetric_difference(expected).area <= 1e-5 * expected.area:
                self.reshape_path(outer_element.id, candidate)
                break
        else:
            try:
                difference = pathops.op(
                    curve_path(outer_geometry, rule),
                    curve_path(cutter, cutter_rule),
                    pathops.PathOp.DIFFERENCE,
                )
            except pathops.PathOpsError as exc:
                raise EditRejectedError("Could not resolve the hole's outline") from exc
            self.replace_geometry(outer_element.id, path_geometry(difference))
        self.delete_objects(frozenset({inner_element.id}))
        return outer_element.id

    def extract_region(
        self,
        polygon: Sequence[tuple[float, float]],
        *,
        cut: bool = True,
        delete: bool = False,
    ) -> tuple[tuple[str, str | None], ...]:
        """Take the contours of the selected paths that lie inside *polygon*
        (root user space) into a path of their own, or delete them.

        Each path with contours inside gets one new path holding them, with
        its attributes, just above it in its group. With *cut*, contours
        crossing the edge are cut along it first (see ``split_geometry``);
        without, only contours wholly inside go. A path lying wholly inside
        is left alone (or deleted). A basic shape (rect, circle, ellipse,
        line) the region cuts becomes a path of its outline first, keeping
        its id and paint. Returns (path, new path or None) pairs for the
        paths it changed.
        """
        from vectrify.document.regions import (
            SHAPES,
            as_path,
            shape_geometry,
            split_geometry,
        )

        with self._change():
            self._whole_objects()
            document = self._working
            changed: list[tuple[str, str | None]] = []
            for element in document.elements():
                if element.id not in self._ids or element.tag not in SHAPES | {"path"}:
                    continue
                if any(
                    e.tag in {"defs", "clipPath"} for e in document.ancestry(element.id)
                ):
                    continue
                style = path_style(document, element)
                shape = element.tag in SHAPES
                filled = style["fill"] != "none" and element.tag != "line"
                if not filled and style["stroke"] == "none":
                    continue
                geometry = (
                    shape_geometry(element)
                    if shape
                    else document.geometry_for(element.id)
                )
                split = split_geometry(
                    geometry,
                    object_matrix(document, element.id),
                    polygon,
                    filled=filled,
                    rule=style["fill-rule"],
                    cut=cut,
                )
                if not split.inside:
                    continue
                self._authorize(document.dependents({element.id}), EditKind.STRUCTURE)
                self._authorize(document.dependents({element.id}), EditKind.GEOMETRY)
                if any(element.id in references(e) for e in document.elements()):
                    raise EditRejectedError(
                        f"{element.id} is referenced; detach its instances first"
                    )
                if (
                    not shape
                    and sum(e.geometry_id == geometry.id for e in document.elements())
                    != 1
                ):
                    raise EditRejectedError("Detach shared geometry first")
                if delete and any(n.pinned for s in split.inside for n in s.nodes):
                    raise EditRejectedError("Unpin the points before deleting them")
                if not split.outside:
                    if delete:
                        self._remove_objects(frozenset({element.id}))
                        changed.append((element.id, None))
                    continue
                surviving = split.outside
                if shape:
                    # Cut: the shape becomes the path of what is left of it.
                    kept = Geometry(new_id("geometry"), split.outside)
                    element = as_path(element, kept)
                    self._add_geometries((kept,))
                    self._working = self._working.replace_element(element)
                else:
                    kept = replace(geometry, subpaths=split.outside)
                    self._working = self._working.replace_geometry(kept)
                made: str | None = None
                if not delete:
                    taken = Geometry(new_id("geometry"), split.inside)
                    surviving += split.inside
                    piece = replace(
                        element, id=new_id("object"), geometry_id=taken.id, name=""
                    )
                    self._insert_siblings(element.id, (piece,), (taken,))
                    made = piece.id
                if not shape:
                    # A basic shape's generated outline never had document node IDs.
                    self._record_removed_nodes(geometry.subpaths, surviving)
                changed.append((element.id, made))
            if not changed:
                raise EditRejectedError(
                    "No contour of these paths lies inside the region"
                    + ("" if cut else "; cut=true takes the parts of ones crossing it")
                )
            return tuple(changed)

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
        return self.insert_objects(
            parent_id, (element,), index=index, geometries=geometries
        )[0]

    def insert_objects(
        self,
        parent_id: str,
        elements: tuple[Element, ...],
        *,
        index: int | None = None,
        geometries: tuple[Geometry, ...] = (),
    ) -> tuple[str, ...]:
        """Insert a batch atomically, including references between its subtrees."""
        with self._change():
            self._whole_objects()
            parent = self._working.element(parent_id)
            if parent.tag not in {"svg", "g", "defs", "clipPath"}:
                raise EditRejectedError("Insertion requires a container")
            self._authorize(self._working.dependents({parent_id}), EditKind.STRUCTURE)
            index = len(parent.children) if index is None else index
            if not 0 <= index <= len(parent.children):
                raise EditRejectedError("Insertion index is outside the container")
            self._add_geometries(geometries)
            self._working = self._working.replace_element(
                replace(
                    parent,
                    children=(
                        *parent.children[:index],
                        *elements,
                        *parent.children[index:],
                    ),
                )
            )
            self._ids |= frozenset(
                e.id for element in elements for e in Document(element).elements()
            )
            return tuple(element.id for element in elements)

    def delete_objects(self, object_ids: frozenset[str]) -> None:
        """Delete explicit subtrees; surviving references must never dangle."""
        with self._change():
            self._whole_objects()
            self._remove_objects(object_ids)

    def _remove_objects(self, object_ids: frozenset[str]) -> None:
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
                raise EditRejectedError("Delete or retarget dependent references first")
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

    def move_objects(
        self, object_ids: frozenset[str], parent_id: str, index: int
    ) -> None:
        """Move objects, in paint order, to *index* among the container's others.

        Zero is the back. An object moved to another group keeps its look:
        the transform and paint it inherited from the groups it leaves are
        written onto it. Group opacity and clipping cannot be carried that
        way, so those moves are refused, as are transform changes to paths
        with shared edges, whose links are kept in their old frame.
        """
        with self._change():
            self._whole_objects()
            document = self._working
            if not object_ids:
                raise EditRejectedError("Choose objects to move")
            if document.root.id in object_ids:
                raise EditRejectedError("Cannot move the document root")
            elements = {e.id: e for e in document.elements()}
            parents = {c.id: e for e in elements.values() for c in e.children}

            def chain(element_id: str) -> tuple[Element, ...]:
                if element_id not in elements:
                    raise EditRejectedError(f"Unknown object: {element_id}")
                ancestors = [elements[element_id]]
                while ancestors[-1].id in parents:
                    ancestors.append(parents[ancestors[-1].id])
                return tuple(reversed(ancestors))

            for object_id in object_ids:
                chain(object_id)
            target = chain(parent_id)
            if target[-1].tag not in {"svg", "g"} or any(
                a.tag in {"defs", "clipPath"} for a in target
            ):
                raise EditRejectedError("Move objects into a group or the drawing")
            if any(a.id in object_ids for a in target):
                raise EditRejectedError("Cannot move a group into itself")
            # A moved group takes its selected descendants along.
            moved = [
                e
                for e in document.elements()
                if e.id in object_ids
                and not any(a.id in object_ids for a in chain(e.id)[:-1])
            ]
            if any(a.tag in {"defs", "clipPath"} for e in moved for a in chain(e.id)):
                raise EditRejectedError(
                    "Definitions and clipping boundaries cannot be restacked"
                )
            moved_ids = {e.id for e in moved}
            self._authorize(document.dependents(moved_ids), EditKind.STRUCTURE)
            updated = [self._carry_context(e, chain(e.id)[:-1], target) for e in moved]

            def prune(element: Element) -> Element:
                return replace(
                    element,
                    children=tuple(
                        prune(c) for c in element.children if c.id not in moved_ids
                    ),
                )

            self._working = replace(document, root=prune(document.root))
            parent = self._working.element(parent_id)
            if not 0 <= index <= len(parent.children):
                raise EditRejectedError("Stacking index is outside the container")
            self._working = self._working.replace_element(
                replace(
                    parent,
                    children=(
                        *parent.children[:index],
                        *updated,
                        *parent.children[index:],
                    ),
                )
            )
            # Groups that now contain the objects, and their instances, change too.
            self._authorize(self._working.dependents(moved_ids), EditKind.STRUCTURE)

    def scale_objects(
        self,
        object_ids: frozenset[str],
        anchor: tuple[float, float],
        scale: tuple[float, float],
    ) -> None:
        """Scale objects by *scale* about *anchor*, both in root user space.

        Each object's geometry stays as it is: the scale is composed onto its
        own transform, in its parent's frame, so it lands where the scale puts
        it whatever transforms its groups have. A selected object inside
        another selected one scales with it, once. Strokes keep their width:
        stroke widths in the scaled objects are divided by the scale's mean,
        except where paint is locked. A non-uniform scale still stretches a
        stroke along its longer axis, as SVG strokes follow their transform.
        Locked position or geometry refuses the edit.
        """
        sx, sy = scale
        if not all(math.isfinite(v) for v in (*anchor, sx, sy)) or min(sx, sy) <= 0:
            raise EditRejectedError("Resize needs a positive, finite scale")
        document = self._working
        if not object_ids:
            raise EditRejectedError("Choose objects to resize")
        if document.root.id in object_ids:
            raise EditRejectedError("Cannot resize the document root")
        ax, ay = anchor
        page = (sx, 0.0, 0.0, sy, ax - sx * ax, ay - sy * ay)
        changes: list[tuple[str, dict[str, str | None]]] = []
        for object_id in sorted(object_ids):
            ancestry = document.ancestry(object_id)
            if any(a.id in object_ids for a in ancestry[:-1]):
                continue
            if any(a.tag in {"defs", "clipPath"} for a in ancestry):
                raise EditRejectedError(
                    "Definitions and clipping boundaries cannot be resized"
                )
            for kind, name in (
                (EditKind.TRANSFORM, "position"),
                (EditKind.GEOMETRY, "geometry"),
            ):
                if any(kind in a.locks for a in ancestry):
                    raise EditRejectedError(
                        f"{object_id}: {name} is locked; unlock it to resize"
                    )
            frame = ancestry_matrix(ancestry[:-1])
            local = multiply(inverse_matrix(frame), multiply(page, frame))
            element = ancestry[-1]
            changes.append(
                (
                    object_id,
                    {
                        "transform": _transform_text(
                            multiply(local, transform(element.get("transform")))
                        )
                    },
                )
            )
            changes.extend(self._stroke_widths(ancestry, math.sqrt(sx * sy)))
        for object_id, attributes in changes:
            self.set_attributes(object_id, attributes)

    def _stroke_widths(
        self, ancestry: tuple[Element, ...], factor: float
    ) -> list[tuple[str, dict[str, str | None]]]:
        """Stroke widths that keep the strokes of a scaled object as wide.

        The object takes its effective width and its descendants their own;
        nothing changes unless something in it paints a stroke.
        """
        element = ancestry[-1]
        stroke, width = INITIAL_PAINT["stroke"], INITIAL_PAINT["stroke-width"]
        for ancestor in ancestry:
            stroke = ancestor.get("stroke", stroke) or stroke
            width = ancestor.get("stroke-width", width) or width
        stroked = stroke != "none"
        widths: dict[str, str] = {element.id: width}

        def walk(item: Element, painted: str) -> None:
            nonlocal stroked
            painted = item.get("stroke", painted) or painted
            stroked = stroked or painted != "none"
            for child in item.children:
                if child.get("stroke-width"):
                    widths[child.id] = str(child.get("stroke-width"))
                walk(child, painted)

        walk(element, stroke)
        if not stroked:
            return []
        found: list[tuple[str, dict[str, str | None]]] = []
        for object_id, value in widths.items():
            match = LENGTH.fullmatch(value)
            locked = any(
                EditKind.PAINT in a.locks or "stroke-width" in a.locks
                for a in self._working.ancestry(object_id)
            )
            if match and not locked:
                found.append(
                    (object_id, {"stroke-width": f"{float(match[1]) / factor:.6g}"})
                )
        return found

    def _carry_context(
        self,
        element: Element,
        old: tuple[Element, ...],
        new: tuple[Element, ...],
    ) -> Element:
        """*element* with what it inherited under *old* kept when under *new*."""
        common = 0
        while common < min(len(old), len(new)) and old[common].id == new[common].id:
            common += 1
        if common == len(old) == len(new):
            return element
        for group in (*old[common:], *new[common:]):
            if (
                float(group.get("opacity", "1") or "1") != 1
                or group.get("clip-path", "none") != "none"
            ):
                raise EditRejectedError(
                    "Group opacity or clipping would change how the moved "
                    "objects look; resolve it first"
                )
        attrs = dict(element.attributes)

        def inherited(ancestors: tuple[Element, ...]) -> dict[str, str]:
            style = dict(INITIAL_PAINT)
            for ancestor in ancestors:
                style.update(
                    (k, v) for k, v in ancestor.attributes if k in PAINT - {"opacity"}
                )
            return style

        before, after = inherited(old), inherited(new)
        changes = {
            key: value
            for key, value in before.items()
            if key not in attrs and after[key] != value
        }
        frames = [ancestry_matrix(ancestors) for ancestors in (old, new)]
        # The local transform that keeps the object where it was on the page.
        delta = tuple(
            0.0 if abs(v) < 1e-12 else v
            for v in multiply(inverse_matrix(frames[1]), frames[0])
        )
        subtree = Document(element).elements()
        if not all(
            math.isclose(a, b, abs_tol=1e-12)
            for a, b in zip(delta, IDENTITY, strict=True)
        ):
            a, b, c, d, e, f = delta
            local = (
                f"translate({e:.15g} {f:.15g})"
                if (a, b, c, d) == (1, 0, 0, 1)
                else f"matrix({' '.join(f'{v:.15g}' for v in delta)})"
            )
            changes["transform"] = f"{local} {attrs.get('transform', '')}".strip()
        if not changes:
            return element
        ids = {e.id for e in subtree}
        if any(set(references(e)) & ids for e in self._working.elements()):
            raise EditRejectedError(
                "Detach instances of the moved objects before moving them "
                "to another group"
            )
        affected = self._working.dependents({element.id})
        for key in changes:
            if key == "transform":
                self._authorize(affected, EditKind.TRANSFORM, key)
            else:
                self._authorize(affected, EditKind.PAINT, key)
        attrs.update(changes)
        validate_attributes(element.tag, attrs)
        return replace(element, attributes=tuple(attrs.items()))

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
        if self._rebase and self._base.revision != self._editor.snapshot.revision:
            current = self._editor.snapshot
            self._working = merge_edit(
                self._base.document, self._working, current.document
            )
            return self._editor._apply(
                self._working,
                self._label,
                current.revision,
                self._remap_selection(current.selection),
            )
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
