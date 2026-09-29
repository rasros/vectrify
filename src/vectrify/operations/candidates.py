"""Replay an edited SVG onto the document as ordinary transaction commands.

Search and LLM methods work on exported SVG text, where every element keeps
its object ID. A candidate is accepted only by diffing it against the document
it came from and repeating each difference through the transaction: deletions,
attribute edits, node moves (in place when the path structure is unchanged,
otherwise a contour replacement), insertions and sibling reorders. The
transaction then enforces selection, permissions, locks and pins exactly as it
does for manual edits.

Strict replay rejects a candidate that changes anything outside the
transaction's scope or permissions. Lenient replay, for LLM replies that were
only asked to stay in scope, skips those changes and counts them instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from vectrify.document import Document, DocumentError, EditKind, import_svg
from vectrify.document.editor import Transaction
from vectrify.document.model import Element, Geometry
from vectrify.document.svg import PAINT
from vectrify.formats.svg.selection import MutationScope
from vectrify.operations.contract import OperationRequest


class CandidateRejectedError(DocumentError):
    """The candidate changes something no command here can express."""


@dataclass(frozen=True)
class Replay:
    edits: int
    # Changes a lenient replay left out because they were not permitted.
    skipped: int = 0


def _same_topology(old: Geometry, new: Geometry) -> bool:
    return len(old.subpaths) == len(new.subpaths) and all(
        a.closed == b.closed
        and len(a.nodes) == len(b.nodes)
        and all(
            m.command == n.command and len(m.values) == len(n.values)
            for m, n in zip(a.nodes, b.nodes, strict=True)
        )
        for a, b in zip(old.subpaths, new.subpaths, strict=True)
    )


def _kind(key: str) -> str:
    if key in PAINT:
        return EditKind.PAINT.value
    if key == "transform":
        return EditKind.TRANSFORM.value
    return EditKind.GEOMETRY.value


def _topmost(ids: set[str], document: Document) -> list[str]:
    """IDs whose parent is not itself in *ids*, in document order."""
    return [
        e.id
        for e in document.elements()
        if e.id in ids
        and not any(a.id in ids for a in document.ancestry(e.id) if a.id != e.id)
    ]


def _subtree_geometries(element: Element, candidate: Document) -> tuple:
    found: dict[str, Geometry] = {}
    for item in Document(element).elements():
        if item.tag == "path" and item.geometry_id:
            geometry = candidate.geometry(item.geometry_id)
            found[geometry.id] = geometry
    return tuple(found.values())


class _Replayer:
    def __init__(
        self,
        tx: Transaction,
        candidate: Document,
        lenient: bool,
        baseline: Document | None = None,
    ):
        self.tx = tx
        self.candidate = candidate
        self.baseline = baseline
        self.lenient = lenient
        self.scope = tx.scope
        self.allowed = tx.allowed
        self.edits = 0
        self.skipped = 0

    def permitted(self, object_id: str | None, *kinds: str) -> bool:
        """In lenient mode, drop what the transaction would refuse anyway."""
        if not self.lenient:
            return True
        ok = (object_id is None or object_id in self.scope) and all(
            k in self.allowed for k in kinds
        )
        if not ok:
            self.skipped += 1
        return ok

    def reject(self, message: str) -> None:
        if not self.lenient:
            raise CandidateRejectedError(message)
        self.skipped += 1

    def run(self) -> Replay:
        document = self.tx.preview
        candidate = self.candidate
        old_ids = {e.id for e in document.elements()} - {document.root.id}
        new_ids = {e.id for e in candidate.elements()} - {candidate.root.id}

        removed = [
            oid
            for oid in _topmost(old_ids - new_ids, document)
            if self.permitted(oid, "structure")
        ]
        if removed:
            self.tx.delete_objects(frozenset(removed))
            self.edits += 1

        if dict(document.root.attributes) != dict(candidate.root.attributes):
            self.reject("Candidate changes the document root")
        for oid in sorted(old_ids & new_ids):
            if oid not in {e.id for e in self.tx.preview.elements()}:
                continue
            self.element(oid)

        for oid in _topmost(new_ids - old_ids, candidate):
            parent = candidate.ancestry(oid)[-2]
            parent_id = document.root.id if parent is candidate.root else parent.id
            if parent_id not in {e.id for e in self.tx.preview.elements()}:
                self.reject(f"{oid}: added under a removed or new container")
                continue
            if not self.permitted(parent_id, "structure"):
                continue
            element = candidate.element(oid)
            self.tx.insert_object(
                parent_id,
                element,
                geometries=_subtree_geometries(element, candidate),
            )
            self.edits += 1

        for parent in [candidate.root, *candidate.elements()]:
            self.order(parent)
        return Replay(self.edits, self.skipped)

    def element(self, oid: str) -> None:
        old = self.tx.preview.element(oid)
        new = self.candidate.element(oid)
        if old.tag != new.tag:
            self.reject(f"{oid}: element type changed")
            return
        before, after = dict(old.attributes), dict(new.attributes)
        same = self._baseline(oid)
        unchanged = dict(same.attributes) if same is not None else {}
        changes = {
            key: after.get(key)
            for key in before.keys() | after.keys()
            if before.get(key) != after.get(key)
            and not (same is not None and unchanged.get(key) == after.get(key))
            and self.permitted(oid, _kind(key))
        }
        if changes:
            self.tx.set_attributes(oid, changes)
            self.edits += 1
        if old.tag == "path":
            self.geometry(oid)

    def geometry(self, oid: str) -> None:
        current = self.tx.preview.geometry_for(oid)
        wanted = self.candidate.geometry_for(oid)
        if current.path_data() == wanted.path_data():
            return
        same = self._baseline(oid)
        if (
            same is not None
            and same.tag == "path"
            and self.baseline is not None
            and self.baseline.geometry_for(oid).path_data() == wanted.path_data()
        ):
            return
        if _same_topology(current, wanted):
            if not self.permitted(oid, "geometry"):
                return
            values = {
                m.id: n.values
                for a, b in zip(current.subpaths, wanted.subpaths, strict=True)
                for m, n in zip(a.nodes, b.nodes, strict=True)
                if m.values != n.values
            }
            self.tx.update_nodes(oid, values)
        else:
            if not self.lenient:
                raise CandidateRejectedError(f"{oid}: path structure changed")
            if not self.permitted(oid, "geometry", "structure"):
                return
            self.tx.replace_geometry(oid, wanted)
        self.edits += 1

    def _baseline(self, oid: str) -> Element | None:
        """The element as the original looks after the same normalization."""
        if self.baseline is None:
            return None
        try:
            return self.baseline.element(oid)
        except DocumentError:
            return None

    def order(self, parent: Element) -> None:
        preview = self.tx.preview
        parent_id = preview.root.id if parent is self.candidate.root else parent.id
        try:
            current_parent = preview.element(parent_id)
        except DocumentError:
            return
        current = [c.id for c in current_parent.children]
        present = set(current)
        target = [c.id for c in parent.children if c.id in present]
        # Children only the document has (kept by a lenient replay) stay put.
        kept = [c for c in current if c not in set(target)]
        target = target + kept if kept else target
        if target == current:
            return
        if not self.permitted(parent_id, "structure"):
            return
        for index, child in enumerate(target):
            if current[index] != child:
                self.tx.reorder_object(child, index)
                current.remove(child)
                current.insert(index, child)
        self.edits += 1


def replay(
    tx: Transaction,
    svg: str,
    *,
    lenient: bool = False,
    baseline: str | None = None,
) -> Replay:
    """Repeat *svg*'s differences from ``tx.preview`` as commands in *tx*.

    *baseline* is the original after the same rewriting the candidate went
    through (e.g. normalization that rounds coordinates). Values the candidate
    shares with it are treated as unchanged, so the rewrite itself is never
    replayed and the document keeps its full precision.
    """
    try:
        candidate = import_svg(svg)
        reference = import_svg(baseline) if baseline is not None else None
    except DocumentError as exc:
        raise CandidateRejectedError(f"Candidate is not editable SVG: {exc}") from exc
    return _Replayer(tx, candidate, lenient, reference).run()


def mutation_scope(request: OperationRequest) -> MutationScope:
    """The elements and edit kinds a search may touch for *request*."""
    document = request.snapshot.document
    selection = request.snapshot.selection
    if selection.whole_document:
        ids = frozenset(child.id for child in document.root.children)
    else:
        ids = frozenset(selection.object_ids)
    if not ids:
        raise DocumentError("Select objects, or the whole drawing, to improve")
    kinds = request.permissions.allowed
    if not kinds:
        raise DocumentError("Allow at least one kind of change")
    return MutationScope(ids, kinds)
