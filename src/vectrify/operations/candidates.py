"""Replay an edited SVG onto the document as ordinary transaction commands.

Methods that work on exported SVG text, where every element keeps its object
ID, hand their result back this way. A candidate is accepted only by diffing it
against the document it came from and repeating each difference through the
transaction: deletions,
attribute edits, node moves (in place when the path structure is unchanged,
otherwise a contour replacement), insertions and sibling reorders. The
transaction then enforces selection, permissions, locks and pins exactly as it
does for manual edits, so a candidate that changes anything outside the
transaction's scope or permissions is rejected.
"""

from __future__ import annotations

from dataclasses import dataclass

from vectrify.document import Document, DocumentError, import_svg
from vectrify.document.editor import Transaction
from vectrify.document.model import Element, Geometry


class CandidateRejectedError(DocumentError):
    """The candidate changes something no command here can express."""


@dataclass(frozen=True)
class Replay:
    edits: int


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
        contours: bool = False,
    ):
        self.contours = contours
        self.tx = tx
        self.candidate = candidate
        self.edits = 0

    def run(self) -> Replay:
        document = self.tx.preview
        candidate = self.candidate
        old_ids = {e.id for e in document.elements()} - {document.root.id}
        new_ids = {e.id for e in candidate.elements()} - {candidate.root.id}

        removed = _topmost(old_ids - new_ids, document)
        if removed:
            self.tx.delete_objects(frozenset(removed))
            self.edits += 1

        if dict(document.root.attributes) != dict(candidate.root.attributes):
            raise CandidateRejectedError("Candidate changes the document root")
        for oid in sorted(old_ids & new_ids):
            if oid not in {e.id for e in self.tx.preview.elements()}:
                continue
            self.element(oid)

        for oid in _topmost(new_ids - old_ids, candidate):
            parent = candidate.ancestry(oid)[-2]
            parent_id = document.root.id if parent is candidate.root else parent.id
            if parent_id not in {e.id for e in self.tx.preview.elements()}:
                raise CandidateRejectedError(
                    f"{oid}: added under a removed or new container"
                )
            element = candidate.element(oid)
            self.tx.insert_object(
                parent_id,
                element,
                geometries=_subtree_geometries(element, candidate),
            )
            self.edits += 1

        for parent in [candidate.root, *candidate.elements()]:
            self.order(parent)
        return Replay(self.edits)

    def element(self, oid: str) -> None:
        old = self.tx.preview.element(oid)
        new = self.candidate.element(oid)
        if old.tag != new.tag:
            raise CandidateRejectedError(f"{oid}: element type changed")
        before, after = dict(old.attributes), dict(new.attributes)
        changes = {
            key: after.get(key)
            for key in before.keys() | after.keys()
            if before.get(key) != after.get(key)
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
        if _same_topology(current, wanted):
            values = {
                m.id: n.values
                for a, b in zip(current.subpaths, wanted.subpaths, strict=True)
                for m, n in zip(a.nodes, b.nodes, strict=True)
                if m.values != n.values
            }
            self.tx.update_nodes(oid, values)
        else:
            if not self.contours:
                raise CandidateRejectedError(f"{oid}: path structure changed")
            self.tx.replace_geometry(oid, wanted)
        self.edits += 1

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
        # Children the candidate does not list stay after the ones it does.
        kept = [c for c in current if c not in set(target)]
        target = target + kept if kept else target
        if target == current:
            return
        for index, child in enumerate(target):
            if current[index] != child:
                self.tx.reorder_object(child, index)
                current.remove(child)
                current.insert(index, child)
        self.edits += 1


def replay(tx: Transaction, svg: str, *, contours: bool = False) -> Replay:
    """Repeat *svg*'s differences from ``tx.preview`` as commands in *tx*.

    *contours* lets the replay replace a path's contours when its structure
    changed.
    """
    try:
        candidate = import_svg(svg)
    except DocumentError as exc:
        raise CandidateRejectedError(f"Candidate is not editable SVG: {exc}") from exc
    return _Replayer(tx, candidate, contours).run()
