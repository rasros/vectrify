"""Replay a mutated SVG onto the document as ordinary transaction commands.

Search methods work on exported SVG text, where every element keeps its object
ID. A candidate is accepted only by diffing it against the document it came
from and repeating each difference through the transaction: attribute edits,
fixed-topology node moves and sibling reorders. The transaction then enforces
selection, permissions, locks and pins exactly as it does for manual edits,
and node identities survive because nodes are updated in place.
"""

from __future__ import annotations

from vectrify.document import Document, DocumentError, import_svg
from vectrify.document.editor import Transaction
from vectrify.document.model import Element, Geometry
from vectrify.formats.svg.selection import MutationScope
from vectrify.operations.contract import OperationRequest


class CandidateRejectedError(DocumentError):
    """The candidate changes something no command here can express."""


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


def _replay_geometry(
    tx: Transaction, document: Document, candidate: Document, element: Element
) -> None:
    if element.tag != "path":
        return
    old = document.geometry_for(element.id)
    new = candidate.geometry_for(element.id)
    if old.path_data() == new.path_data():
        return
    if not _same_topology(old, new):
        raise CandidateRejectedError(f"{element.id}: path structure changed")
    values = {
        m.id: n.values
        for a, b in zip(old.subpaths, new.subpaths, strict=True)
        for m, n in zip(a.nodes, b.nodes, strict=True)
        if m.values != n.values
    }
    tx.update_nodes(element.id, values)


def _replay_order(tx: Transaction, old: Element, new: Element) -> None:
    current = [c.id for c in old.children]
    target = [c.id for c in new.children]
    if sorted(current) != sorted(target):
        raise CandidateRejectedError(f"{old.id}: children were added or removed")
    for index, child in enumerate(target):
        if current[index] != child:
            tx.reorder_object(child, index)
            current.remove(child)
            current.insert(index, child)


def replay(tx: Transaction, svg: str) -> int:
    """Repeat *svg*'s differences from ``tx.preview`` in *tx*; returns edits made."""
    document = tx.preview
    try:
        candidate = import_svg(svg)
    except DocumentError as exc:
        raise CandidateRejectedError(f"Candidate is not editable SVG: {exc}") from exc
    old_ids = {e.id for e in document.elements()}
    new_ids = {e.id for e in candidate.elements()}
    # The root's id is regenerated on every import; everything else is kept.
    old_ids.discard(document.root.id)
    new_ids.discard(candidate.root.id)
    if old_ids != new_ids:
        raise CandidateRejectedError("Candidate adds or removes objects")
    edits = 0
    pairs = [(document.root, candidate.root)] + [
        (document.element(i), candidate.element(i)) for i in sorted(old_ids)
    ]
    for old, new in pairs:
        if old.tag != new.tag:
            raise CandidateRejectedError(f"{old.id}: element type changed")
        before, after = dict(old.attributes), dict(new.attributes)
        changes = {
            key: after.get(key)
            for key in before.keys() | after.keys()
            if before.get(key) != after.get(key)
        }
        if changes:
            if old is document.root:
                raise CandidateRejectedError("Candidate changes the document root")
            tx.set_attributes(old.id, changes)
            edits += 1
        if old.tag == "path":
            before_geometry = tx.preview.geometry_for(old.id)
            _replay_geometry(tx, document, candidate, old)
            edits += tx.preview.geometry_for(old.id) != before_geometry
        if [c.id for c in old.children] != [c.id for c in new.children]:
            _replay_order(tx, old, new)
            edits += 1
    return edits


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
