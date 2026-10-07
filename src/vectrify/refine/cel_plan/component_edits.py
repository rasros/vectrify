"""Sealed, fully declared replacements inside one bounded owned component.

This is a separate edit contract from the small local-path contract. It keeps
every changed path visible to scoring and diagnostics. A full source/document
seal prevents reuse after unrelated edits; native context and checkpoint checks
remain independent of the seal.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace

from vectrify.document import Document
from vectrify.refine.cel_plan.model import StageInterruptedError, Work
from vectrify.refine.cel_plan.ownership import Partition
from vectrify.refine.cel_plan.refine import _bounds

# Reuse the established search seed footprint, without widening local edits.
MAX_OBJECTS = 8_192
MAX_NODES = 32_000
MAX_ELEMENTS = MAX_OBJECTS + MAX_NODES
MAX_SERIAL_BYTES = 16 * 1024 * 1024


def _check(work):
    if work.interrupted:
        raise StageInterruptedError("Component dependency validation interrupted")


def _guard(document: Document, work: Work):
    objects = nodes = 0
    for count, element in enumerate(document.elements(), start=1):
        _check(work)
        objects += element.tag in {"path", "use", "rect", "ellipse", "circle", "line"}
        if count > MAX_ELEMENTS or objects > MAX_OBJECTS:
            raise ValueError("Component document exceeds object bounds")
    for geometry in document.geometries:
        _check(work)
        nodes += sum(len(s.nodes) for s in geometry.subpaths)
        if nodes > MAX_NODES:
            raise ValueError("Component document exceeds geometry bounds")


def signature(document: Document, partition: Partition, work: Work) -> str:
    _guard(document, work)
    payload = repr((document, partition.metadata())).encode()
    if len(payload) > MAX_SERIAL_BYTES:
        raise ValueError("Component dependency payload exceeds byte bounds")
    _check(work)
    return hashlib.sha256(payload).hexdigest()


def _outside(document, parent, ids, work):
    """Allow only private paint resources of declared objects outside the group.

    Document validation proves these resources cannot be shared by other paint.
    All other definitions, geometry, attributes and external order stay exact.
    """

    def visit(element):
        _check(work)
        if element.id == parent:
            return ("component", parent)
        if element.tag == "linearGradient" and element.paint_owner in ids:
            return None
        children = tuple(v for c in element.children if (v := visit(c)) is not None)
        if element.tag == "defs" and not children and not element.attributes:
            return None
        geometry = (
            document.geometry(element.geometry_id) if element.geometry_id else None
        )
        return replace(element, children=()), geometry, children

    return visit(document.root)


@dataclass(frozen=True)
class ComponentEdit:
    parent: str
    source: str

    @classmethod
    def bind(cls, document: Document, partition: Partition, parent: str, work: Work):
        return cls(parent, signature(document, partition, work))

    def validate(self, before, after, original, proposed, ids, box, work):
        if original is None or proposed is None:
            raise ValueError("Component replacement requires complete source ownership")
        if self.source != signature(before, original, work):
            raise ValueError("Component replacement has stale source dependencies")
        _guard(after, work)
        after.validate()
        _check(work)
        if len(ids) > 2 * MAX_OBJECTS or len(ids) != len(set(ids)):
            raise ValueError("Component replacement has invalid object declarations")
        declared = set(ids)
        groups = []
        frames = []
        records = []
        for document in (before, after):
            _check(work)
            ancestry = document.ancestry(self.parent)
            group = ancestry[-1]
            if group.tag not in {"g", "svg"}:
                raise ValueError("Component replacement requires one path container")
            if any(c.tag != "path" or c.children for c in group.children):
                raise ValueError("Component replacement cannot edit nested objects")
            groups.append(group)
            frames.append(tuple(replace(a, children=()) for a in ancestry))
            records.append(
                {c.id: (c, document.geometry_for(c.id)) for c in group.children}
            )
        if frames[0] != frames[1]:
            raise ValueError("Component replacement changed its parent frame")
        known = set(records[0]) | set(records[1])
        if not declared or not declared.issubset(known):
            raise ValueError(
                "Component replacement declares objects outside its parent"
            )
        if any(
            records[0].get(i) != records[1].get(i) and i not in declared for i in known
        ):
            raise ValueError("Component replacement changed an undeclared path")
        for oid in declared & records[0].keys():
            old, geometry = records[0][oid]
            if records[0][oid] == records[1].get(oid):
                continue
            if any(a.locks for a in before.ancestry(oid)) or any(
                n.pinned for s in geometry.subpaths for n in s.nodes
            ):
                raise ValueError("Component replacement changed a protected path")
            if oid in records[1] and old.locks != records[1][oid][0].locks:
                raise ValueError("Component replacement changed path protections")
        remaining = (set(records[0]) & set(records[1])) - declared
        if [c.id for c in groups[0].children if c.id in remaining] != [
            c.id for c in groups[1].children if c.id in remaining
        ]:
            raise ValueError("Component replacement reordered undeclared paths")
        if _outside(before, self.parent, declared, work) != _outside(
            after, self.parent, declared, work
        ):
            raise ValueError("Component replacement changed external dependencies")
        for document, record in zip((before, after), records, strict=True):
            for oid in declared & record.keys():
                _check(work)
                a, b, c, d = _bounds(document, oid)
                if (
                    math.floor(a) < box.x
                    or math.floor(b) < box.y
                    or math.ceil(c) > box.right
                    or math.ceil(d) > box.bottom
                ):
                    raise ValueError("Component replacement bounds omit changed paint")
        return signature(after, proposed, work)
