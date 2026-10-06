"""Bounded local order for bases, a continuing surface and enclosed marks."""

from __future__ import annotations

from dataclasses import replace

import pathops

from vectrify.document import Document, Editor, Geometry, Selection
from vectrify.document.hit_test import multiply
from vectrify.document.join import curve_path, path_style, transformed_geometry
from vectrify.document.redraw import root_matrix
from vectrify.document.topology import inverse_matrix
from vectrify.refine.cel_plan.model import Work

MAX_PROOFS = 16
MAX_NODES = 6_000


def ordered(
    document: Document,
    survivor: str,
    bases: set[str],
    marks: tuple[str, ...],
    footprint: Geometry,
    work: Work,
    diagnostics: dict,
) -> Document | None:
    """Retain other order unless the changed crossing is geometrically disjoint."""
    from vectrify.refine.cel_plan.proposals import bounds

    parent = document.ancestry(survivor)[-2]
    children = list(parent.children)
    positions = {c.id: i for i, c in enumerate(children)}
    moving = {survivor, *marks}
    # Source label order does not imply occlusion. Inside marks stay in their
    # existing relative order, immediately above the newly continuing surface.
    block = [document.element(survivor), *(c for c in children if c.id in marks)]
    remaining = [c for c in children if c.id not in moving]
    at = sum(c.id not in moving for c in children[: positions[survivor]])
    if bases:
        at = max(at, 1 + max(i for i, c in enumerate(remaining) if c.id in bases))
    proposed = [*remaining[:at], *block, *remaining[at:]]
    target = {c.id: i for i, c in enumerate(proposed)}
    inverse = inverse_matrix(root_matrix(document, survivor))
    paths = {survivor: curve_path(footprint, "nonzero")}
    probe = document.replace_geometry(
        replace(footprint, id=document.geometry_for(survivor).id)
    )
    native = {survivor: bounds(probe, probe, (survivor,))}
    for oid in marks:
        style = path_style(document, document.element(oid))
        shape = transformed_geometry(
            document.geometry_for(oid), multiply(inverse, root_matrix(document, oid))
        )
        paths[oid] = curve_path(shape, style["fill-rule"])
        native[oid] = bounds(document, document, (oid,))
    proofs = 0
    for oid in sorted(moving, key=positions.__getitem__):
        for child in children:
            if work.interrupted:
                return None
            if child.id in moving or child.id in bases:
                continue
            if (positions[oid] < positions[child.id]) == (
                target[oid] < target[child.id]
            ):
                continue
            if child.tag != "path":
                return None
            if (
                not native[oid]
                .intersection(bounds(document, document, (child.id,)))
                .area
            ):
                continue
            style = path_style(document, child)
            shape = document.geometry_for(child.id)
            if (
                proofs >= MAX_PROOFS
                or sum(len(s.nodes) for s in shape.subpaths) > MAX_NODES
            ):
                diagnostics["order_proof_limits"] += 1
                return None
            if style["stroke"] != "none" or child.get("clip-path", "none") != "none":
                return None
            proofs += 1
            diagnostics["order_proofs"] += 1
            shape = transformed_geometry(
                shape, multiply(inverse, root_matrix(document, child.id))
            )
            overlap = pathops.op(
                paths[oid],
                curve_path(shape, style["fill-rule"]),
                pathops.PathOp.INTERSECTION,
            )
            if abs(overlap.area) > 1e-8:
                return None
    if work.interrupted:
        return None
    editor = Editor(document, selection=Selection(whole_document=True))
    with editor.transaction(
        "Order supported material and its nested marks"
    ) as transaction:
        for index, child in enumerate(proposed):
            if transaction.preview.element(parent.id).children[index].id != child.id:
                transaction.reorder_object(child.id, index)
    return editor.snapshot.document
