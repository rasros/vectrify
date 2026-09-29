"""Simplify: remove redundant vertices and merge compatible paths.

Wraps ``svg.cleanup.cleanup_svg_geometry`` for the selection. Objects
outside the selection are swapped for inert placeholders while it runs, so they
are neither simplified nor merged, and a placeholder between two selected
paths keeps them from merging across it. The result is replayed as ordinary
commands; coordinates are never rounded.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import ClassVar

from vectrify.document import DocumentError, export_svg
from vectrify.operations.candidates import replay
from vectrify.operations.contract import (
    OperationRequest,
    OperationResult,
    Proposal,
    RunContext,
    register,
)
from vectrify.operations.previews import render_previews
from vectrify.svg.cleanup import cleanup_svg_geometry

SVG_NS = "http://www.w3.org/2000/svg"


def _placeholders(svg: str, scope: frozenset[str]) -> tuple[str, dict[str, str]]:
    """Swap every out-of-scope element that has no in-scope descendant."""
    root = ET.fromstring(svg)
    kept: dict[str, str] = {}

    def has_scope(element: ET.Element) -> bool:
        return element.get("id") in scope or any(has_scope(c) for c in element)

    def visit(parent: ET.Element) -> None:
        for index, child in enumerate(list(parent)):
            oid = child.get("id")
            if oid in scope:
                continue
            if has_scope(child):
                visit(child)
                continue
            kept[oid or ""] = ET.tostring(child, encoding="unicode")
            parent.remove(child)
            parent.insert(index, ET.Element(f"{{{SVG_NS}}}rect", {"id": oid or ""}))

    visit(root)
    ET.register_namespace("", SVG_NS)
    return ET.tostring(root, encoding="unicode"), kept


def _restore(svg: str, kept: dict[str, str]) -> str:
    root = ET.fromstring(svg)

    def visit(parent: ET.Element) -> None:
        for index, child in enumerate(list(parent)):
            oid = child.get("id")
            if oid in kept and child.tag == f"{{{SVG_NS}}}rect" and not len(child):
                parent.remove(child)
                parent.insert(index, ET.fromstring(kept[oid]))
            else:
                visit(child)

    visit(root)
    ET.register_namespace("", SVG_NS)
    return ET.tostring(root, encoding="unicode")


class GeometryCleanup:
    action: ClassVar[str] = "simplify"
    name: ClassVar[str] = "cleanup"
    background: ClassVar[bool] = False
    needs_reference: ClassVar[bool] = False
    resources: ClassVar[frozenset[str]] = frozenset()

    def validate(self, request: OperationRequest) -> None:
        if request.settings:
            raise DocumentError("Geometry cleanup takes no settings")
        if not (request.permissions.geometry and request.permissions.structure):
            raise DocumentError("Allow geometry and structure changes to clean up")
        document = request.snapshot.document
        if not document.selection_ids(request.snapshot.selection):
            raise DocumentError("Select objects, or the whole drawing, to clean up")

    def run(self, request: OperationRequest, _context: RunContext) -> OperationResult:
        document = request.snapshot.document
        scope = frozenset(document.selection_ids(request.snapshot.selection))
        svg, kept = _placeholders(export_svg(document), scope)
        # Merging or dropping a path is only safe when nothing references it.
        unreferenced = frozenset(
            oid for oid in scope if document.dependents({oid}) == {oid}
        )
        cleaned, stats = cleanup_svg_geometry(svg, unreferenced=unreferenced)
        tx = request.transaction("Clean up geometry")
        outcome = replay(tx, _restore(cleaned, kept), contours=True)
        return OperationResult(
            Proposal(
                tx,
                outcome.edits > 0,
                metrics={"cleanup": dict(stats), "edits": outcome.edits},
                previews=render_previews(document, tx.preview, request.bounds),
            ),
            message=None if outcome.edits else "Nothing to clean up here",
        )


register(GeometryCleanup())
