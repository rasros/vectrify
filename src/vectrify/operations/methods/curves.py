"""Simplify: refit selected contours with fewer lines and cubic curves."""

from __future__ import annotations

from typing import ClassVar

from vectrify.document import Document
from vectrify.document.simplify import SimplifyOptions
from vectrify.operations.contract import (
    OperationRequest,
    OperationResult,
    Proposal,
    RunContext,
    register,
)
from vectrify.operations.previews import render_previews


def _stats(document: Document, geometries: set[str]) -> dict[str, int]:
    assets = [document.geometry(gid) for gid in geometries]
    return {
        "nodes": sum(len(s.nodes) for g in assets for s in g.subpaths),
        "coordinates": sum(
            len(n.values) for g in assets for s in g.subpaths for n in s.nodes
        ),
        "bytes": sum(len(g.path_data().encode()) for g in assets),
    }


class CurveSimplify:
    action: ClassVar[str] = "simplify"
    name: ClassVar[str] = "curves"
    background: ClassVar[bool] = False
    needs_reference: ClassVar[bool] = False
    resources: ClassVar[frozenset[str]] = frozenset()

    def validate(self, request: OperationRequest) -> None:
        SimplifyOptions(**request.settings)

    def run(self, request: OperationRequest, _context: RunContext) -> OperationResult:
        before = request.snapshot.document
        tx = request.transaction("Smooth / simplify shapes")
        tx.simplify_shapes(SimplifyOptions(**request.settings))
        after = tx.preview
        selected = before.selection_ids(request.snapshot.selection)
        geometries = {
            before.geometry_for(oid).id
            for oid in selected
            if before.element(oid).tag in {"path", "use"}
        }
        changed = any(g != after.geometry(g.id) for g in before.geometries)
        return OperationResult(
            Proposal(
                tx,
                changed,
                metrics={
                    "before": _stats(before, geometries),
                    "after": _stats(after, geometries),
                },
                previews=render_previews(before, after, request.bounds),
            )
        )


register(CurveSimplify())
