"""Snap: move touching edges of selected regions onto each other."""

from __future__ import annotations

import math
from typing import ClassVar

from vectrify.document import DocumentError
from vectrify.operations.contract import (
    OperationRequest,
    OperationResult,
    Proposal,
    RunContext,
    register,
)
from vectrify.operations.previews import render_previews


def _tolerance(request: OperationRequest) -> float:
    unknown = set(request.settings) - {"tolerance"}
    if unknown:
        raise DocumentError(f"Unknown snap setting: {sorted(unknown)[0]}")
    tolerance = float(request.settings.get("tolerance", 1))
    if not math.isfinite(tolerance):
        raise DocumentError("Enter a finite contact distance")
    return tolerance


class EdgeSnap:
    action: ClassVar[str] = "snap"
    name: ClassVar[str] = "edges"
    background: ClassVar[bool] = False
    needs_reference: ClassVar[bool] = False
    resources: ClassVar[frozenset[str]] = frozenset()

    def validate(self, request: OperationRequest) -> None:
        _tolerance(request)

    def run(self, request: OperationRequest, _context: RunContext) -> OperationResult:
        tx = request.transaction("Snap edges")
        spans = tx.snap_edges(_tolerance(request))
        return OperationResult(
            Proposal(
                tx,
                tx.preview != request.snapshot.document,
                metrics={"edges": len(spans)},
                previews=render_previews(
                    request.snapshot.document,
                    tx.preview,
                    request.bounds,
                    highlight=spans,
                ),
            )
        )


register(EdgeSnap())
