"""Link: match touching edges of selected regions into shared boundaries."""

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
        raise DocumentError(f"Unknown boundary setting: {sorted(unknown)[0]}")
    tolerance = float(request.settings.get("tolerance", 1))
    if not math.isfinite(tolerance):
        raise DocumentError("Enter a finite contact distance")
    return tolerance


class BoundaryMatch:
    action: ClassVar[str] = "link"
    name: ClassVar[str] = "boundaries"
    background: ClassVar[bool] = False
    needs_reference: ClassVar[bool] = False
    resources: ClassVar[frozenset[str]] = frozenset()

    def validate(self, request: OperationRequest) -> None:
        _tolerance(request)

    def run(self, request: OperationRequest, _context: RunContext) -> OperationResult:
        tx = request.transaction("Share boundaries")
        edges = tx.share_boundaries(_tolerance(request))
        return OperationResult(
            Proposal(
                tx,
                edges > 0,
                metrics={"edges": edges},
                previews=render_previews(
                    request.snapshot.document,
                    tx.preview,
                    request.bounds,
                    highlight=True,
                ),
            )
        )


register(BoundaryMatch())
