"""Snap: move touching edges of selected regions onto each other."""

from __future__ import annotations

from typing import ClassVar

from vectrify.operations.contract import (
    OperationRequest,
    OperationResult,
    Proposal,
    RunContext,
    register,
)
from vectrify.operations.previews import render_previews
from vectrify.operations.settings import Setting, read_settings

SETTINGS = {"tolerance": Setting(float, 1.0, minimum=0, label="contact distance")}


def _tolerance(request: OperationRequest) -> float:
    return read_settings(request.settings, SETTINGS, "snap")["tolerance"]


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
