"""Snap: move touching edges of selected regions onto each other."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from vectrify.operations.contract import (
    OperationRequest,
    OperationResult,
    PreparedOperation,
    Proposal,
    RunContext,
    SettingsMethod,
    register,
)
from vectrify.operations.previews import render_previews
from vectrify.operations.settings import Setting

SETTINGS = {"tolerance": Setting(float, 1.0, minimum=0, label="contact distance")}


class EdgeSnap(SettingsMethod):
    settings_schema: ClassVar[Mapping[str, Setting]] = SETTINGS
    settings_label: ClassVar[str] = "snap"

    action: ClassVar[str] = "snap"
    name: ClassVar[str] = "edges"
    background: ClassVar[bool] = False
    needs_reference: ClassVar[bool] = False
    resources: ClassVar[frozenset[str]] = frozenset()

    def validate(self, request: OperationRequest | PreparedOperation) -> None:
        self.prepare(request)

    def run(
        self, request: OperationRequest | PreparedOperation, _context: RunContext
    ) -> OperationResult:
        prepared = self.prepare(request)
        request = prepared.request
        tx = request.transaction("Snap edges")
        spans = tx.snap_edges(prepared.settings["tolerance"])
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
