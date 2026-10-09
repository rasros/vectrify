"""Improve: gradient fitting of one selected path against the reference."""

from __future__ import annotations

from typing import ClassVar

from vectrify.document import DocumentError
from vectrify.operations.contract import (
    OperationRequest,
    OperationResult,
    Proposal,
    RunContext,
    register,
)
from vectrify.operations.settings import Setting, read_settings
from vectrify.refine.selected import (
    FitOptions,
    fit_problem,
    fit_selected_path,
    validate_selection,
)

DEFAULT_STEPS = 8

SETTINGS = {
    "nodes": Setting(bool, True),
    "handles": Setting(bool, True),
    "color": Setting(bool, False),
    "snap": Setting(bool, True),
    "displacement": Setting(float, 2.0, minimum=0, maximum=100),
    "resolution": Setting(int, 768, minimum=64, maximum=2048),
}


def fit_options(request: OperationRequest) -> FitOptions:
    options = FitOptions(
        steps=request.budget.steps or DEFAULT_STEPS,
        **read_settings(request.settings, SETTINGS, "path-fit"),
    )
    if (options.nodes or options.handles) and not request.permissions.geometry:
        raise DocumentError("Allow geometry changes to move nodes or handles")
    if options.color and not request.permissions.paint:
        raise DocumentError("Allow paint changes to fit color")
    return options


class PathFit:
    action: ClassVar[str] = "improve"
    name: ClassVar[str] = "path-fit"
    background: ClassVar[bool] = True
    needs_reference: ClassVar[bool] = True
    resources: ClassVar[frozenset[str]] = frozenset({"gpu"})

    def validate(self, request: OperationRequest) -> None:
        if request.reference is None:
            raise DocumentError("Add a reference image before optimizing a path")
        problem = fit_problem()
        if problem:
            raise DocumentError(problem)
        # Fills and round strokes fit on CPU too; miter outlines require CUDA.
        validate_selection(
            request.snapshot.document, request.snapshot.selection, fit_options(request)
        )

    def run(self, request: OperationRequest, context: RunContext) -> OperationResult:
        assert request.reference is not None
        options = fit_options(request)
        context.progress(0, "Preparing…", total=options.steps)
        fit = fit_selected_path(
            request.snapshot.document,
            request.snapshot.selection,
            request.reference,
            options,
            stop=context.stop,
            progress=context.progress,
        )
        tx = request.transaction("Optimize path")
        fit.write(tx)
        return OperationResult(
            Proposal(
                tx,
                fit.changed,
                metrics={
                    "before": {"error": fit.before},
                    "after": {"error": fit.after},
                    "size": list(fit.size),
                    "steps": fit.steps,
                },
                previews=fit.previews,
            )
        )


register(PathFit())
