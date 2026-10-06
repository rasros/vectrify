"""Generate an illustration with a structural complexity budget."""

from __future__ import annotations

from typing import ClassVar

from vectrify.operations.contract import (
    OperationCancelledError,
    OperationRequest,
    OperationResult,
    RunContext,
    register,
)
from vectrify.operations.generate import (
    generated_result,
    target_region,
    validate_generate,
)
from vectrify.operations.settings import Setting, read_settings

SETTINGS = {
    "complexity": Setting(int, 50, minimum=0, maximum=100),
    "quality": Setting(str, "balanced", choices=("fast", "balanced", "high")),
    "refine": Setting(bool, True),
    "gradients": Setting(bool, True),
    "line_width": Setting(float, 0.0, minimum=0, maximum=32),
    "palette": Setting(int, 0, minimum=0, maximum=256),
    "tolerance": Setting(float, 0.0, minimum=0, maximum=20),
    "protection": Setting(float, 1.0, minimum=0, maximum=10),
    "node_budget": Setting(int, 0, minimum=0, maximum=100_000),
}


class CelPlanned:
    action: ClassVar[str] = "generate"
    name: ClassVar[str] = "cel-planned"
    background: ClassVar[bool] = True
    needs_reference: ClassVar[bool] = True
    resources: ClassVar[frozenset[str]] = frozenset()

    def validate(self, request: OperationRequest) -> None:
        read_settings(request.settings, SETTINGS, "planned cel")
        validate_generate(request)

    def run(self, request: OperationRequest, context: RunContext) -> OperationResult:
        from vectrify.refine.cel_plan.model import Options, PlanningStoppedError
        from vectrify.refine.cel_plan.pipeline import vectorize

        options = Options(**read_settings(request.settings, SETTINGS, "planned cel"))
        region = target_region(request)
        context.progress(0, "Reading ink and surface evidence…", total=3)
        try:
            candidate = vectorize(
                region.image,
                alpha=region.alpha,
                options=options,
                seconds=request.budget.seconds,
                stop=context.stop,
            )
        except PlanningStoppedError as exc:
            raise OperationCancelledError(str(exc)) from exc
        context.progress(2, "Placing planned shapes…")
        result = generated_result(
            request,
            candidate.svg,
            region,
            label="Generate planned cel drawing",
            name="Planned cel",
            metrics=candidate.metrics,
        )
        for alternative in candidate.alternatives:
            proposal = generated_result(
                request,
                alternative.svg,
                region,
                label="Generate planned cel drawing",
                name="Planned cel",
                metrics=alternative.metrics,
            ).recommended
            proposal.label = alternative.label
            result.alternatives.append(proposal)
        context.progress(3, "Preview ready")
        return result


register(CelPlanned())
