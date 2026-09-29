"""Generate: fit a colour palette on the GPU and trace its regions as SVG."""

from __future__ import annotations

from typing import ClassVar

from vectrify.document import DocumentError
from vectrify.operations.contract import (
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
    "colours": Setting(int, 32, minimum=2, maximum=256),
    "steps": Setting(int, 24, minimum=1, maximum=1000, label="palette steps"),
    "min_pixels": Setting(int, 64, minimum=1, label="minimum region pixels"),
    "tolerance": Setting(float, 1.25, minimum=0, maximum=50, label="tolerance"),
    "smooth_sigma": Setting(float, 1.5, minimum=0, maximum=20, label="smoothing"),
    "preserve_outlines": Setting(bool, False),
    "outline_style": Setting(str, "preserve", choices=("preserve", "clean")),
    "outline_radius": Setting(int, 3, minimum=1, maximum=50),
    "outline_contrast": Setting(float, 6, minimum=0, maximum=255),
    "outline_regions": Setting(int, 3, minimum=1, maximum=64),
    "outline_width": Setting(float, 1.5, minimum=0.1, maximum=50),
    "texture_tolerance": Setting(float, 5.0, minimum=0, maximum=50),
    "geometry_cleanup": Setting(bool, True),
}


class ColourRegions:
    action: ClassVar[str] = "generate"
    name: ClassVar[str] = "colour-regions"
    background: ClassVar[bool] = True
    needs_reference: ClassVar[bool] = True
    resources: ClassVar[frozenset[str]] = frozenset({"gpu"})

    def validate(self, request: OperationRequest) -> None:
        read_settings(request.settings, SETTINGS, "colour-region")
        validate_generate(request)

    def run(self, request: OperationRequest, context: RunContext) -> OperationResult:
        import torch

        from vectrify.refine.colour_regions import vectorize

        if not torch.cuda.is_available():
            raise DocumentError("Colour regions need a CUDA GPU")
        settings = read_settings(request.settings, SETTINGS, "colour-region")
        region = target_region(request)
        context.progress(0, "Fitting the palette and tracing regions…", total=2)
        svg, details = vectorize(region.image, **settings)
        context.progress(1, "Placing traced regions…")
        result = generated_result(
            request,
            svg,
            region,
            label="Generate colour regions",
            name="Colour regions",
            metrics={
                "colours": details.get("palette_colours", settings["colours"]),
                "seconds": details["seconds"],
                **(
                    {"cleanup": dict(details["geometry_cleanup"])}
                    if "geometry_cleanup" in details
                    else {}
                ),
            },
        )
        context.progress(2, "Preview ready")
        return result


register(ColourRegions())
