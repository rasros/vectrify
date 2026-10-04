"""Generate: trace cel art as flat regions inside its drawn lines, the lines
drawn over them as strokes."""

from __future__ import annotations

from typing import ClassVar

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
    # How many filled regions the trace merges down to; 0 sizes it to the
    # traced area (see auto_regions).
    "regions": Setting(int, 0, minimum=0, maximum=2000),
    # The lines' stroke width in reference pixels; 0 measures each line.
    "line_width": Setting(float, 0.0, minimum=0.0, maximum=32.0, label="line width"),
    # How far an outline or line may stray from the traced pixels.
    "tolerance": Setting(float, 0.75, minimum=0.1, maximum=10.0, label="tolerance"),
    # Draw the lines as strokes, tapered ones too; off, they are filled shapes.
    "strokes": Setting(bool, True),
    # One unbroken stroke round the drawing's silhouette, in place of the
    # traced outer line.
    "outline": Setting(bool, False, label="continuous outer outline"),
    # Each region's colour fitted to the image under the lines as drawn,
    # rather than its pixels' median.
    "fit_colours": Setting(bool, True, label="fit colours"),
    # A region whose colour clearly ramps takes a linear gradient.
    "gradients": Setting(bool, True, label="gradients"),
}


# By default a trace keeps one region per this many reference pixels (about
# 100 x 100), within these bounds: a larger image has room for more detail.
PIXELS_PER_REGION = 10_000
AUTO_REGIONS = (50, 2000)


def auto_regions(size: tuple[int, int]) -> int:
    """The regions a trace of an area *size* pixels keeps by default."""
    low, high = AUTO_REGIONS
    return min(high, max(low, round(size[0] * size[1] / PIXELS_PER_REGION)))


class Cel:
    action: ClassVar[str] = "generate"
    name: ClassVar[str] = "cel"
    background: ClassVar[bool] = True
    needs_reference: ClassVar[bool] = True
    resources: ClassVar[frozenset[str]] = frozenset()

    def validate(self, request: OperationRequest) -> None:
        read_settings(request.settings, SETTINGS, "cel")
        validate_generate(request)

    def run(self, request: OperationRequest, context: RunContext) -> OperationResult:
        from vectrify.refine.cel import vectorize

        settings = read_settings(request.settings, SETTINGS, "cel")
        region = target_region(request)
        if not settings["regions"]:
            settings["regions"] = auto_regions(region.image.size)
        context.progress(0, "Finding the lines and filling the regions…", total=2)
        svg, details = vectorize(region.image, alpha=region.alpha, **settings)
        context.progress(1, "Placing traced regions and lines…")
        result = generated_result(
            request,
            svg,
            region,
            label="Generate cel trace",
            name="Cel trace",
            metrics={
                key: details[key]
                for key in (
                    "regions",
                    "fill_paths",
                    "gradients",
                    "line_paths",
                    "line_style",
                    "outline",
                    "seconds",
                )
                if key in details
            },
        )
        context.progress(2, "Preview ready")
        return result


register(Cel())
