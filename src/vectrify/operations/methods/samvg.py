"""Generate: trace SAM segments of the reference into filled SVG paths."""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

from PIL import Image

from vectrify.operations.contract import (
    OperationRequest,
    OperationResult,
    RunContext,
    register,
)
from vectrify.operations.generate import (
    Region,
    generated_result,
    target_region,
    validate_generate,
)
from vectrify.operations.settings import Setting, read_settings
from vectrify.refine.samvg import MIN_PIXELS, MIN_WIDTH, SAMVG_MODEL, TOLERANCE

SETTINGS = {
    "max_layers": Setting(int, 512, minimum=1, maximum=4096, label="maximum shapes"),
    # The longest side SAM segments at; 0 segments at the reference's own size.
    "max_side": Setting(int, 0, minimum=0, maximum=4096),
    "model": Setting(
        str,
        SAMVG_MODEL,
        choices=(SAMVG_MODEL, "facebook/sam-vit-base"),
    ),
}


def _enlarged(region: Region, settings: dict) -> tuple[Region, dict]:
    """*region* smoothly enlarged to SAM's working size, if it is smaller.

    A reference smaller than SAM works at comes back as masks on its own
    pixel grid, and tracing those follows every pixel's edge: outlines turn
    into staircases one reference pixel a step. Enlarged first, SAM's masks
    are interpolated between the pixels and the outlines come out smooth.
    The tracer's pixel sizes are scaled with it, so they still count
    reference pixels.
    """
    scale = max(1.0, settings["max_side"] / max(region.image.size))
    if scale > 1:
        width, height = region.image.size
        image = region.image.resize(
            (round(width * scale), round(height * scale)), Image.Resampling.BICUBIC
        )
        region = replace(region, image=image)
    return region, {
        **settings,
        "min_width": round(MIN_WIDTH * scale),
        "tolerance": TOLERANCE * scale,
        "min_pixels": round(MIN_PIXELS * scale * scale),
    }


class Samvg:
    action: ClassVar[str] = "generate"
    name: ClassVar[str] = "samvg"
    background: ClassVar[bool] = True
    needs_reference: ClassVar[bool] = True
    resources: ClassVar[frozenset[str]] = frozenset({"gpu"})

    def validate(self, request: OperationRequest) -> None:
        read_settings(request.settings, SETTINGS, "SAMVG")
        validate_generate(request)

    def run(self, request: OperationRequest, context: RunContext) -> OperationResult:
        from vectrify.refine.samvg import generate_svg

        settings = read_settings(request.settings, SETTINGS, "SAMVG")
        region = target_region(request)
        if not settings["max_side"]:
            settings["max_side"] = max(region.image.size)
        traced, settings = _enlarged(region, settings)
        context.progress(0, "Segmenting the reference with SAM…", total=2)
        svg = generate_svg(traced.image, **settings)
        context.progress(1, "Placing traced shapes…")
        result = generated_result(
            request,
            svg,
            region,
            label="Generate with SAMVG",
            name="SAMVG trace",
            traced=traced,
        )
        context.progress(2, "Preview ready")
        return result


register(Samvg())
