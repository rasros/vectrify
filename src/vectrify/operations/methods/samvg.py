"""Generate: trace SAM segments of the reference into filled SVG paths."""

from __future__ import annotations

from typing import ClassVar

from vectrify.image_utils import rasterize_svg
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
from vectrify.refine.samvg import (
    SAMVG_MAX_SIDE,
    SAMVG_MODEL,
    SAMVG_POINTS_PER_BATCH,
)

SETTINGS = {
    "min_pixels": Setting(int, 32, minimum=1, label="minimum region pixels"),
    "min_impact": Setting(float, 3e-6, minimum=0, label="minimum impact"),
    "max_layers": Setting(int, 512, minimum=1, maximum=4096, label="maximum layers"),
    "segments": Setting(int, 16, minimum=4, maximum=256, label="curve segments"),
    "fill_holes": Setting(bool, True),
    "hybrid_strokes": Setting(bool, True, label="thin strokes"),
    "max_side": Setting(int, SAMVG_MAX_SIDE, minimum=64, maximum=4096),
    "model": Setting(
        str,
        SAMVG_MODEL,
        choices=tuple(
            dict.fromkeys(
                (SAMVG_MODEL, "facebook/sam-vit-huge", "facebook/sam-vit-base")
            )
        ),
    ),
    "points_per_batch": Setting(int, SAMVG_POINTS_PER_BATCH, minimum=1, maximum=1024),
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
        context.progress(0, "Segmenting the reference with SAM…", total=2)
        svg = generate_svg(
            region.image,
            # Text layers need the unsupported <text> element in the editor.
            ocr=False,
            rasterize=rasterize_svg,
            **settings,
        )
        context.progress(1, "Placing traced shapes…")
        result = generated_result(
            request, svg, region, label="Generate with SAMVG", name="SAMVG trace"
        )
        context.progress(2, "Preview ready")
        return result


register(Samvg())
