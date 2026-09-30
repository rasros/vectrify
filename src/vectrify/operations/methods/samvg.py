"""Generate: trace SAM segments of the reference into filled SVG paths."""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

from PIL import Image

from vectrify.image_utils import rasterize_svg
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
from vectrify.refine.samvg import (
    LINE_WIDTH,
    SAMVG_MAX_SIDE,
    SAMVG_MODEL,
    SAMVG_POINTS_PER_BATCH,
)

SETTINGS = {
    "min_pixels": Setting(int, 32, minimum=1, label="minimum region pixels"),
    "min_impact": Setting(float, 3e-6, minimum=0, label="minimum impact"),
    "max_layers": Setting(int, 512, minimum=1, maximum=4096, label="maximum layers"),
    "segments": Setting(int, 16, minimum=4, maximum=256, label="curve segments"),
    # How far an outline may stray from its region, in reference pixels: each
    # is traced densely and simplified to it. 0 traces *segments* curves each.
    "tolerance": Setting(float, 0.5, minimum=0.0, maximum=10.0, label="tolerance"),
    "fill_holes": Setting(bool, True),
    # Regions narrower than this everywhere, in reference pixels, are left
    # out: SAM returns outlines and hairlines as regions of their own.
    "min_width": Setting(int, 3, minimum=0, maximum=64, label="minimum width"),
    # Cut every region down to its visible part, so none overlap.
    "flatten": Setting(bool, False),
    # Fold small patches into a neighbour of a similar colour.
    "merge": Setting(bool, True),
    # Trace the reference's drawn lines as a layer of their own on top.
    "outlines": Setting(bool, True),
    # Move region edges onto the reference's own, finer than SAM's masks.
    "refine": Setting(bool, True),
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

# Flattened regions are traced one by one, so neighbours' outlines stray from
# the common seam by up to about a traced pixel; link edges within this many.
SEAM_PIXELS = 1.5


def _enlarged(region: Region, settings: dict) -> tuple[Region, dict]:
    """*region* smoothly enlarged to SAM's working size, if it is smaller.

    A reference smaller than SAM works at comes back as masks on its own
    pixel grid, and tracing those follows every pixel's edge: outlines turn
    into staircases one reference pixel a step. Enlarged first, SAM's masks
    are interpolated between the pixels and the outlines come out smooth.
    The pixel sizes in the settings are scaled with it, so they still count
    reference pixels.
    """
    longest = max(region.image.size)
    if longest >= settings["max_side"]:
        return region, settings
    scale = settings["max_side"] / longest
    width, height = region.image.size
    image = region.image.resize(
        (round(width * scale), round(height * scale)), Image.Resampling.BICUBIC
    )
    return replace(region, image=image), {
        **settings,
        "min_width": round(settings["min_width"] * scale),
        "tolerance": settings["tolerance"] * scale,
        "min_pixels": round(settings["min_pixels"] * scale * scale),
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
        traced, settings = _enlarged(region, settings)
        # The widest line looked for grows with the image.
        settings["outlines"] = (
            max(LINE_WIDTH, round(max(traced.image.size) / 200))
            if settings["outlines"]
            else 0
        )
        context.progress(0, "Segmenting the reference with SAM…", total=2)
        svg = generate_svg(
            traced.image,
            # Text layers need the unsupported <text> element in the editor.
            ocr=False,
            rasterize=rasterize_svg,
            # Regions hidden by those above paint nothing, and SAM leaves drawn
            # outlines to neither neighbour: drop the one, fill beneath the other.
            drop_hidden=True,
            backdrop=True,
            **settings,
        )
        context.progress(1, "Placing traced shapes…")
        result = generated_result(
            request,
            svg,
            region,
            label="Generate with SAMVG",
            name="SAMVG trace",
            traced=traced,
            seams=SEAM_PIXELS if settings["flatten"] else None,
        )
        context.progress(2, "Preview ready")
        return result


register(Samvg())
