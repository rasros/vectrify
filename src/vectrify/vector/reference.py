"""The target image, prepared once, and how a candidate render is measured.

Every candidate in a search is compared with the same reference: its edges,
colour, shape moments and a detail budget, plus one error per attention
segment. Building this is the expensive part (resizing, edge maps, segment
masks), so it happens once per run and is shared by every caller that scores.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

from PIL import Image

from vectrify.image_utils import resize_long_side
from vectrify.score.base import DEFAULT_CONFIG
from vectrify.score.compare import Reference as PixelReference
from vectrify.score.compare import compare, prepare
from vectrify.score.complexity import detail, detail_excess
from vectrify.score.edges import overlap_distance
from vectrify.score.metrics import COLOUR, DETAIL, EDGE, SHAPE
from vectrify.score.segments import Segment, segment_error, segment_target

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Reference:
    # The target at render resolution; candidates are rasterized at this size.
    image: Image.Image
    png: bytes
    # The smaller copy the pixel measures compare against.
    scoring_image: Image.Image
    pixel: PixelReference
    segments: tuple[Segment, ...]
    # Compressed-size detail of the target at render resolution.
    detail: float

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height

    @classmethod
    def build(
        cls,
        image: Image.Image,
        *,
        score_resolution: int | None = None,
        edge_tolerance: float | None = None,
        segment_count: int = 8,
    ) -> Reference:
        image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        png = buffer.getvalue()
        scoring = resize_long_side(
            image, score_resolution or DEFAULT_CONFIG.target_long_side
        )
        segments = segment_target(scoring, max_regions=segment_count)
        if not segments:
            raise ValueError("target segmentation returned no regions")
        # Measured at the size candidates are rasterized at -- the render
        # resolution, not the smaller scoring copy. Compressed size grows with
        # pixel count, so reading the two at different sizes charges every
        # candidate for the difference: measured on the duck, the same image
        # reads 7,607 at 256 and 41,602 at 700.
        return cls(
            image=image,
            png=png,
            scoring_image=scoring,
            pixel=prepare(scoring, tolerance=edge_tolerance),
            segments=tuple(segments),
            detail=detail(png),
        )

    def measure(self, png: bytes, *, segments: bool = True) -> dict[str, float]:
        """The objectives for one candidate render, lower is better."""
        comparison = compare(self.pixel, png)
        metrics = {
            EDGE: overlap_distance(
                comparison.reference_edges, comparison.candidate_edges
            ),
            COLOUR: float(comparison.colour.mean()),
            SHAPE: comparison.shape,
            DETAIL: detail_excess(self.detail, png),
        }
        if segments:
            for segment in self.segments:
                metrics[segment.metric_name] = segment_error(
                    comparison, segment.mask, detail=segment.detail
                )
        return metrics
