"""Before/after optimization cleanup submits bridges to the retained-fit policy."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from cairosvg.colors import color
from PIL import Image

from vectrify.refine.crossings import crossings
from vectrify.refine.snap import _Frame

if TYPE_CHECKING:
    from vectrify.document import Geometry, Selection
    from vectrify.refine.selected_fit.policy import RetainedFit


class CleanupPass:
    """Reference reconstruction and local repairs share one acceptance owner."""

    def __init__(
        self, retained: RetainedFit, selection: Selection, target: Image.Image
    ):
        self.retained, self.selection, self.target = retained, selection, target
        self.context, self.options, self.objective = (
            retained.context,
            retained.options,
            retained.objective,
        )
        self.smoothing = 1
        self.bridge_crossings = retained.best.crossings
        self.held = frozenset(
            node.id
            for node in self.context.nodes
            if (selection.node_ids and node.id not in selection.node_ids)
            or node.pinned
            or node.feature is not None
        )

    def accept_bridge(self, candidate: Geometry, *, allow_error=False):
        count = crossings(candidate)
        if count > self.bridge_crossings:
            return False
        paint = color(self.retained.best.fill or self.context.style["fill"])[:3]
        proposal = self.retained.evaluate(
            candidate, np.array(paint), smoothing=self.smoothing
        )
        if not self.retained.consider_bridge(
            proposal, self.bridge_crossings, allow_error=allow_error
        ):
            return False
        self.bridge_crossings = count
        if self.objective.guide is not None:
            self.objective.guide.coverage = self.objective.outline_coverage(candidate)
        return True

    def run(self, stopped: Callable[[], bool], *, polish=False):
        if not (
            self.options.cleanup
            and self.options.nodes
            and self.options.handles
            and not self.context.stroke_only
        ):
            return
        from vectrify.refine.cleanup import cleaned

        context, objective = self.context, self.objective
        geometry = self.retained.best.geometry
        self.smoothing = 4 if polish else 1
        self.retained.rescore_quality(self.smoothing)
        guide = objective.guide
        if guide is not None:
            guide.coverage = objective.outline_coverage(geometry)
        self.bridge_crossings = crossings(geometry)
        vx, vy, vw, vh = context.document.artboard()
        pixels = np.diag([self.target.width / vw, self.target.height / vh])
        if guide is not None:
            from vectrify.refine.boundary import field_loops, reconstructed

            scale = float(np.mean(objective.fairness_scale))
            padding = max(8, round(8 / scale))
            contours = field_loops(
                guide.raw_signal,
                guide.power,
                objective.outline_coverage(geometry, padding),
                padding,
            )
            geometry = reconstructed(
                geometry,
                context.geometry,
                guide.frame,
                contours,
                scale,
                self.options.displacement,
                self.held,
                lambda candidate: self.accept_bridge(candidate, allow_error=True),
                stopped,
            )
        cleaned(
            geometry,
            context.geometry,
            _Frame(pixels @ context.linear, pixels @ (context.offset - [vx, vy])),
            self.options.displacement,
            self.held,
            self.accept_bridge,
            stopped,
            guide=guide,
        )
