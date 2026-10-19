"""The single owner and acceptance policy for retained exact-render candidates."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from PIL import Image

from vectrify.document import Geometry
from vectrify.refine.crossings import crossings

if TYPE_CHECKING:
    from vectrify.refine.selected import FitContext, FitOptions
    from vectrify.refine.selected_fit.objective import ExactObjective


@dataclass(frozen=True)
class Candidate:
    geometry: Geometry
    values: dict[str, tuple[float, ...]]
    image: Image.Image
    error: float
    quality: float
    crossings: int


class RetainedFit:
    """Only this owner replaces the published candidate and error allowance.

    Gradient and edge proposals share one comparison. Cleanup bridges also
    require a non-increasing crossing count and an explicit error allowance.
    """

    def __init__(
        self, context: FitContext, options: FitOptions, objective: ExactObjective
    ):
        self.context, self.options, self.objective = context, options, objective
        self.before = objective.score(context.before_image)
        self.least_difference = self.before
        self.best = Candidate(
            context.geometry,
            {},
            context.before_image,
            self.before,
            objective.quality(context.geometry, self.before),
            crossings(context.geometry),
        )
        self.original_values = {node.id: node.values for node in context.nodes}

    def evaluate(self, geometry: Geometry, *, smoothing=1) -> Candidate:
        image = self.context.candidate(geometry)
        actual = self.objective.score(image)
        return Candidate(
            geometry,
            {
                node.id: node.values
                for subpath in geometry.subpaths
                for node in subpath.nodes
                if node.values != self.original_values[node.id]
            },
            image,
            actual,
            self.objective.quality(geometry, actual, smoothing),
            crossings(geometry),
        )

    def consider(self, candidate: Candidate) -> bool:
        if candidate.error >= self.before:
            return False
        if self.options.cleanup:
            accepted = (
                candidate.error <= self.least_difference + self.before * 0.01
                and (candidate.crossings, candidate.quality)
                < (self.best.crossings, self.best.quality)
            )
        else:
            accepted = candidate.error < self.best.error
        if accepted:
            self._retain(candidate)
        return accepted

    def consider_bridge(
        self, candidate: Candidate, bridge_crossings: int, *, allow_error: bool
    ) -> bool:
        if (
            candidate.crossings > bridge_crossings
            or (not allow_error and candidate.error >= self.best.error)
            or candidate.error >= self.before
            or candidate.error > self.least_difference + self.before * 0.01
            or (candidate.crossings, candidate.quality)
            >= (bridge_crossings, self.best.quality)
        ):
            return False
        self._retain(candidate)
        return True

    def rescore_quality(self, smoothing: int) -> None:
        self.best = replace(
            self.best,
            quality=self.objective.quality(
                self.best.geometry, self.best.error, smoothing
            ),
        )

    def _retain(self, candidate: Candidate) -> None:
        self.best = candidate
        self.least_difference = min(self.least_difference, candidate.error)
