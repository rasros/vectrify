"""One comparison of a candidate against the reference, read by every objective.

Structure and colour are measured once per candidate and then reduced over
whatever area an objective cares about -- the whole canvas for the measures,
a grid cell for the region objectives. Sharing the arrays is what makes the
region objectives structure-aware for less work than the colour-only version
cost: the Lab conversion is the expensive half and it used to happen twice, once
inside the score and again inside the region metrics.

The reference side never changes during a run, so its Lab array and edge map are
built once and reused for every candidate.
"""

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from vectrify.score.base import DEFAULT_CONFIG
from vectrify.score.edges import edge_map, overlap_distance
from vectrify.score.moments import shape_distance
from vectrify.score.utils import clamp01, lab_array


@dataclass(frozen=True)
class Reference:
    image: Image.Image
    lab: np.ndarray
    edges: np.ndarray
    # Carried so a candidate is measured the same way the reference was,
    # without threading the setting through every comparison.
    tolerance: float | None = None


def prepare(reference_rgb: Image.Image, tolerance: float | None = None) -> Reference:
    return Reference(
        image=reference_rgb,
        lab=lab_array(reference_rgb),
        edges=edge_map(reference_rgb, tolerance=tolerance),
        tolerance=tolerance,
    )


@dataclass(frozen=True)
class Comparison:
    """Per-pixel structure and colour, ready to reduce over any area."""

    colour: np.ndarray
    reference_edges: np.ndarray
    candidate_edges: np.ndarray

    @property
    def shape(self) -> float:
        """Difference in shape, with position and scale divided out."""
        return shape_distance(self.reference_edges > 0.2, self.candidate_edges > 0.2)

    def blend(self) -> float:
        """Structure and colour over the whole canvas: the search measures."""
        if self.colour.size == 0:
            return 0.0
        structure = overlap_distance(self.reference_edges, self.candidate_edges)
        weight = DEFAULT_CONFIG.w_structure
        return clamp01(weight * structure + (1.0 - weight) * float(self.colour.mean()))


def compare(reference: Reference, candidate_png: bytes) -> Comparison:
    candidate = Image.open(io.BytesIO(candidate_png)).convert("RGB")
    if candidate.size != reference.image.size:
        candidate = candidate.resize(
            reference.image.size, resample=Image.Resampling.BILINEAR
        )
    return Comparison(
        colour=np.abs(reference.lab - lab_array(candidate)).mean(axis=2) / 255.0,
        reference_edges=reference.edges,
        candidate_edges=edge_map(candidate, tolerance=reference.tolerance),
    )
