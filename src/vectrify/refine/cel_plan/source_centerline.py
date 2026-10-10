"""Bounded whole-rail centerline seeds with measured raw corners kept exact.

The caller measures and freezes nongap source evidence after binding a new guide
to actual existing stroke ports. This helper constructs geometry only. Actual
source body/paint/gaps, complete native alpha and ownership need separate proof.
It is not enabled in production search.
"""

from dataclasses import dataclass, replace
from itertools import pairwise

import numpy as np

from vectrify.document import Geometry
from vectrify.refine import cel
from vectrify.refine.cel_plan.band_fit import MAX_MOVEMENT, MAX_PARAMETERS
from vectrify.refine.cel_plan.filled_bands import _check
from vectrify.refine.cel_plan.geometry import fitted
from vectrify.refine.cel_plan.ink_replace import identified
from vectrify.refine.cel_plan.source_bands import MAX_EXTENT, MAX_POINTS

TOLERANCES = (0.75, 0.5, 0.25, 0.15)


@dataclass(frozen=True)
class SourceCenterline:
    geometry: Geometry
    points: np.ndarray
    corners: tuple[tuple[float, float], ...]
    tolerance: float
    parameters: int


def construct(guide, measured, work) -> SourceCenterline | None:
    """Resolve the whole native guide within the existing fitting parameter cap.

    Guide/measurement ports must already coincide. Only guide vertices nearest
    measured raw corners change, within the existing two-pixel construction
    bound. Fitting each segment separately makes those corners exact vertices.
    The most resolved bounded interpretation wins; extra nodes do not relax any
    native/source acceptance gate or authorize original observation movement.
    """
    _check(work)
    for points in (guide, measured):
        if (
            not isinstance(points, np.ndarray)
            or points.ndim != 2
            or points.shape[1] != 2
            or points.dtype.kind not in "fiu"
            or not 4 <= len(points) <= MAX_POINTS
            or not np.isfinite(points).all()
            or np.ptp(points, axis=0).max() > MAX_EXTENT
        ):
            return None
    if not np.array_equal(guide[[0, -1]], measured[[0, -1]]):
        return None
    corners = tuple(
        sorted({tuple(measured[i]) for i in cel.run_corners(measured, False)})
    )
    result = np.array(guide, dtype=float, copy=True)
    fixed = {0, len(guide) - 1}
    for corner in corners:
        _check(work)
        distances = np.linalg.norm(result - corner, axis=1)
        index = int(distances.argmin())
        if index in fixed or distances[index] > MAX_MOVEMENT:
            return None
        result[index] = corner
        fixed.add(index)
    indices = sorted(fixed)
    options = []
    for tolerance in TOLERANCES:
        _check(work)
        segments = [
            fitted(result[a : b + 1], tolerance).contour for a, b in pairwise(indices)
        ]
        nodes = (
            *segments[0].nodes,
            *(node for segment in segments[1:] for node in segment.nodes[1:]),
        )
        # Match the local-normal fitter: two controls per cubic, movable
        # noncorner interior vertices, and one width parameter.
        parameters = 1 + sum(
            3 - (node.endpoint in corners or i == len(nodes) - 1)
            for i, node in enumerate(nodes)
            if node.command == "C"
        )
        if parameters <= MAX_PARAMETERS:
            geometry = identified(
                Geometry("source-centerline", (replace(segments[0], nodes=nodes),)),
                "source-centerline",
            )
            options.append((tolerance, geometry, parameters))
    if not options:
        return None
    tolerance, geometry, parameters = min(options, key=lambda row: row[0])
    result.setflags(write=False)
    _check(work)
    return SourceCenterline(geometry, result, corners, tolerance, parameters)
