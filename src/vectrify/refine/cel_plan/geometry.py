"""Evidence-bounded straight and ellipse proposals in ordinary path geometry.

Canonical fill boundaries use the same fitter on both sides. Endpoints and
supported corners stay anchored. These are proposals, not semantic labels;
the enclosing planner still performs exact visibility/score validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

from vectrify.document import PathNode, Subpath
from vectrify.refine import cel


@dataclass(frozen=True)
class Model:
    contour: Subpath
    kind: str
    residual: float | None


def _sample(points: np.ndarray, count: int = 256) -> np.ndarray:
    lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if lengths[-1] == 0:
        return points[:1]
    keep = np.r_[True, np.diff(lengths) > 1e-8]
    at = np.linspace(0, lengths[-1], min(count, max(8, len(points))))
    return np.column_stack(
        [np.interp(at, lengths[keep], points[keep, axis]) for axis in (0, 1)]
    )


def _straight(points: np.ndarray, tolerance: float) -> Model | None:
    if cel.run_corners(points, False):
        return None
    start, end = points[0], points[-1]
    direction = end - start
    length2 = float(direction @ direction)
    if length2 < 4:
        return None
    along = (points - start) @ direction / length2
    nearest = start + np.clip(along, 0, 1)[:, None] * direction
    distance = np.linalg.norm(points - nearest, axis=1)
    if float(distance.max()) > tolerance or np.min(np.diff(along)) < -0.02:
        return None
    contour = Subpath(
        "model",
        (PathNode("start", "M", tuple(start)), PathNode("end", "L", tuple(end))),
    )
    return Model(contour, "straight", float(distance.max()))


def ellipse(points: np.ndarray, tolerance: float) -> Model | None:
    """A complete, corner-free ellipse fitted to bounded perimeter evidence."""
    if len(points) < 16 or not np.array_equal(points[0], points[-1]):
        return None
    # Pixel staircases and single-pixel extrema are not supported corners.
    # A true turn persists under this small perimeter smoothing; only the
    # classification is smoothed, while fitting still uses the original points.
    smooth = gaussian_filter1d(points[:-1], 1, axis=0, mode="wrap")
    if cel.run_corners(np.vstack((smooth, smooth[0])), True):
        return None
    sample = _sample(points)
    center = sample.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov((sample - center).T))
    radii = np.sqrt(np.maximum(eigenvalues, 1e-6) * 2)
    if radii.min() < 3 or radii.max() / radii.min() > 5:
        return None
    angle = float(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    span = np.ptp(sample, axis=0)

    def coordinates(parameters, values):
        x, y, log_a, log_b, rotation = parameters
        cosine, sine = np.cos(rotation), np.sin(rotation)
        matrix = np.array([[cosine, -sine], [sine, cosine]])
        axes = np.exp([log_a, log_b])
        local = (values - [x, y]) @ matrix / axes
        return local, matrix, axes

    def residual(parameters):
        local, _, axes = coordinates(parameters, sample)
        return (np.linalg.norm(local, axis=1) - 1) * axes.min()

    parameters = np.r_[center, np.log(radii), angle]
    lower = np.r_[sample.min(axis=0) - span, np.log([2, 2]), angle - np.pi]
    upper = np.r_[
        sample.max(axis=0) + span, np.log(np.maximum(span * 2, 4)), angle + np.pi
    ]
    fitted = least_squares(residual, parameters, bounds=(lower, upper), max_nfev=60)
    if not fitted.success or not np.isfinite(fitted.x).all():
        return None
    local, rotation, axes = coordinates(fitted.x, points)
    radial = np.linalg.norm(local, axis=1)
    projected = (local / np.maximum(radial[:, None], 1e-6) * axes) @ rotation.T
    projected += fitted.x[:2]
    # A closed chain has no other junction, but its serialization start remains
    # exact. Translate the fitted ellipse by its bounded start-point residual.
    shift = points[0] - projected[0]
    projected += shift
    error = float(np.linalg.norm(projected - points, axis=1).max())
    error += float(axes.max()) * 0.0003  # Bound the four-cubic ellipse approximation.
    if error > tolerance:
        return None
    # Whole-perimeter evidence prevents completing an ellipse from a short arc.
    theta = np.mod(np.arctan2(local[:-1, 1], local[:-1, 0]), 2 * np.pi)
    sectors = np.unique(np.floor(theta / (np.pi / 8)).astype(int))
    if len(sectors) < 15:
        return None
    signed_area = np.sum(
        points[:-1, 0] * points[1:, 1] - points[1:, 0] * points[:-1, 1]
    )
    direction = 1 if signed_area > 0 else -1
    first = float(np.arctan2(local[0, 1], local[0, 0]))
    origin = fitted.x[:2] + shift

    def position(value):
        return origin + (axes * [np.cos(value), np.sin(value)]) @ rotation.T

    def tangent(value):
        return (axes * [-np.sin(value), np.cos(value)]) @ rotation.T

    nodes = [PathNode("start", "M", tuple(points[0]))]
    kappa = 4 / 3 * np.tan(np.pi / 8) * direction
    for index in range(4):
        start = first + direction * index * np.pi / 2
        end = start + direction * np.pi / 2
        a = position(start) + kappa * tangent(start)
        b = position(end) - kappa * tangent(end)
        endpoint = points[0] if index == 3 else position(end)
        nodes.append(PathNode(f"curve{index}", "C", tuple(np.r_[a, b, endpoint])))
    return Model(Subpath("model", tuple(nodes), closed=True), "ellipse", error)


def fitted(points: np.ndarray, tolerance: float) -> Model:
    """Pick a bounded compact competitor, retaining the unrestricted curve."""
    unrestricted = Model(cel._contour(points, tolerance), "curve", None)
    compact = (
        ellipse(points, tolerance)
        if unrestricted.contour.closed
        else _straight(points, tolerance)
    )
    if compact is not None:
        return compact
    return unrestricted


class Boundaries:
    """One callback per canonical shared chain, with diagnostic model choices."""

    def __init__(self):
        self.decisions: list[dict] = []

    def __call__(self, points: np.ndarray, tolerance: float):
        model = fitted(points, tolerance)
        self.decisions.append(
            {
                "operator": "boundary-model",
                "model": model.kind,
                "residual": model.residual,
                "nodes": len(model.contour.nodes),
            }
        )
        if model.kind == "curve":
            # Keep the established fill fit as the unrestricted competitor;
            # stroke fitting's looser cuts have different acceptance behavior.
            return cel.curve_nodes(
                points, tolerance, smooth=cel.FILL_SMOOTH, fit=cel.FILL_FIT
            )
        nodes = [(node.command, node.values) for node in model.contour.nodes[1:]]
        if model.contour.closed and nodes[-1][1][-2:] != tuple(points[-1]):
            nodes.append(("L", tuple(points[-1])))
        return nodes
