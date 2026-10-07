"""Continuous ink hypotheses supported by canonical color boundaries.

Shade edges without a dark ridge on both sides are not ink. Each hypothesis
keeps its local color/width; the complete export still competes under the exact
score, including any gap completion that the coarse evidence proposes.
Source visibility can support one-sided exterior ink from painted interior
contrast; unpainted RGB supplies no evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from vectrify.document import Geometry
from vectrify.refine import cel
from vectrify.refine.cel_plan.geometry import fitted
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.model import Evidence, Options
from vectrify.refine.colour_regions import colour
from vectrify.refine.crossings import crossings


@dataclass(frozen=True)
class Ink:
    points: np.ndarray
    width: float
    paint: np.ndarray
    support: float
    peak_gap: float


def measure(
    points: np.ndarray,
    target: np.ndarray,
    width: float,
    *,
    light: np.ndarray | None = None,
    visible: np.ndarray | None = None,
) -> Ink | None:
    """Measure a dark ridge; source visibility permits a one-sided exterior.

    An unpainted side supplies no brightness evidence. At least one painted
    surface must prove contrast; two painted sides still require a trough.
    Source profiles use contiguous coverage centroids rather than plateau
    argmins, retaining the original junction/end anchors.
    """
    if len(points) < 4:
        return None
    step = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if step.sum() < max(8, 4 * width):
        return None
    tangent = np.gradient(points, axis=0)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
    reach = max(3, 2.5 * width)
    offsets = np.arange(-reach, reach + 0.25, 0.5)
    samples = points[:, None, :] + offsets[None, :, None] * normal[:, None, :]
    coordinates = [samples[..., 1] - 0.5, samples[..., 0] - 0.5]
    if light is None:
        if visible is None:
            light = gaussian_filter(cel.lightness(target), 0.5)
        else:
            weight = gaussian_filter(visible.astype(np.float32), 0.5)
            light = gaussian_filter(cel.lightness(target) * visible, 0.5)
            light /= np.maximum(weight, 1e-12)
    light = map_coordinates(light, coordinates, order=1, mode="nearest")
    # Contrast must be darker than both adjacent surfaces, not just one side
    # of a shade discontinuity. Search only near the proposed centerline.
    surface = np.minimum(light[:, 0], light[:, -1])
    middle = np.broadcast_to(np.abs(offsets) <= max(1, width), light.shape)
    seen = None
    if visible is not None:
        seen = map_coordinates(
            visible.astype(np.float32), coordinates, order=1, mode="constant", cval=0
        )
        surface = np.minimum(
            np.where(seen[:, 0] >= 0.5, light[:, 0], np.inf),
            np.where(seen[:, -1] >= 0.5, light[:, -1], np.inf),
        )
        middle = middle & (seen >= 0.5)
        active = (surface[:, None] - light > 1e-6) & (seen > 0)
        components = np.cumsum(~active, axis=1)
        center = int(np.argmin(np.abs(offsets)))
        middle &= active & (components == components[:, center, None])
    location = np.where(middle, light, np.inf).argmin(axis=1)
    dark = light[np.arange(len(points)), location]
    contrast = surface - dark
    supported = np.isfinite(surface) & (contrast >= 12) & (dark <= 150)
    if seen is not None:
        supported &= middle[np.arange(len(points)), location]
    share = float(supported.mean())
    if share < 0.45:
        return None
    # A short arc cannot justify completing a whole contour or arbitrary gap.
    closed = np.array_equal(points[0], points[-1])
    gaps = []
    sequence = np.r_[supported[:-1], supported[:-1]] if closed else supported
    current = 0.0
    gap_steps = np.r_[step, step] if closed else np.r_[step, 0]
    for observed, distance in zip(sequence, gap_steps, strict=True):
        current = 0.0 if observed else current + distance
        gaps.append(current)
    peak_gap = max(gaps, default=0.0)
    if peak_gap > min(float(step.sum()) * 0.25, max(12, 8 * width)):
        return None
    peak_coordinates = np.asarray(coordinates)[:, np.arange(len(points)), location]
    if visible is None:
        colors = np.stack(
            [
                map_coordinates(
                    target[..., channel], peak_coordinates, order=1, mode="nearest"
                )
                for channel in range(3)
            ],
            axis=-1,
        )
    else:
        # Normalize bilinear color only over painted neighbors of the peak.
        # This uses four samples per run point, without full-image RGB copies
        # or admitting arbitrary transparent RGB into the stroke paint.
        origin = np.floor(peak_coordinates).astype(int)
        fraction = peak_coordinates - origin
        colors, weight = np.zeros((len(points), 3)), np.zeros(len(points))
        for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
            y, x = origin + np.array([[dy], [dx]])
            valid = (
                (y >= 0) & (x >= 0) & (y < visible.shape[0]) & (x < visible.shape[1])
            )
            y = np.clip(y, 0, visible.shape[0] - 1)
            x = np.clip(x, 0, visible.shape[1] - 1)
            amount = np.prod(np.abs(np.array([[1 - dy], [1 - dx]]) - fraction), axis=0)
            amount *= valid & visible[y, x]
            colors += amount[:, None] * target[y, x]
            weight += amount
        colors /= np.maximum(weight[:, None], 1e-12)
    ink = np.percentile(colors[supported], 15, axis=0)
    surface = np.where(np.isfinite(surface), surface, dark)
    cover = np.clip(
        (surface[:, None] - light) / np.maximum((surface - dark)[:, None], 1), 0, 1
    )
    # Limit width integration to the ridge near the center, not other marks
    # sampled outside it. A line override is applied later by the caller.
    cover[:, np.abs(offsets) > max(2, 2 * width)] = 0
    if seen is not None:
        cover *= seen
        # Keep the one-dimensional component containing the supported peak.
        # A second nearby dark mark cannot lend width or pull the centroid
        # across an actual zero-coverage profile gap.
        active = cover > 1e-6
        components = np.cumsum(~active, axis=1)
        peak = components[np.arange(len(points)), location]
        cover *= active & (components == peak[:, None])
    widths = cover.sum(axis=1) * 0.5
    measured_width = float(np.median(widths[supported]))
    displacement = offsets[location]
    if seen is not None:
        displacement = (cover * offsets).sum(axis=1) / np.maximum(
            cover.sum(axis=1), 1e-12
        )
    centered = points + displacement[:, None] * normal
    # Unsupported pixels keep their original canonical boundary position.
    centered = np.where(supported[:, None], centered, points)
    centered[0], centered[-1] = points[0], points[-1]
    return Ink(centered, max(0.8, measured_width), ink, share, peak_gap)


def boundaries(
    evidence: Evidence, labels: np.ndarray, options: Options, typical: float
):
    graph = build(evidence, labels)
    scale = float(np.sqrt(np.prod(evidence.scale)))
    parts = []
    decisions = []
    for boundary in graph.boundaries:
        if (
            min(boundary.left, boundary.right) < 0
            or boundary.left in graph.hidden
            or boundary.right in graph.hidden
        ):
            continue
        proof = measure(boundary.points, evidence.target, typical)
        if proof is None:
            continue
        width = options.line_width * scale or proof.width
        model = fitted(proof.points, options.boundary_tolerance * scale)
        if crossings(Geometry("proof", (model.contour,))):
            model = fitted(boundary.points, options.boundary_tolerance * scale)
        if crossings(Geometry("proof", (model.contour,))):
            continue
        parts.append(
            cel._stroke([model.contour], colour(proof.paint), width, faint=False)
        )
        decisions.append(
            {
                "operator": "boundary-ink",
                "boundary": boundary.id,
                "model": model.kind,
                "support": proof.support,
                "peak_gap": proof.peak_gap / scale,
                "width": width / scale,
            }
        )
    return parts, decisions
