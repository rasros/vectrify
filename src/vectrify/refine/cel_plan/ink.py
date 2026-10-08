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
    opacity: np.ndarray | None = None,
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
    return _profile(
        points, target, width, light=light, visible=visible, opacity=opacity
    )


def measure_link(
    points,
    target,
    width,
    *,
    light=None,
    visible=None,
    opacity=None,
    junctions=None,
    paint=None,
):
    """Prove a short existing junction link at every original source sample.

    The caller must establish two incident supported source chains at these
    exact junctions. Existing incident bodies may supply junction geometry;
    every raw source sample must still agree with the established ink paint.
    Elsewhere a raw painted-side trough is required. No unsupported interval
    or smoothed contrast alone can justify a link.
    """
    if len(points) < 2 or np.array_equal(points[0], points[-1]):
        return None
    # Junction centres can have fractional coordinates farther apart than an
    # ordinary skeleton step. Inspect the intervening source at half-pixel
    # spacing, keeping the original ends exactly; sparse ports cannot bridge a
    # missing pixel. The caller charges this sample count to its point budget.
    steps = np.maximum(
        1, np.ceil(2 * np.linalg.norm(np.diff(points, axis=0), axis=1))
    ).astype(int)
    points = np.vstack(
        [
            *[
                np.linspace(a, b, n, endpoint=False)
                for a, b, n in zip(points[:-1], points[1:], steps, strict=True)
            ],
            points[-1:],
        ]
    )
    joined = junctions(points) if junctions is not None else np.zeros(len(points), bool)
    return _profile(
        points,
        target,
        width,
        light=light,
        visible=visible,
        opacity=opacity,
        strict=True,
        joined=joined,
        paint=paint,
    )


def _profile(
    points,
    target,
    width,
    *,
    light=None,
    visible=None,
    opacity=None,
    strict=False,
    joined=None,
    paint=None,
):
    if opacity is not None and (opacity.shape != target.shape[:2] or visible is None):
        raise ValueError("Source opacity must align with target and visibility")
    step = np.linalg.norm(np.diff(points, axis=0), axis=1)
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
    if strict:
        # A junction's side probes may fall inside its incident ink. Existing
        # supported stroke bodies supply geometry there, not imaginary bright
        # sides. Keep those source positions and sample their original paint.
        location[joined] = int(np.argmin(np.abs(offsets)))
    dark = light[np.arange(len(points)), location]
    contrast = surface - dark
    supported = np.isfinite(surface) & (contrast >= 12) & (dark <= 150)
    if seen is not None:
        supported &= middle[np.arange(len(points)), location]
    if strict:
        # Sample original centres and both sides directly from painted source
        # RGB. Gaussian filtering or a nearby trough cannot fill a real gap.
        raw_points = (
            points[:, None, :]
            + np.array((-reach, 0, reach))[None, :, None] * normal[:, None, :]
        )
        raw_coordinates = np.stack(
            (raw_points[..., 1] - 0.5, raw_points[..., 0] - 0.5)
        ).reshape(2, -1)
        raw_colors, raw_seen = _colors(target, raw_coordinates, visible)
        raw_light = cel.lightness(raw_colors).reshape(-1, 3)
        raw_seen = raw_seen.reshape(-1, 3)
        sides = np.minimum(
            np.where(raw_seen[:, 0] >= 0.5, raw_light[:, 0], np.inf),
            np.where(raw_seen[:, 2] >= 0.5, raw_light[:, 2], np.inf),
        )
        raw_support = (raw_seen[:, 1] >= 0.5) & (raw_light[:, 1] <= 150)
        if paint is not None:
            raw_support &= np.all(
                np.abs(raw_colors.reshape(-1, 3, 3)[:, 1] - paint) <= 24, axis=1
            )
        supported = raw_support & (
            joined | (supported & np.isfinite(sides) & (sides - raw_light[:, 1] >= 12))
        )
        if not supported.all():
            return None
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
    colors, _ = _colors(target, peak_coordinates, visible)
    ink = np.percentile(colors[supported], 15, axis=0)
    surface = np.where(np.isfinite(surface), surface, dark)
    cover = np.clip(
        (surface[:, None] - light) / np.maximum((surface - dark)[:, None], 1), 0, 1
    )
    # Limit width integration to the ridge near the center, not other marks
    # sampled outside it. A line override is applied later by the caller.
    cover[:, np.abs(offsets) > max(2, 2 * width)] = 0
    if seen is not None:
        if opacity is not None:
            alpha = map_coordinates(
                opacity, coordinates, order=1, mode="constant", cval=0
            )
            if not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1)):
                raise ValueError("Sampled source opacity must be finite in [0, 1]")
            # A faint exterior fringe contributes fractional line coverage.
            # Normalize within this source profile so uniform half/quarter
            # opacity retains its intrinsic width and centroid. Source alpha
            # uses the same cropped analysis grid as RGB; hidden RGB and other
            # disconnected marks still cannot supply contrast or coverage.
            level = np.maximum(alpha.max(axis=1, keepdims=True), 1e-12)
            cover *= seen * np.clip(alpha / level, 0, 1)
        else:
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
    if strict:
        centered[joined] = points[joined]
    centered[0], centered[-1] = points[0], points[-1]
    return Ink(centered, max(0.8, measured_width), ink, share, peak_gap)


def _colors(target, coordinates, visible):
    """Bilinear RGB normalized only over painted neighbours, without RGB copies."""
    count = coordinates.shape[1]
    if visible is None:
        return np.stack(
            [
                map_coordinates(target[..., k], coordinates, order=1, mode="nearest")
                for k in range(3)
            ],
            axis=-1,
        ), np.ones(count)
    origin = np.floor(coordinates).astype(int)
    fraction = coordinates - origin
    colors, weight = np.zeros((count, 3)), np.zeros(count)
    for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
        y, x = origin + np.array([[dy], [dx]])
        valid = (y >= 0) & (x >= 0) & (y < visible.shape[0]) & (x < visible.shape[1])
        y = np.clip(y, 0, visible.shape[0] - 1)
        x = np.clip(x, 0, visible.shape[1] - 1)
        amount = np.prod(np.abs(np.array([[1 - dy], [1 - dx]]) - fraction), axis=0)
        amount *= valid & visible[y, x]
        colors += amount[:, None] * target[y, x]
        weight += amount
    colors /= np.maximum(weight[:, None], 1e-12)
    return colors, weight


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
