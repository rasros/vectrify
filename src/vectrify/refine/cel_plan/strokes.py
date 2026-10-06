"""Coherent ink chains and supported silhouette strokes before width grouping."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter

from vectrify.document import PathNode, Subpath
from vectrify.refine import cel
from vectrify.refine.cel_plan.geometry import fitted
from vectrify.refine.cel_plan.ink import boundaries
from vectrify.refine.cel_plan.layers import Overlay
from vectrify.refine.cel_plan.model import Evidence, Options
from vectrify.refine.colour_regions import colour
from vectrify.refine.tracing import _loops


def strokes(
    evidence: Evidence,
    options: Options,
    *,
    labels: np.ndarray | None = None,
    overlays: tuple[Overlay, ...] = (),
    conservative: bool = False,
):
    scale = float(np.sqrt(np.prod(evidence.scale)))
    tolerance = options.boundary_tolerance * scale
    skeleton = cel.thin(evidence.drawn)
    if not skeleton.any() and labels is None and not overlays:
        return [], {"line_paths": 0, "line_pieces": 0}
    light = cel.lightness(evidence.target)
    share = evidence.darkness / np.maximum(evidence.darkness + light, 1)
    ink = (
        np.percentile(evidence.target[skeleton], 5, axis=0)
        if skeleton.any()
        else np.zeros(3)
    )
    paint = colour(ink)
    depth = np.asarray(distance_transform_edt(evidence.drawn))
    width = options.line_width * scale or (
        max(0.8, float(np.percentile(depth[skeleton], 65))) if skeleton.any() else 1.0
    )
    if conservative:
        # Preserve the detected ink as separate unsmoothed centreline runs.
        # Curved outline projection and joins are structural competitors, not
        # assumptions required to produce the initial safe checkpoint.
        runs = cel.line_runs(skeleton, spur=0, depth=depth)
        parts = []
        for run in runs:
            closed = len(run) > 3 and np.array_equal(run[0], run[-1])
            points = cel.simplify(run, 0)
            if closed:
                points = points[:-1]
            if len(points) < 2:
                continue
            contour = Subpath(
                "ink",
                tuple(
                    PathNode(f"n{i}", "M" if i == 0 else "L", tuple(point))
                    for i, point in enumerate(points)
                ),
                closed,
            )
            parts.append(cel._stroke([contour], paint, width, faint=False))
        return parts, {
            "line_paths": len(parts),
            "line_pieces": len(parts),
            "line_style": "strokes",
            "outline": False,
            "outline_width": width / scale,
        }
    silhouette = evidence.foreground & ~evidence.empty
    # An outer stroke is allowed only where detected ink supports a meaningful
    # share of the component's edge. The outline has one stable width/model.
    edge = silhouette & (np.asarray(distance_transform_edt(silhouette)) <= 1.5)
    supported = edge & binary_dilation(evidence.drawn & (share >= 0.2), iterations=2)
    outer = bool(edge.any() and supported.sum() / edge.sum() >= 0.3)
    parts = []
    if outer:
        smooth = gaussian_filter(silhouette.astype(float), 0.8) >= 0.5
        field = np.asarray(distance_transform_edt(smooth))
        slopes = tuple(np.gradient(field))
        contours = [
            cel._contour(
                cel._onto_level(
                    np.array([*points, points[0]]), field, slopes, 0.5 + width / 2
                ),
                tolerance,
            )
            for points in _loops(smooth)
        ]
        if contours:
            parts.append(cel._stroke(contours, paint, width, faint=False))
        # The border ink belongs to the outline, rather than dozens of width
        # pieces. Interior holes remain explicitly outlined above.
        distance = np.asarray(distance_transform_edt(silhouette))
        skeleton &= distance > max(2.5, width * 1.5)
    ink_decisions = []
    overlay_parts = []
    for overlay in overlays:
        if overlay.ink is not None:
            ink = overlay.ink
            local_width = options.line_width * scale or ink.width
            overlay_parts.append(
                cel._stroke(
                    [overlay.model.contour], colour(ink.paint), local_width, faint=False
                )
            )
            ink_decisions.append(
                {
                    "operator": "overlay-ink",
                    "region": overlay.region,
                    "model": overlay.model.kind,
                    "support": ink.support,
                    "width": local_width / scale,
                }
            )
    if overlay_parts:
        covered, _ = cel.line_layer(overlay_parts, light.shape[1], light.shape[0])
        skeleton &= ~binary_dilation(covered[..., 0] > 0.1, iterations=1)
        parts.extend(overlay_parts)
    if labels is not None:
        boundary_parts, boundary_decisions = boundaries(
            evidence, labels, options, width
        )
        ink_decisions.extend(boundary_decisions)
        if boundary_parts:
            covered, _ = cel.line_layer(boundary_parts, light.shape[1], light.shape[0])
            skeleton &= ~binary_dilation(covered[..., 0] > 0.1, iterations=1)
            parts.extend(boundary_parts)
    runs = cel.line_runs(skeleton, spur=2 * width + 1, depth=depth)
    contours = []
    rejected = 0
    for run in runs:
        x = np.clip(run[:, 0].astype(int), 0, light.shape[1] - 1)
        y = np.clip(run[:, 1].astype(int), 0, light.shape[0] - 1)
        strength = float(np.median(share[y, x]))
        length = float(np.linalg.norm(np.diff(run, axis=0), axis=1).sum())
        closed = np.array_equal(run[0], run[-1])
        if strength < 0.15 or (length < 4 and not closed):
            rejected += 1
            continue
        contours.append(
            fitted(run, tolerance).contour
            if labels is not None
            else cel._contour(run, tolerance)
        )
    joined = cel._joined_runs(
        contours, max(2, 2 * width), light=gaussian_filter(light, 0.8)
    )
    if joined:
        parts.append(cel._stroke(joined, paint, width, faint=False))
    return parts, {
        "line_paths": len(parts),
        "line_pieces": len(joined) + int(outer) + len(ink_decisions),
        "line_runs_joined": len(contours) - len(joined),
        "line_rejected_texture": rejected,
        "line_style": "strokes",
        "outline": outer,
        "outline_width": width / scale,
        "ink_models": ink_decisions,
    }
