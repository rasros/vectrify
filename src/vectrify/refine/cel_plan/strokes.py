"""Coherent ink chains and supported silhouette strokes before width grouping."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter

from vectrify.refine import cel
from vectrify.refine.cel_plan.model import Evidence, Options
from vectrify.refine.colour_regions import colour
from vectrify.refine.tracing import _loops


def strokes(evidence: Evidence, options: Options):
    scale = float(np.sqrt(np.prod(evidence.scale)))
    tolerance = options.boundary_tolerance * scale
    skeleton = cel.thin(evidence.drawn)
    if not skeleton.any():
        return [], {"line_paths": 0, "line_pieces": 0}
    light = cel.lightness(evidence.target)
    share = evidence.darkness / np.maximum(evidence.darkness + light, 1)
    ink = np.percentile(evidence.target[skeleton], 5, axis=0)
    paint = colour(ink)
    depth = np.asarray(distance_transform_edt(evidence.drawn))
    width = options.line_width * scale or max(
        0.8, float(np.percentile(depth[skeleton], 65))
    )
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
        contours.append(cel._contour(run, tolerance))
    joined = cel._joined_runs(
        contours, max(2, 2 * width), light=gaussian_filter(light, 0.8)
    )
    if joined:
        parts.append(cel._stroke(joined, paint, width, faint=False))
    return parts, {
        "line_paths": len(parts),
        "line_pieces": len(joined) + int(outer),
        "line_runs_joined": len(contours) - len(joined),
        "line_rejected_texture": rejected,
        "line_style": "strokes",
        "outline": outer,
        "outline_width": width / scale,
    }
