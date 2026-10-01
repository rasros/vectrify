"""A thin filled shape as the line down its middle, to draw as a stroke.

The shape is rasterised a few pixels across its thickness, thinned to a
centreline the way the cel tracer thins its lines, and the centreline is
fitted with curves. The stroke width is the shape's thickness, measured
along the centreline; round caps give back the length thinning takes off.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

from vectrify.document.holes import filled_region
from vectrify.document.lines import contour_ends, end_pairs, joined
from vectrify.document.model import DocumentError, Geometry, PathNode, Subpath, new_id
from vectrify.refine.cel import curve_nodes, line_runs, thin

# Pixels across the shape's thickness when it is rasterised, and the most
# pixels along its longer side.
ACROSS = 12
LARGEST = 3000


def centreline(
    geometry: Geometry, rule: str = "nonzero", tolerance: float = 0.25
) -> tuple[Geometry, float]:
    """*geometry*'s filled area as open and closed centrelines, and the
    stroke width that covers it, both in its own coordinates.

    The curves follow the centreline within *tolerance*, or a pixel of the
    raster, whichever is more. Lines that meet at a junction run on through
    it the straightest way. A shape too wide for its length is not a line.
    """
    region = filled_region(geometry, rule)
    if region.is_empty or region.area <= 0:
        raise DocumentError("This shape fills no area to find a line in")
    thickness = 2 * region.area / region.length
    left, top, right, bottom = region.bounds
    scale = min(ACROSS / thickness, LARGEST / max(right - left, bottom - top))
    pad = 2
    width = int(np.ceil((right - left) * scale)) + 2 * pad
    height = int(np.ceil((bottom - top) * scale)) + 2 * pad
    canvas = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(canvas)

    def pixels(ring) -> list[tuple[float, float]]:
        return [
            ((x - left) * scale + pad, (y - top) * scale + pad) for x, y in ring.coords
        ]

    for polygon in getattr(region, "geoms", (region,)):
        if polygon.geom_type != "Polygon":
            continue
        draw.polygon(pixels(polygon.exterior), fill=1)
        for hole in polygon.interiors:
            draw.polygon(pixels(hole), fill=0)
    mask = np.asarray(canvas, dtype=bool)
    skeleton = thin(mask)
    if not skeleton.any():
        raise DocumentError("This shape is too small to find a line in")
    # A pixel inside lies half a pixel further from the nearest outside pixel
    # than from the edge between them.
    across = (
        2 * float(np.median(np.asarray(distance_transform_edt(mask))[skeleton])) - 1
    )
    runs = line_runs(skeleton, spur=across)
    length = sum(float(np.linalg.norm(np.diff(r, axis=0), axis=1).sum()) for r in runs)
    if not runs or length < across:
        raise DocumentError(
            "This shape is not a thin line: it is about as wide as it is long"
        )
    # The raster's staircase is smoothed out over a quarter of the thickness.
    step, smooth = max(1.5, tolerance * scale), ACROSS / 4
    contours = []
    for run in runs:
        points = (run - pad) / scale + (left, top)
        nodes = [PathNode(new_id("node"), "M", tuple(points[0]))]
        closed = len(run) > 3 and np.array_equal(run[0], run[-1])
        fitted = curve_nodes(run, step, smooth=smooth)
        if closed and fitted and fitted[-1][0] == "L":
            fitted = fitted[:-1]
        for command, values in fitted:
            local = np.reshape(values, (-1, 2))
            local = (local - pad) / scale + (left, top)
            nodes.append(PathNode(new_id("node"), command, tuple(local.ravel())))
        if len(nodes) > 1:
            contours.append((tuple(nodes), closed))
    open_lines = [nodes for nodes, closed in contours if not closed]
    ends = [e for i, nodes in enumerate(open_lines) for e in contour_ends(i, nodes)]
    chains, _ = joined(open_lines, end_pairs(ends, 0.0))
    used = {c for members, _ in chains for c in members}
    subpaths = [
        Subpath(new_id("subpath"), nodes, closed)
        for nodes, closed in contours
        if closed
    ]
    subpaths.extend(
        Subpath(new_id("subpath"), nodes, False)
        for i, nodes in enumerate(open_lines)
        if i not in used
    )
    subpaths.extend(subpath for _, subpath in chains)
    return Geometry(geometry.id, tuple(subpaths)), across / scale
