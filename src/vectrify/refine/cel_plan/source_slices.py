"""Remove a complete source-supported thin band from a compound shadow owner.

Keep the original shadow curves and subtract only the selected band by directed
winding. An independent Boolean area comparison verifies that the resulting
filled region is the exact difference. This is construction, not an ownership,
alpha, source-gap or painted native proof.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pathops

from vectrify.document import Geometry
from vectrify.document.holes import reversed_subpath
from vectrify.document.join import (
    curve_path,
    path_geometry,
    path_style,
    transformed_geometry,
)
from vectrify.document.redraw import root_matrix
from vectrify.document.topology import inverse_matrix
from vectrify.refine.cel_plan.filled_bands import MAX_NODES, MAX_WIDTH, _check
from vectrify.refine.cel_plan.geometry import fitted
from vectrify.refine.cel_plan.ink_models import footprint
from vectrify.refine.cel_plan.source_bands import MAX_EXTENT, MAX_POINTS

MAX_COMPOUND_NODES = 1024
MAX_SELECTED_SUBPATHS = 16


def source_slice(document, oid, observed, work, *, tolerance=0.75):
    """A bounded complete nongap profile, never a fragment or a shade edge.

    The selected fill must be thin along the physical source chain. Broad
    shadow interiors and unsupported winding interpretations remain excluded.
    Callers must verify actual raster locality and replay ownership atomically.
    """
    _check(work)
    if not np.isfinite(tolerance) or not 0 < tolerance <= 3:
        raise ValueError("Source slicing requires a bounded fitting tolerance")
    points = observed.anchors
    geometry = document.geometry_for(oid)
    if (
        points is None
        or path_style(document, document.element(oid))["fill-rule"] != "nonzero"
        or not 4 <= len(points) <= MAX_POINTS
        or observed.gaps.any()
        or observed.qualified.mean() < 0.6
        or np.array_equal(points[0], points[-1])
        or np.ptp(points, axis=0).max() > MAX_EXTENT
        or any(not s.closed for s in geometry.subpaths)
        or sum(len(s.nodes) for s in geometry.subpaths) > MAX_COMPOUND_NODES
    ):
        return None
    frame = root_matrix(document, oid)
    shape = Geometry("source-slice", (fitted(points, tolerance).contour,))
    if len(shape.subpaths[0].nodes) > 8:
        return None
    clip = curve_path(
        transformed_geometry(
            footprint(shape, MAX_WIDTH, cap="butt"), inverse_matrix(frame)
        )
    )
    old = curve_path(geometry)
    selected = path_geometry(pathops.op(old, clip, pathops.PathOp.INTERSECTION))
    remainder = pathops.op(old, clip, pathops.PathOp.DIFFERENCE)
    _check(work)
    if (
        not 0 < len(selected.subpaths) <= MAX_SELECTED_SUBPATHS
        or sum(len(s.nodes) for s in selected.subpaths) > MAX_NODES
        or not remainder.area
    ):
        return None
    native = transformed_geometry(selected, frame)
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    if not length or abs(curve_path(native).area) / length > MAX_WIDTH / 2:
        return None
    # Opposite winding removes this complete field without resolving or
    # reparameterizing every independent curve in the compound shadow.
    retained = replace(
        geometry,
        subpaths=(
            *geometry.subpaths,
            *(reversed_subpath(s) for s in selected.subpaths),
        ),
    )
    if sum(len(s.nodes) for s in retained.subpaths) > MAX_COMPOUND_NODES:
        return None
    if pathops.op(curve_path(retained), remainder, pathops.PathOp.XOR).area != 0:
        return None
    if (
        pathops.op(
            curve_path(retained), curve_path(selected), pathops.PathOp.INTERSECTION
        ).area
        != 0
    ):
        return None
    _check(work)
    return selected, retained
