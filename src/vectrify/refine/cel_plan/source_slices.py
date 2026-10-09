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
MAX_PORT_EXTENSION = 2


def _extended_field(shape, extensions):
    """Extend the removal domain, keeping the physical centerline untouched."""
    sub = shape.subpaths[0]
    nodes = sub.nodes
    start, end = np.asarray(nodes[0].endpoint), np.asarray(nodes[-1].endpoint)
    first = np.asarray(nodes[1].values[:2]) - start
    last = end - np.asarray(
        nodes[-1].values[-4:-2] if nodes[-1].command == "C" else nodes[-2].endpoint
    )
    for amount, tangent in zip(extensions, (first, last), strict=True):
        if amount and np.linalg.norm(tangent) <= 1e-12:
            return None
    if extensions[0]:
        nodes = (
            replace(
                nodes[0],
                id="field-start",
                values=tuple(start - extensions[0] * first / np.linalg.norm(first)),
            ),
            replace(nodes[0], command="L"),
            *nodes[1:],
        )
    if extensions[1]:
        nodes = (
            *nodes,
            replace(
                nodes[-1],
                id="field-end",
                command="L",
                values=tuple(end + extensions[1] * last / np.linalg.norm(last)),
            ),
        )
    return replace(shape, subpaths=(replace(sub, nodes=nodes),))


def source_slice(
    document, oid, observed, work, *, tolerance=0.75, port_extension=(0.0, 0.0)
):
    """A bounded complete nongap profile, never a fragment or a shade edge.

    The selected fill must be thin along the physical source chain. Broad
    shadow interiors and unsupported winding interpretations remain excluded.
    Callers must verify actual raster locality and replay ownership atomically.
    """
    _check(work)
    if not np.isfinite(tolerance) or not 0 < tolerance <= 3:
        raise ValueError("Source slicing requires a bounded fitting tolerance")
    extensions = np.asarray(port_extension, dtype=float)
    if (
        extensions.shape != (2,)
        or not np.isfinite(extensions).all()
        or (extensions < 0).any()
        or (extensions > MAX_PORT_EXTENSION).any()
    ):
        raise ValueError("Source slicing requires bounded port extensions")
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
    selected_path = pathops.op(old, clip, pathops.PathOp.INTERSECTION)
    if extensions.any():
        extended = _extended_field(shape, extensions)
        if extended is None:
            return None
        extended_clip = curve_path(
            transformed_geometry(
                footprint(extended, MAX_WIDTH, cap="butt"), inverse_matrix(frame)
            )
        )
        extended_path = pathops.op(old, extended_clip, pathops.PathOp.INTERSECTION)
        # Numerical intersections must never trim the original complete field.
        if pathops.op(selected_path, extended_path, pathops.PathOp.DIFFERENCE).area:
            extended_path = pathops.op(
                selected_path, extended_path, pathops.PathOp.UNION
            )
        if pathops.op(selected_path, extended_path, pathops.PathOp.DIFFERENCE).area:
            return None
        selected_path = extended_path
    selected = path_geometry(selected_path)
    remainder = pathops.op(
        old, selected_path if extensions.any() else clip, pathops.PathOp.DIFFERENCE
    )
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
