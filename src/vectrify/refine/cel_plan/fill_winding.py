"""Resolve filled Boolean seams without exempting crossings from validation."""

from __future__ import annotations

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from vectrify.document import Geometry, PathNode, Subpath
from vectrify.document.hit_test import _filled
from vectrify.document.join import curve_path, path_geometry, transformed_geometry
from vectrify.document.topology import inverse_matrix
from vectrify.refine.cel_plan.ink_replace import MAX_NODES
from vectrify.refine.crossings import crossings

TOLERANCE = 0.02


def flatten(subpath, work, allowance):
    """Bounded chord subdivision, including collinear overshoot and cusps."""
    points = [np.array(subpath.nodes[0].endpoint)]
    for node in subpath.nodes[1:]:
        if node.command == "L":
            points.append(np.array(node.endpoint))
        else:
            control = np.vstack((points[-1], np.array(node.values).reshape(3, 2)))
            stack = [(control, 0)]
            while stack:
                if work.interrupted or len(points) + len(stack) > allowance:
                    return None
                control, depth = stack.pop()
                a, _b, _c, d = control
                chord = d - a
                length = float(chord @ chord)
                along = np.clip((control[1:3] - a) @ chord / max(length, 1e-30), 0, 1)
                distance = np.linalg.norm(
                    control[1:3] - a - along[:, None] * chord, axis=1
                )
                if distance.max() <= TOLERANCE:
                    points.append(d)
                    continue
                if depth >= 24:
                    return None
                ab, bc, cd = (control[:-1] + control[1:]) / 2
                abc, bcd = (ab + bc) / 2, (bc + cd) / 2
                middle = (abc + bcd) / 2
                stack.append((np.array((middle, bcd, cd, d)), depth + 1))
                stack.append((np.array((a, ab, abc, middle)), depth + 1))
        if len(points) > allowance:
            return None
    return Subpath(
        subpath.id,
        tuple(
            PathNode(f"flat-{i}", "M" if i == 0 else "L", tuple(p))
            for i, p in enumerate(points)
        ),
        subpath.closed,
    )


def resolved(geometry, matrix, rule, work):
    """Preserve curves first; polygonize only persistent crossed contours.

    The 0.02-pixel chord bound applies in native coordinates, regardless of
    the paint frame. Filled winding is resolved again after subdivision. All
    resulting pixels, alpha and geometry still face the ordinary exact policy.
    """
    if work.interrupted or sum(len(s.nodes) for s in geometry.subpaths) > MAX_NODES:
        return None
    native = transformed_geometry(geometry, matrix)
    path = curve_path(native, rule)
    path.simplify()
    native = path_geometry(path)
    if crossings(native):
        parts = []
        used = 0
        for subpath in native.subpaths:
            if work.interrupted:
                return None
            subpath = flatten(subpath, work, MAX_NODES - used)
            if subpath is None:
                return None
            used += len(subpath.nodes)
            if used > MAX_NODES:
                return None
            parts.append(subpath)
        # Float32 Boolean winding can reproduce a tiny crossing at a nearly
        # coincident vertex. Node/classify these bounded native polylines in
        # double precision using the editor's existing fill-rule semantics.
        shape = _filled(
            tuple((tuple(n.endpoint for n in s.nodes), s.closed) for s in parts),
            "nonzero",
        )
        if isinstance(shape, Polygon):
            polygons = [shape]
        elif isinstance(shape, MultiPolygon):
            polygons = list(shape.geoms)
        else:
            return None
        parts = []
        used = 0
        for polygon in polygons:
            if work.interrupted:
                return None
            for ring in (polygon.exterior, *polygon.interiors):
                coordinates = tuple(ring.coords)[:-1]
                used += len(coordinates)
                if used > MAX_NODES:
                    return None
                parts.append(
                    Subpath(
                        f"ring-{len(parts)}",
                        tuple(
                            PathNode(
                                f"ring-{len(parts)}-{i}", "M" if i == 0 else "L", p
                            )
                            for i, p in enumerate(coordinates)
                        ),
                        True,
                    )
                )
        native = Geometry("resolved", tuple(parts))
    if (
        work.interrupted
        or sum(len(s.nodes) for s in native.subpaths) > MAX_NODES
        or crossings(native)
    ):
        return None
    local = transformed_geometry(native, inverse_matrix(matrix))
    return None if work.interrupted or crossings(local) else local
