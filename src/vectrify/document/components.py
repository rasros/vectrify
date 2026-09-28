"""Connected-subpath partitioning without rewriting any curves."""

from __future__ import annotations

from shapely import STRtree
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union

from vectrify.document.hit_test import _flatten
from vectrify.document.model import Geometry, Subpath


def disconnected_parts(
    geometry: Geometry, *, filled: bool
) -> tuple[tuple[Subpath, ...], ...]:
    """Keep touching/overlapping contours and nested fill rings together.

    Flattening is only used for classification. The original subpaths, nodes,
    pins and exact cubic coordinates are returned untouched. Only intersecting
    footprints connect; stroke styling and positive gaps do not connect paths.
    Filled rings include their interiors, so holes cannot become solid shapes.
    """
    if len(geometry.subpaths) < 2:
        return (geometry.subpaths,)
    tolerance = 0.01
    footprints = []
    for subpath in geometry.subpaths:
        points = [subpath.nodes[0].endpoint]
        for node in subpath.nodes[1:]:
            if node.command == "C":
                v = node.values
                points.extend(
                    _flatten(
                        (points[-1], (v[0], v[1]), (v[2], v[3]), node.endpoint),
                        tolerance,
                    )
                )
            else:
                points.append(node.endpoint)
        if (subpath.closed or filled) and points[-1] != points[0]:
            points.append(points[0])
        shape = LineString(points) if len(points) > 1 else Point(points[0])
        if filled and len(points) >= 4:
            polygon = Polygon(points)
            # Include all bounded faces, irrespective of winding. This may keep
            # extra islands together, but never separates a ring from its hole.
            interior = (
                polygon
                if polygon.is_valid
                else unary_union(list(polygonize(unary_union(shape))))
            )
            shape = unary_union((shape, interior))
        footprints.append(shape)
    tree = STRtree(footprints)
    parents = list(range(len(footprints)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for i, footprint in enumerate(footprints):
        for candidate in tree.query(footprint, predicate="intersects"):
            j = int(candidate)
            if j < i:
                parents[root(i)] = root(j)
    groups: dict[int, list[Subpath]] = {}
    for i, subpath in enumerate(geometry.subpaths):
        groups.setdefault(root(i), []).append(subpath)
    return tuple(tuple(group) for group in groups.values())
