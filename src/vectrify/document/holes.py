"""Identify removable hole contours without rewriting the surrounding curves."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose

from shapely import STRtree, make_valid
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from vectrify.document.hit_test import (
    HitIndex,
    _filled,
    _flatten,
    mapped,
)
from vectrify.document.join import path_style
from vectrify.document.model import (
    Document,
    DocumentError,
    Geometry,
    PathNode,
    Subpath,
    new_id,
)
from vectrify.document.transforms import root_matrix


def ring_points(sub: Subpath) -> list[tuple[float, float]]:
    """A contour's outline, with curves flattened to 0.01 document units."""
    points = [sub.nodes[0].endpoint]
    for node in sub.nodes[1:]:
        if node.command == "C":
            v = node.values
            points.extend(
                _flatten((points[-1], (v[0], v[1]), (v[2], v[3]), node.endpoint), 0.01)
            )
        else:
            points.append(node.endpoint)
    return points


def signed_area(points: list[tuple[float, float]]) -> float:
    return (
        sum(
            a[0] * b[1] - b[0] * a[1]
            for a, b in zip(points, points[1:] + points[:1], strict=True)
        )
        / 2
    )


def filled_region(geometry: Geometry, rule: str) -> BaseGeometry:
    """The area a geometry fills under *rule*, in its own coordinates."""
    return _filled(
        tuple((tuple(ring_points(sub)), True) for sub in geometry.subpaths), rule
    )


def reversed_subpath(sub: Subpath) -> Subpath:
    """The same closed contour traced the other way round, with new node IDs.

    It starts at the same point; the closing line becomes the first segment
    and each curve swaps its handles.
    """
    nodes = sub.nodes
    start = nodes[0].endpoint
    result = [PathNode(new_id("node"), "M", start)]
    if nodes[-1].endpoint != start:
        result.append(PathNode(new_id("node"), "L", nodes[-1].endpoint))
    for i in range(len(nodes) - 1, 0, -1):
        node, target = nodes[i], nodes[i - 1].endpoint
        if node.command == "C":
            v = node.values
            result.append(PathNode(new_id("node"), "C", (*v[2:4], *v[0:2], *target)))
        elif i > 1:
            result.append(PathNode(new_id("node"), "L", target))
    return Subpath(new_id("subpath"), tuple(result), True)


@dataclass(frozen=True)
class Hole:
    id: str
    subpath_ids: frozenset[str]
    shape: BaseGeometry
    path_data: str


def find_holes(document: Document, object_id: str) -> tuple[Hole, ...]:
    element = document.element(object_id)
    if element.tag != "path" or any(
        a.tag in {"defs", "clipPath"} for a in document.ancestry(object_id)
    ):
        raise DocumentError("Choose a drawing path to inspect its holes")
    style = path_style(document, element)
    if style["fill"] == "none":
        return ()
    geometry = document.geometry_for(object_id)
    rings = []
    for sub in geometry.subpaths:
        points = ring_points(sub)
        if len(points) < 3:
            continue
        area = signed_area(points)
        polygon = Polygon(points)
        shape = make_valid(polygon)
        if shape.area:
            # Traced contours can touch themselves at a vertex. Accept the
            # resulting lobes only when repair preserves their signed area;
            # crossing/cancelling contours remain ambiguous.
            valid = polygon.is_valid or isclose(
                abs(area), shape.area, rel_tol=1e-10, abs_tol=1e-7
            )
            rings.append((sub, shape, 1 if area > 0 else -1, valid))
    tree = STRtree([r[1] for r in rings])
    holes = []
    for i, (sub, shape, sign, valid) in enumerate(rings):
        if not valid:
            continue
        ancestors = []
        ambiguous = False
        for raw in tree.query(shape):
            j = int(raw)
            if i == j:
                continue
            other = rings[j][1]
            if (not rings[j][3] and shape.intersects(other)) or shape.equals(other):
                ambiguous = True
                break
            if other.area > shape.area and other.covers(shape):
                ancestors.append(j)
            elif not shape.covers(other) and shape.intersection(other).area > 1e-7:
                ambiguous = True
                break
        if ambiguous:
            continue
        winding = sum(rings[j][2] for j in ancestors)
        is_hole = (
            len(ancestors) % 2 == 1
            if style["fill-rule"] == "evenodd"
            else winding != 0 and winding + sign == 0
        )
        if not is_hole:
            continue
        descendants = {sub.id}
        for raw in tree.query(shape):
            child, child_shape, _, _ = rings[int(raw)]
            if shape.covers(child_shape):
                descendants.add(child.id)
        holes.append(
            Hole(
                sub.id,
                frozenset(descendants),
                shape,
                Geometry("preview", (sub,)).path_data(),
            )
        )
    return tuple(sorted(holes, key=lambda h: h.shape.area))


def document_hole_shape(
    document: Document, object_id: str, holes: tuple[Hole, ...]
) -> BaseGeometry:
    matrix = root_matrix(document, object_id)
    return mapped(unary_union([h.shape for h in holes]), matrix)


def enclosed_objects(
    document: Document, object_id: str, holes: tuple[Hole, ...]
) -> frozenset[str]:
    """Only entire painted leaf objects fit; never remove an overlapping neighbour."""
    area = document_hole_shape(document, object_id, holes)
    if area.is_empty:
        return frozenset()
    index = HitIndex(document, tolerance=0.01)
    excluded = {a.id for a in document.ancestry(object_id)}
    return frozenset(
        e.id
        for e in document.elements()
        if e.id not in excluded
        and e.tag not in {"g", "svg", "defs", "clipPath"}
        and not any(a.tag in {"defs", "clipPath"} for a in document.ancestry(e.id))
        and index.wholly_inside(e.id, area)
    )
