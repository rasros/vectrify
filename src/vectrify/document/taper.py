"""Variable-width ends derived from an editable centerline.

Lengths are local drawing units. Overlapping tapers shrink proportionally so
short contours still reach the requested stroke width between their tips.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import TYPE_CHECKING

from shapely.geometry import GeometryCollection, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

if TYPE_CHECKING:
    import pathops

    from vectrify.document.model import Document, Element, Geometry

Point = tuple[float, float]

START = "data-vectrify-taper-start"
END = "data-vectrify-taper-end"
TAPERS = frozenset({START, END})
OWNER = "data-vectrify-taper-owner"
SOURCE_OPACITY = "data-vectrify-taper-source-opacity"


def active(style: dict[str, str]) -> bool:
    # Import examines metadata before validating the complete document.
    try:
        return any(float(style.get(key, "0")) > 0 for key in TAPERS)
    except ValueError:
        return False


def polygons(
    points: list[Point] | tuple[Point, ...],
    width: float,
    style: dict[str, str],
    tolerance: float,
) -> list[list[Point]]:
    """Non-overlapping rendering is obtained by unioning these stroke pieces."""
    points = [p for i, p in enumerate(points) if i == 0 or p != points[i - 1]]
    if len(points) < 2:
        return []
    distances = [0.0]
    for a, b in pairwise(points):
        distances.append(distances[-1] + math.dist(a, b))
    length = distances[-1]
    start, end = (float(style.get(key, "0")) for key in (START, END))
    # Dividing before adding avoids overflowing otherwise valid large lengths.
    scale = max(start, end, length)
    total = start / scale + end / scale
    if total > length / scale:
        start, end = length * (start / scale) / total, length * (end / scale) / total

    def radius(distance):
        return (
            width
            / 2
            * min(
                1.0,
                distance / start if start else 1.0,
                (length - distance) / end if end else 1.0,
            )
        )

    samples = [(points[0], 0.0)]
    for i, (a, b) in enumerate(pairwise(points)):
        low, high = distances[i : i + 2]
        for distance in sorted({start, length - end}):
            if low < distance < high:
                t = (distance - low) / (high - low)
                samples.append(
                    ((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])), distance)
                )
        samples.append((b, high))
    points = [p for p, _ in samples]
    radii = [max(0, radius(d)) for _, d in samples]
    directions = []
    result = []

    def offset(p, normal, r):
        return p[0] + normal[0] * r, p[1] + normal[1] * r

    for i, (a, b) in enumerate(pairwise(points)):
        size = math.dist(a, b)
        u = ((b[0] - a[0]) / size, (b[1] - a[1]) / size)
        directions.append(u)
        normal = (-u[1], u[0])
        result.append(
            [
                offset(a, normal, radii[i]),
                offset(b, normal, radii[i + 1]),
                offset(b, normal, -radii[i + 1]),
                offset(a, normal, -radii[i]),
            ]
        )

    def arc(center, r, angle, sweep):
        step = 2 * math.acos(max(-1, min(1, 1 - tolerance / max(r, tolerance))))
        count = max(1, math.ceil(abs(sweep) / max(step, 1e-6)))
        if count > 4096:
            from vectrify.document.model import DocumentError

            raise DocumentError("Taper exceeds outline subdivision budget")
        return [
            center,
            *[
                (
                    center[0] + r * math.cos(angle + sweep * i / count),
                    center[1] + r * math.sin(angle + sweep * i / count),
                )
                for i in range(count + 1)
            ],
        ]

    for i, (u, v) in enumerate(pairwise(directions), 1):
        cross = u[0] * v[1] - u[1] * v[0]
        dot = u[0] * v[0] + u[1] * v[1]
        if abs(cross) < 1e-12:
            if dot < 0 and style.get("stroke-linejoin", "miter") == "round":
                result.append(arc(points[i], radii[i], 0, 2 * math.pi))
            continue
        side = -1 if cross > 0 else 1
        n, m = (-u[1] * side, u[0] * side), (-v[1] * side, v[0] * side)
        p, r = points[i], radii[i]
        a, b = offset(p, n, r), offset(p, m, r)
        join = style.get("stroke-linejoin", "miter")
        if join == "round":
            result.append(arc(p, r, math.atan2(n[1], n[0]), math.atan2(cross, dot)))
        elif join == "miter" and 1 + dot > 1e-12:
            tip = (
                p[0] + r * (n[0] + m[0]) / (1 + dot),
                p[1] + r * (n[1] + m[1]) / (1 + dot),
            )
            if math.dist(p, tip) <= r * float(style.get("stroke-miterlimit", "4")):
                result.append([p, a, tip, b])
            else:
                result.append([p, a, b])
        else:
            result.append([p, a, b])
    cap = style.get("stroke-linecap", "butt")
    for i, u, tapered in ((0, directions[0], start), (-1, directions[-1], end)):
        if tapered:
            continue
        p, r = points[i], radii[i]
        outward = (-u[0], -u[1]) if i == 0 else u
        angle = math.atan2(outward[1], outward[0])
        if cap == "round":
            result.append(arc(p, r, angle - math.pi / 2, math.pi))
        elif cap == "square":
            normal = (-u[1], u[0])
            q = offset(p, outward, r)
            result.append(
                [
                    offset(p, normal, r),
                    offset(q, normal, r),
                    offset(q, normal, -r),
                    offset(p, normal, -r),
                ]
            )
    return result


def area(
    points: list[Point] | tuple[Point, ...],
    width: float,
    style: dict[str, str],
    tolerance: float,
) -> BaseGeometry:
    pieces = [Polygon(ring) for ring in polygons(points, width, style, tolerance)]
    return unary_union(pieces) if pieces else GeometryCollection()


def outline(geometry: Geometry, width: float, style: dict[str, str]) -> pathops.Path:
    """The entire stroke, including ordinary closed contours, as a Skia path."""
    import pathops

    from vectrify.document.hit_test import _flatten
    from vectrify.document.lines import open_path
    from vectrify.document.model import Geometry

    result = pathops.Path()
    tolerance = max(1e-5, min(0.02, width * 0.005))
    for sub in geometry.subpaths:
        if sub.closed:
            path = open_path(Geometry(geometry.id, (sub,)))
            path.stroke(
                width,
                pathops.LineCap.BUTT_CAP,
                {
                    "round": pathops.LineJoin.ROUND_JOIN,
                    "bevel": pathops.LineJoin.BEVEL_JOIN,
                }.get(
                    style.get("stroke-linejoin", "miter"), pathops.LineJoin.MITER_JOIN
                ),
                float(style.get("stroke-miterlimit", "4")),
            )
            path.convertConicsToQuads(tolerance)
            result = pathops.op(result, path, pathops.PathOp.UNION)
            continue
        points = [sub.nodes[0].endpoint]
        for node in sub.nodes[1:]:
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
        shape = area(points, width, style, tolerance)
        shapes = (
            [shape]
            if shape.geom_type == "Polygon"
            else list(getattr(shape, "geoms", []))
        )
        from shapely.geometry.polygon import orient

        path = pathops.Path()
        for polygon in shapes:
            if not isinstance(polygon, Polygon):
                continue
            polygon = orient(polygon, sign=1)
            for ring in (polygon.exterior, *polygon.interiors):
                path.moveTo(*ring.coords[0])
                for point in list(ring.coords)[1:-1]:
                    path.lineTo(*point)
                path.close()
        result = pathops.op(result, path, pathops.PathOp.UNION)
    return result


def geometry_for(document: Document, element: Element) -> Geometry:
    """The editable centerline of a path or SVG line."""
    if element.tag == "path":
        return document.geometry_for(element.id)
    from vectrify.document.svg import parse_path

    start = " ".join(element.get(key, "0") or "0" for key in ("x1", "y1"))
    end = " ".join(element.get(key, "0") or "0" for key in ("x2", "y2"))
    return parse_path(f"M{start} L{end}")
