"""Where each object of a snapshot paints, over clipped vector geometry.

Curves and round strokes are flattened with an explicit document-unit tolerance.
The index is independent of rendering resolution and does not include occlusion
by other objects: covered objects remain selectable, as in a vector editor.
"""

from __future__ import annotations

import math
import re
from itertools import pairwise

import numpy as np
from cairosvg.colors import color
from shapely import BufferCapStyle, BufferJoinStyle, clip_by_rect, prepare
from shapely.affinity import affine_transform
from shapely.geometry import GeometryCollection, LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize, unary_union

from vectrify.document.model import Document, DocumentError, Element
from vectrify.document.svg import PAINT

Matrix = tuple[float, float, float, float, float, float]
IDENTITY: Matrix = (1, 0, 0, 1, 0, 0)
Point2 = tuple[float, float]
Contour = tuple[tuple[Point2, ...], bool]


def multiply(a: Matrix, b: Matrix) -> Matrix:
    return (
        a[0] * b[0] + a[2] * b[1],
        a[1] * b[0] + a[3] * b[1],
        a[0] * b[2] + a[2] * b[3],
        a[1] * b[2] + a[3] * b[3],
        a[0] * b[4] + a[2] * b[5] + a[4],
        a[1] * b[4] + a[3] * b[5] + a[5],
    )


def transform(value: str | None) -> Matrix:
    result = IDENTITY
    for name, text in re.findall(r"(\w+)\s*\(([^()]*)\)", value or ""):
        args = [float(v) for v in text.replace(",", " ").split()]
        if name == "matrix":
            matrix = (args[0], args[1], args[2], args[3], args[4], args[5])
        elif name == "translate":
            matrix = (1, 0, 0, 1, args[0], args[1] if len(args) > 1 else 0)
        elif name == "scale":
            matrix = (args[0], 0, 0, args[-1], 0, 0)
        elif name == "rotate":
            angle = math.radians(args[0])
            c, s = math.cos(angle), math.sin(angle)
            matrix = (c, s, -s, c, 0, 0)
            if len(args) == 3:
                x, y = args[1:]
                matrix = multiply(
                    (1, 0, 0, 1, x, y), multiply(matrix, (1, 0, 0, 1, -x, -y))
                )
        else:
            skew = math.tan(math.radians(args[0]))
            matrix = (1, 0, skew, 1, 0, 0) if name == "skewX" else (1, skew, 0, 1, 0, 0)
        result = multiply(result, matrix)
    if not all(math.isfinite(v) for v in result):
        raise DocumentError("Transform exceeds finite coordinate range")
    return result


def mapped(shape: BaseGeometry, matrix: Matrix) -> BaseGeometry:
    a, b, c, d, e, f = matrix
    return affine_transform(shape, (a, c, b, d, e, f))


def _flatten(
    points: tuple[Point2, ...], tolerance: float, depth: int = 0
) -> list[Point2]:
    a, b, c, d = points
    # Distance to the finite chord also catches collinear overshoot/cusps.
    chord = LineString((a, d))
    if max(chord.distance(Point(b)), chord.distance(Point(c))) <= tolerance:
        return [d]
    if depth >= 24:
        raise DocumentError("Curve exceeds hit-testing subdivision budget")
    ab, bc, cd = [((p[0] + q[0]) / 2, (p[1] + q[1]) / 2) for p, q in pairwise(points)]
    abc = ((ab[0] + bc[0]) / 2, (ab[1] + bc[1]) / 2)
    bcd = ((bc[0] + cd[0]) / 2, (bc[1] + cd[1]) / 2)
    mid = ((abc[0] + bcd[0]) / 2, (abc[1] + bcd[1]) / 2)
    return _flatten((a, ab, abc, mid), tolerance, depth + 1) + _flatten(
        (mid, bcd, cd, d), tolerance, depth + 1
    )


def _quadrants(radius: float, tolerance: float) -> int:
    if radius <= tolerance:
        return 1
    angle = math.acos(max(-1, 1 - tolerance / radius))
    if angle == 0:
        raise DocumentError("Round geometry exceeds hit-testing precision")
    count = math.ceil(math.pi / (4 * angle))
    if count > 4096:
        raise DocumentError("Round geometry exceeds hit-testing subdivision budget")
    return max(1, count)


def _filled(contours: tuple[Contour, ...], rule: str) -> BaseGeometry:
    rings = [
        (*points, points[0]) if points[-1] != points[0] else points
        for points, _ in contours
        if len(points) >= 3
    ]
    if not rings:
        return GeometryCollection()
    if len(rings) == 1:
        polygon = Polygon(rings[0])
        if polygon.is_valid:
            return polygon
    # Polygonize noded edges, then classify each face with the original directed
    # contours. This retains nonzero winding, even-odd holes and self crossings.
    linework = unary_union([LineString(ring) for ring in rings])
    edges = np.array([(a, b) for ring in rings for a, b in pairwise(ring)])
    a, b = edges[:, 0], edges[:, 1]
    faces = []
    for face in polygonize(linework):
        p = face.representative_point()
        cross = (b[:, 0] - a[:, 0]) * (p.y - a[:, 1]) - (p.x - a[:, 0]) * (
            b[:, 1] - a[:, 1]
        )
        winding = int(
            np.count_nonzero((a[:, 1] <= p.y) & (b[:, 1] > p.y) & (cross > 0))
        )
        winding -= int(
            np.count_nonzero((a[:, 1] > p.y) & (b[:, 1] <= p.y) & (cross < 0))
        )
        if (winding % 2 != 0) if rule == "evenodd" else (winding != 0):
            faces.append(face)
    return unary_union(faces)


class HitIndex:
    """Build once for an immutable document; rebuild after document changes.

    Coordinates and tolerance use the root SVG user space (viewBox coordinates),
    not screen pixels. An object's painted area is its fill and stroke after
    transforms and clips; a group's is the union of its children's.
    """

    def __init__(self, document: Document, *, tolerance: float = 0.1):
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise DocumentError("Hit-testing tolerance must be finite and positive")
        document.validate()
        self.document = document
        self.tolerance = tolerance
        self._elements = {e.id: e for e in document.elements()}
        self._geometries = {g.id: g for g in document.geometries}
        self._parts: dict[int, tuple[BaseGeometry, tuple[BaseGeometry, ...]]] = {}
        _, areas = self._visit(document.root, IDENTITY, {}, record=True)
        del self._parts
        self._areas = {
            oid: area
            for oid, area in areas.items()
            if not area.is_empty and area.area > 0
        }

    def _collection(self, parts: list[BaseGeometry]) -> BaseGeometry:
        members = tuple(part for part in parts if not part.is_empty)
        if len(members) == 1:
            return members[0]
        shape = GeometryCollection(members)
        # Retain the originals: GEOS collection members are new Python wrappers,
        # which would defeat per-object clipping reuse between groups and leaves.
        self._parts[id(shape)] = (shape, members)
        return shape

    def painted_area(self, object_id: str) -> float:
        """Clipped painted area, with holes excluded and stroke included."""
        self.document.element(object_id)
        shape = self._areas.get(object_id)
        # Groups retain separate contributions for fast rectangle queries. Only
        # merge overlapping contributions when an actual area is requested.
        return 0.0 if shape is None else float(unary_union(shape).area)

    def wholly_inside(self, object_id: str, region: BaseGeometry) -> bool:
        """Whether the complete clipped painted object fits in a region."""
        area = self._areas.get(object_id)
        return area is not None and not area.is_empty and region.covers(area)

    def area(self, object_id: str) -> BaseGeometry | None:
        """Where *object_id* paints, or None if it paints nothing."""
        return self._areas.get(object_id)

    def bounds(
        self, object_ids: frozenset[str]
    ) -> tuple[float, float, float, float] | None:
        """The painted bounds of *object_ids* together, or None if none paint."""
        shapes = [self._areas[oid] for oid in object_ids if oid in self._areas]
        if not shapes:
            return None
        lefts, tops, rights, bottoms = zip(*(s.bounds for s in shapes), strict=True)
        return min(lefts), min(tops), max(rights), max(bottoms)

    def _contours(self, element: Element, tolerance: float) -> tuple[Contour, ...]:
        def number(name: str, default: float = 0) -> float:
            return float(element.get(name, str(default)) or default)

        tag = element.tag
        if tag == "path":
            assert element.geometry_id is not None  # Document.validate checked this.
            contours = []
            for subpath in self._geometries[element.geometry_id].subpaths:
                if len(subpath.nodes) < 2:
                    continue
                points = [subpath.nodes[0].endpoint]
                for node in subpath.nodes[1:]:
                    if node.command == "C":
                        values = node.values
                        points.extend(
                            _flatten(
                                (
                                    points[-1],
                                    (values[0], values[1]),
                                    (values[2], values[3]),
                                    node.endpoint,
                                ),
                                tolerance,
                            )
                        )
                    else:
                        points.append(node.endpoint)
                contours.append((tuple(points), subpath.closed))
            return tuple(contours)
        if tag == "line":
            return (
                (((number("x1"), number("y1")), (number("x2"), number("y2"))), False),
            )
        if tag in {"circle", "ellipse"}:
            rx, ry = (
                (number("r"), number("r"))
                if tag == "circle"
                else (number("rx"), number("ry"))
            )
            if rx == 0 or ry == 0:
                return ()
            count = _quadrants(max(rx, ry), tolerance) * 4
            return (
                (
                    tuple(
                        (
                            number("cx") + rx * math.cos(i * 2 * math.pi / count),
                            number("cy") + ry * math.sin(i * 2 * math.pi / count),
                        )
                        for i in range(count)
                    ),
                    True,
                ),
            )
        if tag == "rect":
            x, y, w, h = number("x"), number("y"), number("width"), number("height")
            if w == 0 or h == 0:
                return ()
            rx = min(number("rx", number("ry")), w / 2)
            ry = min(number("ry", number("rx")), h / 2)
            if rx == 0 or ry == 0:
                return ((((x, y), (x + w, y), (x + w, y + h), (x, y + h)), True),)
            count = _quadrants(max(rx, ry), tolerance)
            points = []
            for cx, cy, start in (
                (x + w - rx, y + ry, -math.pi / 2),
                (x + w - rx, y + h - ry, 0),
                (x + rx, y + h - ry, math.pi / 2),
                (x + rx, y + ry, math.pi),
            ):
                points.extend(
                    (
                        cx + rx * math.cos(start + i * math.pi / (2 * count)),
                        cy + ry * math.sin(start + i * math.pi / (2 * count)),
                    )
                    for i in range(count + 1)
                )
            return ((tuple(points), True),)
        return ()

    def _raw(
        self, element: Element, tolerance: float, *, own_transform: bool = False
    ) -> BaseGeometry:
        """Unclipped geometric bounds, excluding stroke and paint visibility."""
        matrix = transform(element.get("transform")) if own_transform else IDENTITY
        scale = math.hypot(*matrix[:4])
        tolerance /= max(scale, 1e-300)
        contours = self._contours(element, tolerance)
        parts: list[BaseGeometry] = [
            LineString(points) if len(points) > 1 else Point(points[0])
            for points, _ in contours
        ]
        if element.tag == "use":
            target = self._elements[(element.get("href") or "")[1:]]
            shape = self._raw(target, tolerance, own_transform=True)
            parts.append(
                mapped(
                    shape,
                    (
                        1,
                        0,
                        0,
                        1,
                        float(element.get("x", "0") or 0),
                        float(element.get("y", "0") or 0),
                    ),
                )
            )
        else:
            parts.extend(
                self._raw(c, tolerance, own_transform=True)
                for c in element.children
                if c.tag not in {"defs", "clipPath"}
            )
        return mapped(GeometryCollection(parts), matrix)

    def _visit(
        self,
        element: Element,
        parent: Matrix,
        inherited: dict[str, str],
        *,
        record: bool,
        clip: bool = False,
    ) -> tuple[BaseGeometry, dict[str, BaseGeometry]]:
        if element.tag == "defs" or (element.tag == "clipPath" and not clip):
            return GeometryCollection(), {}
        attrs = dict(element.attributes)
        style = inherited | {
            k: v for k, v in attrs.items() if k in PAINT and k != "opacity"
        }
        matrix = multiply(parent, transform(element.get("transform")))
        if not all(math.isfinite(v) for v in matrix):
            raise DocumentError("Transform exceeds finite coordinate range")
        if matrix[0] * matrix[3] - matrix[1] * matrix[2] == 0:
            return GeometryCollection(), {}
        if not clip and float(attrs.get("opacity", "1")) == 0:
            return GeometryCollection(), {}
        scale = math.hypot(*matrix[:4])
        tolerance = self.tolerance / max(scale, 1e-300)
        contours = self._contours(element, tolerance)
        parts: list[BaseGeometry] = []
        if clip or (
            style.get("fill", "black") != "none"
            and color(style.get("fill", "black"))[3] > 0
            and float(style.get("fill-opacity", "1")) > 0
        ):
            parts.append(
                _filled(
                    contours, style.get("clip-rule" if clip else "fill-rule", "nonzero")
                )
            )
        width = float(style.get("stroke-width", "1"))
        if (
            not clip
            and width > 0
            and style.get("stroke", "none") != "none"
            and color(style["stroke"])[3] > 0
            and float(style.get("stroke-opacity", "1")) > 0
        ):
            cap = {
                "butt": BufferCapStyle.flat,
                "round": BufferCapStyle.round,
                "square": BufferCapStyle.square,
            }[style.get("stroke-linecap", "butt")]
            join = {
                "miter": BufferJoinStyle.mitre,
                "round": BufferJoinStyle.round,
                "bevel": BufferJoinStyle.bevel,
            }[style.get("stroke-linejoin", "miter")]
            for points, closed in contours:
                if closed and points[-1] != points[0]:
                    points = (*points, points[0])
                line = LineString(points) if len(set(points)) > 1 else Point(points[0])
                parts.append(
                    line.buffer(
                        width / 2,
                        quad_segs=_quadrants(width / 2, tolerance),
                        cap_style=cap,
                        join_style=join,
                        mitre_limit=max(1, float(style.get("stroke-miterlimit", "4"))),
                    )
                )
        painted = mapped(unary_union(parts), matrix)
        areas: dict[str, BaseGeometry] = {}
        children = []
        if element.tag == "use":
            target = self._elements[(element.get("href") or "")[1:]]
            offset = (
                1,
                0,
                0,
                1,
                float(element.get("x", "0") or 0),
                float(element.get("y", "0") or 0),
            )
            instance, _ = self._visit(
                target, multiply(matrix, offset), style, record=False, clip=clip
            )
            children.append(instance)
        else:
            for child in element.children:
                child_shape, child_areas = self._visit(
                    child, matrix, style, record=record, clip=clip
                )
                children.append(child_shape)
                areas.update(child_areas)
        shape = self._collection([painted, *children])
        clipping = attrs.get("clip-path", "none")
        if clipping != "none":
            definition = self._elements[clipping[5:-1]]
            clip_matrix = matrix
            if definition.get("clipPathUnits") == "objectBoundingBox":
                raw = self._raw(element, tolerance)
                if raw.is_empty:
                    return GeometryCollection(), {}
                x, y, right, bottom = raw.bounds
                if right == x or bottom == y:
                    return GeometryCollection(), {}
                clip_matrix = multiply(matrix, (right - x, 0, 0, bottom - y, x, y))
            clipped, _ = self._visit(
                definition, clip_matrix, {}, record=False, clip=True
            )
            prepare(clipped)
            cache: dict[int, BaseGeometry] = {}

            def apply_clip(area: BaseGeometry) -> BaseGeometry:
                key = id(area)
                if key not in cache:
                    if key in self._parts:
                        result = self._collection(
                            [apply_clip(part) for part in self._parts[key][1]]
                        )
                    elif (
                        area.is_empty
                        or clipped.is_empty
                        or not clipped.intersects(area)
                    ):
                        result = GeometryCollection()
                    elif clipped.covers(area):
                        result = area
                    else:
                        # Limit the expensive boolean to nearby clip edges. A
                        # small object should not intersect the entire landscape
                        # boundary. Rectangle clipping is fast but can produce
                        # invalid polygons, so retain the exact full-clip fallback.
                        left, top, right, bottom = area.bounds
                        padding = self.tolerance
                        local_clip = clip_by_rect(
                            clipped,
                            left - padding,
                            top - padding,
                            right + padding,
                            bottom + padding,
                        )
                        result = area.intersection(
                            local_clip if local_clip.is_valid else clipped
                        )
                    cache[key] = result
                return cache[key]

            # Clip each contribution once; flattening the full scene into one
            # boolean union is unnecessary for containment/intersection queries.
            shape = self._collection(
                [apply_clip(painted), *(apply_clip(c) for c in children)]
            )
            areas = {oid: apply_clip(area) for oid, area in areas.items()}
        if record:
            areas[element.id] = shape
        return shape, areas
