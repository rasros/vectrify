"""Paths' contours by where they lie: which a region holds, and taking the part
of a path inside a region away from the rest.

A region is a polygon in root user space (document coordinates); a path's
nodes are in its own coordinates, so each is mapped through the path's
transforms first. Taking a part never moves a point: contours wholly inside
go as they are, keeping their point IDs, and with *cut* a stroked contour
crossing the region's edge comes apart where it crosses it, a filled shape
is split along the edge as the knife splits it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import pairwise

import numpy as np
import pathops
from shapely import STRtree
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep

from vectrify.document.hit_test import IDENTITY, Matrix, multiply, transform
from vectrify.document.join import curve_path, path_geometry
from vectrify.document.knife import _cut_open, _Line
from vectrify.document.model import (
    Document,
    DocumentError,
    Geometry,
    PathNode,
    Subpath,
    new_id,
)
from vectrify.document.topology import inverse_matrix, mapped_point, subdivide

Point2 = tuple[float, float]


def object_matrix(document: Document, object_id: str) -> Matrix:
    """What maps the coordinates of *object_id*'s geometry into root user
    space: its and its groups' transforms, and for an instance the offset and
    transform of what it shows."""
    matrix: Matrix = IDENTITY
    for ancestor in document.ancestry(object_id):
        matrix = multiply(matrix, transform(ancestor.get("transform")))
    element = document.element(object_id)
    while element.tag == "use":
        x = float(element.get("x", "0") or 0)
        y = float(element.get("y", "0") or 0)
        element = document.element((element.get("href") or "#")[1:])
        matrix = multiply(matrix, (1, 0, 0, 1, x, y))
        matrix = multiply(matrix, transform(element.get("transform")))
    return matrix


def region_polygon(region: Sequence) -> list[Point2]:
    """A region's corners: [x, y, w, h] as a rectangle, or [[x, y], ...]."""
    if len(region) == 4 and all(
        isinstance(v, int | float) and not isinstance(v, bool) for v in region
    ):
        x, y, w, h = (float(v) for v in region)
        if not all(map(math.isfinite, (x, y, w, h))) or w <= 0 or h <= 0:
            raise DocumentError("A region [x, y, w, h] needs a positive size")
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    points = []
    for point in region:
        if not isinstance(point, list | tuple) or len(point) != 2:
            raise DocumentError(
                "A region is [x, y, width, height] or a polygon [[x, y], ...]"
            )
        x, y = (float(v) for v in point)
        if not math.isfinite(x) or not math.isfinite(y):
            raise DocumentError("Region points must be finite")
        points.append((x, y))
    if len(points) < 3 or Polygon(points).area <= 0:
        raise DocumentError("A region polygon needs three or more corners and area")
    return points


def samples(subpath: Subpath, steps: int = 4) -> list[Point2]:
    """Points along a contour: its nodes and, on curves, between them."""
    nodes = subpath.nodes
    found = [nodes[0].endpoint]
    for before, node in pairwise(nodes):
        if node.command == "C":
            v = node.values
            curve = (before.endpoint, (v[0], v[1]), (v[2], v[3]), node.endpoint)
            for i in range(1, steps):
                found.append(_at(curve, i / steps))
        found.append(node.endpoint)
    if subpath.closed and found[-1] != found[0]:
        found.append(found[0])
    return found


def _at(curve: tuple[Point2, ...], t: float) -> Point2:
    return subdivide(curve, t)[0][-1]


def _middle(piece: Sequence[PathNode]) -> Point2:
    """The middle of a piece of contour: halfway along its middle segment."""
    index = max(1, len(piece) // 2)
    before, node = piece[index - 1], piece[index]
    points = (before.endpoint, *zip(node.values[::2], node.values[1::2], strict=True))
    return _at(points, 0.5)


@dataclass(frozen=True)
class Split:
    """A geometry's contours outside a region, and those inside it."""

    outside: tuple[Subpath, ...]
    inside: tuple[Subpath, ...]
    # Whether a contour was cut, giving its pieces new points.
    cut: bool = False


def split_geometry(
    geometry: Geometry,
    matrix: Matrix,
    polygon: Sequence[Point2],
    *,
    filled: bool,
    rule: str = "nonzero",
    cut: bool = True,
) -> Split:
    """Which of *geometry*'s contours lie inside *polygon* (root user space)
    and which outside, *matrix* mapping the geometry there.

    A contour is inside when all of it is. A filled shape's holes stay with
    the shape around them, so one moves only with its hole and the other way
    round. With *cut*, contours crossing the edge are cut along it: a
    stroke's into pieces, each going to the side it lies on; a filled
    shape whose outline (or a hole's) crosses the edge is split in two, both
    meeting along the edge. A shape merely around the region stays whole.
    """
    inverse = inverse_matrix(matrix)
    local = [mapped_point(p, inverse) for p in polygon]
    region = Polygon(local)
    if not region.is_valid:
        region = region.buffer(0)
    inside_test = prep(region)
    contours = geometry.subpaths

    def lies(subpath: Subpath) -> str:
        line = samples(subpath)
        shape = LineString(line) if len(set(line)) > 1 else Point(line[0])
        if inside_test.covers(shape):
            return "in"
        if not inside_test.intersects(shape):
            return "out"
        return "across"

    where = [lies(s) for s in contours]
    if filled:
        inside_ids: set[int] = set()
        crossing: set[int] = set()
        for family in _families(contours):
            states = {where[i] for i in family}
            if states == {"in"}:
                inside_ids.update(family)
            elif cut and "across" in states:
                crossing.update(family)
        outside = [s for i, s in enumerate(contours) if i not in inside_ids | crossing]
        inside = [s for i, s in enumerate(contours) if i in inside_ids]
        if not crossing:
            return Split(tuple(outside), tuple(inside))
        # Only the shapes whose outline crosses the edge are split along it.
        pieces = _split_filled(
            Geometry(geometry.id, tuple(contours[i] for i in sorted(crossing))),
            rule,
            local,
        )
        return Split((*outside, *pieces.outside), (*inside, *pieces.inside), cut=True)
    outside: list[Subpath] = []
    inside: list[Subpath] = []
    was_cut = False
    for subpath, state in zip(contours, where, strict=True):
        if state == "in":
            inside.append(subpath)
            continue
        if state == "out" or not cut:
            outside.append(subpath)
            continue
        pieces = _cut_stroke(subpath, local)
        if len(pieces) == 1:
            # It touches the edge without crossing it.
            middle = _middle(subpath.nodes) if len(subpath.nodes) > 1 else None
            if middle is not None and inside_test.contains(Point(middle)):
                inside.append(subpath)
            else:
                outside.append(subpath)
            continue
        was_cut = True
        for piece in pieces:
            target = inside if inside_test.contains(Point(_middle(piece))) else outside
            target.append(Subpath(new_id("subpath"), tuple(piece), False))
    return Split(tuple(outside), tuple(inside), was_cut)


def _cut_stroke(subpath: Subpath, polygon: Sequence[Point2]) -> list[list[PathNode]]:
    """A contour's pieces between the places the polygon's edges cross it."""
    nodes = list(subpath.nodes)
    if subpath.closed:
        segments = list(nodes[1:])
        if len(nodes) < 3 or segments[-1].endpoint != nodes[0].endpoint:
            segments.append(PathNode(new_id("node"), "L", nodes[0].endpoint))
        nodes = [nodes[0], *segments]
    scale = max(
        1.0,
        *(abs(v) for n in nodes for v in n.values),
        *(abs(v) for p in polygon for v in p),
    )
    pieces = [nodes]
    for a, b in pairwise([*polygon, polygon[0]]):
        if math.dist(a, b) == 0:
            continue
        line = _Line(a, b, 1e-9 * scale)
        pieces = [part for piece in pieces for part in _cut_open(piece, line)]
    pieces = [p for p in pieces if len(p) > 1]
    if subpath.closed and len(pieces) > 1:
        # The first and last pieces meet again at the contour's start.
        first, last = pieces[0], pieces.pop()
        pieces[0] = [*last, *first[1:]]
    return pieces or [list(subpath.nodes)]


def _families(contours: Sequence[Subpath]) -> list[list[int]]:
    """Contours grouped with the ones they lie inside or around: a filled
    shape and its holes, and the islands inside those."""
    present: list[int] = []
    shapes: list[BaseGeometry] = []
    for index, subpath in enumerate(contours):
        line = samples(subpath)
        if len(set(line)) < 3:
            continue
        polygon: BaseGeometry = Polygon(line)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        present.append(index)
        shapes.append(polygon)
    parent = list(range(len(contours)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if present:
        tree = STRtree(shapes)
        probes = [Point(s.nodes[0].endpoint) for s in contours]
        found = tree.query(probes, predicate="within")
        for i, j in zip(found[0], found[1], strict=True):
            j = present[int(j)]
            if int(i) != j:
                parent[root(int(i))] = root(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(contours)):
        groups.setdefault(root(i), []).append(i)
    return list(groups.values())


def _split_filled(geometry: Geometry, rule: str, polygon: Sequence[Point2]) -> Split:
    """A filled region's part inside a polygon and the rest, both in its
    own coordinates, meeting along the polygon's edge."""
    if any(n.pinned for s in geometry.subpaths for n in s.nodes):
        raise DocumentError("Unpin the shape's points before cutting it")
    path = curve_path(geometry, rule)
    region = pathops.Path()
    region.moveTo(*polygon[0])
    for point in polygon[1:]:
        region.lineTo(*point)
    region.close()
    try:
        inside = pathops.op(path, region, pathops.PathOp.INTERSECTION)
        outside = pathops.op(path, region, pathops.PathOp.DIFFERENCE)
    except pathops.PathOpsError as exc:
        raise DocumentError("Could not cut this shape along the region") from exc
    # Skia works in float32; restore the original coordinates it passes through.
    exact = {}
    for subpath in geometry.subpaths:
        for node in subpath.nodes:
            for x, y in zip(node.values[::2], node.values[1::2], strict=True):
                exact[float(np.float32(x)), float(np.float32(y))] = (x, y)

    def restored(path: pathops.Path) -> tuple[Subpath, ...]:
        if not list(path):
            return ()
        result = path_geometry(path)
        return tuple(
            replace(
                s,
                nodes=tuple(
                    replace(
                        n,
                        values=tuple(
                            v
                            for x, y in zip(n.values[::2], n.values[1::2], strict=True)
                            for v in exact.get((x, y), (x, y))
                        ),
                    )
                    for n in s.nodes
                ),
            )
            for s in result.subpaths
        )

    return Split(restored(outside), restored(inside), True)
