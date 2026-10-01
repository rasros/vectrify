"""Cut geometry along a straight line: filled regions into two pieces that
meet exactly, stroked lines apart where the line crosses them."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from itertools import pairwise

import numpy as np
import pathops

from vectrify.document.join import curve_path, path_geometry
from vectrify.document.model import (
    DocumentError,
    Geometry,
    PathNode,
    Subpath,
    new_id,
)
from vectrify.document.topology import subdivide

Point = tuple[float, float]
# A segment's control points (none for a line) and its end point.
Segment = tuple[tuple[Point, ...], Point]


@dataclass(frozen=True)
class Cut:
    """The two sides of a cut."""

    first: Geometry
    second: Geometry


class _Line:
    def __init__(self, start: Point, end: Point, tolerance: float):
        self.origin = start
        length = math.dist(start, end)
        self.length = length
        self.direction = ((end[0] - start[0]) / length, (end[1] - start[1]) / length)
        self.tolerance = tolerance

    def along(self, point: Point) -> float:
        (x, y), (ox, oy), (ux, uy) = point, self.origin, self.direction
        return (x - ox) * ux + (y - oy) * uy

    def off(self, point: Point) -> float:
        (x, y), (ox, oy), (ux, uy) = point, self.origin, self.direction
        return (y - oy) * ux - (x - ox) * uy

    def holds(self, start: Point, segment: Segment) -> bool:
        """Whether a segment is a straight run along the line."""
        controls, end = segment
        return (
            not controls
            and abs(self.off(start)) <= self.tolerance
            and abs(self.off(end)) <= self.tolerance
            and math.dist(start, end) > self.tolerance
        )


def _half_plane(line: _Line, reach: float, side: float) -> pathops.Path:
    (ox, oy), (ux, uy) = line.origin, line.direction
    nx, ny = -uy * side, ux * side
    path = pathops.Path()
    corners = (
        (ox - ux * reach, oy - uy * reach),
        (ox + ux * reach, oy + uy * reach),
        (ox + ux * reach + nx * reach, oy + uy * reach + ny * reach),
        (ox - ux * reach + nx * reach, oy - uy * reach + ny * reach),
    )
    path.moveTo(*corners[0])
    for corner in corners[1:]:
        path.lineTo(*corner)
    path.close()
    return path


def _rings(geometry: Geometry, exact: dict[Point, Point]) -> list[list[Segment]]:
    """Closed rings as segments ending back at their start, with the float32
    rounding of untouched original coordinates undone."""
    rings = []
    for subpath in geometry.subpaths:
        points = [
            exact.get((x, y), (x, y))
            for node in subpath.nodes
            for x, y in zip(node.values[::2], node.values[1::2], strict=True)
        ]
        start, rest = points[0], points[1:]
        ring = []
        for node in subpath.nodes[1:]:
            count = len(node.values) // 2
            ring.append((tuple(rest[: count - 1]), rest[count - 1]))
            rest = rest[count:]
        if not ring or ring[-1][1] != start:
            ring.append(((), start))
        rings.append([((), start), *ring])
    return rings


def _crossings(line: _Line, start: Point, segment: Segment) -> list[float]:
    """Parameters strictly inside a segment where it crosses the line."""
    distances = [line.off(p) for p in (start, *segment[0], segment[1])]
    if len(distances) == 2:
        a, b = distances
        return [a / (a - b)] if a * b < 0 else []
    # The distance along a cubic is itself a cubic in Bernstein form.
    d0, d1, d2, d3 = distances
    coefficients = (
        -d0 + 3 * d1 - 3 * d2 + d3,
        3 * d0 - 6 * d1 + 3 * d2,
        -3 * d0 + 3 * d1,
        d0,
    )
    found = []
    for root in np.roots(np.trim_zeros(coefficients, "f") or [0.0]):
        if abs(root.imag) > 1e-9:
            continue
        t = float(root.real)
        for _ in range(3):
            value = ((coefficients[0] * t + coefficients[1]) * t + coefficients[2]) * t
            slope = (3 * coefficients[0] * t + 2 * coefficients[1]) * t
            slope += coefficients[2]
            if slope:
                t -= (value + coefficients[3]) / slope
        if 1e-9 < t < 1 - 1e-9:
            found.append(t)
    return sorted(found)


def _split_at_line(rings: list[list[Segment]], line: _Line) -> list[list[Segment]]:
    """Put nodes where the line crosses, in double precision, so both sides
    of the cut start from the same seam points."""
    result = []
    for ring in rings:
        split = [ring[0]]
        for segment in ring[1:]:
            start, previous = split[-1][1], 0.0
            points = (start, *segment[0], segment[1])
            if not segment[0]:
                # An axis-aligned edge keeps its exact coordinate on that axis.
                for t in _crossings(line, start, segment):
                    (ax, ay), (bx, by) = start, segment[1]
                    along = line.along((ax + (bx - ax) * t, ay + (by - ay) * t))
                    ox, oy = line.origin
                    ux, uy = line.direction
                    point = (
                        ax if ax == bx else ox + ux * along,
                        ay if ay == by else oy + uy * along,
                    )
                    split.append(((), point))
                split.append(segment)
                continue
            for t in _crossings(line, start, segment):
                if t - previous < 1e-9:
                    continue
                head, points = subdivide(points, (t - previous) / (1 - previous))
                split.append((head[1:-1], head[-1]))
                previous = t
            split.append((points[1:-1], points[-1]))
        result.append(split)
    return result


def _geometry(rings: list[list[Segment]], geometry_id: str) -> Geometry:
    subpaths = []
    for ring in rings:
        (_, start), segments = ring[0], ring[1:]
        # The last line back to the start is the implicit closing edge.
        if not segments[-1][0]:
            segments = segments[:-1]
        nodes = [PathNode(new_id("node"), "M", start)]
        nodes.extend(
            PathNode(
                new_id("node"),
                "C" if controls else "L",
                tuple(v for point in (*controls, end) for v in point),
            )
            for controls, end in segments
        )
        subpaths.append(Subpath(new_id("subpath"), tuple(nodes), closed=True))
    return Geometry(geometry_id, tuple(subpaths))


def cut_geometry(
    geometry: Geometry,
    rule: str,
    start: Point,
    end: Point,
    first_id: str,
    second_id: str,
) -> Cut | None:
    """Split a filled region along the line through *start* and *end*.

    The whole line cuts the region, but only if the segment between the two
    points enters and leaves it: both points lie outside the fill and the
    segment spans a piece of the cut. Each side of the line becomes one
    geometry, compound where the line leaves several parts on that side.
    Curves stay curves, and both sides share the same seam nodes.
    """
    if math.dist(start, end) == 0:
        raise DocumentError("Drag a longer knife line")
    path = curve_path(geometry, rule)
    left, top, right, bottom = path.bounds
    scale = max(1.0, *(abs(v) for v in (left, top, right, bottom, *start, *end)))
    line = _Line(start, end, 1e-5 * scale)
    if path.contains(start) or path.contains(end):
        return None
    geometry = _geometry(_split_at_line(_rings(geometry, {}), line), geometry.id)
    path = curve_path(geometry, rule)
    reach = 4 * (scale + math.dist(start, end))
    try:
        sides = [
            pathops.op(path, _half_plane(line, reach, s), pathops.PathOp.INTERSECTION)
            for s in (1, -1)
        ]
    except pathops.PathOpsError as exc:
        raise DocumentError("Could not cut this path's contours") from exc
    area = abs(path.area)
    if any(not list(side) or abs(side.area) <= 1e-9 * area for side in sides):
        return None
    # Skia works in float32; restore the original coordinates it passes through.
    exact = {}
    for subpath in geometry.subpaths:
        for node in subpath.nodes:
            for x, y in zip(node.values[::2], node.values[1::2], strict=True):
                exact[float(np.float32(x)), float(np.float32(y))] = (x, y)
    pieces = [_rings(path_geometry(side), exact) for side in sides]
    # Skia recomputes where the line meets the contours, and both sides meet it
    # at the same points only up to rounding: settle each on one coordinate,
    # preferring the exact crossings, and break seam lines at all of them.
    stops = [
        node.endpoint
        for subpath in geometry.subpaths
        for node in subpath.nodes
        if abs(line.off(node.endpoint)) <= line.tolerance
    ]
    for rings in pieces:
        for ring in rings:
            for (_, a), segment in pairwise(ring):
                if line.holds(a, segment):
                    for point in (a, segment[1]):
                        if all(math.dist(point, s) > line.tolerance for s in stops):
                            stops.append(point)

    def snap(point: Point) -> Point:
        return next((s for s in stops if math.dist(point, s) <= line.tolerance), point)

    crossed = False
    geometries = []
    for rings, geometry_id in zip(pieces, (first_id, second_id), strict=True):
        snapped = []
        for ring in rings:
            ring = [(controls, snap(point)) for controls, point in ring]
            broken = [ring[0]]
            for segment in ring[1:]:
                a = broken[-1][1]
                if line.holds(a, segment):
                    s0, s1 = line.along(a), line.along(segment[1])
                    inside = sorted(
                        (
                            stop
                            for stop in stops
                            if abs(line.off(stop)) <= line.tolerance
                            and min(s0, s1) + line.tolerance
                            < line.along(stop)
                            < max(s0, s1) - line.tolerance
                        ),
                        key=line.along,
                        reverse=s1 < s0,
                    )
                    broken.extend(((), stop) for stop in inside)
                    middle = (s0 + s1) / 2
                    crossed |= 0 <= middle <= line.length
                broken.append(segment)
            snapped.append(broken)
        geometries.append(_geometry(snapped, geometry_id))
    if not crossed:
        return None
    return Cut(*geometries)


def cut_strokes(
    geometry: Geometry, start: Point, end: Point, second_id: str
) -> Cut | None:
    """Cut a stroked path's contours where the segment from *start* to *end*
    crosses them.

    Each crossed contour comes apart at the crossings, a closed one opening
    up, and both pieces end on their own copy of the crossing point. The
    pieces on the side of the line where less of the cut contours lies go to
    a second geometry, so a loop cut across comes away whole; the rest, and
    every contour the segment misses, stay. Untouched points keep their IDs.
    """
    if math.dist(start, end) == 0:
        raise DocumentError("Drag a longer knife line")
    scale = max(
        1.0,
        *(abs(v) for s in geometry.subpaths for n in s.nodes for v in n.values),
        *map(abs, (*start, *end)),
    )
    line = _Line(start, end, 1e-9 * scale)
    kept: list[Subpath] = []
    pieces: list[tuple[list[PathNode], float]] = []
    for subpath in geometry.subpaths:
        nodes = list(subpath.nodes)
        if subpath.closed:
            # Open the contour at its start; its first and last pieces meet
            # there again.
            segments = list(nodes[1:])
            if len(nodes) < 3 or segments[-1].endpoint != nodes[0].endpoint:
                segments.append(PathNode(new_id("node"), "L", nodes[0].endpoint))
            nodes = [nodes[0], *segments]
        split = _cut_open(nodes, line)
        if len(split) == 1:
            kept.append(subpath)
            continue
        if subpath.closed:
            first, last = split[0], split.pop()
            split[0] = [*last, *first[1:]]
        for piece in split:
            pieces.append((piece, _side(piece, line)))
    if not pieces:
        return None
    lengths = {1.0: 0.0, -1.0: 0.0}
    for piece, side in pieces:
        lengths[side] += sum(
            math.dist(a.endpoint, b.endpoint) for a, b in pairwise(piece)
        )
    away = 1.0 if lengths[1.0] < lengths[-1.0] else -1.0

    def contours(side: float) -> list[Subpath]:
        return [
            Subpath(new_id("subpath"), tuple(piece), False)
            for piece, s in pieces
            if s == side
        ]

    second = contours(away)
    first = [*kept, *contours(-away)]
    return Cut(
        replace(geometry, subpaths=tuple(first)), Geometry(second_id, tuple(second))
    )


def _cut_open(nodes: list[PathNode], line: _Line) -> list[list[PathNode]]:
    """An open contour's pieces between the places the knife segment
    crosses it."""
    pieces = [[nodes[0]]]
    for node in nodes[1:]:
        start = pieces[-1][-1].endpoint
        controls = tuple(
            (node.values[i], node.values[i + 1])
            for i in range(0, len(node.values) - 2, 2)
        )
        points = (start, *controls, node.endpoint)
        previous = 0.0
        for t in _crossings(line, start, (controls, node.endpoint)):
            head, tail = subdivide(points, (t - previous) / (1 - previous))
            if not 0 <= line.along(head[-1]) <= line.length:
                continue
            values = tuple(v for p in head[1:] for v in p)
            pieces[-1].append(PathNode(new_id("node"), node.command, values))
            pieces.append([PathNode(new_id("node"), "M", head[-1])])
            points, previous = tail, t
        values = tuple(v for p in points[1:] for v in p)
        pieces[-1].append(replace(node, values=values))
    return pieces


def _side(piece: list[PathNode], line: _Line) -> float:
    """Which side of the line a piece lies on, next to its cut end."""
    if abs(line.off(piece[0].endpoint)) <= line.tolerance:
        before, node = piece[0], piece[1]
    else:
        before, node = piece[-2], piece[-1]
    points = (before.endpoint, *zip(node.values[::2], node.values[1::2], strict=True))
    middle = subdivide(points, 0.5)[0][-1]
    return 1.0 if line.off(middle) > 0 else -1.0
