"""Cut geometry along a straight segment: filled regions into pieces that
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
    """The pieces of a cut."""

    first: Geometry
    second: Geometry
    additional: tuple[Geometry, ...] = ()

    @property
    def pieces(self) -> tuple[Geometry, ...]:
        return (self.first, self.second, *self.additional)


def preserve_cut_pins(geometry: Geometry, cut: Cut) -> Cut | None:
    """Carry pinned endpoint identities onto the pieces, or refuse their loss.

    Filled booleans rebuild nodes and can merge straight or curved segments,
    so a surviving pin may need to be inserted back into a boundary edge.
    Stroke cuts already keep endpoint identities and need only validation.
    """
    from vectrify.document.contact import projection

    pieces = list(cut.pieces)
    pins = [n for s in geometry.subpaths for n in s.nodes if n.pinned]
    if not pins:
        return cut
    tolerance = 1e-7 * max(1.0, *(abs(v) for n in pins for v in n.endpoint))
    claimed: set[str] = set()
    for pin in pins:
        existing = next(
            (n for g in pieces for s in g.subpaths for n in s.nodes if n.id == pin.id),
            None,
        )
        if existing is not None:
            if existing.endpoint != pin.endpoint or not existing.pinned:
                return None
            claimed.add(pin.id)
            continue
        match = next(
            (
                (i, n)
                for i, g in enumerate(pieces)
                for s in g.subpaths
                for n in s.nodes
                if n.endpoint == pin.endpoint and n.id not in claimed
            ),
            None,
        )
        if match is not None:
            i, node = match
            restored = replace(
                node,
                id=pin.id,
                pinned=True,
                handles_aligned=pin.handles_aligned,
                feature=pin.feature,
            )
            pieces[i] = replace(
                pieces[i],
                subpaths=tuple(
                    replace(
                        s,
                        nodes=tuple(
                            restored if n.id == node.id else n for n in s.nodes
                        ),
                    )
                    for s in pieces[i].subpaths
                ),
            )
            claimed.add(pin.id)
            continue
        found = False
        for i, piece in enumerate(pieces):
            for subpath in piece.subpaths:
                nodes = list(subpath.nodes)
                segments = [*nodes[1:]]
                if subpath.closed:
                    segments.append(PathNode(new_id("node"), "L", nodes[0].endpoint))
                for j, node in enumerate(segments, 1):
                    points = (
                        nodes[j - 1].endpoint,
                        *zip(node.values[::2], node.values[1::2], strict=True),
                    )
                    t, distance = projection(pin.endpoint, points)
                    if distance > tolerance or not 0 < t < 1:
                        continue
                    head, tail = subdivide(points, t)
                    inserted = replace(
                        pin,
                        command=node.command,
                        values=tuple(v for p in head[1:-1] for v in p) + pin.endpoint,
                    )
                    nodes.insert(j, inserted)
                    if j + 1 < len(nodes):
                        nodes[j + 1] = replace(
                            node, values=tuple(v for p in tail[1:] for v in p)
                        )
                    pieces[i] = replace(
                        piece,
                        subpaths=tuple(
                            replace(s, nodes=tuple(nodes)) if s.id == subpath.id else s
                            for s in piece.subpaths
                        ),
                    )
                    found = True
                    break
                if found:
                    break
            if found:
                break
        if not found:
            return None
        claimed.add(pin.id)
    return Cut(pieces[0], pieces[1], tuple(pieces[2:]))


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


def _separate_touching_rings(geometry: Geometry) -> Geometry:
    """Separate boundary loops that only meet at a vertex into their own rings."""
    from shapely import STRtree
    from shapely.geometry import Point as ShapePoint
    from shapely.geometry import box

    rings = []
    for ring in _rings(geometry, {}):
        points = list({point for _, point in ring})
        tree = STRtree([ShapePoint(point) for point in points])
        noded = [ring[0]]
        for segment in ring[1:]:
            start, end = noded[-1][1], segment[1]
            dx, dy = end[0] - start[0], end[1] - start[1]
            length_squared = dx * dx + dy * dy
            if not segment[0] and length_squared:
                # Skia can merge a straight edge through a touching vertex.
                # Restore that vertex before extracting the separate loops.
                stops = []
                for candidate in tree.query(
                    box(
                        *(
                            min(start[0], end[0]),
                            min(start[1], end[1]),
                            max(start[0], end[0]),
                            max(start[1], end[1]),
                        )
                    )
                ):
                    point = points[candidate]
                    px, py = point[0] - start[0], point[1] - start[1]
                    t = (px * dx + py * dy) / length_squared
                    if 0 < t < 1 and abs(px * dy - py * dx) <= 1e-10 * length_squared:
                        stops.append((t, point))
                noded.extend(((), point) for _, point in sorted(stops))
            noded.append(segment)
        walk = [ring[0]]
        positions = {ring[0][1]: 0}
        for segment in noded[1:]:
            point = segment[1]
            if point not in positions:
                positions[point] = len(walk)
                walk.append(segment)
                continue
            index = positions[point]
            loop = [((), point), *walk[index + 1 :], segment]
            if len(loop) > 2 or segment[0]:
                rings.append(loop)
            for _, endpoint in walk[index + 1 :]:
                del positions[endpoint]
            walk = walk[: index + 1]
    return (
        geometry
        if len(rings) == len(geometry.subpaths)
        else _geometry(rings, geometry.id)
    )


def _finite_pieces(sides: list[Geometry], line: _Line) -> Cut | None:
    """Reconnect half-plane pieces across seams outside the knife segment.

    Opposite seam edges have identical endpoints. Removing both joins their
    boundary walks, without rounding or flattening any of the retained curves.
    Fill components track holes and identify cuts that only open a slit from
    an outside boundary to a hole without separating any material.
    """
    sides = [_separate_touching_rings(g) for g in sides]
    edges: list[tuple[Point, Segment]] = []
    following: list[int] = []
    owners: list[str] = []
    seams: dict[tuple[Point, Point], int] = {}
    pairs: list[tuple[int, int, bool]] = []
    for geometry in sides:
        for subpath, ring in zip(geometry.subpaths, _rings(geometry, {}), strict=True):
            offset = len(edges)
            for (_, start), segment in pairwise(ring):
                index = len(edges)
                edges.append((start, segment))
                following.append(index + 1)
                owners.append(subpath.id)
                if line.holds(start, segment):
                    end = segment[1]
                    twin = seams.pop((end, start), None)
                    if twin is None:
                        seams[start, end] = index
                    else:
                        a, b = sorted((line.along(start), line.along(end)))
                        inside = (
                            a >= -line.tolerance and b <= line.length + line.tolerance
                        )
                        pairs.append((twin, index, inside))
            following[-1] = offset
    if not any(inside for _, _, inside in pairs):
        return None
    if all(inside for _, _, inside in pairs):
        return Cut(sides[0], sides[1])

    cells = {}
    parents: list[int] = []
    for geometry in sides:
        contours = [
            (s, curve_path(replace(geometry, subpaths=(s,)))) for s in geometry.subpaths
        ]
        # PathOps normalizes exterior contours counterclockwise and holes
        # clockwise. Each exterior is a material component, including islands
        # inside holes; attach each hole only to its nearest containing exterior.
        exteriors = [(s, path) for s, path in contours if not path.clockwise]
        for subpath, _ in exteriors:
            cell = len(parents)
            parents.append(cell)
            cells[subpath.id] = cell
        for subpath, path in contours:
            if not path.clockwise:
                continue
            containers = [
                (outer, boundary)
                for outer, boundary in exteriors
                if boundary.contains(subpath.nodes[0].endpoint)
            ]
            if not containers:
                raise DocumentError("Could not match this path's hole contours")
            outer, _ = min(containers, key=lambda item: item[1].area)
            cells[subpath.id] = cells[outer.id]

    def root(cell: int) -> int:
        while parents[cell] != cell:
            parents[cell] = parents[parents[cell]]
            cell = parents[cell]
        return cell

    # The material remains connected everywhere the gesture did not cut it.
    for a, b, inside in pairs:
        if not inside:
            parents[root(cells[owners[a]])] = root(cells[owners[b]])
    removed: dict[int, int] = {}
    active: set[int] = set()
    for a, b, inside in pairs:
        if not inside or root(cells[owners[a]]) == root(cells[owners[b]]):
            removed[a], removed[b] = b, a
        else:
            active.update((a, b))
    if not active:
        return None

    rings_by_cell: dict[int, list[list[Segment]]] = {}
    cut_cells: set[int] = set()
    visited = set(removed)
    for first, (start, _) in enumerate(edges):
        if first in visited:
            continue
        cell = root(cells[owners[first]])
        ring: list[Segment] = [((), start)]
        current = first
        while current not in visited:
            visited.add(current)
            ring.append(edges[current][1])
            if current in active:
                cut_cells.add(cell)
            current = following[current]
            while current in removed:
                current = following[removed[current]]
        rings_by_cell.setdefault(cell, []).append(ring)
    if len(cut_cells) < 2:
        return None
    pieces = [
        _geometry(rings, sides[0].id)
        for cell, rings in rings_by_cell.items()
        if cell in cut_cells
    ]
    pieces.sort(key=lambda g: abs(curve_path(g).area), reverse=True)
    untouched = [
        subpath
        for cell, rings in rings_by_cell.items()
        if cell not in cut_cells
        for subpath in _geometry(rings, sides[0].id).subpaths
    ]
    pieces[0] = replace(pieces[0], subpaths=(*pieces[0].subpaths, *untouched))
    for i in range(1, len(pieces)):
        pieces[i] = replace(pieces[i], id=sides[1].id if i == 1 else new_id("geometry"))
    return Cut(pieces[0], pieces[1], tuple(pieces[2:]))


def cut_geometry(
    geometry: Geometry,
    rule: str,
    start: Point,
    end: Point,
    first_id: str,
    second_id: str,
) -> Cut | None:
    """Split a filled region along the segment from *start* to *end*.

    Both points lie outside the fill and the segment enters and leaves it.
    Half-plane booleans provide curved boundaries; splits beyond the gesture
    are reconnected. Curves stay curves, and pieces share exact seam nodes.
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
                broken.append(segment)
            snapped.append(broken)
        geometries.append(_geometry(snapped, geometry_id))
    return _finite_pieces(geometries, line)


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
