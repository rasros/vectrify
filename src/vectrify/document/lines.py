"""Breaking, joining and converting drawn lines.

A traced line is an open contour, stroked. These edits cut contours at their
points, take out the segment between two points, join the open ends that
continue each other, and turn a stroke into its filled outline. Points keep
their IDs where they survive; a point a cut doubles gets a new copy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from itertools import pairwise

import pathops

from vectrify.document.join import path_geometry
from vectrify.document.model import (
    DocumentError,
    Geometry,
    PathNode,
    Subpath,
    new_id,
)

Point = tuple[float, float]
# The most a joined line turns where its ends meet, in degrees, unless the
# two ends are the only ones at one spot: then it is a corner of one line.
TURN = 60.0


def ring(subpath: Subpath) -> tuple[list[PathNode], dict[str, set[str]]]:
    """A closed contour's segments in order, each ending at its point.

    The closing line is a segment too, holding the start point's ID. A
    contour ending on its moveto shows one point for two nodes: the last one
    stands for both, and the moveto's ID maps to it.
    """
    nodes = subpath.nodes
    start = nodes[0]
    segments = list(nodes[1:])
    if len(nodes) > 2 and segments[-1].endpoint == start.endpoint:
        last = segments[-1]
        segments[-1] = replace(last, pinned=last.pinned or start.pinned)
        return segments, {start.id: {last.id}}
    segments.append(PathNode(start.id, "L", start.endpoint, start.pinned))
    return segments, {}


def opened(segments: list[PathNode], index: int, start_id: str) -> list[PathNode]:
    """The ring *segments* as an open contour from the point of segment
    *index* round to it again; its start is a new node *start_id*."""
    at = segments[index]
    move = PathNode(start_id, "M", at.endpoint, at.pinned)
    return [move, *segments[index + 1 :], *segments[: index + 1]]


def reversed_nodes(nodes: list[PathNode] | tuple[PathNode, ...]) -> list[PathNode]:
    """An open contour traced the other way, each point keeping its ID."""
    result = [PathNode(nodes[-1].id, "M", nodes[-1].endpoint, nodes[-1].pinned)]
    for i in range(len(nodes) - 1, 0, -1):
        node, before = nodes[i], nodes[i - 1]
        if node.command == "C":
            v = node.values
            values = (*v[2:4], *v[0:2], *before.endpoint)
        else:
            values = before.endpoint
        result.append(PathNode(before.id, node.command, values, before.pinned))
    return result


def _locate(geometry: Geometry, node_id: str) -> tuple[int, int]:
    for s, subpath in enumerate(geometry.subpaths):
        for n, node in enumerate(subpath.nodes):
            if node.id == node_id:
                return s, n
    raise DocumentError(f"Unknown path node: {node_id}")


def _with(geometry: Geometry, index: int, pieces: list[list[PathNode]]) -> Geometry:
    """*geometry* with contour *index* replaced by the open *pieces*, those
    with a segment; the first keeps the contour's ID."""
    kept = [p for p in pieces if len(p) > 1]
    old = geometry.subpaths[index]
    contours = [
        Subpath(old.id if i == 0 else new_id("subpath"), tuple(nodes), False)
        for i, nodes in enumerate(kept)
    ]
    return replace(
        geometry,
        subpaths=(
            *geometry.subpaths[:index],
            *contours,
            *geometry.subpaths[index + 1 :],
        ),
    )


def break_at(geometry: Geometry, node_id: str) -> tuple[Geometry, dict[str, set[str]]]:
    """Cut the contour at a point: an open one becomes two contours, each
    ending on its own copy of the point, and a closed one opens there.

    Returns the geometry and the point's IDs, old to new. An open
    contour's end is already a free end and is left alone.
    """
    s, n = _locate(geometry, node_id)
    subpath = geometry.subpaths[s]
    nodes = list(subpath.nodes)
    copy = new_id("node")
    if subpath.closed:
        segments, merged = ring(subpath)
        point = next(iter(merged[node_id])) if node_id in merged else node_id
        index = next(i for i, node in enumerate(segments) if node.id == point)
        updated = _with(geometry, s, [opened(segments, index, copy)])
        return updated, {**merged, point: {point, copy}}
    if n in {0, len(nodes) - 1}:
        return geometry, {}
    at = nodes[n]
    second = [PathNode(copy, "M", at.endpoint, at.pinned), *nodes[n + 1 :]]
    return _with(geometry, s, [nodes[: n + 1], second]), {node_id: {node_id, copy}}


def delete_segment(
    geometry: Geometry, segment_id: str
) -> tuple[Geometry, dict[str, set[str]]]:
    """Take out the segment ending at node *segment_id*, which must not be a
    contour's start. An open contour splits in two there and a closed one
    opens; a piece left without a segment goes. Returns the geometry and the
    IDs of nodes that went or merged."""
    s, n = _locate(geometry, segment_id)
    subpath = geometry.subpaths[s]
    if subpath.closed:
        segments, merged = ring(subpath)
        if segment_id in merged:
            segment_id = next(iter(merged[segment_id]))
        index = next(i for i, node in enumerate(segments) if node.id == segment_id)
        at = segments[index]
        rest = [*segments[index + 1 :], *segments[:index]]
        nodes = [PathNode(at.id, "M", at.endpoint, at.pinned), *rest]
        return _with(geometry, s, [nodes]), merged
    if n == 0:
        raise DocumentError("A contour's start has no segment leading into it")
    nodes = list(subpath.nodes)
    at = nodes[n]
    first, second = nodes[:n], [PathNode(at.id, "M", at.endpoint, at.pinned)]
    second.extend(nodes[n + 1 :])
    removed = {p[0].id: set() for p in (first, second) if len(p) == 1}
    return _with(geometry, s, [first, second]), removed


def segments_among(geometry: Geometry, node_ids: frozenset[str]) -> list[str]:
    """The segments both of whose points are among *node_ids*, by the node
    each ends at."""
    found = []
    for subpath in geometry.subpaths:
        if subpath.closed:
            segments, merged = ring(subpath)
            chosen = {
                next(iter(merged[n])) if n in merged else n
                for n in node_ids
                if any(n == m.id for m in subpath.nodes)
            }
            ids = [node.id for node in segments]
            found.extend(
                b
                for a, b in zip([ids[-1], *ids[:-1]], ids, strict=True)
                if a in chosen and b in chosen and a != b
            )
        else:
            found.extend(
                b.id
                for a, b in pairwise(subpath.nodes)
                if a.id in node_ids and b.id in node_ids
            )
    return found


@dataclass(frozen=True)
class End:
    """An open contour's free end: which contour, which end (0 its start, 1
    its end), the point, and the way the line was heading out of it."""

    contour: int
    side: int
    point: Point
    heading: Point


def _unit(x: float, y: float) -> Point:
    length = math.hypot(x, y)
    return (x / length, y / length) if length else (0.0, 0.0)


def contour_ends(index: int, nodes: tuple[PathNode, ...]) -> tuple[End, End]:
    """The two ends of an open contour, heading out of it along its tangent."""

    def heading(point: Point, toward: list[Point]) -> Point:
        for other in toward:
            if other != point:
                return _unit(point[0] - other[0], point[1] - other[1])
        return (0.0, 0.0)

    def points(node: PathNode) -> list[Point]:
        return list(zip(node.values[::2], node.values[1::2], strict=True))

    start, inward = nodes[0].endpoint, points(nodes[1])
    last, before = nodes[-1], nodes[-2].endpoint
    back = points(last)
    return (
        End(index, 0, start, heading(start, inward)),
        End(index, 1, last.endpoint, heading(last.endpoint, [*back[-2::-1], before])),
    )


def end_pairs(
    ends: list[End], reach: float, turn: float = TURN
) -> list[tuple[End, End]]:
    """Pairs of ends to join, the nearest and straightest first, each end
    used once.

    Two ends pair when they are at most *reach* apart and the line runs on
    across the gap, turning at most *turn* degrees between them and at each
    end. Ends at one spot pair whatever the angle when they are the only two
    there: that is a corner of one line.
    """
    limit = math.cos(math.radians(turn))
    tiny = 1e-9 * max([1.0, *(abs(v) for e in ends for v in e.point)])

    def dot(u: Point, v: Point) -> float:
        return u[0] * v[0] + u[1] * v[1]

    # Ends by cell of a grid at least *reach* wide: an end's partners lie in
    # its cell or the eight around it, so only those are measured.
    size = max(reach, tiny)
    cells: dict[tuple[int, int], list[int]] = {}

    def cell(point: Point) -> tuple[int, int]:
        return math.floor(point[0] / size), math.floor(point[1] / size)

    for i, e in enumerate(ends):
        cells.setdefault(cell(e.point), []).append(i)

    def around(point: Point) -> list[int]:
        x, y = cell(point)
        return [
            j
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for j in cells.get((x + dx, y + dy), ())
        ]

    candidates = []
    for i, a in enumerate(ends):
        nearby = around(a.point)
        for b in (ends[j] for j in sorted(j for j in nearby if j > i)):
            if a.contour == b.contour and a.side == b.side:
                continue
            gap = math.dist(a.point, b.point)
            if gap > reach:
                continue
            ahead = -dot(a.heading, b.heading)
            if gap > tiny:
                across = _unit(b.point[0] - a.point[0], b.point[1] - a.point[1])
                if min(ahead, dot(a.heading, across), -dot(b.heading, across)) < limit:
                    continue
            elif (
                ahead < limit
                and sum(math.dist(ends[j].point, a.point) <= tiny for j in nearby) != 2
            ):
                continue
            candidates.append((gap + reach * (1 - ahead), i, a, b))
    candidates.sort(key=lambda c: (c[0], c[1]))
    used: set[tuple[int, int]] = set()
    pairs = []
    for _, _, a, b in candidates:
        if (a.contour, a.side) in used or (b.contour, b.side) in used:
            continue
        used |= {(a.contour, a.side), (b.contour, b.side)}
        pairs.append((a, b))
    return pairs


def joined(
    contours: list[tuple[PathNode, ...]],
    pairs: list[tuple[End, End]],
    *,
    curve: bool = True,
) -> tuple[list[tuple[list[int], Subpath]], dict[str, set[str]]]:
    """The open *contours* joined at the paired ends.

    Returns each joined contour with the indices of the contours it is made
    of, in order, and the IDs of points that merged into another. Ends that
    meet become one point; the others are bridged by a curve leaving each end
    along its line, or with *curve* false by a straight line. Pairing a chain's
    last end with its first closes it.
    """
    partner: dict[tuple[int, int], End] = {}
    for a, b in pairs:
        partner[a.contour, a.side] = b
        partner[b.contour, b.side] = a
    seen: set[int] = set()

    def walk(first: int, forward: bool) -> tuple[list[tuple[int, bool]], bool]:
        """The chain from *first*, entered at its start when *forward*, and
        whether it comes back round to it."""
        chain, contour, ahead = [], first, forward
        while True:
            seen.add(contour)
            chain.append((contour, ahead))
            following = partner.get((contour, 1 if ahead else 0))
            if following is None:
                return chain, False
            if following.contour == first:
                return chain, True
            contour, ahead = following.contour, following.side == 0

    involved = sorted({c for c, _ in partner})
    chains = []
    for contour in involved:
        if contour not in seen and (contour, 0) not in partner:
            chains.append(walk(contour, True))
        elif contour not in seen and (contour, 1) not in partner:
            chains.append(walk(contour, False))
    # What is left goes round in loops.
    chains.extend(walk(c, True) for c in involved if c not in seen)
    merged: dict[str, set[str]] = {}
    result = []
    for chain, closed in chains:
        nodes: list[PathNode] = []
        for contour, ahead in chain:
            piece = list(contours[contour])
            if not ahead:
                piece = reversed_nodes(piece)
            if nodes:
                entry = (contour, 0 if ahead else 1)
                nodes.extend(
                    _bridge(
                        nodes, piece, partner[entry], *entry, contours, curve, merged
                    )
                )
            else:
                nodes = piece
        if closed:
            first, ahead = chain[0]
            entry = (first, 0 if ahead else 1)
            start = PathNode(new_id("node"), "M", nodes[0].endpoint)
            closing = _bridge(
                nodes, [start], partner[entry], *entry, contours, curve, {}
            )
            if closing:
                nodes.extend(closing)
            elif len(nodes) > 2:
                # The ends met: the last point now lies on the start.
                last = nodes[-1]
                nodes[-1] = replace(
                    last, values=(*last.values[:-2], *nodes[0].endpoint)
                )
            else:
                closed = False
        result.append(
            ([c for c, _ in chain], Subpath(new_id("subpath"), tuple(nodes), closed))
        )
    return result, merged


def _bridge(
    nodes: list[PathNode],
    piece: list[PathNode],
    leaving: End,
    contour: int,
    side: int,
    contours: list[tuple[PathNode, ...]],
    curve: bool,
    merged: dict[str, set[str]],
) -> list[PathNode]:
    """*piece*'s nodes after its start, led into from the end of *nodes*:
    *leaving* is the end the chain leaves by, and *piece* is entered at end
    *side* of *contour*."""
    p, start = nodes[-1].endpoint, piece[0]
    q = start.endpoint
    if math.dist(p, q) <= 1e-9 * max(1.0, *map(abs, (*p, *q))):
        merged[start.id] = {nodes[-1].id}
        if start.pinned:
            nodes[-1] = replace(nodes[-1], pinned=True)
        return piece[1:]
    if not curve:
        return [PathNode(start.id, "L", q, start.pinned), *piece[1:]]
    entering = contour_ends(contour, contours[contour])[side]
    u, v, reach = leaving.heading, entering.heading, math.dist(p, q) / 3
    values = (
        p[0] + u[0] * reach,
        p[1] + u[1] * reach,
        q[0] + v[0] * reach,
        q[1] + v[1] * reach,
        *q,
    )
    return [PathNode(start.id, "C", values, start.pinned), *piece[1:]]


def open_path(geometry: Geometry) -> pathops.Path:
    """The contours as Skia sees a stroked path: open ones stay open."""
    path = pathops.Path()
    for subpath in geometry.subpaths:
        for node in subpath.nodes:
            if node.command == "M":
                path.moveTo(*node.values)
            elif node.command == "L":
                path.lineTo(*node.values)
            else:
                path.cubicTo(*node.values)
        if subpath.closed:
            path.close()
    return path


def stroke_outline(geometry: Geometry, style: dict[str, str]) -> Geometry:
    """The area a stroke of *style* paints along *geometry*, as filled
    contours with no overlaps."""
    width = float(style.get("stroke-width", "1") or 1)
    if not width > 0:
        raise DocumentError("A stroke needs a positive width to outline")
    caps = {
        "round": pathops.LineCap.ROUND_CAP,
        "square": pathops.LineCap.SQUARE_CAP,
    }
    joins = {
        "round": pathops.LineJoin.ROUND_JOIN,
        "bevel": pathops.LineJoin.BEVEL_JOIN,
    }
    path = open_path(geometry)
    path.stroke(
        width,
        caps.get(style.get("stroke-linecap", ""), pathops.LineCap.BUTT_CAP),
        joins.get(style.get("stroke-linejoin", ""), pathops.LineJoin.MITER_JOIN),
        float(style.get("stroke-miterlimit", "4") or 4),
    )
    path.convertConicsToQuads(0.01 * width)
    try:
        path = pathops.simplify(path)
    except pathops.PathOpsError as exc:
        raise DocumentError("Could not outline this stroke") from exc
    if not list(path):
        raise DocumentError("This stroke paints nothing to outline")
    return replace(path_geometry(path), id=geometry.id)
