"""Shared edges: the outline a selected path has in common with a neighbour.

Neighbouring regions of a trace meet along one edge drawn in both paths,
the same segments run either way. When Tidy reshapes one side, the other
has to follow or a gap opens between them (or one covers the other). A
`Link` records such a run of segments: where it lies in the selected path
(by the IDs of the points it runs between) and where in the neighbour. The
points a run ends at, where a third region meets the two, are frozen, so
they survive every step and stay put; `follow` then copies the selected
path's outline between them, as it now runs, into the neighbour.

A closed contour drawn back to its start (its last point on its first) is
read as a ring of segments, as is one that closes with an implicit line.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from vectrify.document import Document, Geometry, PathNode, Subpath
from vectrify.document.model import new_id

# Points this close, in a path's own units, are the same point.
SAME = 1e-6


@dataclass(frozen=True)
class Link:
    """A run of segments a selected path shares with a neighbour."""

    path: str
    subpath: str
    # The selected path's points the run goes from and to, along its
    # contour; the same point all the way round a whole shared contour.
    start: str
    end: str
    neighbour: str
    neighbour_subpath: str
    # The neighbour's points at the run's ends, as the selected path's
    # *start* and *end*; with *reversed* the neighbour runs the other way.
    neighbour_start: str
    neighbour_end: str
    reversed: bool


@dataclass(frozen=True)
class _Ring:
    """A contour as segments: *ends[k]* is the point segment k ends at,
    *segments[k]* the node drawing it (None for an implicit closing line),
    segment k running from ends[k - 1]."""

    ends: list[PathNode]
    segments: list[PathNode | None]
    closed: bool


def _point(node: PathNode) -> tuple[float, float]:
    return float(node.values[-2]), float(node.values[-1])


def _key(point) -> tuple[int, int]:
    return round(point[0] / SAME), round(point[1] / SAME)


def _ring(subpath: Subpath) -> _Ring:
    nodes = list(subpath.nodes)
    if not subpath.closed:
        return _Ring(nodes, [None, *nodes[1:]], False)
    if len(nodes) > 1 and _key(_point(nodes[-1])) == _key(_point(nodes[0])):
        # Drawn back to its start: the last node closes onto the first point.
        ends = [nodes[0], *nodes[1:-1]]
        return _Ring(ends, [nodes[-1], *nodes[1:-1]], True)
    return _Ring(nodes, [None, *nodes[1:]], True)


def _controls(ring: _Ring, k: int) -> tuple[tuple[float, float], ...]:
    """Segment k's points from its start to its end."""
    start = _point(ring.ends[k - 1])
    node = ring.segments[k]
    if node is None or node.command == "L":
        return (start, _point(ring.ends[k]))
    values = node.values
    inner = tuple(
        (float(values[i]), float(values[i + 1])) for i in range(0, len(values) - 2, 2)
    )
    return (start, *inner, _point(ring.ends[k]))


def _indices(ring: _Ring) -> range:
    """The segments of *ring*: all of a closed one, from 1 of an open one."""
    return range(len(ring.ends)) if ring.closed else range(1, len(ring.ends))


def links(document: Document, oids, candidates) -> list[Link]:
    """The runs of segments the paths *oids* share with the paths
    *candidates*, the same segments drawn either way, in the same frame."""
    from vectrify.document.transforms import object_matrix

    owned: dict[tuple, list[tuple[str, str, int, bool]]] = {}
    matrices = {}
    sizes: dict[tuple[str, str], int] = {}
    wanted: dict[tuple, set[tuple]] = {}
    selected = []
    for oid in oids:
        matrix = object_matrix(document, oid)
        wanted_ends = wanted.setdefault(matrix, set())
        for subpath in document.geometry_for(oid).subpaths:
            ring = _ring(subpath)
            indices = list(_indices(ring))
            keys = {k: tuple(map(_key, _controls(ring, k))) for k in indices}
            for key in keys.values():
                wanted_ends.add((key[0], key[-1]))
                wanted_ends.add((key[-1], key[0]))
            selected.append((oid, subpath.id, matrix, ring, indices, keys))
    for other in candidates:
        matrix = matrices[other] = object_matrix(document, other)
        if matrix not in wanted:
            continue
        for subpath in document.geometry_for(other).subpaths:
            ring = _ring(subpath)
            sizes[other, subpath.id] = len(ring.ends)
            ring_ends = [_key(_point(node)) for node in ring.ends]
            for k in _indices(ring):
                # A shared segment must first have the same two endpoints.
                # Most of the drawing cannot meet the selected path at all;
                # avoid constructing and indexing all of its cubic controls.
                if (ring_ends[k - 1], ring_ends[k]) not in wanted[matrix]:
                    continue
                points = _controls(ring, k)
                owned.setdefault(tuple(map(_key, points)), []).append(
                    (other, subpath.id, k, False)
                )
                owned.setdefault(tuple(map(_key, points[::-1])), []).append(
                    (other, subpath.id, k, True)
                )
    found: list[Link] = []
    for oid, subpath_id, matrix, ring, indices, keys in selected:
        matches: dict[int, tuple[str, str, int, bool]] = {}
        for k in indices:
            hits = [
                h
                for h in owned.get(keys[k], ())
                if h[0] != oid and matrices[h[0]] == matrix
            ]
            if len(hits) == 1:
                matches[k] = hits[0]
        found += _runs(oid, subpath_id, ring, indices, matches, sizes, document)
    return found


def _runs(
    oid, subpath_id, ring: _Ring, indices, matches, sizes, document
) -> list[Link]:
    """The maximal runs of consecutive segments of *ring* that match
    consecutive segments of one neighbour contour."""

    def follows(k: int, j: int) -> bool:
        """Whether segment j continues segment k's run in the neighbour."""
        a, b = matches.get(k), matches.get(j)
        if a is None or b is None or a[:2] != b[:2] or a[3] != b[3]:
            return False
        size = sizes[a[0], a[1]]
        step = -1 if a[3] else 1
        return b[2] == (a[2] + step) % size

    if not matches:
        return []
    count = len(indices)
    if (
        ring.closed
        and len(matches) == count
        and all(follows(indices[i], indices[(i + 1) % count]) for i in range(count))
    ):
        # A whole contour shared: the run goes round from the first point,
        # through segment 1 first and segment 0 last.
        first = matches[indices[1 % count]]
        return [_link(oid, subpath_id, ring, 0, 0, first, matches[0], document)]
    runs = []
    position = 0
    if ring.closed:
        # Start where a run begins, so none wraps past the start unseen.
        position = next(
            (
                i
                for i in range(count)
                if not follows(indices[i - 1], indices[i]) and indices[i] in matches
            ),
            0,
        )
    seen = 0
    while seen < count:
        k = indices[position % count]
        if k not in matches:
            position += 1
            seen += 1
            continue
        first = k
        last = k
        position += 1
        seen += 1
        while seen < count and follows(last, indices[position % count]):
            last = indices[position % count]
            position += 1
            seen += 1
        runs.append(
            _link(
                oid,
                subpath_id,
                ring,
                first - 1,
                last,
                matches[first],
                matches[last],
                document,
            )
        )
    return runs


def _neighbour_ring(document: Document, oid: str, subpath_id: str) -> _Ring:
    geometry = document.geometry_for(oid)
    return _ring(next(s for s in geometry.subpaths if s.id == subpath_id))


def _link(oid, subpath_id, ring, start, end, first, last, document) -> Link:
    """The run of *ring* from point *start* to point *end*, whose first and
    last segments are the neighbour's *first* and *last*."""
    other = _neighbour_ring(document, first[0], first[1])
    size = len(other.ends)
    if first[3]:
        # Run the other way: the neighbour's first segment ends at our start.
        neighbour_start = other.ends[first[2] % size].id
        neighbour_end = other.ends[(last[2] - 1) % size].id
    else:
        neighbour_start = other.ends[(first[2] - 1) % size].id
        neighbour_end = other.ends[last[2] % size].id
    count = len(ring.ends)
    return Link(
        oid,
        subpath_id,
        ring.ends[start % count].id,
        ring.ends[end % count].id,
        first[0],
        first[1],
        neighbour_start,
        neighbour_end,
        first[3],
    )


def frozen_points(document: Document, found: list[Link]) -> frozenset[str]:
    """Both paths' points the runs *found* end at, and the nodes
    closing a contour onto one of them: no step may move or remove them."""
    ends = {
        i
        for link in found
        for i in (link.start, link.end, link.neighbour_start, link.neighbour_end)
    }
    held = set(ends)
    for oid in {oid for link in found for oid in (link.path, link.neighbour)}:
        for subpath in document.geometry_for(oid).subpaths:
            ring = _ring(subpath)
            closing = ring.segments[0] if ring.closed else None
            if closing is not None and ring.ends[0].id in ends:
                held.add(closing.id)
    return frozenset(held)


def follow(document: Document, found: list[Link]) -> tuple[Document, set[str]]:
    """*document* with each neighbour's shared runs redrawn along the
    selected paths' outlines as they now are; and the neighbours changed.
    A run whose ends are gone from either path is left as it was."""
    changed: set[str] = set()
    by_neighbour: dict[str, list[Link]] = {}
    for link in found:
        by_neighbour.setdefault(link.neighbour, []).append(link)
    for neighbour, its in by_neighbour.items():
        geometry = document.geometry_for(neighbour)
        new = geometry
        for link in its:
            segments = _segments(document.geometry_for(link.path), link)
            if segments is None:
                continue
            redrawn = _redrawn(new, link, segments)
            if redrawn is not None:
                new = redrawn
        if new != geometry:
            document = document.replace_geometry(new)
            changed.add(neighbour)
    return document, changed


def intact(document: Document, link: Link) -> bool:
    """Whether a recorded shared run still has the same controls on both sides.

    A jointly fitted overlap intentionally stops being an exact shared edge.
    Do not copy its earlier boundary back over that accepted improvement.
    """
    source = _segments(document.geometry_for(link.path), link)
    other = replace(
        link,
        subpath=link.neighbour_subpath,
        start=link.neighbour_end if link.reversed else link.neighbour_start,
        end=link.neighbour_start if link.reversed else link.neighbour_end,
    )
    target = _segments(document.geometry_for(link.neighbour), other)
    if source is None or target is None:
        return False
    if link.reversed:
        target = [segment[::-1] for segment in target[::-1]]
    return [tuple(map(_key, s)) for s in source] == [
        tuple(map(_key, s)) for s in target
    ]


def _segments(geometry: Geometry, link: Link):
    """The selected path's run for *link*, as the segments' points from its
    start to its end, or None when its ends are gone."""
    subpath = next((s for s in geometry.subpaths if s.id == link.subpath), None)
    if subpath is None:
        return None
    ring = _ring(subpath)
    ids = [n.id for n in ring.ends]
    if link.start not in ids or link.end not in ids:
        return None
    count = len(ids)
    start, end = ids.index(link.start), ids.index(link.end)
    if not ring.closed and end <= start:
        return None
    steps = (end - start) % count or (count if ring.closed else 0)
    return [_controls(ring, (start + i) % count) for i in range(1, steps + 1)]


def coordinate_indices(geometry: Geometry, subpath_id: str, start_id: str, end_id: str):
    """Independent coordinate rows along a shared run, in contour order.

    Rows index all the path's node values as pairs. An implicit closing line
    has only its two endpoint rows; derived straight controls are not separate
    editor coordinates. The same ring convention as following handles drawn
    closures, wrapped runs and open contours.
    """
    rows, offset = {}, 0
    for subpath in geometry.subpaths:
        for node in subpath.nodes:
            count = len(node.values) // 2
            rows[node.id] = tuple(range(offset, offset + count))
            offset += count
    subpath = next((s for s in geometry.subpaths if s.id == subpath_id), None)
    if subpath is None:
        return None
    ring = _ring(subpath)
    ids = [n.id for n in ring.ends]
    if start_id not in ids or end_id not in ids:
        return None
    start, end = ids.index(start_id), ids.index(end_id)
    if not ring.closed and end <= start:
        return None
    count = len(ids)
    steps = (end - start) % count or (count if ring.closed else 0)
    segments = []
    for i in range(1, steps + 1):
        k = (start + i) % count
        head = rows[ring.ends[k - 1].id][-1]
        node = ring.segments[k]
        tail = (rows[ring.ends[k].id][-1],) if node is None else rows[node.id]
        segments.append((head, *tail))
    return segments


def _redrawn(geometry: Geometry, link: Link, segments) -> Geometry | None:
    """*geometry* with *link*'s run in its neighbour contour drawn as
    *segments* (the selected path's, from its start to its end)."""
    subpaths = list(geometry.subpaths)
    at = next(
        (i for i, s in enumerate(subpaths) if s.id == link.neighbour_subpath), None
    )
    if at is None:
        return None
    subpath = subpaths[at]
    ring = _ring(subpath)
    ids = [n.id for n in ring.ends]
    if link.neighbour_start not in ids or link.neighbour_end not in ids:
        return None
    count = len(ids)
    if link.reversed:
        # The neighbour runs from our end to our start.
        segments = [s[::-1] for s in segments[::-1]]
        first, last = ids.index(link.neighbour_end), ids.index(link.neighbour_start)
    else:
        first, last = ids.index(link.neighbour_start), ids.index(link.neighbour_end)
    if not ring.closed and last <= first:
        return None
    steps = (last - first) % count or (count if ring.closed else 0)
    nodes = list(subpath.nodes)
    # Segment k of the ring is node k from 1 on; a run past the last node
    # goes round the start, so the contour is redrawn from the run's start.
    if ring.closed and first + steps > len(nodes) - 1:
        nodes = _rotated(ring, first)
        first = 0
    old = nodes[first + 1 : first + steps + 1]
    if len(old) != steps:
        return None
    # The run's end keeps its point's ID; its points in between keep theirs
    # as far as they go.
    spare = [n.id for n in old[:-1]]
    new = []
    for i, points in enumerate(segments):
        if i == len(segments) - 1:
            node_id = old[-1].id
        else:
            node_id = spare.pop(0) if spare else new_id("node")
        end = points[-1]
        if len(points) == 2:
            new.append(PathNode(node_id, "L", (end[0], end[1])))
        else:
            new.append(PathNode(node_id, "C", tuple(v for p in points[1:] for v in p)))
    if any(n.pinned for n in old):
        return None
    nodes[first + 1 : first + steps + 1] = new
    subpaths[at] = replace(subpath, nodes=tuple(nodes))
    return replace(geometry, subpaths=tuple(subpaths))


def _rotated(ring: _Ring, head: int) -> list[PathNode]:
    """A closed *ring*'s nodes starting from its point *head*, drawn back to
    it: a new moveto there, then every segment in turn."""
    count = len(ring.ends)
    # The moveto is now the canonical occurrence of this endpoint. Keep its
    # ID so existing junction/edge links still find it after rotation; the
    # drawn closure below gets a separate segment ID.
    nodes = [replace(ring.ends[head], command="M", values=_point(ring.ends[head]))]
    for i in range(1, count + 1):
        k = (head + i) % count
        segment = ring.segments[k]
        if segment is None:
            # The implicit closing line, drawn out.
            segment = PathNode(ring.ends[k].id, "L", _point(ring.ends[k]))
        if i == count:
            segment = replace(segment, id=new_id("node"))
        nodes.append(segment)
    return nodes
