"""The moves Optimize nodes tries on the selected paths.

A state is each selected path's contours and stroke width. Every move returns
a new state or None when it found nothing it may change. Moves work in each
path's own coordinates and size their steps to the path, so a transformed or
tiny path is handled the same way as any other.

Two kinds of node never move: pinned endpoints, and nodes on a linked
boundary edge, whose partner path the search does not see.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from vectrify.document import Document, Geometry, PathNode
from vectrify.document.model import new_id
from vectrify.document.topology import edge

# One step, as a share of the path's larger side.
STEP = 0.05
# Where along a segment a split lands.
SPLIT_RANGE = (0.25, 0.75)
# Stroke width moves by up to this factor either way.
STROKE_FACTOR = 1.25


@dataclass(frozen=True)
class Paths:
    """The part of the drawing a node search changes."""

    geometries: dict[str, Geometry]
    # Stroke widths of the stroked paths only.
    strokes: dict[str, float]

    def key(self) -> tuple:
        return (
            tuple((oid, g.path_data()) for oid, g in sorted(self.geometries.items())),
            tuple(sorted(self.strokes.items())),
        )

    def nodes(self) -> int:
        return sum(len(s.nodes) for g in self.geometries.values() for s in g.subpaths)


@dataclass(frozen=True)
class Frozen:
    """What no move may touch, and how far a step goes, per path."""

    # Whole nodes: the ends of linked boundary edges, handles included.
    nodes: frozenset[str]
    # Endpoints only: pinned nodes and the starts of linked edges.
    endpoints: frozenset[str]
    step: dict[str, float]


def frozen(document: Document, paths: Paths) -> Frozen:
    linked: set[str] = set()
    starts: set[str] = set()
    ids = {g.id for g in paths.geometries.values()}
    for boundary in document.boundaries:
        for member in boundary.members:
            if member.geometry_id in ids:
                linked_edge = edge(document, member)
                linked.add(linked_edge.end.id)
                starts.add(linked_edge.start.id)
    pinned = {
        n.id
        for g in paths.geometries.values()
        for s in g.subpaths
        for n in s.nodes
        if n.pinned
    }
    return Frozen(frozenset(linked), frozenset(pinned | starts), step_sizes(paths))


def step_sizes(paths: Paths) -> dict[str, float]:
    steps = {}
    for oid, geometry in paths.geometries.items():
        points = [
            (n.values[i], n.values[i + 1])
            for s in geometry.subpaths
            for n in s.nodes
            for i in range(0, len(n.values), 2)
        ]
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        steps[oid] = STEP * max(max(xs) - min(xs), max(ys) - min(ys), 1e-3)
    return steps


def _with_subpath(geometry: Geometry, index: int, nodes: list[PathNode]) -> Geometry:
    subpaths = list(geometry.subpaths)
    subpaths[index] = replace(subpaths[index], nodes=tuple(nodes))
    return replace(geometry, subpaths=tuple(subpaths))


def _replace(paths: Paths, oid: str, geometry: Geometry) -> Paths:
    return replace(paths, geometries={**paths.geometries, oid: geometry})


def _shifted(node: PathNode, dx: float, dy: float, *, points: range) -> PathNode:
    values = list(node.values)
    for i in points:
        values[2 * i] += dx
        values[2 * i + 1] += dy
    return replace(node, values=tuple(values))


def nudge(paths: Paths, rng: random.Random, fixed: Frozen) -> Paths | None:
    """Move one point or curve handle of one path.

    An endpoint takes its two neighbouring handles along, the way dragging a
    point in an editor does, so the curve keeps its shape either side of it.
    """
    choices = []
    for oid, geometry in paths.geometries.items():
        for si, subpath in enumerate(geometry.subpaths):
            for ni, node in enumerate(subpath.nodes):
                if node.id in fixed.nodes:
                    continue
                if node.id not in fixed.endpoints:
                    choices.append((oid, si, ni, "endpoint"))
                if node.command == "C":
                    choices.extend([(oid, si, ni, "c1"), (oid, si, ni, "c2")])
    if not choices:
        return None
    oid, si, ni, part = rng.choice(choices)
    geometry = paths.geometries[oid]
    nodes = list(geometry.subpaths[si].nodes)
    step = fixed.step[oid]
    dx, dy = rng.uniform(-step, step), rng.uniform(-step, step)
    node = nodes[ni]
    if part == "c1":
        nodes[ni] = _shifted(node, dx, dy, points=range(1))
    elif part == "c2":
        nodes[ni] = _shifted(node, dx, dy, points=range(1, 2))
    else:
        own = range(1, 3) if node.command == "C" else range(1)
        nodes[ni] = _shifted(node, dx, dy, points=own)
        following = nodes[ni + 1] if ni + 1 < len(nodes) else None
        if (
            following is not None
            and following.command == "C"
            and following.id not in fixed.nodes
        ):
            nodes[ni + 1] = _shifted(following, dx, dy, points=range(1))
    return _replace(paths, oid, _with_subpath(geometry, si, nodes))


def _lerp(a: tuple[float, ...], b: tuple[float, ...], t: float):
    return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t


def split(paths: Paths, rng: random.Random, fixed: Frozen) -> Paths | None:
    """Add a point partway along one segment, then push it a random step.

    A cubic is split exactly (de Casteljau), so the new point starts on the
    curve and only the push changes the shape.
    """
    choices = []
    for oid, geometry in paths.geometries.items():
        for si, subpath in enumerate(geometry.subpaths):
            for ni in range(1, len(subpath.nodes)):
                if subpath.nodes[ni].id not in fixed.nodes:
                    choices.append((oid, si, ni))
            first = subpath.nodes[0]
            if (
                subpath.closed
                and len(subpath.nodes) > 1
                and first.id not in fixed.nodes
            ):
                # The closing line, from the last node back to the first.
                choices.append((oid, si, len(subpath.nodes)))
    if not choices:
        return None
    oid, si, ni = rng.choice(choices)
    geometry = paths.geometries[oid]
    nodes = list(geometry.subpaths[si].nodes)
    t = rng.uniform(*SPLIT_RANGE)
    start = nodes[ni - 1].endpoint
    if ni == len(nodes):
        middle = PathNode(new_id("node"), "L", _lerp(start, nodes[0].endpoint, t))
        nodes.append(middle)
    elif nodes[ni].command == "L":
        middle = PathNode(new_id("node"), "L", _lerp(start, nodes[ni].endpoint, t))
        nodes.insert(ni, middle)
    else:
        node = nodes[ni]
        c1, c2, end = node.values[0:2], node.values[2:4], node.endpoint
        a, b, c = _lerp(start, c1, t), _lerp(c1, c2, t), _lerp(c2, end, t)
        d, e = _lerp(a, b, t), _lerp(b, c, t)
        f = _lerp(d, e, t)
        middle = PathNode(new_id("node"), "C", (*a, *d, *f))
        nodes[ni] = replace(node, values=(*e, *c, *end))
        nodes.insert(ni, middle)
    index = nodes.index(middle)
    step = fixed.step[oid]
    dx, dy = rng.uniform(-step, step), rng.uniform(-step, step)
    own = range(1, 3) if middle.command == "C" else range(1)
    nodes[index] = _shifted(middle, dx, dy, points=own)
    if index + 1 < len(nodes) and nodes[index + 1].command == "C":
        nodes[index + 1] = _shifted(nodes[index + 1], dx, dy, points=range(1))
    return _replace(paths, oid, _with_subpath(geometry, si, nodes))


def remove(paths: Paths, rng: random.Random, fixed: Frozen) -> Paths | None:
    """Drop one point, joining the segments either side of it.

    The joined segment leaves along the removed segment's first handle and
    arrives along the next one's last, so both neighbours keep their tangents.
    """
    choices = []
    for oid, geometry in paths.geometries.items():
        for si, subpath in enumerate(geometry.subpaths):
            least = 3 if subpath.closed else 2
            if len(subpath.nodes) <= least:
                continue
            for ni in range(1, len(subpath.nodes)):
                node = subpath.nodes[ni]
                if node.id not in fixed.nodes and node.id not in fixed.endpoints:
                    choices.append((oid, si, ni))
    if not choices:
        return None
    oid, si, ni = rng.choice(choices)
    geometry = paths.geometries[oid]
    nodes = list(geometry.subpaths[si].nodes)
    gone = nodes.pop(ni)
    if ni < len(nodes):
        following = nodes[ni]
        if gone.command == "C" and following.command == "C":
            nodes[ni] = replace(
                following, values=(*gone.values[0:2], *following.values[2:])
            )
    return _replace(paths, oid, _with_subpath(geometry, si, nodes))


def shift(paths: Paths, rng: random.Random, fixed: Frozen) -> Paths | None:
    """Move one whole path, unless it holds a node that may not move."""
    movable = [
        oid
        for oid, g in paths.geometries.items()
        if not any(
            n.id in fixed.nodes or n.id in fixed.endpoints
            for s in g.subpaths
            for n in s.nodes
        )
    ]
    if not movable:
        return None
    oid = rng.choice(movable)
    geometry = paths.geometries[oid]
    step = fixed.step[oid]
    dx, dy = rng.uniform(-step, step), rng.uniform(-step, step)
    moved = replace(
        geometry,
        subpaths=tuple(
            replace(
                s,
                nodes=tuple(
                    _shifted(n, dx, dy, points=range(len(n.values) // 2))
                    for n in s.nodes
                ),
            )
            for s in geometry.subpaths
        ),
    )
    return _replace(paths, oid, moved)


def stroke(paths: Paths, rng: random.Random, _fixed: Frozen) -> Paths | None:
    """Scale one stroked path's stroke width."""
    if not paths.strokes:
        return None
    oid = rng.choice(sorted(paths.strokes))
    factor = STROKE_FACTOR ** rng.uniform(-1, 1)
    width = max(0.05, paths.strokes[oid] * factor)
    return replace(paths, strokes={**paths.strokes, oid: width})


MOVES = {
    "shape": nudge,
    "detail": split,
    "simplify": remove,
    "position": shift,
    "strokes": stroke,
}
