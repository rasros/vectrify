"""Redraw a stretch of a contour: where a stroke attaches, and the splice.

A contour is read as its segments in order, each named by the node it ends
at; a closed contour's closing line is named by its moveto and left out when
the last node already sits on the first. A place on it is the segment's node
and a t along that segment, 1 at the node itself.

The stretch between two places is cut out and a new run of segments put in.
The ends are split exactly where they are, so every node outside the stretch
keeps its ID, place and handles. Along a closed contour either way round
joins the two places: the shorter one is replaced unless asked otherwise.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import pairwise

import numpy as np

from vectrify.document.hit_test import IDENTITY
from vectrify.document.model import (
    Document,
    DocumentError,
    Geometry,
    PathNode,
    Subpath,
    new_id,
)
from vectrify.document.topology import mapped_point, subdivide
from vectrify.document.transforms import root_matrix as root_matrix

Point = tuple[float, float]
# Samples per segment when looking for the nearest place and measuring length.
SAMPLES = 64
# Places closer than this share of a segment's t to its ends are its ends.
AT_END = 1e-6


@dataclass(frozen=True)
class Attachment:
    """Where a stroke's end meets a contour, in root user space."""

    subpath_id: str
    node_id: str
    t: float
    point: Point
    distance: float


@dataclass(frozen=True)
class _Segment:
    id: str
    # Start, the two handles of a cubic, end; in local coordinates.
    controls: tuple[Point, ...]

    @property
    def node(self) -> PathNode:
        command = "C" if len(self.controls) == 4 else "L"
        return PathNode(
            self.id, command, tuple(v for p in self.controls[1:] for v in p)
        )


def _segments(subpath: Subpath) -> list[_Segment]:
    nodes = subpath.nodes
    found = []
    for before, node in pairwise(nodes):
        handles = (
            ((node.values[0], node.values[1]), (node.values[2], node.values[3]))
            if node.command == "C"
            else ()
        )
        found.append(_Segment(node.id, (before.endpoint, *handles, node.endpoint)))
    if subpath.closed and len(nodes) > 1 and nodes[-1].endpoint != nodes[0].endpoint:
        found.append(_Segment(nodes[0].id, (nodes[-1].endpoint, nodes[0].endpoint)))
    return found


def _bezier(controls: Sequence[Point], t: np.ndarray) -> np.ndarray:
    points = np.asarray(controls, dtype=np.float64)
    if len(points) == 2:
        return points[0] + t[:, None] * (points[1] - points[0])
    u = (1 - t)[:, None]
    t = t[:, None]
    return (
        u**3 * points[0]
        + 3 * u**2 * t * points[1]
        + 3 * u * t**2 * points[2]
        + t**3 * points[3]
    )


def _mapped(controls: Sequence[Point], matrix) -> tuple[Point, ...]:
    return tuple(mapped_point(p, matrix) for p in controls)


def attachment(
    document: Document,
    object_id: str,
    point: Point,
    reach: float,
    node_reach: float,
    subpath_id: str | None = None,
) -> Attachment | None:
    """Where *point*, in root user space, attaches to the object's outline.

    On the nearest point within *node_reach*, else on the nearest place on the
    outline within *reach*; only on *subpath_id*'s contour when given.
    """
    geometry = document.geometry_for(object_id)
    matrix = root_matrix(document, object_id)
    target = np.asarray(point, dtype=np.float64)
    nearest_node: Attachment | None = None
    nearest: Attachment | None = None
    t = np.linspace(0, 1, SAMPLES + 1)
    for subpath in geometry.subpaths:
        if subpath_id is not None and subpath.id != subpath_id:
            continue
        segments = _segments(subpath)
        if not segments:
            continue
        # Each point as the end of its segment; an open contour's start as
        # the start of its first.
        ends = [(s.id, 1.0, s.controls[-1]) for s in segments]
        if not subpath.closed:
            ends.insert(0, (segments[0].id, 0.0, segments[0].controls[0]))
        for node_id, at, local in ends:
            mapped = mapped_point(local, matrix)
            distance = math.dist(mapped, point)
            if distance <= node_reach and (
                nearest_node is None or distance < nearest_node.distance
            ):
                nearest_node = Attachment(subpath.id, node_id, at, mapped, distance)
        for segment in segments:
            controls = _mapped(segment.controls, matrix)
            samples = _bezier(controls, t)
            gaps = np.linalg.norm(samples - target, axis=1)
            i = int(np.argmin(gaps))
            # Narrow the best sample down between its neighbours.
            low, high = t[max(0, i - 1)], t[min(SAMPLES, i + 1)]
            for _ in range(40):
                a, b = low + (high - low) / 3, high - (high - low) / 3
                near = _bezier(controls, np.array([a, b]))
                if np.linalg.norm(near[0] - target) <= np.linalg.norm(near[1] - target):
                    high = b
                else:
                    low = a
            at = float((low + high) / 2)
            spot = _bezier(controls, np.array([at]))[0]
            distance = float(np.linalg.norm(spot - target))
            if distance <= reach and (nearest is None or distance < nearest.distance):
                nearest = Attachment(
                    subpath.id,
                    segment.id,
                    at,
                    (float(spot[0]), float(spot[1])),
                    distance,
                )
    return nearest_node or nearest


def _length(controls: Sequence[Point], matrix) -> float:
    samples = _bezier(_mapped(controls, matrix), np.linspace(0, 1, SAMPLES // 4 + 1))
    return float(np.linalg.norm(np.diff(samples, axis=0), axis=1).sum())


def _position(segments: list[_Segment], subpath: Subpath, place) -> float:
    """A place as a number along the contour: segment index plus t."""
    node_id, t = place
    t = float(t)
    if not math.isfinite(t) or not 0 <= t <= 1:
        raise DocumentError("A place on the outline needs a t between 0 and 1")
    for i, segment in enumerate(segments):
        if segment.id == node_id:
            return i + t
    # A closed contour ending on its moveto names its start by both nodes.
    if node_id == subpath.nodes[0].id:
        return 0.0
    raise DocumentError("The place is not on this contour")


def _split(
    segments: list[_Segment], positions: list[float], closed: bool
) -> tuple[list[_Segment], list[int]]:
    """*segments* split at *positions*, and the index of the point at each."""
    count = len(segments)
    places = []
    cuts: dict[int, set[float]] = {}
    for u in positions:
        k = min(math.floor(u), count - 1)
        t = u - k
        if t <= AT_END:
            places.append((k, 0.0))
        elif t >= 1 - AT_END:
            places.append((k + 1, 0.0))
        else:
            places.append((k, t))
            cuts.setdefault(k, set()).add(t)
    pieces: list[_Segment] = []
    # Where each original point and each cut lands in the split list.
    starts = []
    at_cut: dict[tuple[int, float], int] = {}
    for k, segment in enumerate(segments):
        starts.append(len(pieces))
        rest, done = segment.controls, 0.0
        for t in sorted(cuts.get(k, ())):
            left, rest = subdivide(rest, (t - done) / (1 - done))
            pieces.append(_Segment(new_id("node"), left))
            at_cut[(k, t)] = len(pieces)
            done = t
        pieces.append(_Segment(segment.id, rest))
    starts.append(len(pieces))
    indices = []
    for k, t in places:
        index = at_cut[(k, t)] if t else starts[k]
        indices.append(index % len(pieces) if closed else index)
    return pieces, indices


def _stretch(
    nodes: Sequence[tuple[str, Sequence[float]]], start: Point, end: Point
) -> list[_Segment]:
    """New segments from *start* to *end*, precisely, through *nodes*."""
    found = []
    here = start
    for i, (command, values) in enumerate(nodes):
        values = tuple(float(v) for v in values)
        if command not in {"L", "C"} or len(values) != (6 if command == "C" else 2):
            raise DocumentError("A redrawn stretch is made of lines and cubics")
        if not all(math.isfinite(v) for v in values):
            raise DocumentError("Path coordinates must be finite")
        there = end if i == len(nodes) - 1 else (values[-2], values[-1])
        handles = (
            ((values[0], values[1]), (values[2], values[3])) if command == "C" else ()
        )
        found.append(_Segment(new_id("node"), (here, *handles, there)))
        here = there
    if not found:
        raise DocumentError("Draw the new outline between the two places")
    return found


def _reversed(stretch: list[_Segment]) -> list[_Segment]:
    return [_Segment(s.id, s.controls[::-1]) for s in reversed(stretch)]


def redrawn(
    geometry: Geometry,
    subpath_id: str,
    start: tuple[str, float],
    end: tuple[str, float],
    nodes: Sequence[tuple[str, Sequence[float]]],
    *,
    matrix=IDENTITY,
    long_way: bool = False,
) -> tuple[Geometry, frozenset[str]]:
    """*geometry* with the contour's stretch from *start* to *end* replaced.

    *nodes* are the new segments' commands and local values, drawn from
    *start* to *end*; their ends are set to those places exactly. *matrix*
    maps local coordinates to the frame lengths are compared in. Returns the
    new geometry and the IDs of the nodes the stretch removed.
    """
    subpath = next((s for s in geometry.subpaths if s.id == subpath_id), None)
    if subpath is None:
        raise DocumentError("The contour is not part of this path")
    segments = _segments(subpath)
    if not segments:
        raise DocumentError("The contour has no outline to redraw")
    closed = subpath.closed
    positions = [_position(segments, subpath, place) for place in (start, end)]
    if closed:
        positions = [u % len(segments) for u in positions]
    pieces, (a, b) = _split(segments, positions, closed)
    if a == b:
        raise DocumentError("Start and end the stroke at two different places")
    point = [
        pieces[i].controls[0] if i < len(pieces) else pieces[-1].controls[-1]
        for i in (a, b)
    ]
    stretch = _stretch(nodes, point[0], point[1])
    count = len(pieces)
    if closed:
        forward = sum(
            _length(pieces[i % count].controls, matrix)
            for i in range(a, a + (b - a) % count)
        )
        total = sum(_length(s.controls, matrix) for s in pieces)
        # The arc from a on to b, else the one from b on to a.
        onward = (forward <= total - forward) != long_way
        first, last = (a, b) if onward else (b, a)
        if not onward:
            stretch = _reversed(stretch)
        closing = segments[-1].id
        rotated = pieces[last:] + pieces[:last]
        kept = rotated[: (first - last) % count]
        stretch[-1] = replace(stretch[-1], id=rotated[-1].id)
        cycle = kept + stretch
        start_id = subpath.nodes[0].id
        ends = [s.id for s in cycle]
        if closing in ends:
            # The contour still starts where it did.
            i = ends.index(closing)
        else:
            start_id = closing = stretch[-1].id
            i = len(cycle) - 1
        cycle = cycle[i + 1 :] + cycle[: i + 1]
        tail = cycle[-1]
        new_nodes = [
            PathNode(start_id, "M", tail.controls[-1]),
            *(s.node for s in cycle[:-1]),
        ]
        if tail.id != start_id:
            new_nodes.append(tail.node)
        elif len(tail.controls) == 4:
            new_nodes.append(replace(tail.node, id=new_id("node")))
        # A line back to the start is the implicit closing line.
    else:
        first, last = sorted((a, b))
        if a > b:
            stretch = _reversed(stretch)
        stretch[-1] = replace(stretch[-1], id=pieces[last - 1].id)
        path = pieces[:first] + stretch + pieces[last:]
        new_nodes = [subpath.nodes[0], *(s.node for s in path)]
    old_pins = {n.id: n.pinned for n in subpath.nodes}
    new_nodes = [replace(n, pinned=old_pins.get(n.id, False)) for n in new_nodes]
    updated = replace(subpath, nodes=tuple(new_nodes))
    kept_ids = {n.id for n in new_nodes}
    removed = frozenset(n.id for n in subpath.nodes if n.id not in kept_ids)
    return (
        replace(
            geometry,
            subpaths=tuple(
                updated if s.id == subpath_id else s for s in geometry.subpaths
            ),
        ),
        removed,
    )
