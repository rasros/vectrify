"""Oriented edges and exact subdivision in local geometry coordinates."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from itertools import pairwise

from vectrify.document.model import (
    Document,
    DocumentError,
    Geometry,
    PathNode,
    new_id,
)

# A coordinate slot in an existing node: geometry, node and value index.
Slot = tuple[str, str, int]
Point = tuple[float, float]


def mapped_point(point, matrix):
    a, b, c, d, e, f = matrix
    x, y = point
    return (a * x + c * y + e, b * x + d * y + f)


def inverse_matrix(matrix):
    a, b, c, d, e, f = matrix
    determinant = a * d - b * c
    if not math.isfinite(determinant) or determinant == 0:
        raise DocumentError("Cannot map nodes through a collapsed transform")
    return (
        d / determinant,
        -b / determinant,
        -c / determinant,
        a / determinant,
        (c * f - d * e) / determinant,
        (b * e - a * f) / determinant,
    )


@dataclass(frozen=True)
class EdgeRef:
    """An edge ending at a node, optionally traversed backwards.

    A moveto identifies the implicit closing line of a closed subpath.
    The matrix maps local coordinates into a common frame, such as root
    user space, so edges of differently transformed paths can be compared.
    """

    geometry_id: str
    node_id: str
    reversed: bool = False
    matrix: tuple[float, ...] = (1, 0, 0, 1, 0, 0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "matrix", tuple(self.matrix))
        if len(self.matrix) != 6 or not all(math.isfinite(v) for v in self.matrix):
            raise DocumentError("Edge transform must be finite")
        a, b, c, d, _, _ = self.matrix
        if not math.isfinite(a * d - b * c) or a * d - b * c == 0:
            raise DocumentError("Edge transform must be invertible")


@dataclass(frozen=True)
class Edge:
    ref: EdgeRef
    start: PathNode
    end: PathNode
    subpath_id: str
    closing: bool

    @property
    def points(self) -> tuple[Point, ...]:
        controls = (
            (
                (self.end.values[0], self.end.values[1]),
                (self.end.values[2], self.end.values[3]),
            )
            if self.end.command == "C"
            else ()
        )
        points = (self.start.endpoint, *controls, self.end.endpoint)
        points = tuple(mapped_point(p, self.ref.matrix) for p in points)
        return points[::-1] if self.ref.reversed else points

    @property
    def slots(self) -> tuple[Slot, ...]:
        nodes = [(self.start, len(self.start.values) - 2)]
        if self.end.command == "C":
            nodes.extend([(self.end, 0), (self.end, 2)])
        nodes.append((self.end, len(self.end.values) - 2))
        if self.ref.reversed:
            nodes.reverse()
        return tuple(
            (self.ref.geometry_id, node.id, index + axis)
            for node, index in nodes
            for axis in (0, 1)
        )


def edge(document: Document, ref: EdgeRef) -> Edge:
    position = document.node_position(ref.geometry_id, ref.node_id)
    if position is None:
        raise DocumentError(f"Unknown edge node: {ref.node_id}")
    subpath, i = position
    if i == 0 and (not subpath.closed or len(subpath.nodes) < 2):
        raise DocumentError("Moveto has no incoming edge in an open subpath")
    node = subpath.nodes[i]
    return Edge(ref, subpath.nodes[i - 1], node, subpath.id, i == 0)


def subdivide(
    points: tuple[Point, ...], t: float
) -> tuple[tuple[Point, ...], tuple[Point, ...]]:
    """De Casteljau; both halves reuse precisely the same split coordinate."""
    levels = [points]
    while len(levels[-1]) > 1:
        level = levels[-1]
        levels.append(
            tuple(
                (a[0] * (1 - t) + b[0] * t, a[1] * (1 - t) + b[1] * t)
                for a, b in pairwise(level)
            )
        )
    return tuple(level[0] for level in levels), tuple(
        level[-1] for level in reversed(levels)
    )


def split_edges(
    document: Document, ref: EdgeRef, t: float
) -> tuple[Document, tuple[str, ...]]:
    """Split one edge at *t*; existing endpoints retain their IDs.

    The caller's t follows the edge as *ref* traverses it.
    """
    if ref.reversed:
        t = 1 - t
    ref = replace(ref, reversed=False)
    original = edge(document, ref)
    left, right = subdivide(original.points, t)
    inverse = inverse_matrix(ref.matrix)
    first = tuple(mapped_point(p, inverse) for p in left)
    second = tuple(mapped_point(p, inverse) for p in right)
    node = PathNode(
        new_id("node"),
        "C" if len(first) == 4 else "L",
        tuple(value for point in first[1:] for value in point),
    )
    end = (
        original.end
        if original.closing
        else replace(
            original.end,
            values=tuple(value for point in second[1:-1] for value in point)
            + original.end.values[-2:],
        )
    )
    geometry = document.geometry(ref.geometry_id)
    subpaths = []
    for subpath in geometry.subpaths:
        if subpath.id != original.subpath_id:
            subpaths.append(subpath)
            continue
        nodes = []
        for old in subpath.nodes:
            if old.id == end.id:
                if not original.closing:
                    nodes.append(node)
                nodes.append(end)
            else:
                nodes.append(old)
        if original.closing:
            nodes.append(node)
        subpaths.append(replace(subpath, nodes=tuple(nodes)))
    document = document.replace_geometry(Geometry(geometry.id, tuple(subpaths)))
    return document, (node.id,)
