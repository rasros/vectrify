"""Explicit edge adjacency and exact subdivision in local geometry coordinates."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from itertools import pairwise

from vectrify.document.hit_test import IDENTITY, multiply
from vectrify.document.model import (
    Document,
    DocumentError,
    EdgeRef,
    Geometry,
    PathNode,
    SharedBoundary,
    new_id,
)

# A coordinate slot in an existing node. Shared endpoints may connect several
# boundaries, so propagation uses equivalence classes rather than pairwise edits.
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
        raise DocumentError("Cannot share nodes through a collapsed transform")
    return (
        d / determinant,
        -b / determinant,
        -c / determinant,
        a / determinant,
        (c * f - d * e) / determinant,
        (b * e - a * f) / determinant,
    )


def close_points(first, second):
    return len(first) == len(second) and all(
        math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-8)
        for p, q in zip(first, second, strict=True)
        for a, b in zip(p, q, strict=True)
    )


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


def validate_boundaries(document: Document, since: Document | None = None) -> None:
    """Check every edge belongs to one link at most and linked edges match.

    With *since*, a valid document this one was edited from, the coordinates
    are compared only for links that are new or touch a changed geometry.
    """
    unchanged: set[str] = set()
    known: set[int] = set()
    if since is not None:
        unchanged = {
            g.id for g in document.geometries if since._index()[0].get(g.id) is g
        }
        known = {id(boundary) for boundary in since.boundaries}
    seen = set()
    for boundary in document.boundaries:
        for member in boundary.members:
            key = member.geometry_id, member.node_id
            if key in seen:
                raise DocumentError("An edge may belong to only one shared boundary")
            seen.add(key)
        if id(boundary) in known and all(
            member.geometry_id in unchanged for member in boundary.members
        ):
            continue
        points = edge(document, boundary.members[0]).points
        for member in boundary.members:
            if not close_points(edge(document, member).points, points):
                raise DocumentError(
                    "Shared edges must have identical oriented coordinates"
                )


def slot_graph(document: Document) -> dict:
    """Map each shared coordinate pair to its peers and the transforms between."""
    # Pair coordinates, rather than individual x/y slots: rotation and skew
    # couple both axes when an endpoint or control handle moves.
    graph = {}
    for boundary in document.boundaries:
        first = boundary.members[0]
        source = edge(document, first).slots[::2]
        for member in boundary.members[1:]:
            for a, b in zip(source, edge(document, member).slots[::2], strict=True):
                graph.setdefault(a, []).append(
                    (b, first.matrix, inverse_matrix(member.matrix))
                )
                graph.setdefault(b, []).append(
                    (a, member.matrix, inverse_matrix(first.matrix))
                )
    return graph


def linked_slots(document: Document, geometry_id: str) -> dict:
    """Every coordinate one geometry shares, with the peers an edit moves.

    Each peer is (geometry, node, index, matrix): the matrix maps this
    geometry's local point to the peer's, through all links on the way.
    """
    graph = slot_graph(document)
    links = {}
    for start in graph:
        if start[0] != geometry_id:
            continue
        seen, stack = {start}, [(start, IDENTITY)]
        while stack:
            slot, matrix = stack.pop()
            for peer, forward, inverse in graph[slot]:
                if peer in seen:
                    continue
                seen.add(peer)
                mapped = multiply(inverse, multiply(forward, matrix))
                links.setdefault(start[1:], []).append((*peer, mapped))
                stack.append((peer, mapped))
    return links


def propagate_node(document: Document, geometry_id: str, node: PathNode) -> Document:
    """Apply a node edit to every equivalent endpoint/control coordinate."""
    graph = slot_graph(document)
    original = document.geometry(geometry_id).node(node.id)
    assignments = {}
    pending = []
    for i in range(0, len(node.values), 2):
        point = tuple(node.values[i : i + 2])
        if point != tuple(original.values[i : i + 2]):
            pending.append(((geometry_id, node.id, i), point))
    while pending:
        slot, point = pending.pop()
        if slot in assignments:
            if not close_points((assignments[slot],), (point,)):
                raise DocumentError("Node edit conflicts with a shared endpoint")
            continue
        assignments[slot] = point
        for peer, matrix, inverse in graph.get(slot, []):
            pending.append((peer, mapped_point(mapped_point(point, matrix), inverse)))
    updates = {}
    for (gid, nid, i), point in assignments.items():
        values = updates.setdefault(
            (gid, nid), list(document.geometry(gid).node(nid).values)
        )
        values[i : i + 2] = point
    for (gid, nid), values in updates.items():
        geometry = document.geometry(gid)
        document = document.replace_geometry(
            geometry.replace_node(replace(geometry.node(nid), values=tuple(values)))
        )
    return document


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
    """Split all members of a boundary; existing endpoints retain their IDs."""
    boundary = next(
        (
            b
            for b in document.boundaries
            if any(
                (m.geometry_id, m.node_id) == (ref.geometry_id, ref.node_id)
                for m in b.members
            )
        ),
        None,
    )
    if boundary:
        members = boundary.members
        selected = next(
            m
            for m in members
            if (m.geometry_id, m.node_id) == (ref.geometry_id, ref.node_id)
        )
        # The caller's t always follows the original SVG edge direction.
        if selected.reversed:
            t = 1 - t
    else:
        members = (ref,)
    left, right = subdivide(edge(document, members[0]).points, t)
    left_refs, right_refs, added = [], [], []
    for member in members:
        original = edge(document, member)
        geometry = document.geometry(member.geometry_id)
        first, second = (right[::-1], left[::-1]) if member.reversed else (left, right)
        inverse = inverse_matrix(member.matrix)
        first = tuple(mapped_point(p, inverse) for p in first)
        second = tuple(mapped_point(p, inverse) for p in second)
        node = PathNode(
            new_id("node"),
            "C" if len(first) == 4 else "L",
            tuple(value for point in first[1:] for value in point),
        )
        added.append(node.id)
        end = (
            original.end
            if original.closing
            else replace(
                original.end,
                values=tuple(value for point in second[1:-1] for value in point)
                + original.end.values[-2:],
            )
        )
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
        inserted = replace(member, node_id=node.id)
        left_refs.append(member if member.reversed else inserted)
        right_refs.append(inserted if member.reversed else member)
    if boundary:
        document = replace(
            document,
            boundaries=(
                *tuple(
                    replace(b, members=tuple(left_refs)) if b.id == boundary.id else b
                    for b in document.boundaries
                ),
                SharedBoundary(new_id("boundary"), tuple(right_refs)),
            ),
        )
    return document, tuple(added)
