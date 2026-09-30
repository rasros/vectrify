"""The selected paths a node step changes, and the nodes it must leave alone.

Two kinds of node never move: pinned endpoints, and nodes on a linked
boundary edge, whose partner path the step does not see.
"""

from __future__ import annotations

from dataclasses import dataclass

from vectrify.document import Document, Geometry
from vectrify.document.topology import edge


@dataclass(frozen=True)
class Paths:
    """Each selected path's geometry, by object ID."""

    geometries: dict[str, Geometry]

    def nodes(self) -> int:
        return sum(len(s.nodes) for g in self.geometries.values() for s in g.subpaths)


@dataclass(frozen=True)
class Frozen:
    """What no step may touch."""

    # Whole nodes: the ends of linked boundary edges, handles included.
    nodes: frozenset[str]
    # Endpoints only: pinned nodes and the starts of linked edges.
    endpoints: frozenset[str]


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
    return Frozen(frozenset(linked), frozenset(pinned | starts))
