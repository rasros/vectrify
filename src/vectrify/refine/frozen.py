"""The selected paths a node step changes, and the nodes it must leave alone.

Pinned endpoints never move; their existing handles still may.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import pairwise

from vectrify.document import Geometry


@dataclass(frozen=True)
class Paths:
    """Each selected path's geometry, by object ID."""

    geometries: dict[str, Geometry]

    def nodes(self) -> int:
        return sum(len(s.nodes) for g in self.geometries.values() for s in g.subpaths)


@dataclass(frozen=True)
class Frozen:
    """What no step may touch."""

    # Endpoints only: pinned nodes.
    endpoints: frozenset[str]


def frozen(paths: Paths) -> Frozen:
    return Frozen(
        frozenset(
            n.id
            for g in paths.geometries.values()
            for s in g.subpaths
            for n in s.nodes
            if n.pinned or n.feature is not None
        )
    )


def without_new_handles(original: Geometry, candidate: Geometry) -> Geometry:
    """Retract newly introduced handles at pinned anchors, on either side.

    Controls belong to segments: an anchor's outgoing handle is stored on
    its successor. Explicit closing knots share both sides of the seam.
    Existing handles remain free to change, including after subdivision.
    """
    incoming, outgoing = set(), set()
    for contour in original.subpaths:
        nodes = contour.nodes
        seam = contour.closed and nodes[0].endpoint == nodes[-1].endpoint
        seam_pinned = seam and (nodes[0].pinned or nodes[-1].pinned)
        for i, node in enumerate(nodes):
            at_seam = seam and i in {0, len(nodes) - 1}
            if not node.pinned and not (at_seam and seam_pinned):
                continue
            arrive = nodes[-1] if at_seam else node
            leave = (
                nodes[1]
                if at_seam and len(nodes) > 1
                else nodes[i + 1]
                if i + 1 < len(nodes)
                else None
            )
            if arrive.command != "C" or arrive.values[2:4] == node.endpoint:
                incoming.add(node.id)
            if (
                leave is None
                or leave.command != "C"
                or leave.values[:2] == node.endpoint
            ):
                outgoing.add(node.id)
    if not incoming and not outgoing:
        return candidate
    subpaths = []
    for contour in candidate.subpaths:
        first, last = contour.nodes[0], contour.nodes[-1]
        blocked_in, blocked_out = incoming, outgoing
        if contour.closed and first.endpoint == last.endpoint:
            aliases = {first.id, last.id}
            if aliases & incoming:
                blocked_in = incoming | aliases
            if aliases & outgoing:
                blocked_out = outgoing | aliases
        nodes = [contour.nodes[0]]
        for previous, node in pairwise(contour.nodes):
            if node.command == "C" and (
                previous.id in blocked_out or node.id in blocked_in
            ):
                values = list(node.values)
                if previous.id in blocked_out:
                    values[:2] = previous.endpoint
                if node.id in blocked_in:
                    values[2:4] = node.endpoint
                node = (
                    replace(node, command="L", values=node.endpoint)
                    if tuple(values[:2]) == previous.endpoint
                    and tuple(values[2:4]) == node.endpoint
                    else replace(node, values=tuple(values))
                )
            nodes.append(node)
        subpaths.append(replace(contour, nodes=tuple(nodes)))
    return replace(candidate, subpaths=tuple(subpaths))
