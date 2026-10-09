"""The selected paths a node step changes, and the nodes it must leave alone.

Pinned endpoints never move; their handles still may.
"""

from __future__ import annotations

from dataclasses import dataclass

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
