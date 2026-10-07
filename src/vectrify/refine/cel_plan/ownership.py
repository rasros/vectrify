"""Immutable source-region ownership across export and structural edits.

The original graph supplies stable region and canonical boundary identities.
Primary surfaces partition its visible regions; coverage bases are explicitly
secondary owners. Geometry/paint edits keep ownership, while merges or splits
replace it atomically. This is internal planning metadata, never SVG geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vectrify.document import Document
from vectrify.refine.cel_plan.atoms import Atoms
from vectrify.refine.cel_plan.model import Evidence, Graph, StageInterruptedError, Work


@dataclass(frozen=True)
class Surface:
    """Primary members and optional hidden overlap with other owned regions.

    Covered membership records secondary paint support, not a containment proof.
    Only actual geometry can establish an opacity core's complete coverage.
    """

    id: str
    members: tuple[int, ...]
    role: str = "surface"
    covered: tuple[int, ...] = ()


@dataclass(frozen=True)
class OwnedEdge:
    boundary: int
    left: str | None
    right: str | None


@dataclass(frozen=True)
class Partition:
    surfaces: tuple[Surface, ...]
    atoms: Atoms | None = None

    def __post_init__(self):
        ids = [surface.id for surface in self.surfaces]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate planned surface identity")
        members = []
        for surface in self.surfaces:
            if surface.role not in {"surface", "overlay", "underlay"}:
                raise ValueError("Unknown planned surface role")
            if not surface.members or surface.members != tuple(
                sorted(set(surface.members))
            ):
                raise ValueError("Surface needs sorted unique source regions")
            if min(surface.members) < 0:
                raise ValueError("Source region identities must be nonnegative")
            if (
                surface.covered != tuple(sorted(set(surface.covered)))
                or any(i < 0 for i in surface.covered)
                or set(surface.covered).intersection(surface.members)
                or (surface.role == "underlay" and surface.covered)
            ):
                raise ValueError("Hidden coverage needs distinct sorted source regions")
            if surface.role != "underlay":
                members.extend(surface.members)
        if len(members) != len(set(members)):
            raise ValueError("A source region has multiple primary surface owners")
        known = set(members)
        if any(set(s.covered) - known for s in self.surfaces):
            raise ValueError("Hidden coverage must reference owned source regions")
        if self.atoms is not None:
            retired = {c.parent for c in self.atoms.cuts}
            count = self.atoms.count + 2 * len(self.atoms.cuts)
            if any(
                i >= count or i in retired
                for s in self.surfaces
                for i in (*s.members, *s.covered)
            ):
                raise ValueError("Ownership must reference active source atoms")

    @property
    def owners(self) -> dict[int, str]:
        return {
            member: surface.id
            for surface in self.surfaces
            if surface.role != "underlay"
            for member in surface.members
        }

    def validate(self, document: Document):
        known = {element.id: element for element in document.elements()}
        if any(
            surface.id not in known or known[surface.id].tag != "path"
            for surface in self.surfaces
        ):
            raise ValueError(
                "Planned surface ownership references missing drawing paths"
            )

    def edges(self, graph: Graph) -> tuple[OwnedEdge, ...]:
        if self.atoms is not None and graph.source_atoms != self.atoms.key:
            raise ValueError("Canonical edges need the current source atom graph")
        owners = self.owners
        return tuple(
            OwnedEdge(edge.id, owners.get(edge.left), owners.get(edge.right))
            for edge in graph.boundaries
            if owners.get(edge.left) != owners.get(edge.right)
        )

    def replace(self, ids: tuple[str, ...], surfaces: tuple[Surface, ...]) -> Partition:
        removed = set(ids)
        selected = tuple(surface for surface in self.surfaces if surface.id in removed)
        if len(selected) != len(removed) or any(s.role == "underlay" for s in selected):
            raise ValueError("Replace known primary planned surfaces")
        if any(s.role == "underlay" for s in surfaces):
            raise ValueError("A primary replacement cannot become secondary coverage")
        expected = sorted(member for s in selected for member in s.members)
        actual = sorted(member for s in surfaces for member in s.members)
        if actual != expected:
            raise ValueError(
                "A structural edit must retain all source-region ownership"
            )
        if {i for s in selected for i in s.covered} != {
            i for s in surfaces for i in s.covered
        }:
            raise ValueError("A structural edit must retain hidden source coverage")
        return Partition(
            tuple(s for s in self.surfaces if s.id not in removed) + surfaces,
            self.atoms,
        )

    def split(self, ids, surfaces, atoms: Atoms) -> Partition:
        if not atoms.extends(self.atoms):
            raise ValueError("Source splits must extend their parent's atom namespace")
        start = len(self.atoms.cuts) if self.atoms is not None else 0
        expanded = Partition(
            tuple(
                Surface(
                    s.id,
                    atoms.descendants(s.members, start),
                    s.role,
                    atoms.descendants(s.covered, start),
                )
                for s in self.surfaces
            ),
            atoms,
        )
        return expanded.replace(ids, surfaces)

    def follows(self, previous: Partition) -> bool:
        if self.atoms == previous.atoms:
            return self.owners.keys() == previous.owners.keys()
        if self.atoms is None or not self.atoms.extends(previous.atoms):
            return False
        start = len(previous.atoms.cuts) if previous.atoms is not None else 0
        return set(self.owners) == set(self.atoms.descendants(previous.owners, start))

    def metadata(self) -> dict:
        return {
            "version": 1,
            "complete": True,
            **(
                {"source_atoms": self.atoms.metadata()}
                if self.atoms is not None
                else {}
            ),
            "surfaces": [
                {
                    "id": s.id,
                    "members": list(s.members),
                    "role": s.role,
                    **({"covered": list(s.covered)} if s.covered else {}),
                }
                for s in self.surfaces
            ],
        }

    @classmethod
    def from_metadata(cls, metadata: dict | None) -> Partition | None:
        if not metadata or metadata.get("version") != 1 or not metadata.get("complete"):
            return None
        return cls(
            tuple(
                Surface(
                    s["id"], tuple(s["members"]), s["role"], tuple(s.get("covered", ()))
                )
                for s in metadata["surfaces"]
            ),
            Atoms.from_metadata(metadata.get("source_atoms")),
        )


def exported(
    evidence: Evidence,
    labels: np.ndarray,
    fill_ids: frozenset[int],
    work: Work,
    *,
    overlays: tuple = (),
    bases: tuple = (),
) -> dict:
    """Map merged export labels back to stable original regions, never offsets.

    A current export normally coarsens the original partition. An independently
    supplied partition that splits an original region cannot claim complete
    atomic ownership; it remains renderable but needs new graph atoms before
    structural edits. Empty source regions are excluded, not silently merged.
    """
    if work.interrupted:
        raise StageInterruptedError("Surface ownership interrupted")
    source = evidence.labels
    count = int(source.max()) + 1
    lookup = np.full(count, -1, dtype=np.int32)
    lookup[source.ravel()] = labels.ravel()
    if not np.array_equal(lookup[source], labels):
        return {"version": 1, "complete": False, "reason": "source-region-split"}
    hidden = {int(i) for i in np.unique(source[evidence.empty])}
    members: dict[int, list[int]] = {}
    for original, current in enumerate(lookup):
        if original not in hidden and current >= 0:
            members.setdefault(int(current), []).append(original)
    overlay_for = {
        member: overlay.region for overlay in overlays for member in overlay.members
    }
    grouped: dict[tuple[str, str], list[int]] = {}
    for current, original in members.items():
        if current in overlay_for:
            key = (f"cel-overlay-{overlay_for[current]}", "overlay")
        elif current in fill_ids:
            key = (f"cel-fill-{current}", "surface")
        else:
            return {"version": 1, "complete": False, "reason": "missing-surface"}
        grouped.setdefault(key, []).extend(original)
    surfaces = [
        Surface(oid, tuple(sorted(values)), role)
        for (oid, role), values in sorted(grouped.items())
    ]
    surfaces.extend(
        Surface(
            f"cel-base-{base.component}",
            tuple(
                sorted(
                    original
                    for current in base.members
                    for original in members.get(current, ())
                )
            ),
            "underlay",
        )
        for base in bases
    )
    if work.interrupted:
        raise StageInterruptedError("Surface ownership interrupted")
    return Partition(tuple(surfaces)).metadata()
