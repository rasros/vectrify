"""Bounded export-chain permissions with an exact protected complement.

Only explicitly recorded interior segments are refinable. All other segments,
including implicit canvas/contour closures, remain exact. Geometry fingerprints
prevent stale permissions from authorizing edits after a union or replacement.
These values belong to planning metadata, not the editor/project format.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from vectrify.document import Document
from vectrify.document.redraw import root_matrix
from vectrify.refine import shared
from vectrify.refine.cel_plan.model import Work

MAX_SEGMENTS = 32_768
MAX_CHAINS = 8_192
MAX_SOURCE_POINTS = 4_096
MAX_PATH_NODES = 512
TOKEN = re.compile(r"[MLCZ]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def fingerprint(data: str) -> str:
    tokens = TOKEN.findall(data)
    canonical = " ".join(
        t if t in {"M", "L", "C", "Z"} else repr(float(t)) for t in tokens
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def key(points) -> tuple[tuple[float, float], ...]:
    values = tuple((float(p[0]), float(p[1])) for p in points)
    return min(values, values[::-1])


def segments(document: Document, oid: str):
    for sub in document.geometry_for(oid).subpaths:
        ring = shared._ring(sub)
        for i in shared._indices(ring):
            yield (
                ring.ends[i - 1].id,
                ring.ends[i].id,
                key(shared._controls(ring, i)),
            )


@dataclass(frozen=True)
class Hold:
    protected: tuple[tuple[tuple[float, float], ...], ...]
    endpoints: frozenset[str]
    matrix: tuple[float, ...]

    def intact(self, document: Document, oid: str) -> bool:
        if tuple(root_matrix(document, oid)) != self.matrix:
            return False
        current = Counter(value for _a, _b, value in segments(document, oid))
        return not Counter(self.protected) - current


def bind(document: Document, oid: str, metadata: dict | None) -> Hold | None:
    if not metadata or metadata.get("version") != 1:
        return None
    record = metadata.get("paths", {}).get(oid)
    if record is None:
        return None
    geometry = document.geometry_for(oid)
    if sum(len(s.nodes) for s in geometry.subpaths) > MAX_PATH_NODES:
        return None
    if fingerprint(geometry.path_data()) != record["geometry"]:
        return None
    if tuple(root_matrix(document, oid)) != tuple(record.get("matrix", ())):
        return None
    free = Counter(key(value) for value in record["free"])
    current = list(segments(document, oid))
    if free - Counter(value for _a, _b, value in current):
        return None
    protected, held = [], set()
    for a, b, value in current:
        if free[value]:
            free[value] -= 1
        else:
            protected.append(value)
            held.update((a, b))
    # Explicit drawn closures have a second node at the serialization start.
    # Hold that endpoint too; the implicit segment itself is checked above.
    points = {tuple(point) for value in protected for point in (value[0], value[-1])}
    held.update(
        n.id for sub in geometry.subpaths for n in sub.nodes if n.endpoint in points
    )
    return Hold(tuple(protected), frozenset(held), tuple(record["matrix"]))


def refresh(
    metadata: dict | None, before: Document, after: Document, ids
) -> dict | None:
    """Fork permissions after a proved geometry edit, never mutate a sibling."""
    if not metadata or metadata.get("version") != 1:
        return metadata
    paths = dict(metadata["paths"])
    for oid in ids:
        if oid not in paths:
            continue
        hold = bind(before, oid, metadata)
        if hold is None or not hold.intact(after, oid):
            paths.pop(oid)
            continue
        protected = Counter(hold.protected)
        free = []
        for _a, _b, value in segments(after, oid):
            if protected[value]:
                protected[value] -= 1
            else:
                free.append(value)
        paths[oid] = {
            **paths[oid],
            "geometry": fingerprint(after.geometry_for(oid).path_data()),
            "free": free,
        }
    return {**metadata, "paths": paths}


def discard(metadata: dict | None, ids) -> dict | None:
    if not metadata or metadata.get("version") != 1:
        return metadata
    return {
        **metadata,
        "paths": {oid: r for oid, r in metadata["paths"].items() if oid not in ids},
    }


class Chains:
    """Record export permissions within independent chain/segment caps."""

    def __init__(self):
        self.records: list[dict] = []
        self.count = 0
        self.omitted = 0

    def add(self, points: np.ndarray, nodes, regions):
        if (
            len(self.records) >= MAX_CHAINS
            or self.count + len(nodes) > MAX_SEGMENTS
            or len(points) > MAX_SOURCE_POINTS
        ):
            self.omitted += 1
            return
        # Match the established exporter's two-decimal serialization exactly.
        start = tuple(float(f"{v:.2f}") for v in points[0])
        free = []
        for _command, values in nodes:
            values = tuple(float(f"{v:.2f}") for v in values)
            end = values[-2:]
            free.append(
                key((start, *tuple(zip(values[::2], values[1::2], strict=True))))
            )
            start = end
        raw = np.asarray(points, dtype=np.float64)
        identity = hashlib.sha256(min(raw.tobytes(), raw[::-1].tobytes())).hexdigest()
        self.records.append({"id": identity, "regions": tuple(regions), "free": free})
        self.count += len(free)

    def metadata(self, outlines, eligible, ownership, matrix, work: Work) -> dict:
        paths: dict[str, dict] = {}
        members = {s["id"]: s["members"] for s in ownership.get("surfaces", ())}
        for record in self.records:
            if work.interrupted:
                break
            for region in record["regions"]:
                oid = f"cel-fill-{region}"
                if region not in eligible or oid not in members:
                    continue
                if sum(outlines[region].count(c) for c in "MLC") > MAX_PATH_NODES:
                    continue
                path = paths.setdefault(
                    oid,
                    {
                        "geometry": fingerprint(outlines[region]),
                        "matrix": matrix,
                        "free": [],
                        "chains": [],
                    },
                )
                path["free"].extend(record["free"])
                path["chains"].append({"id": record["id"], "members": members[oid]})
        # Nodes and emitted segment counts differ because of closure commands.
        # Binding applies the exact node cap before these permissions are used.
        return {"version": 1, "paths": paths, "omitted_chains": self.omitted}
