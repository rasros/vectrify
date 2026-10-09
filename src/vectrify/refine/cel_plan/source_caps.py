"""Recover truncated stroke terminals within their own measured source interval.

Only original internal gaps authorize a different cap. Physical terminals and
junctions stay exact. This builds one complete geometry competitor; the caller
must prove the complete painted edit, current ownership and actual native gaps.
It never connects across a gap or uses a human redraw.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from vectrify.document import Editor, Selection
from vectrify.document.join import path_style
from vectrify.document.model import paint_server
from vectrify.document.redraw import root_matrix
from vectrify.document.topology import inverse_matrix
from vectrify.refine.cel_plan.model import StageInterruptedError

MAX_PATHS = 32
MAX_NODES = 1024
MAX_MOVEMENT = 3.0
MATCH_TOLERANCE = 1e-6


def _check(work):
    if work.interrupted:
        raise StageInterruptedError("Source cap reconstruction interrupted")


def _native(points, matrix):
    a, b, c, d, e, f = matrix
    return np.asarray(points) @ np.array([[a, b], [c, d]]) + (e, f)


def _terminal(observed, index, other, radius, margin):
    """An outward qualified sample, separated from its own original gap."""
    points, qualified, gaps = observed.points, observed.qualified, observed.gaps
    low, high = sorted((index, other))
    if low == high or gaps[low : high + 1].any():
        return None
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if index < other:
        before = np.flatnonzero(gaps[:index])
        if not len(before):
            return None
        gap = int(before[-1])
        candidates = np.flatnonzero(
            qualified[gap + 1 : index]
            & (distance[gap + 1 : index] >= distance[gap] + radius + margin)
        )
        new = gap + 1 + int(candidates[0]) if len(candidates) else index
    else:
        after = np.flatnonzero(gaps[index + 1 :])
        if not len(after):
            return None
        gap = index + 1 + int(after[0])
        candidates = np.flatnonzero(
            qualified[index + 1 : gap]
            & (distance[index + 1 : gap] <= distance[gap] - radius - margin)
        )
        new = index + 1 + int(candidates[-1]) if len(candidates) else index
    # Existing original terminals/junctions are never moved. All traversed
    # source samples must remain inside this one non-gap interval.
    if (
        new == index
        or index in (0, len(points) - 1)
        or np.linalg.norm(points[new] - points[index]) > MAX_MOVEMENT
    ):
        return None
    return new


class SourceCaps:
    """Bounded source-only cap alternatives for an explicitly affected region."""

    def __init__(self, guard):
        self.guard = guard

    def extend(self, document, ids, bounds, work, *, margin=0.5):
        """Return an atomic document and source witnesses, or no competitor.

        Both existing endpoints must match the same original physical profile.
        An ambiguous match excludes the whole subpath. Only endpoint values
        change: paint, width, frame, other controls and original junctions stay
        exact. A margin is a proposal parameter, not permission to fill a gap.
        """
        _check(work)
        bounds = np.asarray(bounds, float)
        if (
            bounds.shape != (4,)
            or not np.isfinite(bounds).all()
            or np.any(bounds[2:] < bounds[:2])
            or not np.isfinite(margin)
            or not 0 <= margin <= 3
        ):
            raise ValueError("Source caps require finite bounds and a bounded margin")
        profiles = tuple(
            (index, self.guard.source_breaks(profile, work=work))
            for index, profile in enumerate(self.guard.original_profiles(work=work))
        )
        changes, witnesses = {}, []
        paths = nodes_seen = 0
        for oid in dict.fromkeys(ids):
            _check(work)
            element = document.element(oid)
            if element.tag != "path":
                continue
            style = path_style(document, element)
            if (
                style["fill"] != "none"
                or style["stroke"] == "none"
                or paint_server(style["stroke"]) is not None
                or style["stroke-linecap"] not in {"round", "butt"}
                or style["stroke-linejoin"] != "round"
                or any(a.locks for a in document.ancestry(oid))
                or any(
                    a.get(key, "none") != "none"
                    for a in document.ancestry(oid)
                    for key in (
                        "clip-path",
                        "mask",
                        "filter",
                        "marker-start",
                        "marker-mid",
                        "marker-end",
                        "stroke-dasharray",
                        "vector-effect",
                    )
                )
            ):
                continue
            width = float(style["stroke-width"])
            matrix = root_matrix(document, oid)
            linear = np.array(matrix[:4]).reshape(2, 2).T
            if (
                not np.isfinite(width)
                or width <= 0
                or abs(np.linalg.det(linear)) < 1e-12
            ):
                continue
            radius = (
                width * float(np.linalg.norm(linear, ord=2)) / 2
                if style["stroke-linecap"] == "round"
                else 0.0
            )
            geometry = document.geometry_for(oid)
            paths += 1
            nodes_seen += sum(len(sub.nodes) for sub in geometry.subpaths)
            if paths > MAX_PATHS or nodes_seen > MAX_NODES:
                return None
            subs = list(geometry.subpaths)
            for sub_index, sub in enumerate(subs):
                _check(work)
                if sub.closed or len(sub.nodes) < 2:
                    continue
                endpoints = _native(
                    (sub.nodes[0].endpoint, sub.nodes[-1].endpoint), matrix
                )
                inside = ((endpoints >= bounds[:2]) & (endpoints <= bounds[2:])).all(
                    axis=1
                )
                if not inside.any():
                    continue
                matches = []
                for profile_index, observed in profiles:
                    _check(work)
                    if observed is None:
                        continue
                    distances = np.linalg.norm(
                        observed.points[:, None, :] - endpoints[None, :, :], axis=2
                    )
                    positions = distances.argmin(axis=0)
                    if (distances[positions, np.arange(2)] <= MATCH_TOLERANCE).all():
                        a, b = map(int, positions)
                        if (
                            a != b
                            and not observed.gaps[min(a, b) : max(a, b) + 1].any()
                        ):
                            matches.append((profile_index, observed, a, b))
                if len(matches) != 1:
                    continue
                profile_index, observed, a, b = matches[0]
                new_nodes = list(sub.nodes)
                for side, index, other in ((0, a, b), (-1, b, a)):
                    _check(work)
                    if not inside[side] or new_nodes[side].pinned:
                        continue
                    new = _terminal(observed, index, other, radius, margin)
                    if new is None:
                        continue
                    point = observed.points[new]
                    if not ((point >= bounds[:2]) & (point <= bounds[2:])).all():
                        continue
                    local = _native(point, inverse_matrix(matrix))
                    node = new_nodes[side]
                    new_nodes[side] = replace(
                        node, values=(*node.values[:-2], *map(float, local))
                    )
                    witnesses.append(
                        {
                            "id": oid,
                            "contour": sub_index,
                            "end": "start" if side == 0 else "end",
                            "profile": profile_index,
                            "from_sample": index,
                            "to_sample": new,
                            "from": endpoints[side].tolist(),
                            "to": point.tolist(),
                            "margin": margin,
                        }
                    )
                subs[sub_index] = replace(sub, nodes=tuple(new_nodes))
            if tuple(subs) != geometry.subpaths:
                changes[oid] = replace(geometry, subpaths=tuple(subs))
        _check(work)
        if not changes:
            return None
        editor = Editor(document, selection=Selection(whole_document=True))
        with editor.transaction("Recover source-supported stroke caps") as tx:
            for oid, geometry in changes.items():
                _check(work)
                tx.replace_geometry(oid, geometry)
        _check(work)
        return editor.snapshot.document, tuple(witnesses)
