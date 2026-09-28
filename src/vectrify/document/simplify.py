"""Tolerance-bounded contour reduction with optional corner preservation.

Fit each contour independently; never merge islands or remove holes. Compare
flattened curves in both directions and retain original topology when a proposed
fit would cross itself or change a contour's containment/contact relationships.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from itertools import pairwise

import numpy as np
from shapely import STRtree, hausdorff_distance, make_valid
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

from vectrify.document.hit_test import _flatten
from vectrify.document.model import DocumentError, Geometry, PathNode, Subpath, new_id
from vectrify.refine.samvg import _fit_cubic


@dataclass(frozen=True)
class SimplifyOptions:
    tolerance: float = 1.0
    corners: bool = True

    def __post_init__(self):
        if not math.isfinite(self.tolerance) or not 0.01 <= self.tolerance <= 100:
            raise DocumentError(
                "Contour tolerance must be between 0.01 and 100 SVG units"
            )
        if type(self.corners) is not bool:
            raise DocumentError("Keep sharp corners must be on or off")


def sampled(subpath: Subpath, tolerance: float):
    points = [subpath.nodes[0].endpoint]
    endpoints = {0: subpath.nodes[0]}
    for node in subpath.nodes[1:]:
        if node.command == "C":
            v = node.values
            points.extend(
                _flatten(
                    (points[-1], (v[0], v[1]), (v[2], v[3]), node.endpoint), tolerance
                )
            )
        else:
            points.append(node.endpoint)
        endpoints[len(points) - 1] = node
    if subpath.closed and points[-1] != points[0]:
        points.append(points[0])
    return points, endpoints


def _junctions(line) -> Counter[tuple[float, float]]:
    if line.is_empty or line.length == 0:
        return Counter()
    noded = unary_union(line)
    pieces = list(getattr(noded, "geoms", (noded,)))
    return Counter(
        (float(p[0]), float(p[1]))
        for piece in pieces
        for p in (piece.coords[0], piece.coords[-1])
    )


def _faces(polygon):
    repaired = make_valid(polygon)

    def counts(shape):
        if shape.geom_type == "Polygon":
            return (1, len(shape.interiors), 0)
        if hasattr(shape, "geoms"):
            values = [counts(part) for part in shape.geoms]
            return tuple(sum(v[i] for v in values) for i in range(3))
        return (0, 0, 1)

    return repaired, counts(repaired)


def _reduce(subpath: Subpath, options: SimplifyOptions, *, filled: bool) -> Subpath:
    tolerance = options.tolerance
    points, endpoints = sampled(subpath, tolerance / 16)
    if len(points) < 3:
        return subpath
    original_line = LineString(points)
    junctions = None
    occurrences = defaultdict(list)
    if not original_line.is_simple:
        junctions = _junctions(original_line)
        for i, point in enumerate(points):
            occurrences[point].append(i)
        # Crossing edges without an explicit shared endpoint are ambiguous.
        # Existing traced self-contacts, however, can be held exactly in place.
        if not set(junctions) <= occurrences.keys():
            return subpath
    source = np.asarray(points, dtype=np.float64)
    distances = np.linalg.norm(np.diff(source, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(distances)))
    anchors = {0, len(points) - 1}
    if junctions is not None:
        for point in junctions:
            for i in occurrences[point]:
                anchors.update((max(0, i - 1), i, min(len(points) - 1, i + 1)))
    for index, node in endpoints.items():
        if node.pinned:
            anchors.add(index)
        if options.corners and 0 < index < len(points) - 1:
            # A wider neighborhood ignores tiny zigzags when deciding which
            # original endpoints are genuine corners worth keeping exactly.
            reach = tolerance * 3
            left = max(0, min(index - 1, int(np.searchsorted(arc, arc[index] - reach))))
            right = min(
                len(points) - 1,
                max(index + 1, int(np.searchsorted(arc, arc[index] + reach))),
            )
            incoming, outgoing = (
                source[index] - source[left],
                source[right] - source[index],
            )
            product = np.linalg.norm(incoming) * np.linalg.norm(outgoing)
            if product and np.dot(incoming, outgoing) / product < math.cos(math.pi / 4):
                anchors.add(index)
    if subpath.closed:
        # A complete loop has identical endpoints; split it before fitting.
        anchors.add(int(np.argmax(np.linalg.norm(source - source[0], axis=1))))
    precision = max(0, math.ceil(-math.log10(tolerance / 128)))

    def endpoint(index: int, command: str, values: tuple[float, ...]):
        existing = endpoints.get(index)
        # The implicit closure gets a new identity only if it needs a curve.
        if index == len(points) - 1 and existing and existing.id == subpath.nodes[0].id:
            existing = None
        return PathNode(
            existing.id if existing else new_id("node"),
            command,
            values,
            existing.pinned if existing else False,
        )

    def fit(first: int, last: int, depth: int = 0) -> list[PathNode]:
        sample = source[first : last + 1]
        end = (float(sample[-1, 0]), float(sample[-1, 1]))
        line = LineString([sample[0], sample[-1]])
        original = LineString(sample)
        # A reduced straight span is smaller than a cubic and avoids turning
        # genuinely straight edges into barely curved ones.
        if original.hausdorff_distance(line) <= tolerance * 0.8:
            return [endpoint(last, "L", end)]
        if len(sample) >= 4:
            a, b = _fit_cubic(sample)
            controls = tuple(
                (round(float(p[0]), precision), round(float(p[1]), precision))
                for p in (a, b)
            )
            candidate = LineString(
                [
                    tuple(sample[0]),
                    *_flatten(
                        ((float(sample[0, 0]), float(sample[0, 1])), *controls, end),
                        tolerance / 16,
                    ),
                ]
            )
            if (
                candidate.is_simple
                and hausdorff_distance(original, candidate, densify=0.25)
                <= tolerance * 0.8
            ):
                return [endpoint(last, "C", (*controls[0], *controls[1], *end))]
        if last - first <= 1 or depth >= 24:
            return [
                endpoint(i, "L", tuple(float(v) for v in source[i]))
                for i in range(first + 1, last + 1)
            ]
        # Split near the largest deviation from the chord; a midpoint avoids
        # unbalanced recursion on long, nearly straight digitized outlines.
        from shapely import points as shapely_points

        deviation = line.distance(shapely_points(sample))
        split = int(np.argmax(deviation))
        if split < len(sample) // 4 or split > 3 * len(sample) // 4:
            split = len(sample) // 2
        split = max(1, min(len(sample) - 2, split)) + first
        return fit(first, split, depth + 1) + fit(split, last, depth + 1)

    ordered = sorted(anchors)
    nodes = [subpath.nodes[0]]
    for first, last in pairwise(ordered):
        nodes.extend(fit(first, last))
    # Keep a pinned closing node; otherwise Z already encodes a straight close.
    if (
        subpath.closed
        and nodes[-1].command == "L"
        and nodes[-1].endpoint == nodes[0].endpoint
        and not nodes[-1].pinned
    ):
        nodes.pop()
    result = replace(subpath, nodes=tuple(nodes))
    old_cost = sum(len(n.values) for n in subpath.nodes)
    new_cost = sum(len(n.values) for n in result.nodes)
    if new_cost >= old_cost or len(result.nodes) > len(subpath.nodes):
        return subpath
    # Keep the original if reducing command count would inflate path data.
    if len(Geometry("size", (result,)).path_data()) >= len(
        Geometry("size", (subpath,)).path_data()
    ):
        return subpath
    reduced, _ = sampled(result, tolerance / 16)
    before, after = LineString(points), LineString(reduced)
    if (
        not after.is_simple if junctions is None else _junctions(after) != junctions
    ) or hausdorff_distance(before, after, densify=0.25) > tolerance * 0.9:
        return subpath
    if subpath.closed or filled:
        if len(reduced) < 3:
            return subpath
        old, new = Polygon(points), Polygon(reduced)
        if junctions is None:
            if not old.is_valid or not new.is_valid or new.area == 0:
                return subpath
        elif _faces(old)[1] != _faces(new)[1]:
            return subpath
        if old.exterior.is_ccw != new.exterior.is_ccw:
            return subpath
    return result


def simplify_geometry(
    geometry: Geometry, options: SimplifyOptions, *, filled: bool = True
) -> Geometry:
    proposals = [_reduce(s, options, filled=filled) for s in geometry.subpaths]
    before, after = [], []
    for old, new in zip(geometry.subpaths, proposals, strict=True):
        old_points, _ = sampled(old, options.tolerance / 16)
        new_points, _ = sampled(new, options.tolerance / 16)

        def footprint(points, closed=old.closed):
            if len(points) < 2:
                return LineString()
            return (
                make_valid(Polygon(points))
                if (closed or filled) and len(points) >= 3
                else LineString(points)
            )

        before.append(footprint(old_points))
        after.append(footprint(new_points))
    # Relations must remain stable between every potentially affected pair.
    envelopes = [
        box(*a.envelope.union(b.envelope).bounds) if not a.is_empty else b.envelope
        for a, b in zip(before, after, strict=True)
    ]
    tree = STRtree(envelopes)
    pairs = {
        (i, int(j)) for i, e in enumerate(envelopes) for j in tree.query(e) if i < j
    }

    def relation(a, b):
        return a.disjoint(b), a.contains(b), b.contains(a), a.touches(b)

    while True:
        rejected = set()
        for i, j in pairs:
            if (
                proposals[i] == geometry.subpaths[i]
                and proposals[j] == geometry.subpaths[j]
            ):
                continue
            if relation(before[i], before[j]) != relation(after[i], after[j]):
                rejected.update((i, j))
        if not rejected:
            break
        for i in rejected:
            proposals[i], after[i] = geometry.subpaths[i], before[i]
    return replace(geometry, subpaths=tuple(proposals))
