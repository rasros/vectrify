"""Bounded whole-primitive removal fields around an original source chain.

Two short transverse cuts delimit opposing arcs of one original closed contour.
Copy those complete arcs instead of reparameterizing a partial curved boundary.
The resulting guide is only a raw-source search domain: neither the field nor
its endpoint midpoints prove a stroke, physical ports, native alpha or ownership.
This constructor is not enabled by the ordinary or attached production search.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np
import pathops
from scipy.spatial import KDTree

from vectrify.document import Geometry
from vectrify.document.holes import reversed_subpath
from vectrify.document.join import curve_path, path_style, transformed_geometry
from vectrify.document.redraw import root_matrix
from vectrify.refine.cel_plan.filled_bands import MAX_NODES, MAX_WIDTH, _check
from vectrify.refine.cel_plan.source_bands import MAX_EXTENT, MAX_POINTS
from vectrify.refine.cel_plan.source_slices import MAX_COMPOUND_NODES

MAX_CHORDS = 256
MAX_FIELDS = 32
GUIDE_POINTS = 128


@dataclass(frozen=True)
class SourceSpan:
    field: Geometry
    retained: Geometry
    guide: np.ndarray


def _arc(nodes, start, end):
    """Complete directed edges, including the original implicit straight Z."""
    result = []
    index = start
    while index != end:
        index = (index + 1) % len(nodes)
        node = nodes[index]
        result.append(
            replace(node, command="L", values=node.endpoint) if not index else node
        )
    return tuple(result)


def _sample(start, commands):
    points = [np.asarray(start)]
    for node in commands:
        if node.command == "L":
            points.append(np.asarray(node.endpoint))
        else:
            a = points[-1]
            b, c, d = np.asarray(node.values).reshape(3, 2)
            t = np.linspace(0, 1, 33)[1:, None]
            points.extend(
                (1 - t) ** 3 * a
                + 3 * (1 - t) ** 2 * t * b
                + 3 * (1 - t) * t**2 * c
                + t**3 * d
            )
    points = np.asarray(points)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(arc) > 1e-10]
    if arc[-1] <= 1e-10:
        return None
    at = np.linspace(0, arc[-1], GUIDE_POINTS)
    return np.column_stack([np.interp(at, arc[keep], points[keep, i]) for i in (0, 1)])


def discover(document, oid, observed, former_field, work):
    """Exact geometric alternatives containing the complete former field.

    A caller must measure the entire extended raw-source guide, protect the
    unchanged original source bank, and prove actual native construction and
    complete component acceptance. Thin geometry cannot stand in for ink proof.
    """
    _check(work)
    original = document.geometry_for(oid)
    if (
        observed.anchors is None
        or not 4 <= len(observed.anchors) <= MAX_POINTS
        or not np.isfinite(observed.anchors).all()
        or np.array_equal(observed.anchors[0], observed.anchors[-1])
        or observed.gaps.any()
        or not len(observed.qualified)
        or observed.qualified.mean() < 0.6
        or path_style(document, document.element(oid))["fill-rule"] != "nonzero"
        or any(not s.closed for s in original.subpaths)
        or sum(len(s.nodes) for s in original.subpaths) > MAX_COMPOUND_NODES
        or not former_field.subpaths
        or any(not s.closed for s in former_field.subpaths)
        or sum(len(s.nodes) for s in former_field.subpaths) > MAX_NODES
    ):
        return ()
    old, required = curve_path(original), curve_path(former_field)
    if not required.area or pathops.op(required, old, pathops.PathOp.DIFFERENCE).area:
        return ()
    frame = root_matrix(document, oid)
    native = transformed_geometry(original, frame)
    required_bounds = np.asarray(
        curve_path(transformed_geometry(former_field, frame)).bounds
    )
    results = {}
    for sub, world in zip(original.subpaths, native.subpaths, strict=True):
        _check(work)
        nodes = sub.nodes
        vertices = np.asarray([n.endpoint for n in world.nodes])
        nearby = np.flatnonzero(
            (
                (vertices >= required_bounds[:2] - MAX_EXTENT)
                & (vertices <= required_bounds[2:] + MAX_EXTENT)
            ).all(axis=1)
        )
        if len(nearby) < 4:
            continue
        pairs = KDTree(vertices[nearby]).query_pairs(MAX_WIDTH, output_type="ndarray")
        if len(pairs) > MAX_CHORDS:
            continue
        chords = sorted(tuple(map(int, nearby[pair])) for pair in pairs)
        for first, second in combinations(chords, 2):
            _check(work)
            a, b = first
            c, d = second
            if len({a, b, c, d}) != 4:
                continue
            if a < c < d < b:
                i, j, k, end_b = a, c, d, b
            elif a < b < c < d:
                i, j, k, end_b = b, c, d, a
            else:
                continue
            edges_a, edges_b = _arc(nodes, i, j), _arc(nodes, k, end_b)
            if not 4 <= 2 + len(edges_a) + len(edges_b) <= MAX_NODES:
                continue
            field_nodes = (
                replace(nodes[i], command="M", values=nodes[i].endpoint),
                *edges_a,
                replace(nodes[k], command="L", values=nodes[k].endpoint),
                *edges_b,
            )
            field = Geometry(
                "primitive-field",
                (
                    replace(
                        sub,
                        id="primitive-field",
                        nodes=tuple(
                            replace(n, id=f"field-{index}")
                            for index, n in enumerate(field_nodes)
                        ),
                    ),
                ),
            )
            world_field = transformed_geometry(field, frame)
            path = curve_path(field)
            bounds = np.asarray(curve_path(world_field).bounds)
            if (
                not path.area
                or not np.isfinite(bounds).all()
                or (bounds[2:] - bounds[:2]).max() > MAX_EXTENT
                or pathops.op(required, path, pathops.PathOp.DIFFERENCE).area
                or pathops.op(path, old, pathops.PathOp.DIFFERENCE).area
            ):
                continue
            rail_a = _sample(world.nodes[i].endpoint, _arc(world.nodes, i, j))
            rail_b = _sample(world.nodes[k].endpoint, _arc(world.nodes, k, end_b))
            if rail_a is None or rail_b is None:
                continue
            guide = (rail_a[::-1] + rail_b) / 2
            length = float(np.linalg.norm(np.diff(guide, axis=0), axis=1).sum())
            if not length or abs(curve_path(world_field).area) / length > MAX_WIDTH / 2:
                continue
            remainder = pathops.op(old, path, pathops.PathOp.DIFFERENCE)
            retained = replace(
                original,
                subpaths=(*original.subpaths, reversed_subpath(field.subpaths[0])),
            )
            if (
                not remainder.area
                or sum(len(s.nodes) for s in retained.subpaths) > MAX_COMPOUND_NODES
                or pathops.op(curve_path(retained), remainder, pathops.PathOp.XOR).area
                or pathops.op(
                    curve_path(retained), path, pathops.PathOp.INTERSECTION
                ).area
            ):
                continue
            guide.flags.writeable = False
            key = field.path_data()
            results.setdefault(
                key, (abs(path.area), SourceSpan(field, retained, guide))
            )
            if len(results) > MAX_FIELDS:
                del results[max(results, key=lambda key: (results[key][0], key))]
    _check(work)
    return tuple(
        results[key][1]
        for key in sorted(results, key=lambda key: (results[key][0], key))
    )
