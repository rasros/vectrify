"""Geometric adjoining-face fields inside a complete primitive removal span.

An original material's complete shared boundary selects an opposing-rail field.
Directed winding partitions the restoration without rewriting the rest of the
span's curve commands. This supplies construction alternatives only; it proves
neither source paint, native alpha, layer order nor original ownership. Ordinary
and experimental production search do not enable this constructor yet.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import islice

import numpy as np
import pathops

from vectrify.document import Geometry
from vectrify.document.hit_test import multiply
from vectrify.document.holes import reversed_subpath
from vectrify.document.join import curve_path, path_style, transformed_geometry
from vectrify.document.redraw import root_matrix
from vectrify.document.topology import inverse_matrix
from vectrify.refine.cel_plan.filled_bands import MAX_NODES, _check
from vectrify.refine.cel_plan.paint_continuation import _paint, _supported
from vectrify.refine.cel_plan.source_bands import MAX_EXTENT
from vectrify.refine.cel_plan.source_spans import discover as primitive_spans

MAX_MATERIALS = 16
MAX_OBSERVATIONS = 32
MAX_SOURCE_POINTS = 16_384
MAX_FACES = 32
MIN_SHARED_LENGTH = 8


@dataclass(frozen=True)
class SourceFace:
    material: str
    source_index: int
    field: Geometry
    remainder: Geometry
    shared_edges: int


def _edges(geometry: Geometry) -> set[tuple[tuple[float, float], ...]]:
    """Complete native C/L/Z edges, independent of direction and SVG start."""
    result: set[tuple[tuple[float, float], ...]] = set()
    for sub in geometry.subpaths:
        start = sub.nodes[0].endpoint
        commands = list(sub.nodes[1:])
        if sub.closed:
            commands.append(replace(sub.nodes[0], command="L"))
        for node in commands:
            points: tuple[tuple[float, float], ...] = (
                start,
                *zip(node.values[::2], node.values[1::2], strict=True),
            )
            if (
                sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
                >= MIN_SHARED_LENGTH
            ):
                result.add(min(points, points[::-1]))
            start = node.endpoint
    return result


def discover(document, oid, span, source_fields, material_ids, base_material, work):
    """Bounded opposing-rail interpretations for one adjoining original face.

    The caller supplies original source observations with their complete former
    fields, and material identities selected through the original partition.
    Complete candidate acceptance must
    separately declare new restoration surfaces and preserve every original
    physical observation. Exact geometric partition is not native coverage.
    """
    _check(work)
    sources = tuple(islice(source_fields, MAX_OBSERVATIONS + 1))
    materials = tuple(islice(material_ids, MAX_MATERIALS + 1))
    if (
        not sources
        or not materials
        or len(sources) > MAX_OBSERVATIONS
        or len(materials) > MAX_MATERIALS
        or sum(len(observed.points) for observed, _ in sources) > MAX_SOURCE_POINTS
        or not _supported(document, oid)
        or not _supported(document, base_material)
        or not _paint(document, path_style(document, document.element(base_material)))
        or document.ancestry(oid)[-2].id != document.ancestry(base_material)[-2].id
        or not span.field.subpaths
        or any(not sub.closed for sub in span.field.subpaths)
        or sum(len(sub.nodes) for sub in span.field.subpaths) > MAX_NODES
    ):
        return ()
    full = curve_path(span.field)
    if (
        not full.area
        or pathops.op(
            full, curve_path(document.geometry_for(oid)), pathops.PathOp.DIFFERENCE
        ).area
    ):
        return ()
    frame = root_matrix(document, oid)
    native = curve_path(transformed_geometry(span.field, frame))
    bounds = np.asarray(native.bounds)
    if not np.isfinite(bounds).all() or (bounds[2:] - bounds[:2]).max() > MAX_EXTENT:
        return ()
    options = []
    for index, (observed, former) in enumerate(sources):
        _check(work)
        points = observed.points[observed.qualified]
        nearby = ((points >= bounds[:2]) & (points <= bounds[2:])).all(axis=1)
        if not any(native.contains(tuple(point)) for point in points[nearby]):
            continue
        options.append((index, primitive_spans(document, oid, observed, former, work)))
    results = {}
    for material in sorted(set(materials) - {oid, base_material}):
        _check(work)
        if (
            not _supported(document, material)
            or not _paint(document, path_style(document, document.element(material)))
            or document.ancestry(material)[-2].id != document.ancestry(oid)[-2].id
        ):
            continue
        original = document.geometry_for(material)
        if any(not sub.closed for sub in original.subpaths):
            continue
        material_frame = root_matrix(document, material)
        edges = _edges(transformed_geometry(original, material_frame))
        coverage = curve_path(
            transformed_geometry(
                original, multiply(inverse_matrix(frame), material_frame)
            ),
            path_style(document, document.element(material))["fill-rule"],
        )
        for index, alternatives in options:
            for alternative in alternatives:
                _check(work)
                field = alternative.field
                shared = len(_edges(transformed_geometry(field, frame)) & edges)
                selected = curve_path(field)
                if (
                    not shared
                    or pathops.op(selected, full, pathops.PathOp.DIFFERENCE).area
                    or pathops.op(selected, coverage, pathops.PathOp.INTERSECTION).area
                ):
                    continue
                remainder = None
                # The full span may use the opposite serialization direction.
                # Prove subtraction geometrically rather than assuming its sign.
                for signed in (
                    tuple(reversed_subpath(sub) for sub in field.subpaths),
                    field.subpaths,
                ):
                    candidate = replace(
                        span.field, subpaths=(*span.field.subpaths, *signed)
                    )
                    remaining = curve_path(candidate)
                    if (
                        remaining.area
                        and sum(len(sub.nodes) for sub in candidate.subpaths)
                        <= 2 * MAX_NODES
                        and not pathops.op(
                            remaining, selected, pathops.PathOp.INTERSECTION
                        ).area
                        and not pathops.op(
                            pathops.op(remaining, selected, pathops.PathOp.UNION),
                            full,
                            pathops.PathOp.XOR,
                        ).area
                    ):
                        remainder = candidate
                        break
                if remainder is None:
                    continue
                key = (material, field.path_data())
                results.setdefault(
                    key,
                    (
                        abs(selected.area),
                        SourceFace(material, index, field, remainder, shared),
                    ),
                )
                if len(results) > MAX_FACES:
                    del results[max(results, key=lambda key: (results[key][0], key))]
    _check(work)
    return tuple(
        results[key][1]
        for key in sorted(results, key=lambda key: (results[key][0], key))
    )
