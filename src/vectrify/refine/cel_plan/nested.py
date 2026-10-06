"""Continue material beneath fully enclosed, opaque, independently owned marks.

An RGB cavity is not an alpha hole. Source ownership, native alpha, current
paint and geometry must agree before its existing mark can stay above a base.
The declaration of hidden coverage never substitutes for a containment proof.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pathops
from cairosvg.colors import color
from scipy.ndimage import binary_fill_holes

from vectrify.document import Geometry
from vectrify.document.hit_test import multiply
from vectrify.document.join import (
    curve_path,
    path_style,
    transformed_geometry,
    union_geometry,
)
from vectrify.document.model import paint_server
from vectrify.document.paint import gradient_stops
from vectrify.document.redraw import root_matrix
from vectrify.document.topology import inverse_matrix
from vectrify.refine.cel_plan.local import Box
from vectrify.refine.cel_plan.model import Evidence, Graph, Work
from vectrify.refine.cel_plan.search import State

MAX_MARKS = 64
MAX_NODES = 6_000


@dataclass(frozen=True)
class Enclosed:
    mask: np.ndarray
    ids: tuple[str, ...] = ()
    members: tuple[int, ...] = ()
    geometries: tuple[Geometry, ...] = ()
    rules: tuple[str, ...] = ()

    def contains(self, shape: Geometry, work: Work) -> bool:
        filled = curve_path(shape, "nonzero")
        for geometry, rule in zip(self.geometries, self.rules, strict=True):
            if work.interrupted:
                return False
            outside = pathops.op(
                curve_path(geometry, rule), filled, pathops.PathOp.DIFFERENCE
            )
            if abs(outside.area) > 1e-8:
                return False
        return not work.interrupted

    def continued(self, shape: Geometry) -> Geometry:
        return union_geometry(
            [shape, *self.geometries],
            [{"fill-rule": "nonzero"}, *({"fill-rule": rule} for rule in self.rules)],
        )


def enclosed(
    evidence: Evidence,
    graph: Graph,
    state: State,
    ids: tuple[str, ...],
    box: Box,
    own: np.ndarray,
    survivor: str,
    work: Work,
    *,
    rejections: dict[str, int] | None = None,
) -> Enclosed | None:
    """Bounded source and paint proof; final geometry is checked separately."""

    def reject(reason):
        if rejections is not None:
            rejections[reason] = rejections.get(reason, 0) + 1
        return

    if work.interrupted:
        return None
    filled = np.asarray(binary_fill_holes(own), dtype=bool)
    inside = filled & ~own
    if not inside.any():
        return Enclosed(own)
    partition = state.partition
    if partition is None:
        return reject("missing-ownership")
    if (inside & evidence.empty[box.slices]).any():
        return reject("alpha-hole")
    owners = partition.owners
    regions = tuple(int(i) for i in np.unique(graph.labels[box.slices][inside]))
    if any(i not in owners for i in regions):
        return reject("missing-mark-owner")
    inner = tuple(sorted({owners[i] for i in regions}))
    if len(inner) > MAX_MARKS or set(inner).intersection(ids):
        return reject("mark-count-or-self-owner")
    surfaces = {s.id: s for s in partition.surfaces}
    members = tuple(sorted(i for oid in inner for i in surfaces[oid].members))
    # A merged surface may also own a distant mark; it cannot be moved as an
    # enclosed child just because one of its source regions is inside this hole.
    if members != regions or sum(graph.regions[i].area for i in members) != int(
        inside.sum()
    ):
        return reject("partially-enclosed-owner")
    document = state.document
    parent = document.ancestry(survivor)[-2].id
    inverse = inverse_matrix(root_matrix(document, survivor))
    group_opacity = float(
        np.prod(
            [
                float(a.get("opacity", "1") or "1")
                for a in document.ancestry(survivor)[:-1]
            ]
        )
    )
    alpha = (
        evidence.opacity[box.slices][inside]
        if evidence.opacity is not None
        else np.ones(int(inside.sum()))
    )
    varying_alpha = bool(np.any(np.abs(alpha - group_opacity) > 0.5 / 255))
    geometries, rules = [], []
    nodes = sum(
        len(s.nodes) for oid in ids for s in document.geometry_for(oid).subpaths
    )
    for oid in inner:
        if work.interrupted:
            return None
        element = document.element(oid)
        style = path_style(document, element)
        if (
            surfaces[oid].covered
            or document.ancestry(oid)[-2].id != parent
            or style["stroke"] != "none"
            or float(style["opacity"]) != 1
            or float(style["fill-opacity"]) != 1
            or style["fill"] == "none"
            or element.get("clip-path", "none") != "none"
        ):
            return reject("mark-style-or-frame")
        server = paint_server(style["fill"])
        if server is None:
            if color(style["fill"])[3] != 1:
                return reject("nonopaque-current-paint")
        else:
            paint = document.element(server)
            if (
                paint.tag != "linearGradient"
                or not gradient_stops(paint)
                or any(stop[1][3] != 1 for stop in gradient_stops(paint))
            ):
                return reject("unsupported-mark-gradient")
        geometry = document.geometry_for(oid)
        nodes += sum(len(s.nodes) for s in geometry.subpaths)
        if nodes > MAX_NODES:
            return reject("mark-node-limit")
        geometries.append(
            transformed_geometry(
                geometry, multiply(inverse, root_matrix(document, oid))
            )
        )
        rules.append(style["fill-rule"])
    if work.interrupted:
        return None
    if varying_alpha:
        # Source bytes can vary even when the existing child paint is opaque
        # within its isolated group. An actual core makes the proposed overlap
        # alpha-neutral; native scoring still judges the changed edge colors.
        # Neither primary membership nor a nearly-opaque source byte proves it.
        frame = inverse_matrix(root_matrix(document, survivor))
        shape = union_geometry(
            [
                transformed_geometry(
                    document.geometry_for(oid),
                    multiply(frame, root_matrix(document, oid)),
                )
                for oid in ids
            ]
            + geometries,
            [path_style(document, document.element(oid)) for oid in ids]
            + [{"fill-rule": rule} for rule in rules],
        )
        selected_members = tuple(
            sorted((*members, *(i for oid in ids for i in surfaces[oid].members)))
        )
        if sum(len(s.nodes) for s in shape.subpaths) > MAX_NODES:
            return reject("source-alpha-proof-node-limit")
        if not in_core(state, selected_members, survivor, shape, work):
            return reject("unproved-source-alpha-variation")
    return Enclosed(filled, inner, members, tuple(geometries), tuple(rules))


def in_core(
    state: State,
    members: tuple[int, ...],
    survivor: str,
    geometry: Geometry,
    work: Work,
) -> bool:
    """Prove the complete continued fill lies inside an actual opaque core.

    The isolated group's opacity may be partial. Its core must be opaque in
    that group, so a continuing child cannot accumulate alpha at mark edges.
    """
    if state.partition is None or work.interrupted:
        return False
    document = state.document
    parent = document.ancestry(survivor)[-2].id
    inverse = inverse_matrix(root_matrix(document, survivor))
    selected = set(members)
    required = curve_path(geometry, "nonzero")
    for surface in state.partition.surfaces:
        if work.interrupted:
            return False
        if (
            not selected.issubset(
                surface.members if surface.role == "underlay" else surface.covered
            )
            or document.ancestry(surface.id)[-2].id != parent
        ):
            continue
        element = document.element(surface.id)
        style = path_style(document, element)
        shape = document.geometry_for(surface.id)
        if (
            style["fill"] == "none"
            or style["stroke"] != "none"
            or float(style["fill-opacity"]) != 1
            or float(style["opacity"]) != 1
            or element.get("clip-path", "none") != "none"
            or paint_server(style["fill"]) is not None
            or color(style["fill"])[3] != 1
            or sum(len(s.nodes) for s in shape.subpaths) > MAX_NODES
        ):
            continue
        shape = transformed_geometry(
            shape, multiply(inverse, root_matrix(document, surface.id))
        )
        outside = pathops.op(
            required, curve_path(shape, style["fill-rule"]), pathops.PathOp.DIFFERENCE
        )
        if not list(outside):
            return not work.interrupted
    return False
