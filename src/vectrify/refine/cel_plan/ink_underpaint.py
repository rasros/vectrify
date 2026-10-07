"""Compact adjacent paints sharing a boundary hidden beneath fitted ink.

An owned RGB cavity and its sole surrounding paint may meet under the rim.
Actual opaque paint/core, complete cavity ownership and native ink coverage
prove this interpretation; unsupported cavities retain traced continuations.
"""

from __future__ import annotations

import math

import numpy as np
import pathops
from scipy.ndimage import (
    binary_fill_holes,
    distance_transform_edt,
)

from vectrify.document import Geometry
from vectrify.document.hit_test import multiply
from vectrify.document.join import (
    curve_path,
    path_style,
    transformed_geometry,
    union_geometry,
)
from vectrify.document.redraw import root_matrix
from vectrify.document.topology import inverse_matrix
from vectrify.refine.cel_plan.geometry import fitted
from vectrify.refine.cel_plan.local import MAX_CROP_PIXELS, Box
from vectrify.refine.cel_plan.nested import opaque_fill
from vectrify.refine.cel_plan.score import render
from vectrify.refine.tracing import _loops

MAX_NODES = 6000
MAX_PERIMETER = 4096


def compact(
    state,
    evidence,
    graph,
    ids,
    members,
    own,
    box,
    original,
    rim,
    survivor,
    neighbors,
    tolerance,
    work,
    diagnostics,
):
    def reject(reason):
        excluded = diagnostics.setdefault("rim_underpaint_exclusions", {})
        excluded[reason] = excluded.get(reason, 0) + 1
        return

    if work.interrupted or state.partition is None:
        return None
    if len(neighbors) != 2:
        return reject("neighbor-count")
    filled = np.asarray(binary_fill_holes(own), bool)
    cavity = filled & ~own
    regions = tuple(int(i) for i in np.unique(graph.labels[box.slices][cavity]))
    owners = state.partition.owners
    cavity_owners = {owners[i] for i in regions if i in owners}
    if not regions or len(cavity_owners) != 1 or any(i not in owners for i in regions):
        return reject("cavity-owners")
    inner = next(iter(cavity_owners))
    if inner not in neighbors:
        return reject("cavity-neighbor")
    outer = next(i for i in neighbors if i != inner)
    surfaces = {s.id: s for s in state.partition.surfaces}
    if surfaces[inner].members != regions or sum(
        graph.regions[i].area for i in regions
    ) != int(cavity.sum()):
        return reject("partial-cavity-owner")
    document = state.document
    for oid in (*ids, inner, outer):
        if work.interrupted:
            return None
        style = path_style(document, document.element(oid))
        if (
            style["stroke"] != "none"
            or float(style["opacity"]) != 1
            or float(style["fill-opacity"]) != 1
            or not opaque_fill(document, style["fill"])
        ):
            return reject("nonopaque-paint")
    node_count = sum(
        len(s.nodes)
        for oid in (inner, outer)
        for s in document.geometry_for(oid).subpaths
    )
    node_count += sum(len(s.nodes) for g in (original, rim) for s in g.subpaths)
    if node_count > MAX_NODES:
        return reject("geometry-limit")
    # Both sides meet near the source rim's medial contour. This boundary has
    # no visible significance if every native pixel it can mix is under ink.
    to_cavity = np.asarray(distance_transform_edt(~cavity))
    to_outside = np.asarray(distance_transform_edt(filled))
    middle = cavity | (own & (to_cavity <= to_outside))
    loops = _loops(middle)
    if len(loops) != 1 or len(loops[0]) > MAX_PERIMETER:
        return reject("perimeter-limit")
    points = np.array([*loops[0], loops[0][0]], dtype=float)
    points += (box.x, box.y)
    points = points / evidence.scale + evidence.offset
    model = fitted(points, tolerance)
    if work.interrupted:
        return None
    matrix = root_matrix(document, survivor)
    middle_shape = transformed_geometry(
        Geometry("underpaint", (model.contour,)), inverse_matrix(matrix)
    )
    ordered_contours = sorted(
        rim.subpaths, key=lambda s: abs(curve_path(Geometry("contour", (s,))).area)
    )
    hole, border = (curve_path(Geometry("contour", (s,))) for s in ordered_contours)
    mid = curve_path(middle_shape)
    inner_shape = transformed_geometry(
        document.geometry_for(inner),
        multiply(inverse_matrix(matrix), root_matrix(document, inner)),
    )
    inner_rule = path_style(document, document.element(inner))["fill-rule"]
    if (
        abs(pathops.op(hole, mid, pathops.PathOp.DIFFERENCE).area) > 1e-8
        or abs(pathops.op(mid, border, pathops.PathOp.DIFFERENCE).area) > 1e-8
        or abs(
            pathops.op(
                curve_path(inner_shape, inner_rule), mid, pathops.PathOp.DIFFERENCE
            ).area
        )
        > 1e-8
    ):
        return reject("uncontained-material")
    native_rim = transformed_geometry(rim, matrix)
    native_mid = transformed_geometry(middle_shape, matrix)
    a, b, c, d = curve_path(native_rim).bounds
    native_box = Box(
        math.floor(a) - 2, math.floor(b) - 2, math.ceil(c) + 2, math.ceil(d) + 2
    )
    if not native_box.area or native_box.area > MAX_CROP_PIXELS:
        return reject("native-pixel-limit")

    def mask(geometry):
        width, height = (
            native_box.right - native_box.x,
            native_box.bottom - native_box.y,
        )
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="{native_box.x} {native_box.y} '
            f'{width} {height}"><path fill="black" fill-rule="nonzero" '
            f'd="{geometry.path_data()}"/></svg>'
        )
        return render(svg, (width, height))[..., 3]

    ink, material = mask(native_rim), mask(native_mid)
    hole_coverage = mask(
        transformed_geometry(Geometry("hole", (ordered_contours[0],)), matrix)
    )
    border_coverage = mask(
        transformed_geometry(Geometry("border", (ordered_contours[1],)), matrix)
    )
    boundary = (material > 0) & (material < 1)
    if work.interrupted:
        return None
    # Every mixed underpaint pixel must be covered by opaque ink. At visible
    # inner/outer antialias pixels the underpaint must be the corresponding
    # pure material. This proves compositing directly at native resolution;
    # a widened pixel-neighborhood band would exclude valid narrow rims.
    visible = ink < 1
    if (
        (ink[boundary] != 1).any()
        or (material[visible & (hole_coverage > 0)] != 1).any()
        or (material[visible & (border_coverage < 1)] != 0).any()
    ):
        return reject("visible-material-boundary")
    background = union_geometry(
        [
            document.geometry_for(outer),
            transformed_geometry(
                original, multiply(inverse_matrix(root_matrix(document, outer)), matrix)
            ),
            transformed_geometry(
                inner_shape,
                multiply(inverse_matrix(root_matrix(document, outer)), matrix),
            ),
        ],
        [
            path_style(document, document.element(outer)),
            {"fill-rule": "nonzero"},
            {"fill-rule": inner_rule},
        ],
    )
    if (
        sum(len(s.nodes) for g in (background, middle_shape) for s in g.subpaths)
        > MAX_NODES
    ):
        return reject("output-geometry-limit")
    diagnostics["rim_compact_underpaints"] += 1
    middle_shape = transformed_geometry(
        middle_shape, multiply(inverse_matrix(root_matrix(document, inner)), matrix)
    )
    return (
        [
            (document.element(outer), background),
            (document.element(inner), middle_shape),
        ],
        {outer: tuple(sorted((*members, *regions))), inner: members},
        model.kind,
    )
