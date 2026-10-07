"""Complete source and visibility proofs for a compact material extension.

An opaque core permits changing coverage of a child without accumulating alpha.
It does not authorize painting over an unrelated primary owner. Newly covered
visible pixels must agree with the actual side paint, and new geometry may only
continue below independently owned paint that already lies above the material.
"""

from __future__ import annotations

import math

import numpy as np
import pathops
from cairosvg.colors import color

from vectrify.document.hit_test import multiply
from vectrify.document.join import (
    curve_path,
    path_geometry,
    path_style,
    transformed_geometry,
)
from vectrify.document.model import paint_server
from vectrify.document.paint import gradient_stops
from vectrify.document.redraw import root_matrix
from vectrify.document.topology import inverse_matrix
from vectrify.refine.cel_plan.local import Box
from vectrify.refine.cel_plan.nested import in_core
from vectrify.refine.cel_plan.refine import _bounds
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.shade_edges import predict
from vectrify.refine.cel_plan.surface_models import prediction

MAX_PIXELS = 262_144
MAX_NODES = 6_000
MAX_PROOFS = 128
CHUNK_PIXELS = 65_536
RESIDUAL = 48.0


def supported(
    state,
    ids,
    survivor,
    members,
    old,
    shape,
    paints,
    normal,
    rho,
    evidence,
    graph,
    work,
    diagnostics: dict,
    *,
    coverage=False,
):
    """Return exact secondary side support, or exclude the complete extension."""

    def reject(reason):
        excluded = diagnostics.setdefault("supported_boundary_exclusions", {})
        excluded[reason] = excluded.get(reason, 0) + 1
        return

    if work.interrupted or state.partition is None:
        return None
    if not in_core(state, members, survivor, shape, work):
        return reject("actual-core")
    extra = pathops.op(curve_path(shape), curve_path(old), pathops.PathOp.DIFFERENCE)
    if work.interrupted:
        return None
    if abs(extra.area) <= 1e-8:
        return reject("no-extension")
    if sum(1 for _ in extra) > MAX_NODES:
        return reject("geometry-limit")
    document = state.document
    matrix = root_matrix(document, survivor)
    inverse = inverse_matrix(matrix)
    parent = document.ancestry(survivor)[-2]
    # Pure material RGB is irrelevant below an already opaque higher owner.
    # Subtract actual upper geometry, not its label mask or a coverage claim.
    # Partial paint and antialiased uncovered geometry remain in the screen.
    bare = extra
    above = False
    occluder_nodes = 0
    upper_proofs = 0
    for child in parent.children:
        if work.interrupted:
            return None
        if child.id == survivor:
            above = True
            continue
        if not above or child.tag != "path":
            continue
        style = path_style(document, child)
        if (
            style["fill"] == "none"
            or style["stroke"] != "none"
            or float(style["opacity"]) != 1
            or float(style["fill-opacity"]) != 1
            or child.get("clip-path", "none") != "none"
        ):
            continue
        server = paint_server(style["fill"])
        if server is None:
            if color(style["fill"])[3] != 1:
                continue
        else:
            stops = gradient_stops(document.element(server))
            if not stops or any(abs(stop[1][3] - 1) > 1e-9 for stop in stops):
                continue
        child_shape = transformed_geometry(
            document.geometry_for(child.id),
            multiply(inverse, root_matrix(document, child.id)),
        )
        upper_path = curve_path(child_shape, style["fill-rule"])
        # Bounds in this common frame include current child transforms.
        a, b, c, d = extra.bounds
        p, q, r, s = upper_path.bounds
        if max(a, p) >= min(c, r) or max(b, q) >= min(d, s):
            continue
        occluder_nodes += sum(len(s.nodes) for s in child_shape.subpaths)
        if occluder_nodes > MAX_NODES or upper_proofs >= MAX_PROOFS:
            return reject("upper-order-limit")
        bare = pathops.op(bare, upper_path, pathops.PathOp.DIFFERENCE)
        upper_proofs += 1
        diagnostics["supported_boundary_upper_proofs"] = (
            diagnostics.get("supported_boundary_upper_proofs", 0) + 1
        )
        if sum(1 for _ in bare) > MAX_NODES:
            return reject("geometry-limit")
    sx, sy = evidence.scale
    ox, oy = evidence.offset
    analysis = transformed_geometry(path_geometry(extra), matrix)
    analysis = transformed_geometry(analysis, (sx, 0, 0, sy, -ox * sx, -oy * sy))
    left, top, right, bottom = curve_path(analysis).bounds
    height, width = graph.labels.shape
    if left < 0 or top < 0 or right > width or bottom > height:
        return reject("source-frame")
    box = Box(math.floor(left), math.floor(top), math.ceil(right), math.ceil(bottom))
    if not box.area or box.area > MAX_PIXELS:
        return reject("pixel-limit")
    # The native renderer's byte coverage includes every visible changed pixel,
    # including subpixel extensions. No fit sample can hide an outlier here.
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{box.right - box.x}" '
        f'height="{box.bottom - box.y}" viewBox="{box.x} {box.y} '
        f'{box.right - box.x} {box.bottom - box.y}"><path fill="black" '
        f'fill-rule="nonzero" d="{analysis.path_data()}"/></svg>'
    )
    raster = render(svg, (box.right - box.x, box.bottom - box.y))[..., 3]
    bare_geometry = transformed_geometry(path_geometry(bare), matrix)
    bare_geometry = transformed_geometry(
        bare_geometry, (sx, 0, 0, sy, -ox * sx, -oy * sy)
    )
    bare_svg = svg.replace(analysis.path_data(), bare_geometry.path_data())
    visible_raster = render(bare_svg, (box.right - box.x, box.bottom - box.y))[..., 3]
    if work.interrupted:
        return None
    covered = [set(), set()]
    primary = set(members)
    owners = state.partition.owners
    for chunk in box.chunks(CHUNK_PIXELS):
        if work.interrupted:
            return None
        shown = visible_raster[chunk.within(box)] > 0
        full_y, full_x = np.nonzero(raster[chunk.within(box)] > 0)
        full_xy = np.column_stack((full_x + chunk.x + 0.5, full_y + chunk.y + 0.5))
        full_side = full_xy @ normal < rho
        full_labels = graph.labels[chunk.slices][full_y, full_x]
        for index, chosen in enumerate((full_side, ~full_side)):
            covered[index].update(
                int(i)
                for i in np.unique(full_labels[chosen])
                if i not in primary and i in owners
            )
        if (shown & evidence.empty[chunk.slices]).any():
            return reject("empty-source")
        yy, xx = np.nonzero(shown)
        if not len(xx):
            continue
        xy = np.column_stack((xx + chunk.x + 0.5, yy + chunk.y + 0.5))
        side = xy @ normal < rho
        rgb = evidence.target[chunk.slices][shown]
        if coverage:
            # The two opaque materials mix through geometric pixel coverage,
            # not intrinsic alpha. Exported gradients still clamp at endpoints.
            predicted = predict(paints, xy, normal, rho, extend=False) * 255
            if np.max(np.abs(predicted - rgb)) > RESIDUAL:
                return reject("paint-residual")
        else:
            for index, selected in enumerate((side, ~side)):
                if not selected.any():
                    continue
                predicted = prediction(paints[index], xy[selected], extend=False) * 255
                if np.max(np.abs(predicted - rgb[selected])) > RESIDUAL:
                    return reject("paint-residual")
        diagnostics["supported_boundary_pixels"] = diagnostics.get(
            "supported_boundary_pixels", 0
        ) + len(xx)
    # A lower path is not hidden merely because its source colors fit. The
    # extension must prove its actual geometry disjoint before rising over it.
    selected_ids = set(ids)
    underlays = {s.id for s in state.partition.surfaces if s.role == "underlay"}
    proofs = 0
    nodes = 0
    extra_bounds = Box(
        math.floor(left), math.floor(top), math.ceil(right), math.ceil(bottom)
    )
    for child in parent.children:
        if work.interrupted:
            return None
        if child.id == survivor:
            break
        if child.id in selected_ids or child.id in underlays:
            continue
        if child.tag != "path":
            return reject("unsupported-lower-object")
        # Avoid a proof or allocation for distant current geometry.
        bounds = _bounds(document, child.id)
        probe = Box(
            math.floor((bounds[0] - ox) * sx),
            math.floor((bounds[1] - oy) * sy),
            math.ceil((bounds[2] - ox) * sx),
            math.ceil((bounds[3] - oy) * sy),
        )
        if not probe.intersection(extra_bounds).area:
            continue
        geometry = document.geometry_for(child.id)
        style = path_style(document, child)
        if style["stroke"] != "none" or child.get("clip-path", "none") != "none":
            return reject("unsupported-lower-paint")
        nodes += sum(len(s.nodes) for s in geometry.subpaths)
        if nodes > MAX_NODES or proofs >= MAX_PROOFS:
            return reject("order-limit")
        geometry = transformed_geometry(
            geometry, multiply(inverse, root_matrix(document, child.id))
        )
        overlap = pathops.op(
            extra, curve_path(geometry, style["fill-rule"]), pathops.PathOp.INTERSECTION
        )
        proofs += 1
        diagnostics["supported_boundary_order_proofs"] = (
            diagnostics.get("supported_boundary_order_proofs", 0) + 1
        )
        if abs(overlap.area) > 1e-8:
            return reject("lower-paint-overlap")
    if work.interrupted:
        return None
    return tuple(tuple(sorted(values)) for values in covered)
