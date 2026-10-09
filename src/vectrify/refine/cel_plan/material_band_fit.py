"""Fit exclusive material restoration and an attached stroke together.

Only appended material and the new centerline move. Original material contours,
paint servers, physical source ports and the sealed shadow stay outside parameter
space. Crop feasibility proposes a drawing; complete native/source checks and
original ownership replay still decide whether it can be published.
"""

from __future__ import annotations

from dataclasses import replace
from xml.etree import ElementTree as ET

import cairocffi as cairo
import numpy as np
import pathops
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize

from vectrify.document import Editor, Selection, export_svg
from vectrify.document.hit_test import multiply
from vectrify.document.join import (
    curve_path,
    path_geometry,
    path_style,
    transformed_geometry,
)
from vectrify.document.redraw import root_matrix
from vectrify.document.topology import inverse_matrix
from vectrify.refine import cel
from vectrify.refine.cel_plan.band_fit import (
    MAX_EVALUATIONS,
    MAX_MOVEMENT,
    MAX_PARAMETERS,
    painted_context,
)
from vectrify.refine.cel_plan.filled_bands import MAX_WIDTH, _check
from vectrify.refine.cel_plan.ink_models import footprint
from vectrify.refine.cel_plan.ink_replace import identified
from vectrify.refine.cel_plan.local import Box, _native_raster
from vectrify.refine.cel_plan.paint_continuation import (
    MAX_PATCH_CONTOURS,
    MAX_PATCH_NODES,
    _paint,
    _supported,
)
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.source_absence import ALPHA_TOLERANCE, SourceAbsence
from vectrify.refine.crossings import crossings

MAX_FLAT_NODES = 4096
NORMAL_MOVEMENT = 0.0625
TANGENT_MOVEMENT = 0.625


def _flat(geometry, frame, work):
    """Native-tolerance flattening for added material, never retained shadows."""
    context = cairo.Context(cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None))
    context.transform(cairo.Matrix(*frame))
    context.set_tolerance(0.1)
    for sub in geometry.subpaths:
        _check(work)
        for node in sub.nodes:
            if node.command == "M":
                context.move_to(*node.values)
            elif node.command == "L":
                context.line_to(*node.values)
            else:
                context.curve_to(*node.values)
        if sub.closed:
            context.close_path()
    result = pathops.Path()
    for count, (kind, values) in enumerate(context.copy_path_flat()):
        _check(work)
        if count >= MAX_FLAT_NODES:
            return None
        if kind == cairo.PATH_MOVE_TO:
            result.moveTo(*values)
        elif kind == cairo.PATH_LINE_TO:
            result.lineTo(*values)
        elif kind == cairo.PATH_CLOSE_PATH:
            result.close()
    return result


def _edge_variables(patch, error_points, limit):
    """Select controlling edges even when their endpoints are far from errors."""
    unique, distances = {}, {}
    for s, sub in enumerate(patch.subpaths):
        for n, node in enumerate(sub.nodes):
            unique.setdefault(node.endpoint, []).append((s, n))
            # Z closes the last endpoint to M with a straight edge. Its middle
            # can control failed pixels even when both endpoints are far away.
            if (not n and not sub.closed) or (n and node.command != "L"):
                continue
            a, b = np.asarray(sub.nodes[n - 1].endpoint), np.asarray(node.endpoint)
            edge = b - a
            squared = float(edge @ edge)
            if squared <= 1e-12 or not len(error_points):
                continue
            position = np.clip((error_points - a) @ edge / squared, 0, 1)
            distance = float(
                np.linalg.norm(
                    error_points - a - position[:, None] * edge, axis=1
                ).min()
            )
            if distance <= 2:
                for point in (tuple(a), tuple(b)):
                    distances[point] = min(distances.get(point, float("inf")), distance)
    ranked = sorted(distances, key=lambda p: (distances[p], p))[:limit]
    return [unique[p] for p in ranked]


class MaterialBandFit:
    def __init__(self, evidence, guard):
        self.evidence, self.guard = evidence, guard

    def fit(self, before, assembled, oid, seed, target, work, *, width_fixed=False):
        _check(work)
        if oid == target or not _supported(before, target):
            return None
        material_style = path_style(before, before.element(target))
        if not _paint(before, material_style) or assembled.element(
            target
        ) != before.element(target):
            return None
        if assembled.ancestry(oid)[-2].id != assembled.ancestry(target)[-2].id:
            return None
        original_material = before.geometry_for(target)
        if any(not s.closed for s in original_material.subpaths):
            return None
        owner_frame, material_frame = (
            root_matrix(assembled, oid),
            root_matrix(before, target),
        )
        linear = np.asarray(owner_frame[:4]).reshape(2, 2).T
        gram = linear.T @ linear
        scale = float(np.sqrt(gram[0, 0]))
        if scale <= 1e-12 or not np.allclose(
            gram, np.eye(2) * scale**2, rtol=1e-8, atol=1e-10
        ):
            return None
        center = seed.band.geometry
        if len(center.subpaths) != 1 or center.subpaths[0].closed:
            return None
        nodes = center.subpaths[0].nodes
        delta = np.asarray(nodes[-1].endpoint) - nodes[0].endpoint
        length = float(np.linalg.norm(delta))
        if length < 8 or len(nodes) > 8:
            return None
        normal, tangent = np.array((-delta[1], delta[0])) / length, delta / length
        corners = {tuple(seed.anchors[i]) for i in cel.run_corners(seed.anchors, False)}
        variables = []
        for i, node in enumerate(nodes):
            for j in range(0, len(node.values), 2):
                endpoint = j == len(node.values) - 2
                if i == 0 or (
                    endpoint
                    and (
                        i == len(nodes) - 1
                        or node.command == "L"
                        or node.endpoint in corners
                    )
                ):
                    continue
                variables.append((i, j))
        width_min = seed.band.width if width_fixed else max(0.8, seed.band.width * 0.5)
        limits = [(-float(MAX_MOVEMENT), float(MAX_MOVEMENT))] * len(variables)
        if not width_fixed:
            limits.append((width_min, min(MAX_WIDTH, seed.band.width * 1.5)))
        if len(limits) > MAX_PARAMETERS:
            return None
        flat = _flat(before.geometry_for(oid), owner_frame, work)
        if flat is None:
            return None
        strip = curve_path(
            transformed_geometry(
                footprint(center, MAX_WIDTH, cap="butt"), inverse_matrix(owner_frame)
            )
        )
        frame = multiply(inverse_matrix(material_frame), owner_frame)
        patch = transformed_geometry(
            path_geometry(pathops.op(flat, strip, pathops.PathOp.INTERSECTION)), frame
        )
        existing = curve_path(original_material, material_style["fill-rule"])
        patch = path_geometry(
            pathops.op(curve_path(patch), existing, pathops.PathOp.DIFFERENCE)
        )
        if (
            not 0 < len(patch.subpaths) <= MAX_PATCH_CONTOURS
            or sum(len(s.nodes) for s in patch.subpaths) > MAX_PATCH_NODES
        ):
            return None
        original_data = original_material.path_data()
        editor = Editor(assembled, selection=Selection(whole_document=True))
        with editor.transaction("Seed exclusive source material restoration") as tx:
            tx.replace_geometry(
                target,
                replace(
                    original_material,
                    subpaths=(
                        *original_material.subpaths,
                        *identified(patch, target + "-joint-restoration").subpaths,
                    ),
                ),
            )
            tx.set_attributes(oid, {"stroke-width": repr(width_min / scale)})
        initial_document = editor.snapshot.document
        bounds, native_size = seed.bounds, self.evidence.source_size
        box = Box(*bounds)
        baseline_full = _native_raster(ET.fromstring(export_svg(before)), native_size)
        baseline = baseline_full.values(box)
        root = painted_context(initial_document, bounds, work, keep=(oid, target))
        initial = _native_raster(root, native_size).values(box)
        if not np.array_equal(
            initial,
            _native_raster(
                ET.fromstring(export_svg(initial_document)), native_size
            ).values(box),
        ):
            return None
        material = next(e for e in root.iter() if e.get("id") == target)
        stroke = next(e for e in root.iter() if e.get("id") == oid)
        ink_style = path_style(initial_document, initial_document.element(oid))
        cap = ink_style["stroke-linecap"]
        if (
            ink_style["fill"] != "none"
            or cap not in {"butt", "round"}
            or ink_style["stroke-linejoin"] != "round"
        ):
            return None
        lines = self.guard.fitting_crop(bounds, work=work)
        if lines is None:
            return None
        seen = lines.observe(baseline.astype(float) / 255, work=work)
        gaps = self.guard.gap_centres(limit=4096, work=work)
        inside = ((gaps >= bounds[:2]) & (gaps < bounds[2:])).all(axis=1)
        queries = gaps[inside] - bounds[:2] - 0.5
        y, x = np.nonzero(initial[..., 3] != baseline[..., 3])
        errors = np.column_stack((x + bounds[0] + 0.5, y + bounds[1] + 0.5))
        native_patch = transformed_geometry(patch, material_frame)
        material_variables = _edge_variables(
            native_patch, errors, (MAX_PARAMETERS - len(limits)) // 2
        )
        material_limits = [
            v
            for _ in material_variables
            for v in (
                (-NORMAL_MOVEMENT, NORMAL_MOVEMENT),
                (-TANGENT_MOVEMENT, TANGENT_MOVEMENT),
            )
        ]
        limits = material_limits + limits
        values = np.r_[
            np.zeros(len(limits) - int(not width_fixed)),
            [] if width_fixed else [width_min],
        ]
        best_value, best_parameters, evaluations = float("inf"), None, 0

        def shapes(parameters):
            updated = [[list(n.values) for n in s.nodes] for s in native_patch.subpaths]
            for index, occurrences in enumerate(material_variables):
                for s, n in occurrences:
                    updated[s][n][-2:] = (
                        np.asarray(native_patch.subpaths[s].nodes[n].endpoint)
                        + parameters[2 * index] * normal
                        + parameters[2 * index + 1] * tangent
                    )
            changed_patch = replace(
                native_patch,
                subpaths=tuple(
                    replace(
                        s,
                        nodes=tuple(
                            replace(n, values=tuple(v))
                            for n, v in zip(s.nodes, coords, strict=True)
                        ),
                    )
                    for s, coords in zip(native_patch.subpaths, updated, strict=True)
                ),
            )
            updated_center = [list(n.values) for n in nodes]
            offset = 2 * len(material_variables)
            for amount, (i, j) in zip(
                parameters[offset : offset + len(variables)], variables, strict=True
            ):
                updated_center[i][j : j + 2] = (
                    np.asarray(nodes[i].values[j : j + 2]) + amount * normal
                )
            changed_center = replace(
                center,
                subpaths=(
                    replace(
                        center.subpaths[0],
                        nodes=tuple(
                            replace(n, values=tuple(v))
                            for n, v in zip(nodes, updated_center, strict=True)
                        ),
                    ),
                ),
            )
            return transformed_geometry(
                changed_patch, inverse_matrix(material_frame)
            ), changed_center

        def score(parameters):
            nonlocal best_value, best_parameters, evaluations
            _check(work)
            evaluations += 1
            changed_patch, changed_center = shapes(parameters)
            width = seed.band.width if width_fixed else float(parameters[-1])
            material.set("d", original_data + " " + changed_patch.path_data())
            stroke.set(
                "d",
                transformed_geometry(
                    changed_center, inverse_matrix(owner_frame)
                ).path_data(),
            )
            stroke.set("stroke-width", repr(width / scale))
            actual = _native_raster(root, native_size).values(box)
            line_penalty = lines.penalty(seen, actual.astype(float) / 255, work=work)
            overlap = float(
                pathops.op(
                    curve_path(changed_patch), existing, pathops.PathOp.INTERSECTION
                ).area
            )
            size = (bounds[2] - bounds[0], bounds[3] - bounds[1])
            body = render(
                f'<svg width="{size[0]}" height="{size[1]}" '
                f'viewBox="{bounds[0]} {bounds[1]} {size[0]} {size[1]}">'
                f'<path d="{changed_center.path_data()}" fill="none" '
                f'stroke="white" stroke-width="{width}" '
                f'stroke-linecap="{cap}" stroke-linejoin="round"/></svg>',
                size,
            )[..., 3]
            gap_alpha = map_coordinates(
                body, [queries[:, 1], queries[:, 0]], order=1, mode="constant", cval=0
            )
            alpha_error = int(
                np.abs(actual[..., 3].astype(int) - baseline[..., 3]).sum()
            )
            regularization = (
                float(parameters @ parameters)
                - (0 if width_fixed else float(parameters[-1]) ** 2)
                + (width - width_min) ** 2
            )
            value = (
                alpha_error
                + regularization * 1e-6
                + line_penalty
                + 1000 * float(np.maximum(0, gap_alpha - ALPHA_TOLERANCE).sum())
                + 1000 * min(1, overlap)
            )
            if (
                alpha_error == 0
                and line_penalty == 0
                and overlap <= 1e-8
                and np.all(gap_alpha <= ALPHA_TOLERANCE)
                and value < best_value
            ):
                best_value, best_parameters = value, np.array(parameters, copy=True)
            _check(work)
            return value

        initial_loss = score(values)
        if len(values):
            minimize(
                score,
                values,
                method="Powell",
                bounds=limits,
                options={
                    "maxfev": MAX_EVALUATIONS - 1,
                    "maxiter": 12,
                    "xtol": 0.001,
                    "ftol": 1e-6,
                },
            )
        if best_parameters is None:
            return None
        changed_patch, changed_center = shapes(best_parameters)
        width = seed.band.width if width_fixed else float(best_parameters[-1])
        if crossings(changed_center) or not SourceAbsence(
            self.evidence, (), work, guard=self.guard
        ).permits(changed_center, width, cap, work):
            return None
        editor = Editor(assembled, selection=Selection(whole_document=True))
        with editor.transaction(
            "Fit source material and editable outline atomically"
        ) as tx:
            tx.replace_geometry(
                target,
                replace(
                    original_material,
                    subpaths=(
                        *original_material.subpaths,
                        *identified(
                            changed_patch, target + "-joint-restoration"
                        ).subpaths,
                    ),
                ),
            )
            tx.replace_geometry(
                oid,
                identified(
                    transformed_geometry(changed_center, inverse_matrix(owner_frame)),
                    oid,
                ),
            )
            tx.set_attributes(oid, {"stroke-width": repr(width / scale)})
        document = editor.snapshot.document
        whole = _native_raster(ET.fromstring(export_svg(document)), native_size)
        if not np.array_equal(whole.root[..., 3], baseline_full.root[..., 3]):
            return None
        material.set("d", document.geometry_for(target).path_data())
        stroke.set("d", document.geometry_for(oid).path_data())
        stroke.set("stroke-width", repr(width / scale))
        if not np.array_equal(
            whole.values(box), _native_raster(root, native_size).values(box)
        ):
            return None
        comparison = self.guard.compare(
            baseline_full.root.astype(float) / 255,
            whole.root.astype(float) / 255,
            work=work,
        )
        if not comparison["qualified_samples"] or comparison["rejections"]:
            return None
        _check(work)
        return document, {
            "profile": seed.profile,
            "mode": "source-material-joint",
            "evaluations": evaluations,
            "initial_loss": initial_loss,
            "fitted_loss": best_value,
            "parameters": best_parameters.tolist(),
            "width": width,
            "native_alpha_exact": True,
            "native_body_absence": True,
            "linecap": cap,
            "source_line_comparison": comparison,
            "source_ports": [list(nodes[0].endpoint), list(nodes[-1].endpoint)],
            "material": target,
            "material_patch_nodes": sum(len(s.nodes) for s in changed_patch.subpaths),
            "material_patch_contours": len(changed_patch.subpaths),
            "material_native_bounds": list(
                curve_path(transformed_geometry(changed_patch, material_frame)).bounds
            ),
            "material_variables": material_variables,
            "material_normal_movement": NORMAL_MOVEMENT,
            "material_tangent_movement": TANGENT_MOVEMENT,
        }
