"""Fit a complete attached centerline over its independently constructed floor.

Local normal parameters move only the new stroke. Actual native stroke bodies
must satisfy every frozen qualified query of affected original profiles and the
fresh whole-guide profile. Existing junction strokes can supply joint support,
but backgrounds cannot. Crop fitting proposes parameters; complete alpha,
locality, painted source and absence checks still decide whether to return them.
Physical family binding and component acceptance remain separate. Only explicit
experimental High schedules this fitter.
"""

from dataclasses import dataclass, replace
from xml.etree import ElementTree as ET

import numpy as np
from cairosvg.colors import color
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize

from vectrify.document import Editor, Selection, export_svg
from vectrify.document.join import curve_path, transformed_geometry
from vectrify.document.redraw import root_matrix
from vectrify.document.topology import inverse_matrix
from vectrify.refine.cel_plan.band_fit import (
    MAX_EVALUATIONS,
    MAX_MOVEMENT,
    MAX_PARAMETERS,
    painted_context,
)
from vectrify.refine.cel_plan.filled_bands import MAX_NATIVE_PIXELS, MAX_WIDTH, _check
from vectrify.refine.cel_plan.ink_replace import identified
from vectrify.refine.cel_plan.line_fidelity import MAX_PROFILES, FittingBody
from vectrify.refine.cel_plan.local import HALO, Box, _native_raster
from vectrify.refine.cel_plan.paint_continuation import _supported
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.source_absence import ALPHA_TOLERANCE, SourceAbsence
from vectrify.refine.cel_plan.source_bands import MAX_EXTENT, MAX_PIXELS
from vectrify.refine.cel_plan.source_centerline import SourceCenterline
from vectrify.refine.cel_plan.span_restoration import SpanRestoration
from vectrify.refine.cel_plan.stroke_inventory import _style
from vectrify.refine.crossings import crossings

MAX_CONTRACTS = 32
MAX_QUERIES = 16_384


@dataclass(frozen=True)
class Bodies:
    """All queries remain in each immutable contract, including outside a crop."""

    contracts: tuple[FittingBody, ...]

    def observe(self, alpha, work):
        observations = tuple(c.observe(alpha, work=work) for c in self.contracts)
        return {
            "penalty": sum(r["penalty"] for r in observations),
            "missing_samples": sum(r["missing_samples"] for r in observations),
            "contracts": list(observations),
        }


def _stroke_root(document, ids, size, work):
    """Actual white stroke bodies in the original shared opacity hierarchy."""
    root = ET.Element("svg", {"width": str(size[0]), "height": str(size[1])})
    groups = {}
    for oid in ids:
        _check(work)
        element = document.element(oid)
        supported = _style(document, element)
        if supported is None:
            raise ValueError("Span body requires a supported genuine stroke")
        style, _opacity = supported
        parent = root
        for ancestor in document.ancestry(oid)[:-1]:
            if ancestor.id not in groups:
                groups[ancestor.id] = ET.SubElement(
                    parent, "g", {"opacity": ancestor.get("opacity", "1")}
                )
            parent = groups[ancestor.id]
        ET.SubElement(
            parent,
            "path",
            {
                "id": oid,
                "d": document.geometry_for(oid).path_data(),
                "transform": "matrix("
                + " ".join(map(str, root_matrix(document, oid)))
                + ")",
                "fill": "none",
                "stroke": "white",
                "stroke-width": style["stroke-width"],
                "stroke-opacity": repr(
                    float(style["stroke-opacity"]) * color(style["stroke"])[3]
                ),
                "stroke-linecap": style["stroke-linecap"],
                "stroke-linejoin": style["stroke-linejoin"],
                "stroke-miterlimit": style["stroke-miterlimit"],
                "opacity": element.get("opacity", "1"),
            },
        )
    _check(work)
    return root


def _variables(center, corners):
    """One local normal per cubic control or unfixed interior cubic vertex."""
    nodes, variables = center.subpaths[0].nodes, []
    for i, node in enumerate(nodes):
        if node.command != "C":
            continue
        for j in (0, 2, 4):
            if j == 4:
                if i == len(nodes) - 1 or node.endpoint in corners:
                    continue
                tangent = np.asarray(nodes[i + 1].endpoint) - nodes[i - 1].endpoint
            elif j == 0:
                tangent = np.asarray(node.values[:2]) - nodes[i - 1].endpoint
            else:
                tangent = np.asarray(node.endpoint) - node.values[2:4]
            norm = np.linalg.norm(tangent)
            if norm >= 1e-8:
                variables.append((i, j, np.array((-tangent[1], tangent[0])) / norm))
    return variables


class SpanBodyFit:
    def __init__(self, evidence, guard):
        self.evidence, self.guard = evidence, guard

    def fit(
        self,
        before,
        floor: SpanRestoration,
        oid,
        centerline: SourceCenterline,
        width,
        paint,
        added_guard,
        added_profile,
        bar,
        work,
    ):
        """Return a fully verified drawing, never the intermediate floor/fit."""
        _check(work)
        size = self.evidence.source_size
        center = centerline.geometry
        if (
            not _supported(before, oid)
            or not _supported(floor.document, oid)
            or self.evidence.rgba.shape != (size[1], size[0], 4)
            or self.guard.shape != self.evidence.rgba.shape
            or added_guard.shape != self.guard.shape
            or np.prod(size) > MAX_NATIVE_PIXELS
            or not np.isfinite(width)
            or not 0.8 <= width <= MAX_WIDTH
            or len(center.subpaths) != 1
            or center.subpaths[0].closed
            or len(center.subpaths[0].nodes) < 2
            or _style(before, before.element(bar)) is None
            or before.ancestry(bar)[-2].id != before.ancestry(oid)[-2].id
            or floor.document.element(bar) != before.element(bar)
            or floor.document.geometry_for(bar) != before.geometry_for(bar)
        ):
            return None
        nodes = center.subpaths[0].nodes
        if not np.isfinite([v for n in nodes for v in n.values]).all():
            return None
        ports = np.array([nodes[i].endpoint for i in (0, -1)])
        existing = transformed_geometry(
            before.geometry_for(bar), root_matrix(before, bar)
        )
        if not any(
            np.array_equal(ports, np.array([s.nodes[i].endpoint for i in ends]))
            for s in existing.subpaths
            if not s.closed and len(s.nodes) >= 2
            for ends in ((0, -1), (-1, 0))
        ) or not set(centerline.corners) <= {n.endpoint for n in nodes}:
            return None
        observed = added_guard.source_breaks(added_profile, work=work)
        if (
            observed is None
            or observed.gaps.any()
            # Qualification is an immutable raw contrast observation, not the
            # caller's whole-guide ink support measurement. Keep every such
            # query mandatory without introducing a new qualification ratio.
            or not observed.qualified.any()
            or not np.array_equal(observed.anchors[[0, -1]], ports)
        ):
            return None
        frame = root_matrix(floor.document, oid)
        linear = np.asarray(frame[:4]).reshape(2, 2).T
        gram = linear.T @ linear
        scale = float(np.sqrt(gram[0, 0]))
        if (
            not np.isfinite(frame).all()
            or scale <= 1e-12
            or not np.allclose(gram, np.eye(2) * scale**2, rtol=1e-8, atol=1e-10)
        ):
            return None
        native_field = transformed_geometry(floor.field, root_matrix(before, oid))
        native_bounds = np.asarray(curve_path(native_field).bounds)
        if (
            not np.isfinite(native_bounds).all()
            or np.max(np.abs(native_bounds)) > 1_000_000
            or np.max(native_bounds[2:] - native_bounds[:2]) > MAX_EXTENT
        ):
            return None
        lo = np.maximum(0, np.floor(native_bounds[:2]).astype(int) - HALO)
        hi = np.minimum(size, np.ceil(native_bounds[2:]).astype(int) + HALO)
        bounds = tuple(map(int, (*lo, *hi)))
        box = Box(*bounds)
        variables = _variables(center, set(centerline.corners))
        if not 0 < box.area <= MAX_PIXELS or len(variables) + 1 > MAX_PARAMETERS:
            return None
        editor = Editor(floor.document, selection=Selection(whole_document=True))
        with editor.transaction("Initialize complete source-supported outline") as tx:
            tx.replace_geometry(
                oid,
                identified(transformed_geometry(center, inverse_matrix(frame)), oid),
            )
            tx.set_attributes(
                oid,
                {
                    "fill": "none",
                    "stroke": paint,
                    "stroke-width": repr(width / scale),
                    "stroke-opacity": "1",
                    "stroke-linecap": "round",
                    "stroke-linejoin": "round",
                },
            )
        assembled = editor.snapshot.document
        if _style(assembled, assembled.element(oid)) is None:
            return None
        baseline_full = _native_raster(ET.fromstring(export_svg(before)), size).root
        baseline = baseline_full[box.slices]
        root = painted_context(
            assembled, bounds, work, keep=tuple(p.id for p in floor.parts)
        )
        if not np.array_equal(
            _native_raster(root, size).values(box),
            _native_raster(ET.fromstring(export_svg(assembled)), size).values(box),
        ):
            return None
        field_root = ET.Element("svg", {"width": str(size[0]), "height": str(size[1])})
        ET.SubElement(
            field_root, "path", {"d": native_field.path_data(), "fill": "white"}
        )
        field_mask = _native_raster(field_root, size).root[..., 3] > 0
        bar_alpha = (
            _native_raster(_stroke_root(before, (bar,), size, work), size).values(box)[
                ..., 3
            ]
            / 255
        )
        own, joint, affected, queries = [], [], [], 0
        profiles = self.guard.original_profiles(work=work)
        if len(profiles) > MAX_PROFILES:
            return None
        for number, profile in enumerate(profiles):
            _check(work)
            original = self.guard.source_breaks(profile, work=work)
            q = np.floor(original.points).astype(int)
            inside = ((q >= 0) & (q < np.array(size))).all(axis=1)
            touched = np.zeros(len(q), bool)
            touched[inside] = field_mask[q[inside, 1], q[inside, 0]]
            if not (touched & original.qualified).any():
                continue
            queries += len(original.points)
            if len(affected) >= MAX_CONTRACTS - 1 or queries > MAX_QUERIES:
                return None
            contract = self.guard.fitting_body(profile, bounds, work=work)
            if contract is None:
                return None
            original_bar = contract.observe(bar_alpha, work=work)
            connected = (
                original_bar["supported_samples"]
                >= 0.5 * original_bar["qualified_samples"]
            )
            (joint if connected else own).append(contract)
            affected.append(
                {
                    "profile": number,
                    "qualified": int(original.qualified.sum()),
                    "touched_qualified": int((touched & original.qualified).sum()),
                    "actual_bar_supported": original_bar["supported_samples"],
                    "role": "joint" if connected else "new-stroke",
                }
            )
        extra = added_guard.fitting_body(added_profile, bounds, work=work)
        if (
            not affected
            or extra is None
            or queries + len(observed.points) > MAX_QUERIES
        ):
            return None
        own.append(extra)
        own_body, joint_body = Bodies(tuple(own)), Bodies(tuple(joint))
        lines = self.guard.fitting_crop(bounds, work=work)
        extra_lines = added_guard.fitting_crop(bounds, work=work)
        if lines is None or extra_lines is None:
            return None
        line_parent = lines.observe(baseline / 255, work=work)
        extra_parent = extra_lines.observe(baseline / 255, work=work)
        stroke = next(e for e in root.iter() if e.get("id") == oid)
        body_root = _stroke_root(assembled, (oid,), size, work)
        joint_root = _stroke_root(assembled, (oid, bar), size, work)
        body_el = next(e for e in body_root.iter() if e.get("id") == oid)
        joint_el = next(e for e in joint_root.iter() if e.get("id") == oid)
        gaps = self.guard.gap_centres(limit=4096, work=work)
        gaps = gaps[((gaps >= bounds[:2]) & (gaps < bounds[2:])).all(axis=1)] - lo - 0.5
        best, best_value, guide, evaluations = None, float("inf"), float("inf"), 0

        def shape(parameters):
            coordinates = [list(n.values) for n in nodes]
            for amount, (i, j, direction) in zip(
                parameters[:-1], variables, strict=True
            ):
                coordinates[i][j : j + 2] = (
                    np.asarray(coordinates[i][j : j + 2]) + amount * direction
                )
            return replace(
                center,
                subpaths=(
                    replace(
                        center.subpaths[0],
                        nodes=tuple(
                            replace(n, values=tuple(map(float, v)))
                            for n, v in zip(nodes, coordinates, strict=True)
                        ),
                    ),
                ),
            )

        def score(parameters):
            nonlocal best, best_value, guide, evaluations
            _check(work)
            evaluations += 1
            geometry, fitted_width = shape(parameters), float(parameters[-1])
            data = transformed_geometry(geometry, inverse_matrix(frame)).path_data()
            for element in (stroke, body_el, joint_el):
                element.set("d", data)
                element.set("stroke-width", repr(fitted_width / scale))
            actual = _native_raster(root, size).values(box)
            body = _native_raster(body_root, size).values(box)[..., 3] / 255
            combined = _native_raster(joint_root, size).values(box)[..., 3] / 255
            body_penalty = (
                own_body.observe(body, work)["penalty"]
                + joint_body.observe(combined, work)["penalty"]
            )
            painted_penalty = lines.penalty(
                line_parent, actual / 255, work=work
            ) + extra_lines.penalty(extra_parent, actual / 255, work=work)
            w, h = bounds[2] - bounds[0], bounds[3] - bounds[1]
            opaque = render(
                f'<svg width="{w}" height="{h}" '
                f'viewBox="{bounds[0]} {bounds[1]} {w} {h}">'
                f'<path d="{geometry.path_data()}" fill="none" stroke="white" '
                f'stroke-width="{fitted_width}" stroke-linecap="round" '
                'stroke-linejoin="round"/></svg>',
                (w, h),
            )[..., 3]
            negative = map_coordinates(
                opaque, [gaps[:, 1], gaps[:, 0]], order=1, mode="constant", cval=0
            )
            gap_penalty = 1000 * float(np.maximum(0, negative - ALPHA_TOLERANCE).sum())
            alpha_error = int(
                np.abs(actual[..., 3].astype(int) - baseline[..., 3]).sum()
            )
            value = (
                alpha_error
                + body_penalty
                + painted_penalty
                + gap_penalty
                + float(parameters[:-1] @ parameters[:-1]) * 1e-6
            )
            guide = min(guide, value)
            if (
                alpha_error == body_penalty == painted_penalty == gap_penalty == 0
                and value < best_value
            ):
                best, best_value = np.array(parameters, copy=True), value
            _check(work)
            return value

        minimum = max(0.8, width * 0.5)
        initial = np.r_[np.zeros(len(variables)), minimum]
        limits = [(-float(MAX_MOVEMENT), float(MAX_MOVEMENT))] * len(variables) + [
            (minimum, min(MAX_WIDTH, width * 1.5))
        ]
        score(initial)
        minimize(
            score,
            initial,
            method="Powell",
            bounds=limits,
            options={
                "maxfev": MAX_EVALUATIONS - 2,
                "maxiter": 12,
                "xtol": 0.001,
                "ftol": 1e-6,
            },
        )
        _check(work)
        if best is None:
            return None
        score(best)
        fitted, fitted_width = shape(best), float(best[-1])
        if crossings(fitted) or not SourceAbsence(
            self.evidence, (), work, guard=self.guard
        ).permits(fitted, fitted_width, "round", work):
            return None
        editor = Editor(assembled, selection=Selection(whole_document=True))
        with editor.transaction("Fit complete attached source body") as tx:
            tx.replace_geometry(
                oid,
                identified(transformed_geometry(fitted, inverse_matrix(frame)), oid),
            )
            tx.set_attributes(oid, {"stroke-width": repr(fitted_width / scale)})
        candidate = editor.snapshot.document
        actual = _native_raster(ET.fromstring(export_svg(candidate)), size).root
        body_full = _native_raster(
            _stroke_root(candidate, (oid,), size, work), size
        ).root
        joint_alpha = (
            _native_raster(
                _stroke_root(candidate, (oid, bar), size, work), size
            ).values(box)[..., 3]
            / 255
        )
        own_proof = own_body.observe(body_full[box.slices][..., 3] / 255, work)
        joint_proof = joint_body.observe(joint_alpha, work)
        old_comparison = self.guard.compare(
            baseline_full / 255, actual / 255, work=work
        )
        new_comparison = added_guard.compare(
            baseline_full / 255, actual / 255, work=work
        )
        allowed = field_mask | (body_full[..., 3] > 0)
        if (
            not np.array_equal(actual[..., 3], baseline_full[..., 3])
            or not np.array_equal(actual[~allowed], baseline_full[~allowed])
            or not np.array_equal(
                actual[box.slices], _native_raster(root, size).values(box)
            )
            or own_proof["missing_samples"]
            or joint_proof["missing_samples"]
            or old_comparison["rejections"]
            or new_comparison["rejections"]
        ):
            return None
        _check(work)
        return candidate, {
            "scope": "complete-attached-source-body-before-family-binding",
            "accepted": False,
            "evaluations": evaluations,
            "parameter_count": len(best),
            "parameters": best.tolist(),
            "width": fitted_width,
            "best_guide": guide,
            "bounds": list(bounds),
            "fixed_ports": ports.tolist(),
            "fixed_raw_corners": [list(p) for p in centerline.corners],
            "affected_original_profiles": affected,
            "own_body": own_proof,
            "joint_body": joint_proof,
            "original_source_comparison": old_comparison,
            "added_source_comparison": new_comparison,
            "native_alpha_exact": True,
            "outside_field_plus_actual_body_rgba_exact": True,
            "complete_context_crop_exact": True,
            "native_body_absence": True,
        }
