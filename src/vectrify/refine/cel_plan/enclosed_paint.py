"""Couple a closed contour with an owned interior material and retained marks.

Source residuals screen whole owners, never assign only their majority pixels.
The resulting layered drawing is a competing hypothesis, accepted only by the
same native local/full policy as every other structural edit.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import cast

import numpy as np
import pathops
from cairosvg.colors import color
from scipy.ndimage import distance_transform_edt

from vectrify.document import Editor, Geometry, Selection
from vectrify.document.hit_test import multiply
from vectrify.document.join import curve_path, transformed_geometry
from vectrify.document.redraw import root_matrix
from vectrify.document.topology import inverse_matrix
from vectrify.refine import cel
from vectrify.refine.cel_plan.constraints import discard
from vectrify.refine.cel_plan.families import Families, _gradient, _opacity
from vectrify.refine.cel_plan.geometry import Model, ellipse
from vectrify.refine.cel_plan.ink_replace import identified
from vectrify.refine.cel_plan.layer_order import ordered
from vectrify.refine.cel_plan.local import Box
from vectrify.refine.cel_plan.model import Options, Work
from vectrify.refine.cel_plan.nested import Enclosed, in_core
from vectrify.refine.cel_plan.opacity import Paint, fit_samples
from vectrify.refine.cel_plan.ownership import Partition, Surface
from vectrify.refine.cel_plan.search import Proposal, State
from vectrify.refine.crossings import crossings
from vectrify.refine.tracing import _loops

MAX_PIXELS = 65_536
MAX_PERIMETER = 4_096
MAX_NODES = 6_000
RESIDUALS = (48.0, 24.0)
COVERAGE_REACH = 1.5  # Native pixels, independent of the analysis scale.


def prediction(paint: Paint, xy: np.ndarray) -> np.ndarray:
    if paint.gradient is None:
        return np.tile((*color(paint.color)[:3], paint.opacity), (len(xy), 1))
    ramp = paint.gradient
    axis = np.asarray(ramp.end) - ramp.start
    u = np.clip((xy - ramp.start) @ axis / float(axis @ axis), 0, 1)[:, None]
    ends = np.array([(*color(s.colour)[:3], s.opacity) for s in ramp.stops])
    return ends[0] * (1 - u) + ends[1] * u


def proposals(
    state: State,
    outer: Proposal,
    rim: str,
    nesting: Enclosed,
    box: Box,
    own: np.ndarray,
    families: Families,
    options: Options,
    work: Work,
    diagnostics: dict,
    *,
    rim_paint: Paint,
):
    from vectrify.refine.cel_plan.proposals import bounds

    if (
        work.interrupted
        or len(nesting.ids) < 2
        or box.area > MAX_PIXELS
        or outer.partition is None
    ):
        return
    evidence, graph = families.evidence, families.graph
    partition = outer.partition
    surfaces = {s.id: s for s in partition.surfaces}
    constrained = set(state.details.get("paint_constraints", ()))
    eligible = [
        oid
        for oid in nesting.ids
        if oid not in constrained
        and surfaces[oid].role == "surface"
        and not surfaces[oid].covered
        and not any(graph.regions[i].fixed for i in surfaces[oid].members)
    ]
    if len(eligible) < 2:
        return
    diagnostics["coherent_interior_attempts"] = (
        diagnostics.get("coherent_interior_attempts", 0) + 1
    )
    seed = min(
        eligible,
        key=lambda oid: (
            -sum(graph.regions[i].area for i in surfaces[oid].members),
            oid,
        ),
    )
    samples = families.samples(surfaces[seed].members, work)
    if samples is None or work.interrupted:
        return
    xy, rgba, size = samples
    if not size:
        return
    initial = fit_samples(xy, rgba, gradients=options.gradients)
    inside = nesting.mask & ~own
    loops = _loops(inside)
    if len(loops) != 1 or len(loops[0]) > MAX_PERIMETER:
        return
    points = np.array([*loops[0], loops[0][0]], dtype=float)
    points += (box.x, box.y)
    points = points / evidence.scale + evidence.offset
    compact = ellipse(points, options.boundary_tolerance)
    models = [] if compact is None else [compact]
    models.append(
        Model(cel._contour(points, options.boundary_tolerance), "contour", None)
    )
    y, x = np.nonzero(inside)
    xy = np.column_stack((x + box.x + 0.5, y + box.y + 0.5))
    observed = evidence.target[box.slices][inside] / 255
    predicted = prediction(initial, xy)
    residual = np.max(np.abs(predicted[:, :3] - observed), axis=1) * 255
    # Edge pixels can represent coverage between the material and its enclosing
    # rim. Test that explicit mixture rather than ignoring their source samples.
    # A high residual in the cavity interior still retains its complete owner.
    rim_values = prediction(rim_paint, xy)
    axis = predicted[:, :3] - rim_values[:, :3]
    u = np.clip(
        np.sum((observed - rim_values[:, :3]) * axis, axis=1)
        / np.maximum(np.sum(axis * axis, axis=1), 1e-12),
        0,
        1,
    )
    mixed = rim_values[:, :3] + u[:, None] * axis
    distance = cast(
        np.ndarray,
        distance_transform_edt(
            inside, sampling=(1 / evidence.scale[1], 1 / evidence.scale[0])
        ),
    )
    fringe = distance[inside] <= COVERAGE_REACH
    fringe &= np.abs(predicted[:, 3] - rim_values[:, 3]) <= 1 / 255
    mixture_error = np.max(np.abs(mixed - observed), axis=1) * 255
    explained = fringe & (mixture_error < residual)
    residual[explained] = mixture_error[explained]
    diagnostics["coherent_coverage_samples"] = diagnostics.get(
        "coherent_coverage_samples", 0
    ) + int(explained.sum())
    labels = graph.labels[box.slices][inside]
    # Coverage beside a distinct retained highlight or ink mark can also explain
    # an edge sample. A lone outlier in a matching owner cannot explain itself.
    for oid in nesting.ids:
        if work.interrupted:
            return
        context = families.samples(surfaces[oid].members, work)
        if context is None or work.interrupted:
            return
        context_xy, context_rgba, size = context
        if not size:
            continue
        mark_paint = fit_samples(context_xy, context_rgba, gradients=False)
        mark_values = prediction(mark_paint, xy)
        contrast = (
            np.max(
                np.abs(
                    np.median(
                        prediction(mark_paint, context_xy)[:, :3]
                        - prediction(initial, context_xy)[:, :3],
                        axis=0,
                    )
                )
            )
            * 255
        )
        if contrast <= max(RESIDUALS):
            continue
        mask = np.isin(graph.labels[box.slices], surfaces[oid].members) & inside
        distance = cast(
            np.ndarray,
            distance_transform_edt(
                ~mask, sampling=(1 / evidence.scale[1], 1 / evidence.scale[0])
            ),
        )
        near = ~mask[inside] & (distance[inside] <= COVERAGE_REACH)
        near &= np.abs(predicted[:, 3] - mark_values[:, 3]) <= 1 / 255
        axis = predicted[:, :3] - mark_values[:, :3]
        u = np.clip(
            np.sum((observed - mark_values[:, :3]) * axis, axis=1)
            / np.maximum(np.sum(axis * axis, axis=1), 1e-12),
            0,
            1,
        )
        mixed = mark_values[:, :3] + u[:, None] * axis
        error = np.max(np.abs(mixed - observed), axis=1) * 255
        explained = near & (error < residual)
        residual[explained] = error[explained]
        diagnostics["coherent_mark_coverage_samples"] = diagnostics.get(
            "coherent_mark_coverage_samples", 0
        ) + int(explained.sum())
    maximum = {
        oid: float(residual[np.isin(labels, surfaces[oid].members)].max())
        for oid in eligible
    }
    document = outer.document
    parent = document.ancestry(rim)[-2]
    seen = set()
    for limit in RESIDUALS:
        if work.interrupted:
            return
        ids = tuple(oid for oid in eligible if maximum[oid] <= limit)
        if len(ids) < 2 or ids in seen or seed not in ids:
            diagnostics["coherent_paint_exclusions"] = (
                diagnostics.get("coherent_paint_exclusions", 0) + 1
            )
            continue
        seen.add(ids)
        selected = [child.id for child in parent.children if child.id in ids]
        survivor = selected[-1]
        retained = tuple(oid for oid in nesting.ids if oid not in ids)
        members = tuple(sorted(i for oid in ids for i in surfaces[oid].members))
        covered = tuple(sorted(i for oid in retained for i in surfaces[oid].members))
        samples = families.samples(members, work)
        if samples is None or work.interrupted:
            return
        xy, rgba, size = samples
        if not size:
            continue
        paint = fit_samples(xy, rgba, gradients=options.gradients)
        paints = [paint]
        if paint.gradient is not None:
            paints.append(fit_samples(xy, rgba, gradients=False))
        opacity = _opacity(document, survivor)
        if opacity <= 0:
            continue
        inverse = inverse_matrix(root_matrix(document, survivor))
        # Enclosed's geometries are expressed in the original rim's frame.
        frame = multiply(inverse, root_matrix(document, rim))
        marks = Enclosed(
            inside,
            retained,
            covered,
            tuple(
                transformed_geometry(g, frame)
                for oid, g in zip(nesting.ids, nesting.geometries, strict=True)
                if oid in retained
            ),
            tuple(
                rule
                for oid, rule in zip(nesting.ids, nesting.rules, strict=True)
                if oid in retained
            ),
        )
        for model, paint in product(models, paints):
            if work.interrupted:
                return
            shape = identified(
                transformed_geometry(Geometry("interior", (model.contour,)), inverse),
                survivor,
            )
            if sum(len(s.nodes) for s in shape.subpaths) > MAX_NODES or crossings(
                shape
            ):
                continue
            if not marks.contains(shape, work):
                diagnostics["coherent_mark_exclusions"] = (
                    diagnostics.get("coherent_mark_exclusions", 0) + 1
                )
                continue
            in_rim = transformed_geometry(
                shape,
                multiply(
                    inverse_matrix(root_matrix(document, rim)),
                    root_matrix(document, survivor),
                ),
            )
            outside = pathops.op(
                curve_path(in_rim, "nonzero"),
                curve_path(document.geometry_for(rim), "nonzero"),
                pathops.PathOp.DIFFERENCE,
            )
            if abs(outside.area) > 1e-8 or work.interrupted:
                diagnostics["coherent_rim_exclusions"] = (
                    diagnostics.get("coherent_rim_exclusions", 0) + 1
                )
                continue
            if evidence.opacity is not None and not in_core(
                state, nesting.members, survivor, shape, work
            ):
                diagnostics["coherent_core_exclusions"] = (
                    diagnostics.get("coherent_core_exclusions", 0) + 1
                )
                continue
            changed = partition.replace(ids, (Surface(survivor, members),))
            changed = Partition(
                tuple(
                    replace(s, covered=covered) if s.id == survivor else s
                    for s in changed.surfaces
                ),
                changed.atoms,
            )
            editor = Editor(document, selection=Selection(whole_document=True))
            with editor.transaction(
                "Propose coherent enclosed material"
            ) as transaction:
                transaction.delete_objects(frozenset(set(ids) - {survivor}))
                transaction.replace_geometry(survivor, shape)
                transaction.set_fill(
                    survivor,
                    _gradient(paint, evidence, document, survivor, opacity)
                    if paint.gradient
                    else paint.color,
                )
                transaction.set_attributes(
                    survivor,
                    {
                        "fill-rule": "nonzero",
                        "stroke": "none",
                        "fill-opacity": "1"
                        if paint.gradient
                        else repr(min(1.0, paint.opacity / opacity)),
                    },
                )
            proposed = ordered(
                editor.snapshot.document,
                survivor,
                {rim},
                retained,
                shape,
                work,
                diagnostics,
            )
            if proposed is None or work.interrupted:
                continue
            edited = tuple(dict.fromkeys((*outer.ids, *nesting.ids)))
            holds = set((outer.details or {}).get("geometry_constraints", ())) - set(
                ids
            )
            holds.add(survivor)
            diagnostics["coherent_interior_proposals"] = (
                diagnostics.get("coherent_interior_proposals", 0) + 1
            )
            yield replace(
                outer,
                operator="closed-material",
                ids=edited,
                parameters=(
                    *outer.parameters,
                    model.kind,
                    "gradient" if paint.gradient else "flat",
                    limit,
                    members,
                ),
                document=proposed,
                bounds=bounds(state.document, proposed, edited),
                details={
                    **(outer.details or {}),
                    "regions": sum(s.role != "underlay" for s in changed.surfaces),
                    "geometry_constraints": sorted(holds),
                    "chain_constraints": discard(
                        state.details.get("chain_constraints"), edited
                    ),
                    "closed_overlay": {
                        **(outer.details or {}).get("closed_overlay", {}),
                        "nested_marks": (survivor, *retained),
                    },
                    "enclosed_material": {
                        "model": model.kind,
                        "paint_model": "gradient" if paint.gradient else "flat",
                        "residual": model.residual,
                        "source_members": members,
                        "retained_marks": retained,
                        "covered_members": covered,
                        "removed_paths": len(ids) - 1,
                        "paint_residual_limit": limit,
                        "coverage_reach": COVERAGE_REACH,
                    },
                },
                partition=changed,
            )
