"""Orchestrate evidence, structural proposals and native-render checkpoints."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, replace
from threading import Event

import numpy as np
from PIL import Image

from vectrify.refine.cel_plan.evidence import collect
from vectrify.refine.cel_plan.export import export
from vectrify.refine.cel_plan.frontier import Frontier, Observation
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.materials import coherent_labels
from vectrify.refine.cel_plan.model import (
    Candidate,
    Options,
    PlanningStoppedError,
    StageInterruptedError,
    Work,
)
from vectrify.refine.cel_plan.planning import merged_labels
from vectrify.refine.cel_plan.policy import Policy, Weights
from vectrify.refine.cel_plan.proposals import Operators
from vectrify.refine.cel_plan.refine import MAX_GEOMETRY_NODES, refine
from vectrify.refine.cel_plan.score import (
    foreground_mask,
    measurements,
    opacity_measurements,
    render,
)
from vectrify.refine.cel_plan.search import MAX_EDIT_OBJECTS
from vectrify.refine.cel_plan.search import search as local_search


def core_material_seed(frontier, evidence, graph, options, work, duration) -> dict:
    """A fitted material silhouette competes with the native RGBA partitions."""
    from vectrify.refine.cel_plan.core_materials import candidate

    began = time.monotonic()
    discovery = Work(
        min(work.deadline, began + min(duration * 0.2, work.remaining * 0.35)),
        work.stop,
        work.timings,
    )
    try:
        result, details = candidate(
            evidence,
            graph,
            frontier.policy,
            options,
            work,
            normalizer=frontier.normalizer,
            discovery=discovery,
        )
    except StageInterruptedError:
        return {"status": "discovery-interrupted", "seconds": time.monotonic() - began}
    if result is None or work.interrupted:
        return {**details, "seconds": time.monotonic() - began, "validation_seconds": 0}
    svg, exported = result
    validation = time.monotonic()
    retained = frontier.add(svg, "Fitted material silhouettes", exported)
    return {
        **details,
        "status": "retained" if retained else "rejected",
        "generation_status": details.get("growth", {}).get("status", "complete"),
        "seconds": time.monotonic() - began,
        "validation_seconds": time.monotonic() - validation,
    }


def material_seed(
    frontier, evidence, graph, options, work, duration, *, starting_labels=None
) -> dict:
    """Optional coherent initialization, retained only by the native frontier."""
    began = time.monotonic()
    discovery = Work(
        min(work.deadline, began + min(duration * 0.12, work.remaining * 0.3)),
        work.stop,
        work.timings,
    )
    try:
        if starting_labels is not None:
            graph = build(evidence, starting_labels, work=discovery)
        labels, details = coherent_labels(
            evidence,
            graph,
            replace(options, complexity=50),
            discovery,
            normalizer=frontier.normalizer,
        )
        details["initial_partition"] = (
            "validated-detailed" if starting_labels is not None else "source-atoms"
        )
    except StageInterruptedError:
        return {"status": "discovery-interrupted", "seconds": time.monotonic() - began}
    if not details["merges"] or work.interrupted:
        return {**details, "seconds": time.monotonic() - began, "validation_seconds": 0}
    try:
        svg, exported = export(
            evidence,
            labels,
            replace(options, complexity=50),
            work,
            layers=True,
            cost_normalizer=frontier.normalizer,
        )
    except StageInterruptedError:
        return {
            **details,
            "status": "export-interrupted",
            "seconds": time.monotonic() - began,
        }
    if work.stop.is_set():
        return {**details, "status": "stopped", "seconds": time.monotonic() - began}
    validation = time.monotonic()
    retained = frontier.add(
        svg, "Coherent RGBA materials", {**exported, "material_models": details}
    )
    return {
        **details,
        "status": "retained" if retained else "rejected",
        "generation_status": details["status"],
        "seconds": time.monotonic() - began,
        "validation_seconds": time.monotonic() - validation,
    }


def vectorize(
    image: Image.Image,
    *,
    alpha: np.ndarray | None = None,
    options: Options | None = None,
    seconds: float | None = None,
    stop: Event | None = None,
    observe: Callable[[Observation], None] | None = None,
) -> Candidate:
    started = time.monotonic()
    options = options or Options()
    work = Work.start(seconds if seconds is not None else options.seconds, stop)
    if work.stop.is_set():
        raise PlanningStoppedError("Stopped before a validated candidate was available")
    evidence = collect(image, alpha, options, work)
    try:
        graph = build(evidence, work=work)
    except StageInterruptedError as exc:
        if work.stop.is_set():
            raise PlanningStoppedError(
                "Stopped before a validated candidate was available"
            ) from exc
        raise ValueError(
            "Time limit reached before a validated candidate was available"
        ) from exc
    if work.stop.is_set():
        raise PlanningStoppedError("Stopped before a validated candidate was available")
    policy = Policy.from_evidence(
        evidence,
        graph,
        weights=Weights(features=Weights().features * options.protection),
    )
    frontier = Frontier(policy, observe)
    decisions = []
    # The same anchors feed every slider selection. Search effort can truncate
    # this sequence; selection on the resulting frozen frontier stays ordered.
    # Detailed geometry establishes the fixed representation normalizer first.
    # Translucent subdivisions need a supported base interpretation here too;
    # otherwise antialias gaps can reject every detailed normalization seed.
    proposals = (
        (100, False, evidence.opacity is not None),
        (50, True, True),
        (0, False, False),
        (50, False, False),
        (25, False, False),
        (75, False, False),
    )
    duration = seconds if seconds is not None else options.seconds
    reserve = max(0.25, duration * 0.1, image.width * image.height / 1_000_000 * 0.5)
    refinement_reserve = duration * 0.25 if options.refine else 0
    last_candidate_seconds = 0.0
    validation_seconds = 0.0
    # A conservative, independently validated plan exists before optional
    # fitting/search. Rejected traced curves must not trigger six expensive
    # attempts after the deadline merely because no checkpoint exists yet.
    if work.interrupted:
        if work.stop.is_set():
            raise PlanningStoppedError(
                "Stopped before a validated candidate was available"
            )
        raise ValueError(
            "Time limit reached before a validated candidate was available"
        )
    # A bounded linear approximation cuts pixel staircases, while native hard
    # checks decide whether it is a safe fallback. Tighten only after rejection.
    bound = min(0.75, options.tolerance) if options.tolerance else 0.75
    # RGBA partitions first establish native coverage. A cheaper
    # approximation competes later; repeated lossy fallbacks waste the reserve.
    tolerances = (0,) if evidence.opacity is not None else (bound, min(bound, 0.25), 0)
    for tolerance in dict.fromkeys(tolerances):
        if work.interrupted:
            break
        try:
            fallback_svg, fallback_details = export(
                evidence,
                evidence.labels,
                replace(options, complexity=100, gradients=False),
                work,
                conservative=True,
                conservative_tolerance=tolerance,
            )
        except StageInterruptedError:
            break
        validated = time.monotonic()
        if work.stop.is_set():
            raise PlanningStoppedError(
                "Stopped before a validated candidate was available"
            )
        frontier.add(fallback_svg, "Conservative CEL fallback", fallback_details)
        validation_seconds += time.monotonic() - validated
        if frontier.baseline:
            break
    dense_fallback = (
        frontier.baseline is not None
        and frontier.baseline.evaluation.structure["nodes"] > MAX_GEOMETRY_NODES
    )
    fragmented_fallback = (
        options.quality == "high"
        and frontier.baseline is not None
        and frontier.baseline.evaluation.structure["paths"] > MAX_EDIT_OBJECTS
    )
    deferred_fitting = dense_fallback or fragmented_fallback
    search = Work(
        work.deadline - reserve - (0 if deferred_fitting else refinement_reserve),
        work.stop,
        work.timings,
    )
    fitting_reserved = not deferred_fitting

    def reserve_fitting():
        nonlocal fitting_reserved
        if (
            options.refine
            and not fitting_reserved
            and any(
                entry.evaluation.structure["nodes"] <= MAX_GEOMETRY_NODES
                and (
                    options.quality != "high"
                    or entry.evaluation.structure["paths"] <= MAX_EDIT_OBJECTS
                    or entry.details.get("joint_cell_search")
                )
                for entry in frontier.entries
            )
        ):
            # High avoids fitting thousands of fragments before a complete
            # owned ink/material interpretation can compete. Once retained,
            # that interpretation gets fitting within the remaining live time.
            search.deadline = min(
                search.deadline,
                work.deadline
                - reserve
                - min(refinement_reserve, work.remaining * 0.35),
            )
            fitting_reserved = True

    normalizer_source = "conservative-fallback"
    structural_search: dict = {
        "status": "unavailable",
        "attempted": 0,
        "accepted": 0,
        "seconds": 0.0,
    }
    search_ran = False
    material_initialization: dict = {
        "status": "unavailable",
        "merges": 0,
        "seconds": 0.0,
    }
    core_initialization: dict = {"status": "unavailable", "seconds": 0.0}
    for level, structure, layers in proposals:
        if search.interrupted or work.interrupted:
            break
        if frontier.baseline and (search.remaining <= last_candidate_seconds):
            break
        if work.stop.is_set():
            if frontier.baseline is None:
                raise PlanningStoppedError(
                    "Stopped before a validated candidate was available"
                )
            break
        began = time.monotonic()
        proposal_options = replace(options, complexity=level)
        labels, edits = merged_labels(graph, proposal_options, search)
        decisions.extend(edits)
        try:
            svg, details = export(
                evidence,
                labels,
                proposal_options,
                search,
                structure=structure,
                layers=layers,
                cost_normalizer=frontier.normalizer
                if frontier.normalizer_fixed
                else None,
            )
        except StageInterruptedError:
            break
        if work.stop.is_set():
            if frontier.baseline is None:
                raise PlanningStoppedError(
                    "Stopped before a validated candidate was available"
                )
            break
        validated = time.monotonic()
        label = f"{'Structured' if structure else 'Traced'} at complexity {level}"
        frontier.add(svg, label, details)
        validation_seconds += time.monotonic() - validated
        last_candidate_seconds = time.monotonic() - began
        reserve_fitting()
        if level == 100:
            detailed_valid = False
            try:
                frontier.freeze_normalizer(svg)
                normalizer_source = "validated-detailed"
                detailed_valid = True
            except ValueError:
                if frontier.baseline:
                    frontier.freeze_normalizer()
            if (
                evidence.opacity is not None
                and frontier.normalizer_fixed
                and not search.interrupted
            ):
                core_started = time.monotonic()
                try:
                    core_initialization = core_material_seed(
                        frontier, evidence, graph, options, search, duration
                    )
                except (
                    ValueError,
                    RuntimeError,
                    ArithmeticError,
                    np.linalg.LinAlgError,
                ) as exc:
                    core_initialization = {
                        "status": "failed",
                        "detail": str(exc),
                        "seconds": time.monotonic() - core_started,
                    }
                validation_seconds += core_initialization.get("validation_seconds", 0)
                work.timings["core_material_initialization"] = core_initialization[
                    "seconds"
                ]
                material_started = time.monotonic()
                try:
                    material_initialization = material_seed(
                        frontier,
                        evidence,
                        graph,
                        options,
                        search,
                        duration,
                        starting_labels=labels if detailed_valid else None,
                    )
                except (
                    ValueError,
                    RuntimeError,
                    ArithmeticError,
                    np.linalg.LinAlgError,
                ) as exc:
                    material_initialization = {
                        "status": "failed",
                        "detail": str(exc),
                        "seconds": time.monotonic() - material_started,
                    }
                validation_seconds += material_initialization.get(
                    "validation_seconds", 0
                )
                work.timings["material_initialization"] = material_initialization[
                    "seconds"
                ]
        if (
            not search_ran
            and frontier.normalizer_fixed
            and not search.interrupted
            and any(
                entry.evaluation.structure["nodes"] <= MAX_GEOMETRY_NODES
                for entry in frontier.entries
            )
        ):
            # Individual operators get an opportunity before complete anchor
            # proposals consume the remaining search prefix. Fitting and final
            # validation keep their independent reservations.
            # Large owned components need a native context before any sealed
            # reconstruction can compete. Give that High-quality opportunity
            # a larger share of the same search prefix; final validation and
            # any previously reserved fitting time still bound its deadline.
            large_core = options.quality == "high" and any(
                entry.details.get("alpha_model") == "material-core-silhouette"
                and entry.evaluation.structure["paths"] > MAX_EDIT_OBJECTS
                for entry in frontier.entries
            )
            local_seconds = duration * (0.3 if large_core else 0.2)
            local_work = Work(
                min(search.deadline, time.monotonic() + local_seconds),
                work.stop,
                work.timings,
            )
            local_started = time.monotonic()
            operators = None
            try:
                operators = Operators(evidence, graph, replace(options, complexity=50))
                if large_core and getattr(operators, "bands", None) is not None:
                    # The explicit complete-band competitor includes its
                    # independent source bank and painted width trials. Keep
                    # their opportunity inside the same global reserves.
                    local_seconds = duration * 0.4
                    local_work.deadline = min(
                        search.deadline, local_started + local_seconds
                    )
                structural_search = local_search(
                    frontier,
                    options,
                    local_work,
                    operators,
                    checkpoint_work=search,
                    minimum_checkpoint_seconds=max(
                        0.05, validation_seconds / max(1, len(frontier.decisions))
                    ),
                )
            except (ValueError, RuntimeError, ArithmeticError) as exc:
                # Optional search never replaces the independently validated
                # checkpoint with a partial working state after failure.
                structural_search = {
                    "status": "failed",
                    "detail": str(exc),
                    "attempted": None,
                    "accepted": None,
                    "seconds": time.monotonic() - local_started,
                    "validation_seconds_unavailable": True,
                }
            if operators is not None:
                structural_search["operator_diagnostics"] = {
                    "scheduling": dict(operators.schedule_diagnostics),
                    "surface_families": dict(operators.families.diagnostics),
                    "material_surfaces": dict(
                        operators.families.surface_models.diagnostics
                    ),
                    "source_splits": dict(operators.families._split_models.diagnostics)
                    if operators.families._split_models is not None
                    else {},
                    "piecewise_surfaces": dict(
                        operators.families._piecewise_models.diagnostics
                    )
                    if operators.families._piecewise_models is not None
                    else {},
                    "core_material_cells": dict(
                        operators.families._core_models.diagnostics
                    )
                    if operators.families._core_models is not None
                    else {},
                    "joint_material_cells": dict(
                        operators.families._joint_models.diagnostics
                    )
                    if operators.families._joint_models is not None
                    else {},
                    "opacity_fields": dict(operators.opacity_fields.diagnostics),
                    "ink_replacement": dict(operators.replacements.diagnostics),
                    "source_ridges": dict(operators.ridges.diagnostics),
                    "source_strokes": dict(operators.strokes.diagnostics),
                    "source_silhouette_strokes": dict(
                        operators.strokes.silhouettes.diagnostics
                    )
                    if operators.strokes.silhouettes is not None
                    else {},
                    "source_ridge_models": dict(operators.ridges.rim_diagnostics),
                    "source_ridge_underpaint_rejections": dict(
                        operators.ridges.underpaint_rejections
                    ),
                    "source_ridge_restoration_rejections": dict(
                        operators.ridges.restoration_rejections
                    ),
                    "closed_overlays": dict(operators.overlays.diagnostics),
                    "nested_surface_rejections": dict(
                        operators.families.nesting_rejections
                    ),
                    "nested_overlay_rejections": dict(
                        operators.overlays.nesting_rejections
                    ),
                    "paint_restoration_rejections": dict(
                        operators.replacements.restoration_rejections
                    ),
                }
            search_ran = True
            structural_search["discovery_allocation_seconds"] = local_seconds
            validation_seconds += structural_search.get("validation_seconds", 0)
            work.timings["structural_search"] = structural_search["seconds"]
            reserve_fitting()
    if frontier.baseline is None:
        if work.stop.is_set():
            raise PlanningStoppedError(
                "Stopped before a validated candidate was available"
            )
        reasons = sorted(
            {
                reason
                for decision in frontier.decisions
                for reason in decision.get("rejections", ())
            }
        )
        raise ValueError(
            "No CEL planning candidate passed exact coverage and topology checks"
            + (f": {', '.join(reasons)}" if reasons else "")
        )
    if not frontier.normalizer_fixed:
        frontier.freeze_normalizer()
    before_refinement = frontier.select(
        options.complexity, node_budget=options.node_budget
    )
    refinement = refine(
        frontier,
        evidence,
        options,
        Work(work.deadline - reserve, work.stop, work.timings),
    )
    validation_seconds += refinement.get("validation_seconds", 0)
    selected = frontier.select(options.complexity, node_budget=options.node_budget)
    refinement["before_objective"] = before_refinement.metrics["objective"]
    refinement["after_objective"] = selected.metrics["objective"]
    actual = render(selected.svg, image.size)
    measured = measurements(actual, evidence.rgba, foreground_mask(evidence.rgba))
    measured["opacity"] = opacity_measurements(actual, evidence.rgba)
    work.timings["validation"] = validation_seconds
    metrics = {
        **selected.metrics,
        "complexity": options.complexity,
        "quality": options.quality,
        "requested_settings": asdict(options),
        "resolved_settings": {
            **asdict(options),
            "palette": options.palette_size,
            "tolerance": selected.metrics.get(
                "boundary_tolerance", options.boundary_tolerance
            ),
            "line_width": selected.metrics.get("outline_width", options.line_width),
        },
        "reference": measured,
        "decisions": len(decisions),
        "candidate_decisions": frontier.decisions,
        "timings": work.timings,
        "seconds": time.monotonic() - started,
        "out_of_time": work.remaining == 0,
        "search_out_of_time": search.remaining == 0,
        "stopped": work.stop.is_set(),
        "deadline_overshoot": max(0.0, time.monotonic() - work.deadline),
        "cost_normalizer_source": normalizer_source,
        "refinement": refinement,
        "structural_search": structural_search,
        "material_initialization": material_initialization,
        "core_material_initialization": core_initialization,
        "refinement_complete": False,
        "fitting_time_reclaimed_after_compaction": deferred_fitting
        and fitting_reserved,
        "fitting_deferred_for_fragmentation": fragmented_fallback,
    }
    alternatives = tuple(
        candidate
        for candidate in frontier.alternatives(
            options.complexity, node_budget=options.node_budget
        )
        if candidate.svg != selected.svg
    )
    return Candidate(
        selected.svg,
        "Planned cel drawing",
        options.complexity,
        metrics,
        tuple(decisions),
        alternatives,
    )
