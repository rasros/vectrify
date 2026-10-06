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
from vectrify.refine.cel_plan.model import (
    Candidate,
    Options,
    PlanningStoppedError,
    StageInterruptedError,
    Work,
)
from vectrify.refine.cel_plan.planning import merged_labels
from vectrify.refine.cel_plan.policy import Policy
from vectrify.refine.cel_plan.refine import refine
from vectrify.refine.cel_plan.score import (
    foreground_mask,
    measurements,
    opacity_measurements,
    render,
)


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
    policy = Policy.from_evidence(evidence, graph)
    frontier = Frontier(policy, observe)
    decisions = []
    # The same anchors feed every slider selection. Search effort can truncate
    # this sequence; selection on the resulting frozen frontier stays ordered.
    # Detailed geometry establishes the fixed representation normalizer first.
    proposals = (
        (100, False, False),
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
    # RGBA byte partitions first establish exact native coverage. A cheaper
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
    search = Work(work.deadline - reserve - refinement_reserve, work.stop, work.timings)
    normalizer_source = "conservative-fallback"
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
        if level == 100:
            try:
                frontier.freeze_normalizer(svg)
                normalizer_source = "validated-detailed"
            except ValueError:
                if frontier.baseline:
                    frontier.freeze_normalizer()
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
        "refinement_complete": False,
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
