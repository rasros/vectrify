"""Orchestrate evidence, structural proposals and native-render checkpoints."""

from __future__ import annotations

import time
from dataclasses import asdict, replace
from threading import Event

import numpy as np
from PIL import Image

from vectrify.refine.cel_plan.evidence import collect
from vectrify.refine.cel_plan.export import export
from vectrify.refine.cel_plan.frontier import Frontier
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.model import (
    Candidate,
    Options,
    PlanningStoppedError,
    Work,
)
from vectrify.refine.cel_plan.planning import merged_labels
from vectrify.refine.cel_plan.policy import Policy
from vectrify.refine.cel_plan.score import (
    foreground_mask,
    measurements,
    render,
)


def vectorize(
    image: Image.Image,
    *,
    alpha: np.ndarray | None = None,
    options: Options | None = None,
    seconds: float | None = None,
    stop: Event | None = None,
) -> Candidate:
    started = time.monotonic()
    options = options or Options()
    work = Work.start(seconds if seconds is not None else options.seconds, stop)
    if work.stop.is_set():
        raise PlanningStoppedError("Stopped before a validated candidate was available")
    evidence = collect(image, alpha, options, work)
    graph = build(evidence)
    if work.stop.is_set():
        raise PlanningStoppedError("Stopped before a validated candidate was available")
    policy = Policy.from_evidence(evidence, graph)
    frontier = Frontier(policy)
    decisions = []
    # The same anchors feed every slider selection. Search effort can truncate
    # this sequence; selection on the resulting frozen frontier stays ordered.
    # Detailed geometry establishes the fixed representation normalizer first.
    levels = (100, 0, 50, 25, 75)
    reserve = max(0.25, image.width * image.height / 1_000_000 * 0.5)
    last_candidate_seconds = 0.0
    validation_seconds = 0.0
    for level in levels:
        if frontier.baseline and (
            work.interrupted or work.remaining <= reserve + last_candidate_seconds
        ):
            break
        if work.stop.is_set():
            raise PlanningStoppedError(
                "Stopped before a validated candidate was available"
            )
        began = time.monotonic()
        proposal_options = replace(options, complexity=level)
        labels, edits = merged_labels(graph, proposal_options, work)
        decisions.extend(edits)
        svg, details = export(evidence, labels, proposal_options, work)
        if work.stop.is_set() and frontier.baseline is None:
            raise PlanningStoppedError(
                "Stopped before a validated candidate was available"
            )
        validated = time.monotonic()
        frontier.add(svg, f"Planned at complexity {level}", details)
        validation_seconds += time.monotonic() - validated
        last_candidate_seconds = time.monotonic() - began
    if frontier.baseline is None:
        raise ValueError(
            "No CEL planning candidate passed exact coverage and topology checks"
        )
    selected = frontier.select(options.complexity, node_budget=options.node_budget)
    actual = render(selected.svg, image.size)
    measured = measurements(actual, evidence.rgba, foreground_mask(evidence.rgba))
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
        "stopped": work.stop.is_set(),
        "deadline_overshoot": max(0.0, time.monotonic() - work.deadline),
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
