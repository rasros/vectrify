"""Orchestrate evidence, structural proposals and native-render checkpoints."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
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
    Evidence,
    Graph,
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


def _operator_diagnostics(operators: Operators) -> dict:
    """Copy optional operator reports without forcing their construction."""
    return {
        "scheduling": dict(operators.schedule_diagnostics),
        "surface_families": dict(operators.families.diagnostics),
        "material_surfaces": dict(operators.families.surface_models.diagnostics),
        "source_splits": dict(operators.families._split_models.diagnostics)
        if operators.families._split_models is not None
        else {},
        "piecewise_surfaces": dict(operators.families._piecewise_models.diagnostics)
        if operators.families._piecewise_models is not None
        else {},
        "core_material_cells": dict(operators.families._core_models.diagnostics)
        if operators.families._core_models is not None
        else {},
        "joint_material_cells": dict(operators.families._joint_models.diagnostics)
        if operators.families._joint_models is not None
        else {},
        "opacity_fields": dict(operators.opacity_fields.diagnostics),
        "ink_replacement": dict(operators.replacements.diagnostics),
        "source_ridges": dict(operators.ridges.diagnostics),
        "source_strokes": dict(operators.strokes.diagnostics),
        "source_silhouette_strokes": dict(operators.strokes.silhouettes.diagnostics)
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
        "nested_surface_rejections": dict(operators.families.nesting_rejections),
        "nested_overlay_rejections": dict(operators.overlays.nesting_rejections),
        "paint_restoration_rejections": dict(
            operators.replacements.restoration_rejections
        ),
    }


@dataclass
class _PlanningRun:
    """Live planning state; only the frontier owns validated candidates."""

    image: Image.Image
    options: Options
    work: Work
    evidence: Evidence
    graph: Graph
    frontier: Frontier
    started: float
    duration: float
    reserve: float = field(init=False)
    refinement_reserve: float = field(init=False)
    search_work: Work = field(init=False)
    deferred_fitting: bool = False
    fragmented_fallback: bool = False
    fitting_reserved: bool = False
    last_candidate_seconds: float = 0.0
    validation_seconds: float = 0.0
    decisions: list[dict] = field(default_factory=list)
    normalizer_source: str = "conservative-fallback"
    search_ran: bool = False
    structural_search: dict = field(
        default_factory=lambda: {
            "status": "unavailable",
            "attempted": 0,
            "accepted": 0,
            "seconds": 0.0,
        }
    )
    material_initialization: dict = field(
        default_factory=lambda: {
            "status": "unavailable",
            "merges": 0,
            "seconds": 0.0,
        }
    )
    core_initialization: dict = field(
        default_factory=lambda: {
            "status": "unavailable",
            "seconds": 0.0,
        }
    )

    def __post_init__(self) -> None:
        self.reserve = max(
            0.25,
            self.duration * 0.1,
            self.image.width * self.image.height / 1_000_000 * 0.5,
        )
        self.refinement_reserve = self.duration * 0.25 if self.options.refine else 0

    def establish_fallback(self) -> None:
        """Validate a conservative drawing before optional discovery."""
        # A conservative, independently validated plan exists before optional
        # fitting/search. Rejected traced curves must not trigger six expensive
        # attempts after the deadline merely because no checkpoint exists yet.
        if self.work.interrupted:
            if self.work.stop.is_set():
                raise PlanningStoppedError(
                    "Stopped before a validated candidate was available"
                )
            raise ValueError(
                "Time limit reached before a validated candidate was available"
            )
        # A bounded linear approximation cuts pixel staircases, while native hard
        # checks decide whether it is a safe fallback. Tighten only after rejection.
        bound = min(0.75, self.options.tolerance) if self.options.tolerance else 0.75
        # RGBA partitions first establish native coverage. A cheaper
        # approximation competes later; repeated lossy fallbacks waste the reserve.
        tolerances = (
            (0,) if self.evidence.opacity is not None else (bound, min(bound, 0.25), 0)
        )
        for tolerance in dict.fromkeys(tolerances):
            if self.work.interrupted:
                break
            try:
                fallback_svg, fallback_details = export(
                    self.evidence,
                    self.evidence.labels,
                    replace(self.options, complexity=100, gradients=False),
                    self.work,
                    conservative=True,
                    conservative_tolerance=tolerance,
                )
            except StageInterruptedError:
                break
            validated = time.monotonic()
            if self.work.stop.is_set():
                raise PlanningStoppedError(
                    "Stopped before a validated candidate was available"
                )
            self.frontier.add(
                fallback_svg, "Conservative CEL fallback", fallback_details
            )
            self.validation_seconds += time.monotonic() - validated
            if self.frontier.baseline:
                break

    def allocate_search(self) -> None:
        """Defer fitting for dense seeds without consuming the final reserve."""
        dense_fallback = (
            self.frontier.baseline is not None
            and self.frontier.baseline.evaluation.structure["nodes"]
            > MAX_GEOMETRY_NODES
        )
        self.fragmented_fallback = (
            self.options.quality == "high"
            and self.frontier.baseline is not None
            and self.frontier.baseline.evaluation.structure["paths"] > MAX_EDIT_OBJECTS
        )
        self.deferred_fitting = dense_fallback or self.fragmented_fallback
        self.search_work = Work(
            self.work.deadline
            - self.reserve
            - (0 if self.deferred_fitting else self.refinement_reserve),
            self.work.stop,
            self.work.timings,
        )
        self.fitting_reserved = not self.deferred_fitting

    def reserve_fitting(self) -> None:
        """Reclaim fitting time only after a usable interpretation is retained."""
        if (
            self.options.refine
            and not self.fitting_reserved
            and any(
                entry.evaluation.structure["nodes"] <= MAX_GEOMETRY_NODES
                and (
                    self.options.quality != "high"
                    or entry.evaluation.structure["paths"] <= MAX_EDIT_OBJECTS
                    or entry.details.get("joint_cell_search")
                )
                for entry in self.frontier.entries
            )
        ):
            # High avoids fitting thousands of fragments before a complete
            # owned ink/material interpretation can compete. Once retained,
            # that interpretation gets fitting within the remaining live time.
            self.search_work.deadline = min(
                self.search_work.deadline,
                self.work.deadline
                - self.reserve
                - min(self.refinement_reserve, self.work.remaining * 0.35),
            )
            self.fitting_reserved = True

    def search_anchors(self) -> None:
        """Visit the fixed anchor prefix, then offer local edits once."""
        # The same anchors feed every slider selection. Search effort can truncate
        # this sequence; selection on the resulting frozen frontier stays ordered.
        # Detailed geometry establishes the fixed representation normalizer first.
        # Translucent subdivisions need a supported base interpretation here too;
        # otherwise antialias gaps can reject every detailed normalization seed.
        proposals = (
            (100, False, self.evidence.opacity is not None),
            (50, True, True),
            (0, False, False),
            (50, False, False),
            (25, False, False),
            (75, False, False),
        )
        for level, structure, layers in proposals:
            if self.search_work.interrupted or self.work.interrupted:
                break
            if self.frontier.baseline and (
                self.search_work.remaining <= self.last_candidate_seconds
            ):
                break
            if self.work.stop.is_set():
                if self.frontier.baseline is None:
                    raise PlanningStoppedError(
                        "Stopped before a validated candidate was available"
                    )
                break
            began = time.monotonic()
            proposal_options = replace(self.options, complexity=level)
            labels, edits = merged_labels(
                self.graph, proposal_options, self.search_work
            )
            self.decisions.extend(edits)
            try:
                svg, details = export(
                    self.evidence,
                    labels,
                    proposal_options,
                    self.search_work,
                    structure=structure,
                    layers=layers,
                    cost_normalizer=self.frontier.normalizer
                    if self.frontier.normalizer_fixed
                    else None,
                )
            except StageInterruptedError:
                break
            if self.work.stop.is_set():
                if self.frontier.baseline is None:
                    raise PlanningStoppedError(
                        "Stopped before a validated candidate was available"
                    )
                break
            validated = time.monotonic()
            label = f"{'Structured' if structure else 'Traced'} at complexity {level}"
            self.frontier.add(svg, label, details)
            self.validation_seconds += time.monotonic() - validated
            self.last_candidate_seconds = time.monotonic() - began
            self.reserve_fitting()
            if level == 100:
                self.initialize_materials(svg, labels)
            if (
                not self.search_ran
                and self.frontier.normalizer_fixed
                and not self.search_work.interrupted
                and any(
                    entry.evaluation.structure["nodes"] <= MAX_GEOMETRY_NODES
                    for entry in self.frontier.entries
                )
            ):
                self.run_structural_search()

    def initialize_materials(self, svg: str, labels: np.ndarray) -> None:
        """Freeze the detailed cost scale before optional material seeds."""
        detailed_valid = False
        try:
            self.frontier.freeze_normalizer(svg)
            self.normalizer_source = "validated-detailed"
            detailed_valid = True
        except ValueError:
            if self.frontier.baseline:
                self.frontier.freeze_normalizer()
        if (
            self.evidence.opacity is not None
            and self.frontier.normalizer_fixed
            and not self.search_work.interrupted
        ):
            core_started = time.monotonic()
            try:
                self.core_initialization = core_material_seed(
                    self.frontier,
                    self.evidence,
                    self.graph,
                    self.options,
                    self.search_work,
                    self.duration,
                )
            except (
                ValueError,
                RuntimeError,
                ArithmeticError,
                np.linalg.LinAlgError,
            ) as exc:
                self.core_initialization = {
                    "status": "failed",
                    "detail": str(exc),
                    "seconds": time.monotonic() - core_started,
                }
            self.validation_seconds += self.core_initialization.get(
                "validation_seconds", 0
            )
            self.work.timings["core_material_initialization"] = (
                self.core_initialization["seconds"]
            )
            material_started = time.monotonic()
            try:
                self.material_initialization = material_seed(
                    self.frontier,
                    self.evidence,
                    self.graph,
                    self.options,
                    self.search_work,
                    self.duration,
                    starting_labels=labels if detailed_valid else None,
                )
            except (
                ValueError,
                RuntimeError,
                ArithmeticError,
                np.linalg.LinAlgError,
            ) as exc:
                self.material_initialization = {
                    "status": "failed",
                    "detail": str(exc),
                    "seconds": time.monotonic() - material_started,
                }
            self.validation_seconds += self.material_initialization.get(
                "validation_seconds", 0
            )
            self.work.timings["material_initialization"] = self.material_initialization[
                "seconds"
            ]

    def run_structural_search(self) -> None:
        """Bound local discovery and publish through the shared search reserve."""
        # Individual operators get an opportunity before complete anchor
        # proposals consume the remaining search prefix. Fitting and final
        # validation keep their independent reservations.
        # Large owned components need a native context before any sealed
        # reconstruction can compete. Give that High-quality opportunity
        # a larger share of the same search prefix; final validation and
        # any previously reserved fitting time still bound its deadline.
        large_core = self.options.quality == "high" and any(
            entry.details.get("alpha_model") == "material-core-silhouette"
            and entry.evaluation.structure["paths"] > MAX_EDIT_OBJECTS
            for entry in self.frontier.entries
        )
        local_seconds = self.duration * (0.3 if large_core else 0.2)
        local_work = Work(
            min(self.search_work.deadline, time.monotonic() + local_seconds),
            self.work.stop,
            self.work.timings,
        )
        local_started = time.monotonic()
        operators = None
        try:
            operators = Operators(
                self.evidence, self.graph, replace(self.options, complexity=50)
            )
            if large_core and getattr(operators, "bands", None) is not None:
                # The explicit complete-band competitor includes its
                # independent source bank and painted width trials. Keep
                # their opportunity inside the same global reserves.
                local_seconds = self.duration * 0.4
                local_work.deadline = min(
                    self.search_work.deadline, local_started + local_seconds
                )
            self.structural_search = local_search(
                self.frontier,
                self.options,
                local_work,
                operators,
                checkpoint_work=self.search_work,
                minimum_checkpoint_seconds=max(
                    0.05, self.validation_seconds / max(1, len(self.frontier.decisions))
                ),
            )
        except (ValueError, RuntimeError, ArithmeticError) as exc:
            # Optional search never replaces the independently validated
            # checkpoint with a partial working state after failure.
            self.structural_search = {
                "status": "failed",
                "detail": str(exc),
                "attempted": None,
                "accepted": None,
                "seconds": time.monotonic() - local_started,
                "validation_seconds_unavailable": True,
            }
        if operators is not None:
            self.structural_search["operator_diagnostics"] = _operator_diagnostics(
                operators
            )
        self.search_ran = True
        self.structural_search["discovery_allocation_seconds"] = local_seconds
        self.validation_seconds += self.structural_search.get("validation_seconds", 0)
        self.work.timings["structural_search"] = self.structural_search["seconds"]
        self.reserve_fitting()

    def require_checkpoint(self) -> None:
        """Resolve failure before selection and freeze the fallback scale if needed."""
        if self.frontier.baseline is None:
            if self.work.stop.is_set():
                raise PlanningStoppedError(
                    "Stopped before a validated candidate was available"
                )
            reasons = sorted(
                {
                    reason
                    for decision in self.frontier.decisions
                    for reason in decision.get("rejections", ())
                }
            )
            raise ValueError(
                "No CEL planning candidate passed exact coverage and topology checks"
                + (f": {', '.join(reasons)}" if reasons else "")
            )
        if not self.frontier.normalizer_fixed:
            self.frontier.freeze_normalizer()

    def refine_candidate(self) -> tuple[Candidate, dict]:
        """Fit within its independent reserve and select a validated drawing."""
        before_refinement = self.frontier.select(
            self.options.complexity, node_budget=self.options.node_budget
        )
        refinement = refine(
            self.frontier,
            self.evidence,
            self.options,
            Work(self.work.deadline - self.reserve, self.work.stop, self.work.timings),
        )
        self.validation_seconds += refinement.get("validation_seconds", 0)
        selected = self.frontier.select(
            self.options.complexity, node_budget=self.options.node_budget
        )
        refinement["before_objective"] = before_refinement.metrics["objective"]
        refinement["after_objective"] = selected.metrics["objective"]
        return selected, refinement

    def report(self, selected: Candidate, refinement: dict) -> Candidate:
        """Measure the selected native drawing and preserve the public report schema."""
        actual = render(selected.svg, self.image.size)
        measured = measurements(
            actual, self.evidence.rgba, foreground_mask(self.evidence.rgba)
        )
        measured["opacity"] = opacity_measurements(actual, self.evidence.rgba)
        self.work.timings["validation"] = self.validation_seconds
        metrics = {
            **selected.metrics,
            "complexity": self.options.complexity,
            "quality": self.options.quality,
            "requested_settings": asdict(self.options),
            "resolved_settings": {
                **asdict(self.options),
                "palette": self.options.palette_size,
                "tolerance": selected.metrics.get(
                    "boundary_tolerance", self.options.boundary_tolerance
                ),
                "line_width": selected.metrics.get(
                    "outline_width", self.options.line_width
                ),
            },
            "reference": measured,
            "decisions": len(self.decisions),
            "candidate_decisions": self.frontier.decisions,
            "timings": self.work.timings,
            "seconds": time.monotonic() - self.started,
            "out_of_time": self.work.remaining == 0,
            "search_out_of_time": self.search_work.remaining == 0,
            "stopped": self.work.stop.is_set(),
            "deadline_overshoot": max(0.0, time.monotonic() - self.work.deadline),
            "cost_normalizer_source": self.normalizer_source,
            "refinement": refinement,
            "structural_search": self.structural_search,
            "material_initialization": self.material_initialization,
            "core_material_initialization": self.core_initialization,
            "refinement_complete": False,
            "fitting_time_reclaimed_after_compaction": self.deferred_fitting
            and self.fitting_reserved,
            "fitting_deferred_for_fragmentation": self.fragmented_fallback,
        }
        alternatives = tuple(
            candidate
            for candidate in self.frontier.alternatives(
                self.options.complexity, node_budget=self.options.node_budget
            )
            if candidate.svg != selected.svg
        )
        return Candidate(
            selected.svg,
            "Planned cel drawing",
            self.options.complexity,
            metrics,
            tuple(self.decisions),
            alternatives,
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
    policy = Policy.from_evidence(
        evidence,
        graph,
        weights=Weights(features=Weights().features * options.protection),
    )
    frontier = Frontier(policy, observe)
    run = _PlanningRun(
        image=image,
        options=options,
        work=work,
        evidence=evidence,
        graph=graph,
        frontier=frontier,
        started=started,
        duration=seconds if seconds is not None else options.seconds,
    )
    run.establish_fallback()
    run.allocate_search()
    run.search_anchors()
    run.require_checkpoint()
    selected, refinement = run.refine_candidate()
    return run.report(selected, refinement)
