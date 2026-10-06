"""Bounded exact-validated alternatives, independent of the slider selection."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable
from dataclasses import dataclass

from vectrify.refine.cel_plan.model import Candidate
from vectrify.refine.cel_plan.policy import Evaluation, Policy

MAX_CANDIDATES = 12
MAX_BYTES = 64 * 1024 * 1024
CHECKPOINTS = (0, 25, 50, 75, 100)


def _dominates(before: Evaluation, after: Evaluation) -> bool:
    # Cost includes paths/contours/paint as well as nodes. A cheaper drawing
    # can therefore have more nodes; keep the lower-node tradeoff so explicit
    # node ceilings do not lose their feasible candidate.
    return (
        before.cost <= after.cost
        and before.visual <= after.visual
        and before.structure["nodes"] <= after.structure["nodes"]
    )


@dataclass(frozen=True)
class Entry:
    svg: str
    label: str
    evaluation: Evaluation
    key: str
    details: dict


@dataclass(frozen=True)
class Observation:
    """A diagnostic copy of an evaluated proposal, including rejected ones.

    An observer receives no policy or frontier reference and cannot supply a
    score. Benchmark clean targets therefore stay outside planning decisions.
    SVG retention belongs to the observer's separately bounded artifact sink.
    """

    svg: str
    key: str
    label: str
    evaluation: Evaluation | None
    details: dict
    decision: dict


class Frontier:
    """Retain nondominated drawings and one separate interruption fallback."""

    def __init__(
        self, policy: Policy, observe: Callable[[Observation], None] | None = None
    ):
        self.policy = policy
        self.observe = observe
        self.entries: list[Entry] = []
        self.baseline: Entry | None = None
        self.decisions: list[dict] = []
        self._normalizer: float | None = None
        self._published = False
        self._validated_scale: tuple[str, float] | None = None
        self._refinement_protected: set[str] | None = None

    def _record(
        self,
        svg: str,
        decision: dict,
        details: dict | None,
        evaluation: Evaluation | None = None,
    ) -> None:
        self.decisions.append(decision)
        if self.observe is not None:
            self.observe(
                Observation(
                    svg,
                    hashlib.sha256(svg.encode()).hexdigest(),
                    decision["candidate"],
                    copy.deepcopy(evaluation),
                    copy.deepcopy(details or {}),
                    copy.deepcopy(decision),
                )
            )

    @property
    def normalizer(self) -> float:
        if self._normalizer is not None:
            return self._normalizer
        return max(1.0, self.baseline.evaluation.cost) if self.baseline else 1.0

    @property
    def normalizer_fixed(self) -> bool:
        return self._normalizer is not None

    def freeze_normalizer(self, svg: str | None = None) -> None:
        """Fix cost scale from the validated detailed candidate before search.

        The coverage baseline stays conservative. Only its representation scale
        is replaced, once, before any selection/refinement is published. Pareto
        retention does not use this scale and the initial pool is not truncated.
        """
        if self._normalizer is not None or self._published:
            raise ValueError("The representation scale is already fixed or in use")
        entry = next((entry for entry in self.entries if entry.svg == svg), None)
        if svg is None:
            entry = self.baseline
        if (
            svg is not None
            and self._validated_scale is not None
            and self._validated_scale[0] == hashlib.sha256(svg.encode()).hexdigest()
        ):
            self._normalizer = max(1.0, self._validated_scale[1])
            return
        if entry is None:
            raise ValueError("A validated candidate is required for the cost scale")
        self._normalizer = max(1.0, entry.evaluation.cost)

    def seeds(self, limit: int) -> tuple[tuple[Entry, int], ...]:
        """Fixed refinement anchors, independent of the requested slider value."""
        found: list[tuple[Entry, int]] = []
        for complexity in (50, 0, 100):
            entry = self._pick(complexity)
            if not any(previous.key == entry.key for previous, _ in found):
                found.append((entry, complexity))
            if len(found) == limit:
                break
        return tuple(found)

    def reject(self, svg: str, label: str, reason: str, details: dict | None = None):
        self._record(
            svg,
            {"candidate": label, "accepted": False, "rejections": [reason]},
            details,
        )

    def add(self, svg: str, label: str, details: dict | None = None) -> bool:
        return self._add(svg, label, details)

    def refine(
        self, svg: str, label: str, details: dict, *, complexity: int, before: float
    ) -> bool:
        """An exact checkpoint that must retain or improve its anchor objective."""
        if not self.normalizer_fixed:
            raise ValueError("Freeze the representation scale before refinement")
        if self._refinement_protected is None:
            # Keep every pre-fitting tradeoff unless a new drawing dominates
            # it at every complexity. Frontier pruning must not turn an
            # improving anchor edit into a regression for another slider value.
            self._refinement_protected = {entry.key for entry in self.entries}
        return self._add(svg, label, details, maximum=(complexity, before))

    def _add(
        self,
        svg: str,
        label: str,
        details: dict | None = None,
        maximum: tuple[int, float] | None = None,
    ) -> bool:
        if len(svg.encode()) > MAX_BYTES:
            self._record(
                svg,
                {
                    "candidate": label,
                    "accepted": False,
                    "rejections": ["candidate-memory-limit"],
                },
                details,
            )
            return False
        key = hashlib.sha256(svg.encode()).hexdigest()
        if any(entry.key == key for entry in self.entries):
            return False
        try:
            evaluation = self.policy.evaluate(svg)
        except ValueError as exc:
            self._record(
                svg,
                {
                    "candidate": label,
                    "accepted": False,
                    "rejections": ["invalid-candidate"],
                    "detail": str(exc),
                },
                details,
            )
            return False
        if not evaluation.valid:
            self._record(
                svg,
                {
                    "candidate": label,
                    "accepted": False,
                    "rejections": list(evaluation.rejections),
                },
                details,
                evaluation,
            )
            return False
        entry = Entry(svg, label, evaluation, key, details or {})
        if self._normalizer is None:
            self._validated_scale = (key, evaluation.cost)
        if maximum is not None:
            complexity, before = maximum
            objective = evaluation.objective(
                complexity, self.normalizer, self.policy.weights.detail
            )
            if objective > before + 1e-12:
                self._record(
                    svg,
                    {
                        "candidate": label,
                        "accepted": False,
                        "rejections": ["objective-regression"],
                        "objective": objective,
                        "before_objective": before,
                    },
                    details,
                    evaluation,
                )
                return False
        if self.baseline is None:
            self.policy.establish(evaluation)
            self.baseline = entry
        if any(_dominates(other.evaluation, evaluation) for other in self.entries):
            self._record(
                svg,
                {"candidate": label, "accepted": False, "rejections": ["dominated"]},
                details,
                evaluation,
            )
            return False
        previous = self.entries
        self.entries = [
            other
            for other in self.entries
            if not _dominates(evaluation, other.evaluation)
        ]
        self.entries.append(entry)
        self.entries.sort(
            key=lambda item: (item.evaluation.cost, item.evaluation.visual, item.key)
        )
        self._bound()
        retained = entry in self.entries
        if not retained:
            # Pruning can reject a large replacement after it dominates old
            # entries. Roll that edit back instead of losing the old frontier.
            self.entries = previous
        self._record(
            svg,
            {
                "candidate": label,
                "accepted": retained,
                "rejections": [] if retained else ["frontier-limit"],
                "visual": evaluation.visual,
                "representation_cost": evaluation.cost,
            },
            details,
            evaluation,
        )
        return retained

    def _objective(self, entry: Entry, complexity: int):
        return entry.evaluation.objective(
            complexity, self.normalizer, self.policy.weights.detail
        )

    def budget(self, complexity: int) -> dict:
        """The initial content-dependent search target, never a coverage waiver."""
        if not 0 <= complexity <= 100:
            raise ValueError("Complexity must lie between zero and one hundred")
        entries = self.entries or ([self.baseline] if self.baseline else [])
        if not entries:
            raise ValueError("A validated drawing is required for the budget floor")
        floor = min(entry.evaluation.cost for entry in entries)
        return {
            "target": max(floor, self.normalizer * 2 ** ((complexity - 100) / 50)),
            "floor": floor,
            "floor_source": "simplest-validated-candidate",
            "schedule_version": 1,
        }

    def _bound(self):
        # Protect the choices at the five advertised checkpoints before filling
        # remaining slots evenly across the representation-cost range.
        protected = {self._pick(complexity).key for complexity in CHECKPOINTS}
        protected.update((self.entries[0].key, self.entries[-1].key))
        protected.add(
            min(
                self.entries,
                key=lambda entry: (
                    entry.evaluation.structure["nodes"],
                    entry.evaluation.cost,
                    entry.key,
                ),
            ).key
        )
        fixed = self._refinement_protected or set()
        protected.update(fixed)
        while len(self.entries) > MAX_CANDIDATES or self._bytes() > MAX_BYTES:
            removable = [entry for entry in self.entries if entry.key not in protected]
            if not removable:
                # Memory is a hard bound. Keep the baseline separately even if
                # exceptionally large alternatives cannot fit the frontier.
                removable = [
                    entry
                    for entry in self.entries
                    if entry is not self.baseline and entry.key not in fixed
                ]
            if not removable:
                break

            def spacing(entry):
                index = self.entries.index(entry)
                return min(
                    entry.evaluation.cost - self.entries[index - 1].evaluation.cost
                    if index
                    else float("inf"),
                    self.entries[index + 1].evaluation.cost - entry.evaluation.cost
                    if index + 1 < len(self.entries)
                    else float("inf"),
                )

            self.entries.remove(
                min(removable, key=lambda entry: (spacing(entry), entry.key))
            )

    def _bytes(self):
        entries = list(self.entries)
        if self.baseline and self.baseline not in entries:
            entries.append(self.baseline)
        return sum(len(entry.svg.encode()) for entry in entries)

    def _pick(self, complexity: int, node_budget: int = 0) -> Entry:
        choices = self.entries or ([self.baseline] if self.baseline else [])
        if not choices:
            raise ValueError("No fully validated candidate is available")
        within_budget = [
            entry
            for entry in choices
            if entry.evaluation.structure["nodes"] <= node_budget
        ]
        if node_budget and within_budget:
            choices = within_budget
        return min(
            choices,
            key=lambda entry: (
                self._objective(entry, complexity),
                entry.evaluation.cost,
                entry.key,
            ),
        )

    def select(self, complexity: int, *, node_budget: int = 0) -> Candidate:
        self._published = True
        entry = self._pick(complexity, node_budget)
        budget = self.budget(complexity)
        return Candidate(
            entry.svg,
            entry.label,
            complexity,
            {
                **entry.details,
                **entry.evaluation.metrics(),
                **self.policy.metadata(),
                "objective": self._objective(entry, complexity),
                "cost_normalizer": self.normalizer,
                "frontier_candidates": len(self.entries),
                "node_budget": node_budget,
                "budget_unmet": bool(
                    node_budget and entry.evaluation.structure["nodes"] > node_budget
                ),
                "representation_budget": {
                    **budget,
                    "achieved": entry.evaluation.cost,
                    "unmet": entry.evaluation.cost > budget["target"] + 1e-12,
                },
            },
            tuple(self.decisions),
        )

    def alternatives(
        self, complexity: int, *, node_budget: int = 0
    ) -> tuple[Candidate, ...]:
        selected = self._pick(complexity, node_budget)
        entries = [selected]
        choices = self.entries or [selected]
        for other in (choices[0], choices[-1]):
            if other.key not in {entry.key for entry in entries}:
                entries.append(other)
        return tuple(
            Candidate(
                entry.svg,
                "Selected"
                if entry is selected
                else "Simpler"
                if entry.evaluation.cost < selected.evaluation.cost
                else "More detailed",
                complexity,
                {
                    **entry.evaluation.metrics(),
                    "node_budget": node_budget,
                    "budget_unmet": bool(
                        node_budget
                        and entry.evaluation.structure["nodes"] > node_budget
                    ),
                    "representation_budget": {
                        **self.budget(complexity),
                        "achieved": entry.evaluation.cost,
                        "unmet": entry.evaluation.cost
                        > self.budget(complexity)["target"] + 1e-12,
                    },
                },
            )
            for entry in entries
        )
