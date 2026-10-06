"""Bounded exact-validated alternatives, independent of the slider selection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from vectrify.refine.cel_plan.model import Candidate
from vectrify.refine.cel_plan.policy import Evaluation, Policy

MAX_CANDIDATES = 12
MAX_BYTES = 64 * 1024 * 1024
CHECKPOINTS = (0, 25, 50, 75, 100)


@dataclass(frozen=True)
class Entry:
    svg: str
    label: str
    evaluation: Evaluation
    key: str
    details: dict


class Frontier:
    """Retain nondominated drawings and one separate interruption fallback."""

    def __init__(self, policy: Policy):
        self.policy = policy
        self.entries: list[Entry] = []
        self.baseline: Entry | None = None
        self.decisions: list[dict] = []

    @property
    def normalizer(self) -> float:
        return max(1.0, self.baseline.evaluation.cost) if self.baseline else 1.0

    def add(self, svg: str, label: str, details: dict | None = None) -> bool:
        if len(svg.encode()) > MAX_BYTES:
            self.decisions.append(
                {
                    "candidate": label,
                    "accepted": False,
                    "rejections": ["candidate-memory-limit"],
                }
            )
            return False
        key = hashlib.sha256(svg.encode()).hexdigest()
        if any(entry.key == key for entry in self.entries):
            return False
        try:
            evaluation = self.policy.evaluate(svg)
        except ValueError as exc:
            self.decisions.append(
                {
                    "candidate": label,
                    "accepted": False,
                    "rejections": ["invalid-candidate"],
                    "detail": str(exc),
                }
            )
            return False
        if not evaluation.valid:
            self.decisions.append(
                {
                    "candidate": label,
                    "accepted": False,
                    "rejections": list(evaluation.rejections),
                }
            )
            return False
        entry = Entry(svg, label, evaluation, key, details or {})
        if self.baseline is None:
            self.policy.establish(evaluation)
            self.baseline = entry
        if any(
            other.evaluation.cost <= evaluation.cost
            and other.evaluation.visual <= evaluation.visual
            for other in self.entries
        ):
            self.decisions.append(
                {"candidate": label, "accepted": False, "rejections": ["dominated"]}
            )
            return False
        self.entries = [
            other
            for other in self.entries
            if not (
                evaluation.cost <= other.evaluation.cost
                and evaluation.visual <= other.evaluation.visual
            )
        ]
        self.entries.append(entry)
        self.entries.sort(
            key=lambda item: (item.evaluation.cost, item.evaluation.visual, item.key)
        )
        self._bound()
        self.decisions.append(
            {
                "candidate": label,
                "accepted": True,
                "visual": evaluation.visual,
                "representation_cost": evaluation.cost,
            }
        )
        return True

    def _objective(self, entry: Entry, complexity: int):
        return entry.evaluation.objective(
            complexity, self.normalizer, self.policy.weights.detail
        )

    def _bound(self):
        # Protect the choices at the five advertised checkpoints before filling
        # remaining slots evenly across the representation-cost range.
        protected = {self._pick(complexity).key for complexity in CHECKPOINTS}
        protected.update((self.entries[0].key, self.entries[-1].key))
        while len(self.entries) > MAX_CANDIDATES or self._bytes() > MAX_BYTES:
            removable = [entry for entry in self.entries if entry.key not in protected]
            if not removable:
                # Memory is a hard bound. Keep the baseline separately even if
                # exceptionally large alternatives cannot fit the frontier.
                removable = [
                    entry for entry in self.entries if entry is not self.baseline
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
        entry = self._pick(complexity, node_budget)
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
                },
            )
            for entry in entries
        )
