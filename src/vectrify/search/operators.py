"""Policies for choosing which mutation operator a task should apply.

The choice lives here rather than in the worker so that one policy sees every
outcome: eight worker processes each running their own copy could never learn
anything, since none of them observes whether its own children survived.
"""

import random
from collections.abc import Mapping
from typing import Protocol


class OperatorPolicy(Protocol):
    def select(self) -> str | None:
        """Name the operator the next mutation task should apply."""
        ...

    def update(self, operator: str | None, survived: bool) -> None:
        """Report whether an operator's child survived its generation.

        *operator* is what the worker actually applied, which may be None or a
        name this policy does not know: crossover falls back to mutation, and
        a backend may not have the operator the task named. Ignore those.
        """
        ...


class FixedWeightPolicy:
    """Draws from static weights and learns nothing.

    One weight table for every image and every stage of every run, which is
    what the backends did on their own before the choice moved here.
    """

    def __init__(self, weights: Mapping[str, float]):
        self._names = list(weights)
        self._weights = [weights[name] for name in self._names]

    def select(self) -> str | None:
        if not self._names:
            return None
        return random.choices(self._names, weights=self._weights, k=1)[0]

    def update(self, operator: str | None, survived: bool) -> None:
        _ = operator, survived
