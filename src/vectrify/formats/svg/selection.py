"""Target-aware selection for SVG mutation operators.

Operators parse their input independently, so target attribution is resolved
against the elements in the parse currently being edited.  Keeping that state
in this small, per-mutation object makes mutations reentrant and lets callers
run targeted edits concurrently without sharing module state.
"""

import random
import xml.etree.ElementTree as ET
from collections.abc import Mapping

from vectrify.formats.svg.ownership import drawable_elements

# Share of the selection mass spread evenly over every candidate.
TARGET_FLOOR = 0.25


def _element_of(item):
    """Operators offer elements, or tuples with an element in them."""
    if isinstance(item, tuple):
        for part in item:
            if isinstance(part, ET.Element):
                return part
        return item[0]
    return item


class MutationContext:
    """Selection weights resolved for one parsed SVG document."""

    def __init__(self, root: ET.Element, targets: Mapping[int, float] | None = None):
        by_index = targets or {}
        self._weights = {
            id(element): by_index.get(index, 0.0)
            for index, (_chain, element) in enumerate(drawable_elements(root))
        }

    def pick(self, candidates: list):
        if not candidates:
            raise NoChangeError
        if not self._weights:
            return random.choice(candidates)

        weights = [self._weights.get(id(_element_of(item)), 0.0) for item in candidates]
        total = sum(weights)
        if total <= 0.0:
            return random.choice(candidates)
        floor = total * TARGET_FLOOR / len(candidates)
        return random.choices(
            candidates, weights=[weight + floor for weight in weights], k=1
        )[0]


class NoChangeError(Exception):
    """Raised by an operator when there is nothing it can mutate."""
