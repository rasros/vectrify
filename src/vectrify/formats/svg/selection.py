"""Target-aware selection for SVG mutation operators.

Operators parse their input independently, so target attribution is resolved
against the elements in the parse currently being edited.  Keeping that state
in this small, per-mutation object makes mutations reentrant and lets callers
run targeted edits concurrently without sharing module state.
"""

import random
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass

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


# Attributes that count as paint rather than geometry when a scope limits kinds.
PAINT_ATTRIBUTES = frozenset(
    {
        "fill",
        "stroke",
        "opacity",
        "fill-opacity",
        "stroke-opacity",
        "stroke-width",
        "stroke-linecap",
        "stroke-linejoin",
        "stroke-miterlimit",
        "fill-rule",
        "style",
    }
)


@dataclass(frozen=True)
class MutationScope:
    """What an editor operation lets a search touch.

    ``object_ids`` names editable elements by their ``id``; their descendants
    are editable too. ``kinds`` are the permitted edit kinds: ``geometry``,
    ``paint``, ``structure`` and ``transform``. Everything else stays fixed.
    """

    object_ids: frozenset[str]
    kinds: frozenset[str]


class MutationContext:
    """Selection weights resolved for one parsed SVG document."""

    def __init__(
        self,
        root: ET.Element,
        targets: Mapping[int, float] | None = None,
        scope: MutationScope | None = None,
    ):
        by_index = targets or {}
        self._weights = {
            id(element): by_index.get(index, 0.0)
            for index, (_chain, element) in enumerate(drawable_elements(root))
        }
        self.scope = scope
        self._editable: set[int] | None = None
        if scope is not None:
            editable: set[int] = set()

            def walk(element: ET.Element, inside: bool) -> None:
                inside = inside or element.get("id") in scope.object_ids
                if inside:
                    editable.add(id(element))
                for child in element:
                    walk(child, inside)

            walk(root, False)
            self._editable = editable

    def editable(self, element: ET.Element) -> bool:
        return self._editable is None or id(element) in self._editable

    def allows(self, kind: str) -> bool:
        return self.scope is None or kind in self.scope.kinds

    def allows_attribute(self, name: str) -> bool:
        if self.scope is None:
            return True
        if name == "transform":
            return self.allows("transform")
        return self.allows("paint" if name in PAINT_ATTRIBUTES else "geometry")

    def pick(self, candidates: list):
        if self._editable is not None:
            candidates = [c for c in candidates if self.editable(_element_of(c))]
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
