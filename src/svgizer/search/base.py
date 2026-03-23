from enum import Enum
from typing import Any, Protocol

from .models import ChainState, Result, SearchNode


class StrategyType(str, Enum):
    GENETIC = "genetic"
    GREEDY = "greedy"


class SearchStrategy(Protocol):
    """Protocol for the 'brains' of the search (selection and evolution)."""

    def select_parent(
        self, nodes: list[SearchNode[Any]], progress: float
    ) -> tuple[int, int | None]:
        """Decides which node(s) to mutate or crossover next."""
        ...

    def create_new_state(
        self, parent_state: ChainState[Any], result: Result[Any]
    ) -> ChainState[Any]:
        """Handles temperature bumping, staleness, and state transition."""
        ...

    @property
    def top_k_count(self) -> int: ...