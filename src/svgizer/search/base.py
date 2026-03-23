from enum import Enum
from typing import Protocol, TypeVar

from svgizer.search.models import ChainState, Result, SearchNode

TState = TypeVar("TState")


class StrategyType(str, Enum):
    GENETIC = "genetic"
    GREEDY = "greedy"


class SearchStrategy(Protocol[TState]):
    """Protocol for the 'brains' of the search (selection and evolution)."""

    def select_parent(
        self, nodes: list[SearchNode[TState]], progress: float
    ) -> tuple[int, int | None]:
        """Decides which node(s) to mutate or crossover next."""
        ...

    def create_new_state(
        self, parent_state: ChainState[TState], result: Result[TState]
    ) -> ChainState[TState]:
        """Handles temperature bumping, staleness, and state transition."""
        ...

    @property
    def top_k_count(self) -> int: ...


class StorageAdapter(Protocol[TState]):
    """
    Protocol for the 'memory' of the search.
    The SearchEngine uses this to persist every accepted step of the evolution.
    """

    def initialize(self) -> None: ...

    def save_node(self, node: SearchNode[TState]) -> None: ...

    def load_resume_nodes(self) -> list[SearchNode[TState]]: ...

    def save_final_svg(self, svg_content: str) -> None: ...

    def load_seed_svg(self, seed_path: str) -> str: ...

    @property
    def max_node_id(self) -> int:
        """Returns the highest ID currently known to the storage backend."""
        ...
