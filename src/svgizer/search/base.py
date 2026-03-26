import zlib
from enum import Enum
from typing import Protocol, TypeVar

from svgizer.search.models import ChainState, Result, SearchNode

TState = TypeVar("TState")


def ncd(a: str, b: str) -> float:
    """Normalised Compression Distance: ~0 for near-identical, ~1 for unrelated."""
    if not a or not b:
        return 0.0
    ca = len(zlib.compress(a.encode("utf-8"), level=6))
    cb = len(zlib.compress(b.encode("utf-8"), level=6))
    cab = len(zlib.compress((a + b).encode("utf-8"), level=6))
    denom = max(ca, cb)
    if denom == 0:
        return 0.0
    return (cab - min(ca, cb)) / denom


class StrategyType(str, Enum):
    GREEDY = "greedy"
    NSGA = "nsga"


class SearchStrategy(Protocol[TState]):
    """Protocol for the 'brains' of the search (selection and evolution)."""

    def select_parent(
        self, nodes: list[SearchNode[TState]], progress: float
    ) -> tuple[int, int | None]:
        """Decides which node(s) to mutate or crossover next."""
        ...

    def create_new_state(self, result: Result[TState]) -> ChainState[TState]:
        """Handles state transition."""
        ...

    def should_diversify(self, pool: list[SearchNode]) -> bool:
        """Return True when the pool has converged and fresh LLM seeds are needed."""
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

    def load_resume_nodes(self) -> list[tuple[int, str]]:
        """Returns raw IDs and SVG content for re-hydration."""
        ...

    @property
    def max_node_id(self) -> int:
        """Returns the highest ID currently known to the storage backend."""
        ...
