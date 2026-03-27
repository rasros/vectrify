from enum import Enum
from pathlib import Path
from typing import Protocol, TypeVar

from svgizer.search.models import ChainState, Result, SearchNode

TState = TypeVar("TState")


class StrategyType(str, Enum):
    GREEDY = "greedy"
    NSGA = "nsga"


class SearchStrategy(Protocol[TState]):
    def select_parent(
        self, nodes: list[SearchNode[TState]], progress: float
    ) -> tuple[int, int | None]: ...

    def create_new_state(self, result: Result[TState]) -> ChainState[TState]: ...

    def should_diversify(self, pool: list[SearchNode]) -> tuple[bool, float]:
        """Return (trigger_epoch, diversity).

        diversity is the mean normalised Hamming distance across sampled node pairs.
        """
        ...

    def epoch_seeds(
        self, pool: list[SearchNode[TState]], max_seeds: int
    ) -> list[SearchNode[TState]]:
        """Select diverse nodes from the pool to seed the next epoch."""
        ...

    @property
    def top_k_count(self) -> int: ...


class StorageAdapter(Protocol[TState]):
    current_run_dir: Path | None

    def initialize(self) -> None: ...

    def save_node(self, node: SearchNode[TState]) -> None: ...

    def load_resume_nodes(self) -> list[tuple[int, str]]: ...

    @property
    def max_node_id(self) -> int: ...
