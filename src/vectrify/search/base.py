from pathlib import Path
from typing import Protocol, TypeVar

from vectrify.search.models import SearchNode

TState = TypeVar("TState")


class SearchStrategy(Protocol[TState]):
    def select_parent(
        self, nodes: list[SearchNode[TState]]
    ) -> tuple[int, int | None]: ...

    def should_diversify(self, pool: list[SearchNode]) -> tuple[bool, float]:
        """Return (trigger_epoch, diversity).

        diversity is the mean normalised Hamming distance across sampled node pairs.
        """
        ...

    def select_survivors(
        self, nodes: list[SearchNode[TState]], max_keep: int
    ) -> list[SearchNode[TState]]:
        """Cut a combined parent+child population down to *max_keep* members.

        Called once per generation rather than once per child, so an
        implementation may do work proportional to the whole population.
        """
        ...

    def epoch_parents(
        self, pool: list[SearchNode[TState]], max_parents: int
    ) -> list[SearchNode[TState]]:
        """Select the nodes the next epoch's LLM edits should start from.

        These are parents, not pool members: the epoch's pool is built from
        their edited children, so a node returned here survives only through
        whatever the LLM makes of it.
        """
        ...


class StorageAdapter(Protocol[TState]):
    current_run_dir: Path | None

    def initialize(self) -> None: ...

    def save_node(self, node: SearchNode[TState], tasks_completed: int = 0) -> None: ...

    def save_best(self, node: SearchNode[TState]) -> None:
        """Write the best final candidate to the top-level output path."""
        ...

    def record_eviction(self, node_id: int, tasks_completed: int) -> None: ...

    def load_resume_nodes(self) -> list[tuple[int, str]]: ...

    @property
    def max_node_id(self) -> int: ...
