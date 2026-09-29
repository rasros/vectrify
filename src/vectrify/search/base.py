from typing import Protocol, TypeVar

from vectrify.search.models import SearchNode

TState = TypeVar("TState")


class SearchStrategy(Protocol[TState]):
    def select_parent(
        self, nodes: list[SearchNode[TState]]
    ) -> tuple[int, int | None]: ...

    def top_tier_ids(self, pool: list[SearchNode]) -> set[int]:
        """Ids of the best-ranked tier. Entry into it is what counts as
        progress, so this replaces comparing a blended score."""
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
        """The leading, distinct candidates for the evaluator to rank."""
        ...
