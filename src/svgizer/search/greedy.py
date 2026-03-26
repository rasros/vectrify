from typing import Generic, TypeVar

from svgizer.search.models import ChainState, Result, SearchNode

TState = TypeVar("TState")


class GreedyHillClimbingStrategy(Generic[TState]):
    @property
    def top_k_count(self) -> int:
        return 1

    def select_parent(
        self, nodes: list[SearchNode[TState]], progress: float
    ) -> tuple[int, int | None]:
        _ = progress
        if not nodes:
            return 0, None

        best_node = min(nodes, key=lambda n: n.score)
        return best_node.id, None

    def should_diversify(self, pool: list[SearchNode[TState]]) -> bool:
        return False

    def create_new_state(self, result: Result) -> ChainState[TState]:
        return ChainState(
            score=result.score,
            payload=result.payload,
        )
