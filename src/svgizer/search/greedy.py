from typing import Generic, TypeVar

from .models import ChainState, Result, SearchNode

TState = TypeVar("TState")


class GreedyHillClimbingStrategy(Generic[TState]):
    def __init__(
        self, patience: int = 3, temp_step: float = 0.3, max_temp: float = 1.6
    ):
        self.patience = patience
        self.temp_step = temp_step
        self.max_temp = max_temp

    @property
    def top_k_count(self) -> int:
        return 1

    def select_parent(
        self,
        nodes: list[SearchNode[TState]],
        progress: float,
    ) -> tuple[int, int | None]:
        _ = progress
        if not nodes:
            return 0, None

        best_node = min(nodes, key=lambda n: n.score)
        return best_node.id, None

    def create_new_state(
        self, parent_state: ChainState[TState], result: Result
    ) -> ChainState[TState]:
        next_temp = parent_state.model_temperature
        stale_hits = parent_state.stale_hits

        if result.score >= parent_state.score:
            stale_hits += 1
            if stale_hits >= self.patience and next_temp < self.max_temp:
                next_temp = min(self.max_temp, next_temp + self.temp_step)
                stale_hits = 0
        else:
            stale_hits = 0

        return ChainState(
            score=result.score,
            model_temperature=next_temp,
            stale_hits=stale_hits,
            payload=result.payload,
        )
