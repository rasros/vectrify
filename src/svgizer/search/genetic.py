import random
from collections.abc import Callable
from typing import Any

from .models import ChainState, Result, SearchNode
from .utils import calculate_elite_prob, choose_from_top_k_weighted


class GeneticPoolStrategy:
    """Pool-based refinement and crossover strategy."""

    def __init__(
        self,
        top_k: int = 3,
        temp_step: float = 0.3,
        max_temp: float = 1.6,
        elite_start: float = 0.70,
        elite_end: float = 0.10,
        stale_threshold: int = 1,
        crossover_prob: float = 0.25,
        is_stale_fn: Callable[[Any, Any], bool] | None = None,
    ):
        self.top_k = top_k
        self.temp_step = temp_step
        self.max_temp = max_temp
        self.elite_start = elite_start
        self.elite_end = elite_end
        self.stale_threshold = stale_threshold
        self.crossover_prob = crossover_prob
        self.is_stale_fn = is_stale_fn or (lambda parent_payload, result_payload: False)

    @property
    def top_k_count(self) -> int:
        return self.top_k

    def select_parent(
        self, nodes: list[SearchNode], progress: float
    ) -> tuple[int, int | None]:
        if not nodes:
            return 0, None

        best_k = sorted(nodes, key=lambda n: n.score)[: self.top_k]

        if len(best_k) >= 2 and random.random() < self.crossover_prob:
            p1, p2 = random.sample(best_k, 2)
            return p1.id, p2.id

        best_node = best_k[0]
        elite_prob = calculate_elite_prob(progress, self.elite_start, self.elite_end)

        if random.random() < elite_prob:
            return choose_from_top_k_weighted(best_k), None
        return best_node.id, None

    def create_new_state(self, parent_state: ChainState, result: Result) -> ChainState:
        next_temp = parent_state.model_temperature
        stale_hits = parent_state.stale_hits

        # Delegate staleness check to the injected domain logic
        if self.is_stale_fn(parent_state.payload, result.payload):
            stale_hits += 1
            if stale_hits >= self.stale_threshold and next_temp < self.max_temp:
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