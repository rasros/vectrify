import random
from typing import List
from svgizer.models import SearchNode, ChainState, Result
from svgizer.utils import is_stale, calculate_elite_prob, choose_from_top_k_weighted

class GeneticPoolStrategy:
    """Pool-based refinement strategy."""

    def __init__(
            self,
            top_k: int = 3,
            temp_step: float = 0.3,
            max_temp: float = 1.6,
            elite_start: float = 0.70,
            elite_end: float = 0.10,
            stale_threshold: int = 1
    ):
        self.top_k = top_k
        self.temp_step = temp_step
        self.max_temp = max_temp
        self.elite_start = elite_start
        self.elite_end = elite_end
        self.stale_threshold = stale_threshold

    @property
    def top_k_count(self) -> int:
        return self.top_k

    def select_parent(self, nodes: List[SearchNode], progress: float) -> int:
        if not nodes:
            return 0

        # Sort by score (ascending)
        best_k = sorted(nodes, key=lambda n: n.score)[:self.top_k]
        best_node = best_k[0]

        elite_prob = calculate_elite_prob(progress, self.elite_start, self.elite_end)

        if random.random() < elite_prob:
            return choose_from_top_k_weighted(best_k)
        return best_node.id

    def create_new_state(self, parent_state: ChainState, result: Result) -> ChainState:
        next_temp = parent_state.model_temperature
        stale_hits = parent_state.stale_hits

        if parent_state.svg and is_stale(parent_state.svg, result.svg):
            stale_hits += 1
            if stale_hits >= self.stale_threshold and next_temp < self.max_temp:
                next_temp = min(self.max_temp, next_temp + self.temp_step)
                stale_hits = 0
        else:
            stale_hits = 0

        return ChainState(
            svg=result.svg,
            raster_data_url=None, # Populated by engine if lineage is on
            raster_preview_data_url=None, # Populated by engine
            score=result.score,
            model_temperature=next_temp,
            stale_hits=stale_hits,
            invalid_msg=None,
            change_summary=result.change_summary,
        )