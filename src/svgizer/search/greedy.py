import random
from typing import Generic, TypeVar

from svgizer.search.models import ChainState, Result, SearchNode

TState = TypeVar("TState")


class GreedyHillClimbingStrategy(Generic[TState]):
    """Parallel random beam search with culling.

    `beams` independent beams are seeded via LLM at the start of each epoch.
    Only the top `cull_keep` fraction of scored beams are eligible for
    expansion; the rest starve and are evicted by better candidates.
    """

    def __init__(self, beams: int = 10, cull_keep: float = 0.5):
        self.beams = beams
        self.cull_keep = cull_keep

    @property
    def top_k_count(self) -> int:
        return self.beams

    def select_parent(
        self, nodes: list[SearchNode[TState]], progress: float
    ) -> tuple[int, int | None]:
        _ = progress
        if not nodes:
            return 0, None

        valid = [n for n in nodes if n.score < float("inf")]
        if not valid:
            # Still in seeding phase — any node is fine (will trigger LLM call)
            return random.choice(nodes).id, None

        # Only expand the top cull_keep fraction; bottom beams starve and die
        n_active = max(1, round(len(valid) * self.cull_keep))
        top_beams = sorted(valid, key=lambda n: n.score)[:n_active]
        return random.choice(top_beams).id, None

    def should_diversify(self, pool: list[SearchNode[TState]]) -> tuple[bool, float]:
        _ = pool
        return False, 0.0

    def epoch_seeds(
        self, pool: list[SearchNode[TState]], max_seeds: int
    ) -> list[SearchNode[TState]]:
        """Return [] to signal a full restart from fresh LLM seeds."""
        _ = pool, max_seeds
        return []

    def create_new_state(self, result: Result) -> ChainState[TState]:
        return ChainState(
            score=result.score,
            payload=result.payload,
        )
