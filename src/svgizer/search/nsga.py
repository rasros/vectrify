import logging
import random
from typing import Generic, TypeVar

from svgizer.search.base import estimate_jaccard
from svgizer.search.models import ChainState, Result, SearchNode

log = logging.getLogger(__name__)

TState = TypeVar("TState")

Objectives = tuple[float, float]


def _dominates(a: Objectives, b: Objectives) -> bool:
    """True if a Pareto-dominates b (better/equal in all, strictly better in one)."""
    return a[0] <= b[0] and a[1] <= b[1] and (a[0] < b[0] or a[1] < b[1])


def non_dominated_sort(
    nodes: list[SearchNode],
    objectives: dict[int, Objectives],
) -> list[list[SearchNode]]:
    """Fast non-dominated sort (Deb 2002). front[0] is the Pareto front."""
    id_to_node = {n.id: n for n in nodes}

    domination_count: dict[int, int] = {n.id: 0 for n in nodes}
    dominated_set: dict[int, list[int]] = {n.id: [] for n in nodes}

    for a in nodes:
        for b in nodes:
            if a.id == b.id:
                continue
            if _dominates(objectives[a.id], objectives[b.id]):
                dominated_set[a.id].append(b.id)
            elif _dominates(objectives[b.id], objectives[a.id]):
                domination_count[a.id] += 1

    fronts: list[list[SearchNode]] = []
    current_front = [n for n in nodes if domination_count[n.id] == 0]

    while current_front:
        fronts.append(current_front)
        next_front: list[SearchNode] = []
        for a in current_front:
            for b_id in dominated_set[a.id]:
                domination_count[b_id] -= 1
                if domination_count[b_id] == 0:
                    next_front.append(id_to_node[b_id])
        current_front = next_front

    return fronts


def crowding_distance(
    front: list[SearchNode],
    objectives: dict[int, Objectives],
) -> dict[int, float]:
    """Compute crowding distance to maintain diversity within a front."""
    if len(front) <= 2:
        return {n.id: float("inf") for n in front}

    distances: dict[int, float] = {n.id: 0.0 for n in front}
    n_objectives = 2

    for m in range(n_objectives):
        sorted_front = sorted(front, key=lambda n: objectives[n.id][m])
        obj_min = objectives[sorted_front[0].id][m]
        obj_max = objectives[sorted_front[-1].id][m]

        distances[sorted_front[0].id] = float("inf")
        distances[sorted_front[-1].id] = float("inf")

        obj_range = obj_max - obj_min
        if obj_range == 0.0:
            continue

        for k in range(1, len(sorted_front) - 1):
            distances[sorted_front[k].id] += (
                objectives[sorted_front[k + 1].id][m]
                - objectives[sorted_front[k - 1].id][m]
            ) / obj_range

    return distances


class NsgaStrategy(Generic[TState]):
    """NSGA-II-style selection balancing visual quality and SVG complexity."""

    def __init__(
        self,
        pool_size: int = 20,
        crossover_prob: float = 0.25,
        similarity_threshold: float = 0.97,
        min_diversity: float = 0.10,
    ):
        self.pool_size = pool_size
        self.crossover_prob = crossover_prob
        self.similarity_threshold = similarity_threshold
        self.min_diversity = min_diversity

    @property
    def top_k_count(self) -> int:
        return self.pool_size

    def _too_similar(self, node: SearchNode[TState], other: SearchNode[TState]) -> bool:
        if not node.signature or not other.signature:
            return False

        sim = estimate_jaccard(node.signature, other.signature)
        return sim >= self.similarity_threshold

    def select_parent(
        self, nodes: list[SearchNode[TState]], progress: float
    ) -> tuple[int, int | None]:
        _ = progress

        valid = [n for n in nodes if n.score < float("inf")]
        if not valid:
            return nodes[0].id if nodes else 0, None

        max_score = max(n.score for n in valid) or 1.0
        max_complexity = max(n.complexity for n in valid) or 1.0

        objectives: dict[int, Objectives] = {
            n.id: (n.score / max_score, n.complexity / max_complexity) for n in valid
        }

        fronts = non_dominated_sort(valid, objectives)
        rank: dict[int, int] = {}
        crowd: dict[int, float] = {}
        for front_idx, front in enumerate(fronts):
            distances = crowding_distance(front, objectives)
            for node in front:
                rank[node.id] = front_idx
                crowd[node.id] = distances[node.id]

        sorted_valid = sorted(valid, key=lambda n: (rank[n.id], -crowd[n.id]))
        pool: list[SearchNode[TState]] = []
        for node in sorted_valid:
            if len(pool) >= self.pool_size:
                break
            if not any(self._too_similar(node, p) for p in pool):
                pool.append(node)
        if not pool:
            pool = sorted_valid[: self.pool_size]

        def _tournament(exclude_id: int | None = None) -> SearchNode[TState]:
            candidates = [n for n in pool if n.id != exclude_id]
            if len(candidates) < 2:
                return candidates[0] if candidates else pool[0]
            a, b = random.sample(candidates, 2)
            ra, da = rank[a.id], crowd[a.id]
            rb, db = rank[b.id], crowd[b.id]
            if ra < rb or (ra == rb and da > db):
                return a
            return b

        p1 = _tournament()

        if len(pool) >= 2 and random.random() < self.crossover_prob:
            p2 = _tournament(exclude_id=p1.id)
            return p1.id, p2.id

        return p1.id, None

    def epoch_seeds(
        self, pool: list[SearchNode[TState]], max_seeds: int
    ) -> list[SearchNode[TState]]:
        """Return a diverse Pareto-front subset to seed the next epoch."""
        valid = [n for n in pool if n.score < float("inf")]
        if not valid:
            return pool[:max_seeds]

        max_score = max(n.score for n in valid) or 1.0
        max_complexity = max(n.complexity for n in valid) or 1.0
        objectives: dict[int, Objectives] = {
            n.id: (n.score / max_score, n.complexity / max_complexity) for n in valid
        }

        fronts = non_dominated_sort(valid, objectives)
        seeds: list[SearchNode[TState]] = []

        for front in fronts:
            if len(seeds) >= max_seeds:
                break
            distances = crowding_distance(front, objectives)
            front_sorted = sorted(front, key=lambda n: -distances[n.id])
            for node in front_sorted:
                if len(seeds) >= max_seeds:
                    break
                if not any(
                    node.signature
                    and s.signature
                    and estimate_jaccard(node.signature, s.signature)
                    >= self.similarity_threshold
                    for s in seeds
                ):
                    seeds.append(node)

        return seeds or valid[:max_seeds]

    def should_diversify(self, pool: list[SearchNode[TState]]) -> bool:
        candidates = [n for n in pool if n.signature and n.score < float("inf")]
        if len(candidates) < 4:
            return False
        n = len(candidates)
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        if len(all_pairs) <= 28:
            sample_pairs = all_pairs
        else:
            sample_pairs = random.sample(all_pairs, 8)

        mean_distance = sum(
            1.0 - estimate_jaccard(candidates[i].signature, candidates[j].signature)
            for i, j in sample_pairs
        ) / len(sample_pairs)

        return mean_distance < self.min_diversity

    def create_new_state(self, result: Result[TState]) -> ChainState[TState]:
        return ChainState(score=result.score, payload=result.payload)
