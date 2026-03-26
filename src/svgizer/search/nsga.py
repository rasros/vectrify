"""
NSGA-II-inspired multi-objective search strategy.

Objectives:
  1. Visual quality  (node.score, lower is better)
  2. SVG complexity  (node.complexity, lower is better)

Both objectives are normalised to [0, 1] within the active population before
Pareto sorting, so they contribute equally to dominance.

Selection uses binary tournament on (Pareto front rank ASC, crowding distance DESC).

Diversity: when building the selection pool, near-duplicate nodes (normalised
edit distance >= diversity_threshold) are skipped to prevent crossover waste.
"""

import random
from difflib import SequenceMatcher
from typing import Generic, TypeVar

from svgizer.search.base import ncd
from svgizer.search.models import ChainState, Result, SearchNode

TState = TypeVar("TState")

# Objectives tuple: (visual_score, complexity)
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
    """Compute crowding distance for each node in a front."""
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
    """
    NSGA-II-style parent selection over visual quality and SVG complexity.

    Complexity is read directly from node.complexity (set by the worker via Result).
    Both objectives are normalised within the active population so neither dominates.

    Args:
        pool_size:          Max nodes in the selection pool, ordered by Pareto rank
                            then crowding distance.
        crossover_prob:     Probability of selecting two parents instead of one.
        diversity_threshold: Normalised edit-distance ratio above which two nodes are
                            considered near-duplicates. The lower-quality duplicate is
                            dropped from the pool. Set to 1.0 to disable.
    """

    def __init__(
        self,
        pool_size: int = 20,
        crossover_prob: float = 0.25,
        diversity_threshold: float = 0.97,
        diversity_boost_threshold: float = 0.10,
    ):
        self.pool_size = pool_size
        self.crossover_prob = crossover_prob
        self.diversity_threshold = diversity_threshold
        self.diversity_boost_threshold = diversity_boost_threshold

    @property
    def top_k_count(self) -> int:
        return self.pool_size

    def _too_similar(self, node: SearchNode[TState], other: SearchNode[TState]) -> bool:
        """True if node and other have nearly identical content."""
        if node.content is None or other.content is None:
            return False
        m = SequenceMatcher(None, node.content, other.content, autojunk=False)
        return m.quick_ratio() >= self.diversity_threshold and (
            m.ratio() >= self.diversity_threshold
        )

    def select_parent(
        self, nodes: list[SearchNode[TState]], progress: float
    ) -> tuple[int, int | None]:
        _ = progress

        # Exclude sentinel / invalid nodes (score=inf means no SVG yet)
        valid = [n for n in nodes if n.score < float("inf")]
        if not valid:
            return nodes[0].id if nodes else 0, None

        # Normalise both objectives to [0, 1] within the current population
        max_score = max(n.score for n in valid) or 1.0
        max_complexity = max(n.complexity for n in valid) or 1.0

        objectives: dict[int, Objectives] = {
            n.id: (n.score / max_score, n.complexity / max_complexity) for n in valid
        }

        # Non-dominated sort → per-node rank and crowding distance
        fronts = non_dominated_sort(valid, objectives)
        rank: dict[int, int] = {}
        crowd: dict[int, float] = {}
        for front_idx, front in enumerate(fronts):
            distances = crowding_distance(front, objectives)
            for node in front:
                rank[node.id] = front_idx
                crowd[node.id] = distances[node.id]

        # Build pool: iterate in quality order; skip near-duplicates of admitted nodes
        # (the admitted node is always better-ranked, so the duplicate is always worse)
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

    def should_diversify(self, pool: list[SearchNode[TState]]) -> bool:
        """True when pool NCD is low enough to warrant fresh LLM seeds."""
        candidates = [n for n in pool if n.content and n.score < float("inf")]
        if len(candidates) < 4:
            return False
        n = len(candidates)
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        sample_pairs = random.sample(all_pairs, min(8, len(all_pairs)))
        mean_ncd = sum(
            ncd(candidates[i].content, candidates[j].content)  # type: ignore[arg-type]
            for i, j in sample_pairs
        ) / len(sample_pairs)
        return mean_ncd < self.diversity_boost_threshold

    def create_new_state(self, result: Result[TState]) -> ChainState[TState]:
        return ChainState(score=result.score, payload=result.payload)
