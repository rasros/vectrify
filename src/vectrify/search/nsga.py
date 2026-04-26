import logging
import random
from typing import Generic, TypeVar

from vectrify.search.diversity import hamming_distance, pool_diversity
from vectrify.search.models import INVALID_SCORE, ChainState, Result, SearchNode

log = logging.getLogger(__name__)

TState = TypeVar("TState")

Objectives = tuple[float, float]


def _dominates(a: Objectives, b: Objectives) -> bool:
    """True if a Pareto-dominates b (better/equal in all, strictly better in one)."""
    return a[0] <= b[0] and a[1] <= b[1] and (a[0] < b[0] or a[1] < b[1])


def _constrained_dominates(
    a: Objectives,
    b: Objectives,
    a_score: float,
    b_score: float,
    threshold: float,
) -> bool:
    """Dominance with a score constraint (constraint-first NSGA-II, Deb 2000).

    A solution whose score is strictly better than *threshold* is considered
    feasible; a feasible solution always dominates an infeasible one.
    """
    a_feasible = a_score < threshold
    b_feasible = b_score < threshold
    if a_feasible and not b_feasible:
        return True
    if not a_feasible and b_feasible:
        return False
    return _dominates(a, b)


def _percentile_75(scores: list[float]) -> float:
    if not scores:
        return INVALID_SCORE
    s = sorted(scores)
    return s[min(int(0.75 * len(s)), len(s) - 1)]


def non_dominated_sort(
    nodes: list[SearchNode],
    objectives: dict[int, Objectives],
    score_threshold: float | None = None,
) -> list[list[SearchNode]]:
    """Fast non-dominated sort (Deb 2002). front[0] is the Pareto front."""
    id_to_node = {n.id: n for n in nodes}

    if score_threshold is not None:
        raw: dict[int, float] = {n.id: n.score for n in nodes}

        def _dom(a_id: int, b_id: int) -> bool:
            return _constrained_dominates(
                objectives[a_id],
                objectives[b_id],
                raw[a_id],
                raw[b_id],
                score_threshold,
            )
    else:

        def _dom(a_id: int, b_id: int) -> bool:
            return _dominates(objectives[a_id], objectives[b_id])

    domination_count: dict[int, int] = {n.id: 0 for n in nodes}
    dominated_set: dict[int, list[int]] = {n.id: [] for n in nodes}

    for a in nodes:
        for b in nodes:
            if a.id == b.id:
                continue
            if _dom(a.id, b.id):
                dominated_set[a.id].append(b.id)
            elif _dom(b.id, a.id):
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
        return {n.id: INVALID_SCORE for n in front}

    distances: dict[int, float] = {n.id: 0.0 for n in front}
    n_objectives = 2

    for m in range(n_objectives):
        sorted_front = sorted(front, key=lambda n: objectives[n.id][m])
        obj_min = objectives[sorted_front[0].id][m]
        obj_max = objectives[sorted_front[-1].id][m]

        distances[sorted_front[0].id] = INVALID_SCORE
        distances[sorted_front[-1].id] = INVALID_SCORE

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
        crossover_distance_threshold: int = 10,
        epoch_diversity: float = 0.0,
    ):
        self.pool_size = pool_size
        self.crossover_distance_threshold = crossover_distance_threshold
        self.epoch_diversity = epoch_diversity

    @property
    def top_k_count(self) -> int:
        return self.pool_size

    def _is_duplicate(
        self, node: SearchNode[TState], other: SearchNode[TState]
    ) -> bool:
        if node.signature is None or other.signature is None:
            return False
        return node.signature == other.signature

    def select_parent(
        self, nodes: list[SearchNode[TState]], progress: float
    ) -> tuple[int, int | None]:
        _ = progress
        valid = [n for n in nodes if n.score < INVALID_SCORE]
        if not valid:
            return nodes[0].id if nodes else 0, None

        max_score = max(n.score for n in valid) or 1.0
        max_complexity = max(n.complexity for n in valid) or 1.0
        objectives: dict[int, Objectives] = {
            n.id: (n.score / max_score, n.complexity / max_complexity) for n in valid
        }

        score_threshold = _percentile_75([n.score for n in valid])
        fronts = non_dominated_sort(valid, objectives, score_threshold=score_threshold)
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
            if not any(self._is_duplicate(node, p) for p in pool):
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
        if len(pool) >= 2:
            p2_candidate = _tournament(exclude_id=p1.id)
            sig1, sig2 = p1.signature, p2_candidate.signature
            if (
                sig1 is not None
                and sig2 is not None
                and hamming_distance(sig1, sig2) > self.crossover_distance_threshold
            ):
                return p1.id, p2_candidate.id

        return p1.id, None

    def epoch_seeds(
        self, pool: list[SearchNode[TState]], max_seeds: int
    ) -> list[SearchNode[TState]]:
        """Return a diverse Pareto-front subset to seed the next epoch."""
        valid = [n for n in pool if n.score < INVALID_SCORE]
        if not valid:
            return pool[:max_seeds]

        max_score = max(n.score for n in valid) or 1.0
        max_complexity = max(n.complexity for n in valid) or 1.0
        objectives: dict[int, Objectives] = {
            n.id: (n.score / max_score, n.complexity / max_complexity) for n in valid
        }

        score_threshold = _percentile_75([n.score for n in valid])
        fronts = non_dominated_sort(valid, objectives, score_threshold=score_threshold)

        pareto_nodes: list[SearchNode[TState]] = []
        for front in fronts:
            for node in front:
                if not any(self._is_duplicate(node, s) for s in pareto_nodes):
                    pareto_nodes.append(node)

        pareto_nodes.sort(key=lambda n: n.score)
        seeds = pareto_nodes[:max_seeds]
        return seeds or valid[:max_seeds]

    def should_diversify(self, pool: list[SearchNode[TState]]) -> tuple[bool, float]:
        diversity = pool_diversity(pool)
        return self.epoch_diversity > 0 and diversity < self.epoch_diversity, diversity

    def create_new_state(self, result: Result[TState]) -> ChainState[TState]:
        return ChainState(score=result.score, payload=result.payload)
