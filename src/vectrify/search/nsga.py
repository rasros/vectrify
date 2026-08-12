import logging
import random
from collections.abc import Callable, Mapping
from typing import Any, Generic, TypeVar

from vectrify.score.complexity import OBJECTIVE_NAMES
from vectrify.search.diversity import hamming_distance, pool_diversity
from vectrify.search.models import INVALID_SCORE, SearchNode

log = logging.getLogger(__name__)

TState = TypeVar("TState")

# An objective vector; lower is better in every component. Any arity is
# accepted so callers can trade off two or more objectives without this module
# needing to know how many.
Objectives = tuple[float, ...]


def _dominates(a: Objectives, b: Objectives) -> bool:
    """True if a Pareto-dominates b (better/equal in all, strictly better in one).

    *a* and *b* must have the same arity; ``strict=True`` makes a mismatch an
    error rather than silently comparing only the shared prefix and ignoring
    the remaining objectives.
    """
    pairs = list(zip(a, b, strict=True))
    return all(x <= y for x, y in pairs) and any(x < y for x, y in pairs)


def pareto_front(items: list, key: "Callable[[Any], Objectives]") -> list:
    """Return the items whose key(item) vector is not Pareto-dominated by any
    other item's (lower is better in every objective).

    *key* may return a vector of any arity as long as it is the same for every
    item.
    """
    points = [key(it) for it in items]
    return [
        items[i]
        for i, p in enumerate(points)
        if not any(_dominates(q, p) for j, q in enumerate(points) if j != i)
    ]


def non_dominated_sort(
    nodes: list[SearchNode],
    objectives: Mapping[int, Objectives],
) -> list[list[SearchNode]]:
    """Fast non-dominated sort (Deb 2002). front[0] is the Pareto front.

    Textbook dominance over the whole objective vector: no objective is
    privileged. Visual error used to be gated ahead of the others, which made
    it primary and the rest tie-breakers; keeping every objective equal is
    what lets a metric like a complexity ratio actually shape the front.
    """
    id_to_node = {n.id: n for n in nodes}

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
    objectives: Mapping[int, Objectives],
) -> dict[int, float]:
    """Compute crowding distance to maintain diversity within a front.

    Works for any objective arity; the count is read off the vectors rather
    than assumed.
    """
    if len(front) <= 2:
        return {n.id: INVALID_SCORE for n in front}

    distances: dict[int, float] = {n.id: 0.0 for n in front}
    n_objectives = len(objectives[front[0].id])

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


def build_objectives(nodes: list[SearchNode]) -> dict[int, Objectives]:
    """Normalize score plus every registered metric into an objective vector.

    The vector is ``(score, *OBJECTIVE_NAMES)`` in registry order. Each objective is
    scaled by its own population maximum, so they are directly comparable and no
    weighting between them is needed -- NSGA trades them off by dominance
    instead. Adding a metric to the registry lengthens the vector, which the
    dominance and crowding helpers handle without changes.

    Callers must pass only valid nodes (score < INVALID_SCORE); an infinite
    score would corrupt the normalization for every other node.
    """
    max_score = max((n.score for n in nodes), default=1.0) or 1.0
    maxima = {
        name: max((n.metrics.get(name, 0.0) for n in nodes), default=1.0) or 1.0
        for name in OBJECTIVE_NAMES
    }
    return {
        n.id: (
            n.score / max_score,
            *(n.metrics.get(name, 0.0) / maxima[name] for name in OBJECTIVE_NAMES),
        )
        for n in nodes
    }


def pareto_select(
    nodes: list[SearchNode],
    objectives: Mapping[int, Objectives],
    max_keep: int,
) -> list[SearchNode]:
    """Walk Pareto fronts in order, taking crowding-distance-diverse nodes
    from each until *max_keep* are selected."""
    fronts = non_dominated_sort(nodes, objectives)
    selected: list[SearchNode] = []
    for front in fronts:
        if len(selected) >= max_keep:
            break
        distances = crowding_distance(front, objectives)
        for node in sorted(front, key=lambda n: -distances[n.id]):
            if len(selected) >= max_keep:
                break
            selected.append(node)
    return selected


class NsgaStrategy(Generic[TState]):
    """NSGA-II-style selection balancing visual quality and SVG complexity."""

    def __init__(
        self,
        pool_size: int = 20,
        crossover_distance_threshold: int = 10,
        epoch_diversity: float = 0.0,
        tournament_size: int = 2,
    ):
        self.pool_size = pool_size
        self.crossover_distance_threshold = crossover_distance_threshold
        self.epoch_diversity = epoch_diversity
        # Selection intensity is a function of the tournament size alone -- the
        # winner's expected quantile is ~1/(size+1) -- so this is an absolute
        # count rather than a fraction of the pool, and stays meaningful when
        # pool_size changes.
        self.tournament_size = max(2, tournament_size)

    def _is_duplicate(
        self, node: SearchNode[TState], other: SearchNode[TState]
    ) -> bool:
        if node.signature is None or other.signature is None:
            return False
        return node.signature == other.signature

    def select_parent(self, nodes: list[SearchNode[TState]]) -> tuple[int, int | None]:
        valid = [n for n in nodes if n.score < INVALID_SCORE]
        if not valid:
            return nodes[0].id if nodes else 0, None

        objectives = build_objectives(valid)

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
            if not any(self._is_duplicate(node, p) for p in pool):
                pool.append(node)
        if not pool:
            pool = sorted_valid[: self.pool_size]

        def _tournament(exclude_id: int | None = None) -> SearchNode[TState]:
            candidates = [n for n in pool if n.id != exclude_id]
            if len(candidates) < 2:
                return candidates[0] if candidates else pool[0]
            sample = random.sample(
                candidates, min(self.tournament_size, len(candidates))
            )
            return min(sample, key=lambda n: (rank[n.id], -crowd[n.id]))

        p1 = _tournament()
        if len(pool) >= 2:
            p2_candidate = _tournament(exclude_id=p1.id)
            sig1, sig2 = p1.signature, p2_candidate.signature
            # Two nodes of one lineage are the same drawing at different
            # stages, so grafting between them mostly reshuffles a candidate
            # into itself: measured on the bench, crossover in a single-seed
            # pool took the largest share of the task budget for 1-10% of the
            # gain. Root 0 means the caller tracks no lineage, which must not
            # read as "all one lineage" and disable crossover outright.
            same_lineage = p1.root_id != 0 and p1.root_id == p2_candidate.root_id
            if (
                not same_lineage
                and sig1 is not None
                and sig2 is not None
                and hamming_distance(sig1, sig2) > self.crossover_distance_threshold
            ):
                return p1.id, p2_candidate.id

        return p1.id, None

    def select_survivors(
        self, nodes: list[SearchNode[TState]], max_keep: int
    ) -> list[SearchNode[TState]]:
        """Truncate parents+children by non-dominated rank, then crowding.

        Nodes that failed to score go first: they have no usable objective
        vector, and a single infinite score would corrupt the population
        normalization build_objectives applies to everyone else.
        """
        valid = [n for n in nodes if n.score < INVALID_SCORE]
        if len(valid) <= max_keep:
            invalid = [n for n in nodes if n.score >= INVALID_SCORE]
            return valid + invalid[: max_keep - len(valid)]
        return pareto_select(valid, build_objectives(valid), max_keep)

    def epoch_parents(
        self, pool: list[SearchNode[TState]], max_parents: int
    ) -> list[SearchNode[TState]]:
        """Return a diverse Pareto-front subset for the next epoch's LLM edits."""
        valid = [n for n in pool if n.score < INVALID_SCORE]
        if not valid:
            return pool[:max_parents]

        objectives = build_objectives(valid)

        fronts = non_dominated_sort(valid, objectives)

        pareto_nodes: list[SearchNode[TState]] = []
        for front in fronts:
            for node in front:
                if not any(self._is_duplicate(node, s) for s in pareto_nodes):
                    pareto_nodes.append(node)

        pareto_nodes.sort(key=lambda n: n.score)
        parents = pareto_nodes[:max_parents]
        return parents or valid[:max_parents]

    def should_diversify(self, pool: list[SearchNode[TState]]) -> tuple[bool, float]:
        diversity = pool_diversity(pool)
        return self.epoch_diversity > 0 and diversity < self.epoch_diversity, diversity
