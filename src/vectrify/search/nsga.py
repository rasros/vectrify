import logging
import random
from collections.abc import Callable, Mapping
from typing import Any, Generic, TypeVar

from vectrify.score.metrics import COLOUR, COLOUR_WEIGHT, EDGE, EDGE_WEIGHT
from vectrify.search.diversity import hamming_distance, pool_diversity
from vectrify.search.models import INVALID_SCORE, SearchNode

log = logging.getLogger(__name__)

TState = TypeVar("TState")

# An objective vector; lower is better in every component. Any arity is
# accepted so callers can trade off two or more objectives without this module
# needing to know how many.
Objectives = tuple[float, ...]


def _dominates(a: Objectives, b: Objectives) -> bool:
    """True if a wins on more objectives than it loses on.

    A majority rather than Pareto's unanimity. The objectives are three
    imperfect measures of the same thing, and requiring all of them to agree
    means requiring three judges that are each wrong 15-20% of the time to be
    right together: measured one mutation from a parent, unanimity accepts
    correctly 55% of the time but only recognises 13% of the real improvements,
    where a majority recognises 46% at 50% correct. Almost nothing dominates
    under unanimity, so rank stops separating candidates and crowding distance
    -- which sorts for spread, not quality -- decides survival instead.

    The price is that a majority relation cycles, so it cannot be peeled into
    fronts the way Pareto dominance can. Callers rank it with _copeland.

    *a* and *b* must have the same arity; ``strict=True`` makes a mismatch an
    error rather than silently comparing only the shared prefix.
    """
    pairs = list(zip(a, b, strict=True))
    wins = sum(1 for x, y in pairs if x < y)
    losses = sum(1 for x, y in pairs if y < x)
    return wins > losses


def _copeland(points: list[Objectives]) -> list[int]:
    """How many rivals each point beats, less how many beat it.

    The majority relation is a tournament, not an order: three candidates can
    each beat the next in a cycle. Counting wins minus losses ranks a
    tournament without needing it to be transitive, and reduces to the usual
    thing when it happens to be -- a point nothing dominates still scores
    highest, and a point everything dominates still scores lowest.
    """
    scores = [0] * len(points)
    for i, a in enumerate(points):
        for j in range(i + 1, len(points)):
            b = points[j]
            if _dominates(a, b):
                scores[i] += 1
                scores[j] -= 1
            elif _dominates(b, a):
                scores[j] += 1
                scores[i] -= 1
    return scores


def pareto_front(items: list, key: "Callable[[Any], Objectives]") -> list:
    """Return the best-ranked items under the majority relation.

    Asking instead for the items nothing dominates returns nothing at all on a
    real population: measured on a 200-node pool, every single node was beaten
    by some other, because a majority relation cycles. This returns the top
    tier of the tournament ranking, which is the same set whenever the relation
    is a genuine order.

    *key* may return a vector of any arity as long as it is the same for every
    item.
    """
    if not items:
        return []
    scores = _copeland([key(it) for it in items])
    best = max(scores)
    return [item for item, score in zip(items, scores, strict=True) if score == best]


def non_dominated_sort(
    nodes: list[SearchNode],
    objectives: Mapping[int, Objectives],
) -> list[list[SearchNode]]:
    """Rank nodes into tiers by the majority relation, best tier first.

    Deb's peeling sort assumes dominance is transitive, and the majority
    relation is not: measured on a 200-node pool, not one node was undominated,
    so the peel produced no fronts at all and every node fell through to the
    unresolved bucket. Rank then separated nothing and crowding distance --
    which sorts for spread, not quality -- decided survival by itself, which is
    what left the best score unchanged across fifteen generations.

    Tiers come from the tournament score instead, so a cycle costs its members
    a tie with each other rather than costing the whole population its
    ordering. No objective is privileged; visual error used to be gated ahead
    of the others, which made it primary and the rest tie-breakers.
    """
    if not nodes:
        return []

    scores = _copeland([objectives[n.id] for n in nodes])
    tiers: dict[int, list[SearchNode]] = {}
    for node, score in zip(nodes, scores, strict=True):
        tiers.setdefault(score, []).append(node)
    return [tiers[score] for score in sorted(tiers, reverse=True)]


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
    """Blend the chromatic and structural distances into an objective vector.

    Each part is scaled by its own population maximum first, so a weighting
    between them means what it says rather than being decided by whichever
    happens to be on the larger scale.

    The result is one component, not three. Trading several measures off by
    dominance is worth the machinery when they disagree about which candidate
    is better in ways that are each defensible; measured against the evaluator
    panel these do not, and the vote scored below the parts it was built from
    (see score.metrics). Ranking still runs through the same tournament helper,
    where a single component simply orders the pool.

    Callers must pass only valid nodes (score < INVALID_SCORE); an infinite
    score would corrupt the normalization for every other node.
    """
    maxima = {
        name: max((n.metrics.get(name, 0.0) for n in nodes), default=1.0) or 1.0
        for name in (COLOUR, EDGE)
    }
    weights = {COLOUR: COLOUR_WEIGHT, EDGE: EDGE_WEIGHT}
    return {
        n.id: (
            sum(
                weight * (n.metrics.get(name, 0.0) / maxima[name])
                for name, weight in weights.items()
            ),
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
    """NSGA-II-style selection over the objective vector."""

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
        # Ranking the pool costs a comparison between every pair, and parent
        # selection asked for it once per dispatched task even though the pool
        # only changes at a generation boundary. Profiled on an 800-task run it
        # was a third of the main thread, which is the thread every worker
        # waits on -- so more workers bought nothing. Keyed on the pool's
        # membership, which is what the ranking depends on; a node's score and
        # metrics never change once it is built.
        self._ranked: tuple[tuple, list, dict, dict] | None = None

    def _rank_pool(
        self, valid: list[SearchNode[TState]]
    ) -> tuple[list[SearchNode[TState]], dict[int, int], dict[int, float]]:
        """Rank, crowd and de-duplicate the pool, reusing the last answer.

        Returns the selectable pool with each node's tier and crowding
        distance.
        """
        # Keyed on what the ranking is computed from, not on identity alone: a
        # node's measures never change once it is built, but a caller is free
        # to hand over a different population carrying the same ids, and an
        # id-only key would answer that with the previous pool's ordering.
        key = tuple(
            (n.id, n.metrics.get(COLOUR, 0.0), n.metrics.get(EDGE, 0.0)) for n in valid
        )
        cached = self._ranked
        if cached is not None and cached[0] == key:
            return cached[1], cached[2], cached[3]

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

        self._ranked = (key, pool, rank, crowd)
        return pool, rank, crowd

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

        pool, rank, crowd = self._rank_pool(valid)

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

        survivors = pareto_select(valid, build_objectives(valid), max_keep)

        # Elitism, which NSGA-II normally gets for free: under Pareto dominance
        # the best node is in front 0 by construction, but the majority
        # relation can outvote it two objectives to one and evict it. Measured,
        # it did -- a generation dropped the run's best candidate and the pool
        # never recovered its score. The best node keeps its place explicitly.
        best = min(valid, key=lambda n: n.score)
        if best.id not in {n.id for n in survivors}:
            survivors[-1] = best
        return survivors

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
