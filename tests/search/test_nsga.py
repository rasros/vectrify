import statistics
from unittest.mock import patch

import pytest

from vectrify.score.metrics import SCORER_METRICS
from vectrify.search import ChainState, SearchNode, nsga
from vectrify.search.diversity import simhash
from vectrify.search.models import INVALID_SCORE
from vectrify.search.nsga import (
    NsgaStrategy,
    _dominates,
    build_objectives,
    crowding_distance,
    non_dominated_sort,
    pareto_front,
)


def make_node(
    node_id: int,
    score: float,
    content: str | None = None,
    edge: float = 0.0,
    colour: float = 0.0,
    shape: float = 0.0,
    detail: float = 0.0,
) -> SearchNode:
    state = ChainState(score=score, payload=None)
    return SearchNode(
        score=score,
        id=node_id,
        parent_id=0,
        state=state,
        metrics={
            "edge": edge,
            "colour": colour,
            "shape": shape,
            "detail": detail,
        },
        signature=simhash(content) if content else None,
    )


def test_dominates_is_decided_by_a_majority_of_the_objectives():
    """Unanimity leaves almost nothing dominating anything, so rank stops
    separating candidates and crowding distance decides survival instead."""
    # Wins two, loses one -> a majority, so it dominates.
    assert _dominates((0.1, 0.1, 0.9), (0.2, 0.2, 0.1))
    assert not _dominates((0.2, 0.2, 0.1), (0.1, 0.1, 0.9))
    # Better in all three.
    assert _dominates((0.1, 0.1, 0.1), (0.2, 0.2, 0.2))
    # Equal in two, better in the third: one win, no losses.
    assert _dominates((0.2, 0.2, 0.1), (0.2, 0.2, 0.2))
    # One win against one loss is not a majority.
    assert not _dominates((0.1, 0.2, 0.5), (0.2, 0.1, 0.5))


def test_a_cycle_of_majorities_still_leaves_every_node_ranked():
    """A majority relation is not transitive: three candidates can each beat
    the next. Peeling fronts alone would never place them, and a node dropped
    from the sort is one deleted from the pool with nothing replacing it."""
    nodes = [make_node(i, float(i)) for i in range(1, 4)]
    objectives = {
        1: (0.0, 1.0, 2.0),
        2: (2.0, 0.0, 1.0),
        3: (1.0, 2.0, 0.0),
    }
    fronts = non_dominated_sort(nodes, objectives)
    assert sorted(n.id for front in fronts for n in front) == [1, 2, 3]


@pytest.mark.parametrize("arity", [1, 2, 3, 5])
def test_dominates_supports_any_arity(arity):
    better = tuple([0.1] * arity)
    worse = tuple([0.2] * arity)
    assert _dominates(better, worse)
    assert not _dominates(worse, better)
    assert not _dominates(better, better)


def test_dominates_rejects_mismatched_arity():
    with pytest.raises(ValueError, match="argument 2 is longer"):
        _dominates((0.1, 0.2), (0.1, 0.2, 0.3))


def test_front_keeps_what_a_majority_cannot_beat():
    items = [
        # One win and one loss against each other, so neither takes a majority.
        {"n": "sharper", "o": (0.1, 0.9, 0.5)},
        {"n": "truer_colour", "o": (0.9, 0.1, 0.5)},
        # Loses a majority to both.
        {"n": "beaten", "o": (0.2, 0.95, 0.6)},
    ]
    front = pareto_front(items, key=lambda it: it["o"])
    assert {it["n"] for it in front} == {"sharper", "truer_colour"}


def test_non_dominated_sort_with_three_objectives():
    nodes = [make_node(i, float(i)) for i in range(1, 4)]
    objectives = {1: (0.1, 0.1, 0.1), 2: (0.2, 0.2, 0.2), 3: (0.3, 0.3, 0.3)}
    fronts = non_dominated_sort(nodes, objectives)
    assert [[n.id for n in f] for f in fronts] == [[1], [2], [3]]


def test_crowding_distance_reads_arity_from_the_vectors():
    """With a hardcoded arity of 2 the third objective's spread was ignored."""
    nodes = [make_node(i, float(i)) for i in range(1, 5)]
    # Identical in the first two objectives, spread only in the third.
    objectives = {
        1: (0.5, 0.5, 0.0),
        2: (0.5, 0.5, 0.1),
        3: (0.5, 0.5, 0.7),
        4: (0.5, 0.5, 1.0),
    }
    dist = crowding_distance(nodes, objectives)
    assert dist[1] == float("inf")
    assert dist[4] == float("inf")
    # Node 3 sits in a sparser neighbourhood than node 2, so it must score
    # higher -- which is only true if objective 3 was measured at all.
    assert dist[3] > dist[2] > 0.0


def test_build_objectives_keeps_one_component_per_measure():
    """The pool is ranked by trading the measures off, so each has to survive
    into the vector as its own axis. Collapsed into one component the majority
    relation degenerates to a plain comparison and crowding distance has no
    second axis to spread along."""
    nodes = [
        make_node(1, 0.5, edge=0.2, colour=1000.0),
        make_node(2, 1.0, edge=0.4, colour=500.0),
    ]

    objectives = build_objectives(nodes)

    assert all(len(v) == len(SCORER_METRICS) for v in objectives.values())
    assert objectives[1] == pytest.approx((0.5, 1.0, 0.0, 0.0))
    assert objectives[2] == pytest.approx((1.0, 0.5, 0.0, 0.0))


def test_build_objectives_scales_each_measure_by_its_own_maximum():
    """Colour arrives on a scale hundreds of times larger than edge. Dominance
    does not care -- it compares component by component -- but crowding
    distance does, and would otherwise spread the pool along colour alone."""
    nodes = [
        make_node(1, 0.5, edge=0.2, colour=1000.0),
        make_node(2, 1.0, edge=0.4, colour=500.0),
    ]

    objectives = build_objectives(nodes)

    assert max(v[0] for v in objectives.values()) == 1.0
    assert max(v[1] for v in objectives.values()) == 1.0


def test_dominance_is_unchanged_by_rescaling_one_measure():
    """Why no weights are applied here: a positive rescale of a component
    cannot change any dominance verdict, so a weight in this vector is inert."""
    a = (0.1, 0.9, 0.5)
    b = (0.2, 0.4, 0.6)
    scaled_a = (0.1 * 1000, 0.9, 0.5)
    scaled_b = (0.2 * 1000, 0.4, 0.6)
    assert nsga._dominates(a, b) == nsga._dominates(scaled_a, scaled_b)


def test_build_objectives_ignores_the_score_it_does_not_rank_on():
    """The round score is the blend of these same two measures, so feeding it
    back in would count them twice."""
    same = build_objectives(
        [
            make_node(1, 0.0, edge=1.0, colour=1.0),
            make_node(2, 9e9, edge=1.0, colour=1.0),
        ]
    )

    assert same[1] == same[2]


def test_build_objectives_separates_candidates_alike_in_score():
    """The measures are carried separately so that two candidates scoring the
    same can still be told apart by structure or by colour."""
    clean = make_node(1, 0.4, edge=0.2, colour=0.2)
    smudged = make_node(2, 0.4, edge=0.8, colour=0.8)
    objectives = build_objectives([clean, smudged])
    assert _dominates(objectives[clean.id], objectives[smudged.id])
    assert not _dominates(objectives[smudged.id], objectives[clean.id])


def test_build_objectives_survives_all_zero_objectives():
    nodes = [make_node(i, 0.0) for i in range(1, 4)]

    objectives = build_objectives(nodes)

    assert all(v == (0.0,) * len(SCORER_METRICS) for v in objectives.values())


def test_non_dominated_sort_all_pareto():
    nodes = [make_node(1, 0.1), make_node(2, 0.5), make_node(3, 0.9)]
    objectives = {1: (0.1, 0.9), 2: (0.5, 0.5), 3: (0.9, 0.1)}
    fronts = non_dominated_sort(nodes, objectives)
    assert len(fronts) == 1
    assert {n.id for n in fronts[0]} == {1, 2, 3}


def test_non_dominated_sort_two_fronts():
    nodes = [make_node(i, float(i)) for i in range(1, 5)]
    objectives = {
        1: (0.1, 0.9),
        2: (0.9, 0.1),
        3: (0.5, 0.95),
        4: (0.95, 0.5),
    }
    fronts = non_dominated_sort(nodes, objectives)
    assert len(fronts) == 2
    assert {n.id for n in fronts[0]} == {1, 2}
    assert {n.id for n in fronts[1]} == {3, 4}


def test_crowding_distance_boundary_nodes_are_infinite():
    nodes = [make_node(i, float(i)) for i in range(1, 5)]
    objectives = {1: (0.0, 0.0), 2: (0.3, 0.3), 3: (0.6, 0.6), 4: (1.0, 1.0)}
    dist = crowding_distance(nodes, objectives)
    assert dist[1] == float("inf")
    assert dist[4] == float("inf")
    assert 0 < dist[2] < float("inf")
    assert 0 < dist[3] < float("inf")


def test_crowding_distance_two_nodes_are_infinite():
    nodes = [make_node(1, 0.1), make_node(2, 0.9)]
    objectives = {1: (0.1, 0.2), 2: (0.9, 0.8)}
    dist = crowding_distance(nodes, objectives)
    assert dist[1] == float("inf")
    assert dist[2] == float("inf")


def test_select_parent_returns_valid_node_id():
    strategy = NsgaStrategy(pool_size=5, crossover_distance_threshold=65)
    nodes = [make_node(i, i * 0.1, edge=i * 0.1) for i in range(1, 6)]
    pid, secondary = strategy.select_parent(nodes)
    assert pid in {n.id for n in nodes}
    assert secondary is None


def test_select_parent_crossover_returns_two_distinct_parents():
    strategy = NsgaStrategy(pool_size=5, crossover_distance_threshold=0)
    nodes = [
        make_node(i, i * 0.1, content=f"<svg><rect id='{i}'/></svg>", edge=i * 0.1)
        for i in range(1, 6)
    ]
    results = set()
    for _ in range(20):
        pid, secondary = strategy.select_parent(nodes)
        if secondary is not None:
            results.add((pid, secondary))
    assert results, "crossover never selected a secondary parent"
    assert all(pair[0] != pair[1] for pair in results)


def test_select_parent_skips_invalid_nodes():
    strategy = NsgaStrategy(pool_size=10, crossover_distance_threshold=65)
    sentinel = SearchNode(
        score=float("inf"),
        id=0,
        parent_id=0,
        state=ChainState(score=float("inf"), payload=None),
        metrics={"edge": 0.0, "colour": 0.0},
    )
    valid = make_node(1, 0.3)
    pid, _ = strategy.select_parent([sentinel, valid])
    assert pid == 1


def test_select_parent_only_invalid_falls_back():
    strategy = NsgaStrategy(pool_size=5, crossover_distance_threshold=65)
    sentinel = SearchNode(
        score=float("inf"),
        id=0,
        parent_id=0,
        state=ChainState(score=float("inf"), payload=None),
        metrics={"edge": 0.0, "colour": 0.0},
    )
    pid, secondary = strategy.select_parent([sentinel])
    assert pid == 0
    assert secondary is None


def test_diversity_rejects_exact_duplicate_with_worse_score():
    content = "<svg><rect width='200' height='200'/></svg>"
    strategy = NsgaStrategy(pool_size=5, crossover_distance_threshold=65)
    good = make_node(1, 0.1, content=content)
    duplicate = make_node(2, 0.9, content=content)  # exact same content, worse score
    different = make_node(3, 0.5, content="<svg><completely different/></svg>")

    selected = set()
    for _ in range(50):
        pid, _ = strategy.select_parent([good, duplicate, different])
        selected.add(pid)
    assert 2 not in selected


def test_colour_can_outweigh_a_better_embedding_distance():
    """Neither part is privileged. Visual error used to be gated ahead of the
    rest, which made it primary and left the other a tie-breaker; a second
    measure cannot shape the front while that is true. Here the worse
    embedding distance wins because it is far better on colour."""
    strategy = NsgaStrategy(pool_size=10, crossover_distance_threshold=65)
    nodes = [
        make_node(1, 0.80, colour=1000.0),
        make_node(2, 1.00, colour=100.0),
    ]

    selected = {strategy.select_parent(nodes)[0] for _ in range(50)}

    assert selected == {2}


def test_pool_size_limits_candidate_set():
    strategy = NsgaStrategy(pool_size=2, crossover_distance_threshold=65)
    nodes = [
        make_node(1, 0.1, edge=10.0),
        make_node(2, 0.2, edge=20.0),
        make_node(3, 0.8, edge=800.0),
        make_node(4, 0.9, edge=900.0),
        make_node(5, 1.0, edge=1000.0),
    ]
    selected = {strategy.select_parent(nodes)[0] for _ in range(50)}
    assert selected <= {1, 2}


def test_pool_size_one_always_returns_same_node():
    strategy = NsgaStrategy(pool_size=1, crossover_distance_threshold=65)
    nodes = [make_node(i, i * 0.1, edge=float(i * 100)) for i in range(1, 6)]
    selected = {strategy.select_parent(nodes)[0] for _ in range(20)}
    assert selected == {1}


def test_epoch_parents_returns_the_best_ranked_tier():
    """Two candidates good at different things both survive; one that is worse
    than a rival on every measure does not."""
    strategy = NsgaStrategy(pool_size=10)
    nodes = [
        make_node(1, 0.0, edge=100.0, colour=900.0),  # wins edge
        make_node(2, 0.0, edge=900.0, colour=100.0),  # wins colour
        make_node(3, 0.0, edge=950.0, colour=950.0),  # loses to both
    ]

    seeds = strategy.epoch_parents(nodes, max_parents=2)

    assert {n.id for n in seeds} == {1, 2}


def test_epoch_parents_respects_max_parents():
    strategy = NsgaStrategy(pool_size=10)
    nodes = [make_node(i, i * 0.1, edge=float(i * 100)) for i in range(1, 8)]
    seeds = strategy.epoch_parents(nodes, max_parents=3)
    assert len(seeds) == 3


def test_epoch_parents_filters_exact_duplicates():
    content = "<svg>" + "".join(str(i) for i in range(500)) + "</svg>"
    strategy = NsgaStrategy(pool_size=10)
    good = make_node(1, 0.1, edge=100.0, content=content)
    duplicate = make_node(2, 0.1, edge=100.0, content=content)  # exact copy
    different = make_node(
        3, 0.2, edge=200.0, content="<svg><completely different/></svg>"
    )
    seeds = strategy.epoch_parents([good, duplicate, different], max_parents=3)
    seed_ids = {n.id for n in seeds}
    assert not (1 in seed_ids and 2 in seed_ids)


def test_epoch_parents_empty_pool_returns_empty():
    strategy = NsgaStrategy(pool_size=10)
    seeds = strategy.epoch_parents([], max_parents=5)
    assert seeds == []


def test_epoch_parents_come_back_in_rank_order():
    """Best-ranked first, with nothing sorted afterwards: a score to sort by
    would have to blend the measures, which is what dominance replaced."""
    strategy = NsgaStrategy(pool_size=10)
    nodes = [
        make_node(1, 0.0, edge=200.0, colour=200.0),
        make_node(2, 0.0, edge=400.0, colour=400.0),
        make_node(3, 0.0, edge=600.0, colour=600.0),
        make_node(4, 0.0, edge=800.0, colour=800.0),
    ]
    seeds = strategy.epoch_parents(nodes, max_parents=4)

    assert seeds[0].id == 1


def test_epoch_parents_all_invalid_falls_back():
    strategy = NsgaStrategy(pool_size=10)
    nodes = [make_node(i, float("inf")) for i in range(1, 4)]
    seeds = strategy.epoch_parents(nodes, max_parents=5)
    assert len(seeds) == 3


def test_accurate_and_simple_candidates_share_the_front():
    """Neither dominates: one wins on error, the other on structure. Under the
    old feasibility gate the accurate one was promoted outright."""
    n1 = make_node(1, score=0.1, edge=5000.0)
    n2 = make_node(2, score=0.9, edge=10.0)
    objectives = {1: (0.1, 1.0), 2: (0.9, 0.0)}
    fronts = non_dominated_sort([n1, n2], objectives)
    assert len(fronts) == 1
    assert {n.id for n in fronts[0]} == {1, 2}


def test_non_dominated_sort_no_threshold_simple_dominates_complex():
    n1 = make_node(1, score=0.1, edge=5000.0)
    n2 = make_node(2, score=0.9, edge=10.0)
    objectives = {1: (0.1, 1.0), 2: (0.9, 0.0)}
    fronts = non_dominated_sort([n1, n2], objectives)
    assert len(fronts) == 1
    assert {n.id for n in fronts[0]} == {1, 2}


def test_tournament_size_defaults_to_two():
    assert NsgaStrategy().tournament_size == 2


def test_tournament_size_is_clamped_to_a_usable_minimum():
    assert NsgaStrategy(tournament_size=1).tournament_size == 2
    assert NsgaStrategy(tournament_size=0).tournament_size == 2


def test_tournament_size_larger_than_the_pool_is_safe():
    strategy = NsgaStrategy(pool_size=10, tournament_size=50)
    nodes = [make_node(i, i * 0.1) for i in range(1, 4)]
    pid, _ = strategy.select_parent(nodes)
    assert pid in {n.id for n in nodes}


def test_tournament_size_of_one_node_pool_is_safe():
    strategy = NsgaStrategy(tournament_size=8)
    nodes = [make_node(1, 0.5)]
    assert strategy.select_parent(nodes) == (1, None)


def test_larger_tournament_biases_harder_toward_the_objective():
    import random as _random

    def better_half_rate(size: int, trials: int = 1500) -> float:
        strategy = NsgaStrategy(
            pool_size=20, crossover_distance_threshold=999, tournament_size=size
        )
        _random.seed(7)
        hits = 0
        for _ in range(trials):
            nodes = [
                make_node(
                    i,
                    _random.random(),
                    content=f"n{i}-{_random.random()}",
                    colour=_random.random() * 5000,
                )
                for i in range(20)
            ]
            # Read the axis these nodes actually differ on. The vector is in
            # registry order, so its first component is not colour.
            axis = SCORER_METRICS.index("colour")
            blended = build_objectives(nodes)
            median = sorted(v[axis] for v in blended.values())[10]
            pid, _secondary = strategy.select_parent(nodes)
            if blended[pid][axis] <= median:
                hits += 1
        return hits / trials

    assert better_half_rate(4) > better_half_rate(2) > 0.5


def test_crossover_is_skipped_within_one_lineage():
    """Two nodes of one lineage are the same drawing at different stages, so
    grafting between them recombines a candidate with itself."""
    strategy = NsgaStrategy(pool_size=4, crossover_distance_threshold=0)
    nodes = [
        make_node(i, 0.1 * i, content=f"<svg><rect id='{i}'/></svg>")
        for i in range(1, 5)
    ]
    for n in nodes:
        n.root_id = 7

    for _ in range(20):
        assert strategy.select_parent(nodes)[1] is None


def test_crossover_fires_across_lineages():
    strategy = NsgaStrategy(pool_size=4, crossover_distance_threshold=0)
    nodes = [
        make_node(i, 0.1 * i, content=f"<svg><rect id='{i}'/></svg>")
        for i in range(1, 5)
    ]
    for n in nodes:
        n.root_id = n.id

    assert any(strategy.select_parent(nodes)[1] is not None for _ in range(20))


def test_untracked_lineage_leaves_crossover_enabled():
    """root_id 0 means the caller tracks no lineage; reading that as one shared
    lineage would disable crossover for every caller outside the engine."""
    strategy = NsgaStrategy(pool_size=4, crossover_distance_threshold=0)
    nodes = [
        make_node(i, 0.1 * i, content=f"<svg><rect id='{i}'/></svg>")
        for i in range(1, 5)
    ]
    assert all(n.root_id == 0 for n in nodes)
    assert any(strategy.select_parent(nodes)[1] is not None for _ in range(20))


def test_select_survivors_does_not_simply_keep_the_best_score():
    """Regression: survival used to evict the worst score outright, so the other
    objectives only reached parent selection and the pool collapsed onto one
    measure. A candidate carrying the worst score still wins a place if it
    takes the majority of the rest.

    Elitism reserves one slot for the best score, so the point is tested on the
    slots elitism does not claim."""
    strategy = NsgaStrategy()
    best_score = make_node(1, 0.1, colour=0.9)
    wins_the_rest = make_node(2, 0.85, colour=0.05)
    middling = [
        make_node(3, 0.5, colour=0.5),
        make_node(4, 0.6, colour=0.6),
    ]

    kept = {
        n.id
        for n in strategy.select_survivors([best_score, wins_the_rest, *middling], 2)
    }

    assert kept == {1, 2}


def test_select_survivors_drops_unscored_nodes_first():
    """An infinite score would normalise every other objective to zero."""
    strategy = NsgaStrategy()
    scored = [make_node(1, 0.1), make_node(2, 0.2)]
    unscored = make_node(3, INVALID_SCORE)

    kept = strategy.select_survivors([*scored, unscored], 2)

    assert [n.id for n in kept] == [1, 2]


def test_select_survivors_keeps_everything_that_fits():
    strategy = NsgaStrategy()
    nodes = [make_node(i, 0.1 * i) for i in range(1, 4)]

    assert len(strategy.select_survivors(nodes, 5)) == 3


def test_ranking_survives_a_population_nothing_wins():
    """Majority dominance is a tournament, and a tournament this dense has no
    undominated member: on a real 200-node pool not one node was unbeaten. A
    sort that only peels undominated nodes returns a single undifferentiated
    front there, which leaves crowding distance to decide survival on spread
    instead of quality."""
    # Varying both measures the objective is built from, so tiers can track
    # quality rather than a number selection no longer reads.
    nodes = [
        make_node(i, 0.0, edge=100.0 * (i % 5), colour=10.0 * (i % 7))
        for i in range(60)
    ]
    objectives = build_objectives(nodes)

    fronts = non_dominated_sort(nodes, objectives)

    assert len(fronts) > 1, "ranking put the whole population in one tier"
    assert sum(len(f) for f in fronts) == len(nodes), "a node was dropped"

    best = statistics.fmean(objectives[n.id][0] for n in fronts[0])
    worst = statistics.fmean(objectives[n.id][0] for n in fronts[-1])
    assert best < worst, "the ranking does not track quality"


def test_a_three_way_cycle_ties_rather_than_vanishing():
    """Rock-paper-scissors over three objectives: each beats the next two-to-one
    and loses to the third. None can outrank the others, and none may be
    dropped -- a dropped node is a candidate deleted from the pool with nothing
    replacing it."""
    rock = make_node(1, 0.1, edge=0.5)
    paper = make_node(2, 0.2, edge=0.1)
    scissors = make_node(3, 0.5, edge=0.2)
    objectives = {
        rock.id: (0.1, 0.5, 0.2),
        paper.id: (0.2, 0.1, 0.5),
        scissors.id: (0.5, 0.2, 0.1),
    }

    fronts = non_dominated_sort([rock, paper, scissors], objectives)

    assert [len(f) for f in fronts] == [3]
    assert {n.id for n in fronts[0]} == {1, 2, 3}


def test_the_best_node_always_survives_its_generation():
    """NSGA-II is elitist because the best node sits in front 0 under Pareto
    dominance. The majority relation has no such guarantee -- two objectives
    can outvote the one the run is judged on -- and a generation that evicts
    the best candidate loses ground it never recovers."""
    best = make_node(1, 0.01, edge=1000.0)
    crowd = [make_node(i, 0.5 + 0.01 * i, edge=float(i)) for i in range(2, 12)]

    survivors = NsgaStrategy(pool_size=5).select_survivors([best, *crowd], max_keep=5)

    assert len(survivors) == 5
    assert best.id in {n.id for n in survivors}


def test_the_pool_ranking_is_reused_until_the_pool_changes():
    """Ranking compares every pair, and selection asked for it once per
    dispatched task on a pool that only changes at a generation boundary. That
    put an O(n^2) sort on the main thread hundreds of times per run, which is
    the thread every worker waits on."""
    strategy = NsgaStrategy(pool_size=5)
    nodes = [make_node(i, 0.1 * i, edge=float(10 - i)) for i in range(1, 8)]

    calls = []
    real = nsga.non_dominated_sort

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    with patch.object(nsga, "non_dominated_sort", counting):
        for _ in range(5):
            strategy.select_parent(nodes)
        assert len(calls) == 1, "the pool ranking was recomputed"

        # A changed pool has to be ranked again, or selection would go on
        # using positions for nodes that are no longer there.
        strategy.select_parent(nodes[:-1])
        assert len(calls) == 2


def test_a_changed_pool_is_never_ranked_from_stale_positions():
    """The cache is keyed on membership, so a node added to the pool must be
    reachable by selection rather than missing from the ranking."""
    strategy = NsgaStrategy(pool_size=10)
    nodes = [make_node(i, 0.5, edge=1.0) for i in range(1, 4)]
    strategy.select_parent(nodes)

    best = make_node(99, 0.001, edge=0.001, content="unique content here")
    pool, rank, _crowd = strategy._rank_pool([*nodes, best])

    assert best.id in rank
    assert best.id in {n.id for n in pool}


# ── The search is multi-objective, or it is not doing what it claims ─────────
#
# These guard the property, not the implementation. The objective vector was
# once collapsed into a single weighted sum, which left every part of NSGA-II
# below it inert -- a majority of one component is a plain comparison, Copeland
# tiers separate only what the sum separated, and crowding distance has no
# second axis to spread along -- while every test still passed. The cost is
# specific: a measure that only ever contributes a fraction of a sum can never
# outrank the others on the candidates it disagrees about, which is the one
# situation it was added for.


def test_every_scored_metric_survives_into_the_objective_vector():
    """Arity tracks the registry. A measure dropped from the vector still shows
    up in the reported score, so nothing else here would notice its loss."""
    nodes = [
        make_node(1, 0.5, edge=0.2, colour=0.9, shape=0.4),
        make_node(2, 0.6, edge=0.8, colour=0.1, shape=0.7),
    ]

    objectives = build_objectives(nodes)

    assert all(len(v) == len(SCORER_METRICS) for v in objectives.values())


def test_measures_that_disagree_can_rank_in_a_cycle():
    """With one component the relation is a total order: transitive, and no
    cycle is expressible. Three measures disagreeing produce one, which is the
    signature that they are genuinely being traded off -- and the reason
    ranking goes through Copeland rather than a peeling sort.

    (Three objectives and no exact ties leaves every *pair* comparable, since
    wins + losses = 3 cannot split evenly. Non-transitivity, not an antichain,
    is what multi-objectivity buys at this arity.)
    """
    a = make_node(1, 0.5, colour=0.1, edge=0.9, shape=0.5)
    b = make_node(2, 0.5, colour=0.5, edge=0.1, shape=0.9)
    c = make_node(3, 0.5, colour=0.9, edge=0.5, shape=0.1)

    objectives = build_objectives([a, b, c])
    va, vb, vc = (objectives[n.id] for n in (a, b, c))

    assert nsga._dominates(va, vb)
    assert nsga._dominates(vb, vc)
    assert nsga._dominates(vc, va)  # impossible under a single component
    assert nsga._copeland([va, vb, vc]) == [0, 0, 0]


def test_the_third_measure_decides_when_the_other_two_disagree():
    """What the vector buys, stated exactly. Majority rule needs two of three,
    so a candidate winning on shape alone stays dominated -- shape does not
    override colour and edge. It decides the pairs where those two split, and
    it decides them on which candidate is better, not by how much.

    A weighted sum cannot do that: shape carries the smallest weight, so a
    large margin on edge buries it. Here the two rules pick opposite winners.
    """
    a = make_node(1, 0.5, colour=0.1, edge=0.9, shape=0.1)
    b = make_node(2, 0.5, colour=0.9, edge=0.1, shape=0.9)

    objectives = build_objectives([a, b])
    va, vb = objectives[1], objectives[2]

    # colour prefers a, edge prefers b, shape prefers a -- so a takes it.
    assert nsga._dominates(va, vb)
    assert not nsga._dominates(vb, va)

    # The blend the pool used to be ranked by prefers b, because edge carries
    # nearly four times shape's weight and wins by a full unit here.
    # And it decides on which candidate is better, not by how much: b wins its
    # one axis by a full unit and still loses, which no weighted sum could
    # reproduce -- the smallest weight is always buried by a large margin
    # elsewhere. That sum no longer exists anywhere in the run.
    ie = SCORER_METRICS.index("edge")
    assert vb[ie] < va[ie]
    assert va[ie] - vb[ie] > 0.8


def test_a_candidate_winning_on_one_measure_alone_is_still_dominated():
    """Recorded so the limit is not mistaken for a bug. Majority rule means two
    of three; a specialist on a single axis loses to any candidate that beats
    it on the other two, and survives only on crowding distance within a tier.
    """
    specialist = make_node(1, 0.9, colour=1.0, edge=1.0, shape=0.0)
    generalist = make_node(2, 0.4, colour=0.4, edge=0.4, shape=0.9)

    objectives = build_objectives([specialist, generalist])

    assert nsga._dominates(objectives[2], objectives[1])


def test_crowding_distance_reads_every_axis():
    """It spreads the front along each objective in turn. Given one component
    it can only ever spread along that one, so a pool diverse in shape and
    uniform in colour would look uniform."""
    front = [
        make_node(1, 0.5, colour=0.5, edge=0.5, shape=0.0),
        make_node(2, 0.5, colour=0.5, edge=0.5, shape=0.5),
        make_node(3, 0.5, colour=0.5, edge=0.5, shape=1.0),
    ]
    objectives = build_objectives(front)

    distances = nsga.crowding_distance(front, objectives)

    # The extremes are pinned, and the middle node was reached through the
    # shape axis alone -- the other two are flat across this front.
    assert distances[1] == INVALID_SCORE
    assert distances[3] == INVALID_SCORE
    assert distances[2] > 0.0
