import pytest

from vectrify.search import ChainState, SearchNode
from vectrify.search.diversity import simhash
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
    zip_complexity: float = 100.0,
    content: str | None = None,
    node_complexity: float = 0.0,
    worst_region_4: float = 0.0,
    worst_region_16: float = 0.0,
    zip_ratio: float = 0.0,
    node_ratio: float = 0.0,
) -> SearchNode:
    state = ChainState(score=score, payload=None)
    return SearchNode(
        score=score,
        id=node_id,
        parent_id=0,
        state=state,
        metrics={
            "zip_complexity": zip_complexity,
            "node_complexity": node_complexity,
            "worst_region_4": worst_region_4,
            "worst_region_16": worst_region_16,
            "zip_ratio": zip_ratio,
            "node_ratio": node_ratio,
        },
        signature=simhash(content) if content else None,
    )


def test_dominates_honours_every_objective():
    """Regression: the comparison used to hardcode indices 0 and 1, so a third
    objective was silently ignored and a worse vector could 'dominate'."""
    # Better in the first two, worse in the third -> not dominance.
    assert not _dominates((0.1, 0.1, 0.9), (0.2, 0.2, 0.1))
    assert not _dominates((0.2, 0.2, 0.1), (0.1, 0.1, 0.9))
    # Better in all three -> dominance.
    assert _dominates((0.1, 0.1, 0.1), (0.2, 0.2, 0.2))
    # Equal in two, strictly better in the third -> dominance.
    assert _dominates((0.2, 0.2, 0.1), (0.2, 0.2, 0.2))


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


def test_pareto_front_supports_three_objectives():
    items = [
        {"n": "best", "o": (0.1, 0.1, 0.1)},
        {"n": "dominated", "o": (0.2, 0.2, 0.2)},
        # Worse on the first two but best on the third -> incomparable, so it
        # survives only if the third objective is actually considered.
        {"n": "third_only", "o": (0.9, 0.9, 0.0)},
    ]
    front = pareto_front(items, key=lambda it: it["o"])
    assert {it["n"] for it in front} == {"best", "third_only"}


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


def test_build_objectives_normalizes_every_registered_metric():
    from vectrify.score.complexity import OBJECTIVE_NAMES

    nodes = [
        make_node(1, 0.5, 200.0, node_ratio=1000.0, worst_region_4=0.2),
        make_node(2, 1.0, 400.0, node_ratio=500.0, worst_region_4=0.4),
    ]
    objectives = build_objectives(nodes)
    assert all(len(v) == len(OBJECTIVE_NAMES) + 1 for v in objectives.values())
    # Each objective is scaled by its own population maximum, so the largest
    # value in every column is exactly 1.0 -- that is what makes them
    # comparable without any weighting between them.
    assert objectives[1] == (0.5, 0.5, 0.0, 0.0, 1.0)
    assert objectives[2] == (1.0, 1.0, 0.0, 0.0, 0.5)


def test_build_objectives_charges_for_source_size():
    """Two candidates alike in score and render must still be separated by how
    much source they spend."""
    lean = make_node(1, 0.4, 100.0, node_ratio=200.0)
    bloated = make_node(2, 0.4, 100.0, node_ratio=8000.0)
    objectives = build_objectives([lean, bloated])
    assert _dominates(objectives[lean.id], objectives[bloated.id])
    assert not _dominates(objectives[bloated.id], objectives[lean.id])


def test_build_objectives_survives_all_zero_objectives():
    from vectrify.score.complexity import OBJECTIVE_NAMES

    nodes = [make_node(i, 0.0, 0.0, node_complexity=0.0) for i in range(1, 4)]
    objectives = build_objectives(nodes)
    zeros = (0.0,) * (len(OBJECTIVE_NAMES) + 1)
    assert all(v == zeros for v in objectives.values())


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
    nodes = [make_node(i, i * 0.1, i * 100.0) for i in range(1, 6)]
    pid, secondary = strategy.select_parent(nodes)
    assert pid in {n.id for n in nodes}
    assert secondary is None


def test_select_parent_crossover_returns_two_distinct_parents():
    strategy = NsgaStrategy(pool_size=5, crossover_distance_threshold=0)
    nodes = [
        make_node(i, i * 0.1, i * 100.0, content=f"<svg><rect id='{i}'/></svg>")
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
        metrics={"zip_complexity": 0.0, "node_complexity": 0.0},
    )
    valid = make_node(1, 0.3, 200.0)
    pid, _ = strategy.select_parent([sentinel, valid])
    assert pid == 1


def test_select_parent_only_invalid_falls_back():
    strategy = NsgaStrategy(pool_size=5, crossover_distance_threshold=65)
    sentinel = SearchNode(
        score=float("inf"),
        id=0,
        parent_id=0,
        state=ChainState(score=float("inf"), payload=None),
        metrics={"zip_complexity": 0.0, "node_complexity": 0.0},
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


def test_tournament_treats_every_objective_equally():
    """No objective is privileged. Visual error used to be gated ahead of the
    rest, which made it primary and left the others as tie-breakers; a
    complexity term cannot shape the front while that is true."""
    strategy = NsgaStrategy(pool_size=10, crossover_distance_threshold=65)
    nodes = [
        make_node(1, 0.1, zip_ratio=500.0),
        make_node(2, 0.9, zip_ratio=10.0),
    ]
    selected = {strategy.select_parent(nodes)[0] for _ in range(50)}
    assert selected == {1, 2}


def test_pool_size_limits_candidate_set():
    strategy = NsgaStrategy(pool_size=2, crossover_distance_threshold=65)
    nodes = [
        make_node(1, 0.1, zip_complexity=10.0),
        make_node(2, 0.2, zip_complexity=20.0),
        make_node(3, 0.8, zip_complexity=800.0),
        make_node(4, 0.9, zip_complexity=900.0),
        make_node(5, 1.0, zip_complexity=1000.0),
    ]
    selected = {strategy.select_parent(nodes)[0] for _ in range(50)}
    assert selected <= {1, 2}


def test_pool_size_one_always_returns_same_node():
    strategy = NsgaStrategy(pool_size=1, crossover_distance_threshold=65)
    nodes = [make_node(i, i * 0.1, zip_complexity=float(i * 100)) for i in range(1, 6)]
    selected = {strategy.select_parent(nodes)[0] for _ in range(20)}
    assert selected == {1}


def test_should_diversify_small_pool_needs_boost():
    strategy = NsgaStrategy(epoch_diversity=0.5)
    nodes = [make_node(i, 0.1, content="<svg><circle/></svg>") for i in range(1, 5)]
    triggered, diversity = strategy.should_diversify(nodes)
    assert triggered is True
    assert 0.0 <= diversity <= 1.0


def test_should_diversify_large_pool_needs_boost():
    strategy = NsgaStrategy(epoch_diversity=0.5)
    nodes = [make_node(i, 0.1, content="<svg><circle/></svg>") for i in range(1, 21)]
    triggered, diversity = strategy.should_diversify(nodes)
    assert triggered is True
    assert 0.0 <= diversity <= 1.0


def test_should_not_diversify_diverse_pool():
    strategy = NsgaStrategy(epoch_diversity=0.01)
    nodes = [
        make_node(
            i, 0.1, content=f"<svg><circle r='{i * 1000}' cx='{i}' cy='{i}'/></svg>"
        )
        for i in range(1, 5)
    ]
    triggered, diversity = strategy.should_diversify(nodes)
    assert triggered is False
    assert 0.0 <= diversity <= 1.0


def test_should_not_diversify_too_few_nodes():
    strategy = NsgaStrategy(epoch_diversity=0.99)
    nodes = [make_node(i, 0.1) for i in range(1, 4)]
    triggered, diversity = strategy.should_diversify(nodes)
    assert triggered is False
    assert diversity == 1.0


def test_epoch_parents_returns_pareto_front():
    strategy = NsgaStrategy(pool_size=10)
    nodes = [
        make_node(1, 0.1, zip_complexity=1000.0),  # good quality, complex
        make_node(
            2, 0.5, zip_complexity=100.0
        ),  # worse quality, simpler (dominates node 3)
        make_node(3, 0.9, zip_complexity=900.0),  # dominated by node 2
    ]
    seeds = strategy.epoch_parents(nodes, max_parents=2)
    seed_ids = {n.id for n in seeds}
    assert 1 in seed_ids
    assert 2 in seed_ids
    assert 3 not in seed_ids


def test_epoch_parents_respects_max_parents():
    strategy = NsgaStrategy(pool_size=10)
    nodes = [make_node(i, i * 0.1, zip_complexity=float(i * 100)) for i in range(1, 8)]
    seeds = strategy.epoch_parents(nodes, max_parents=3)
    assert len(seeds) == 3


def test_epoch_parents_filters_exact_duplicates():
    content = "<svg>" + "".join(str(i) for i in range(500)) + "</svg>"
    strategy = NsgaStrategy(pool_size=10)
    good = make_node(1, 0.1, zip_complexity=100.0, content=content)
    duplicate = make_node(2, 0.1, zip_complexity=100.0, content=content)  # exact copy
    different = make_node(
        3, 0.2, zip_complexity=200.0, content="<svg><completely different/></svg>"
    )
    seeds = strategy.epoch_parents([good, duplicate, different], max_parents=3)
    seed_ids = {n.id for n in seeds}
    assert not (1 in seed_ids and 2 in seed_ids)


def test_epoch_parents_empty_pool_returns_empty():
    strategy = NsgaStrategy(pool_size=10)
    seeds = strategy.epoch_parents([], max_parents=5)
    assert seeds == []


def test_epoch_parents_sorted_by_visual_score():
    strategy = NsgaStrategy(pool_size=10)
    nodes = [
        make_node(1, 0.1, zip_complexity=800.0),
        make_node(2, 0.3, zip_complexity=600.0),
        make_node(3, 0.5, zip_complexity=400.0),
        make_node(4, 0.7, zip_complexity=200.0),
    ]
    seeds = strategy.epoch_parents(nodes, max_parents=4)
    scores = [n.score for n in seeds]
    assert scores == sorted(scores)
    assert seeds[0].id == 1


def test_epoch_parents_all_invalid_falls_back():
    strategy = NsgaStrategy(pool_size=10)
    nodes = [make_node(i, float("inf")) for i in range(1, 4)]
    seeds = strategy.epoch_parents(nodes, max_parents=5)
    assert len(seeds) == 3


def test_accurate_and_simple_candidates_share_the_front():
    """Neither dominates: one wins on error, the other on complexity. Under the
    old feasibility gate the accurate one was promoted outright."""
    n1 = make_node(1, score=0.1, zip_complexity=5000.0)
    n2 = make_node(2, score=0.9, zip_complexity=10.0)
    objectives = {1: (0.1, 1.0), 2: (0.9, 0.0)}
    fronts = non_dominated_sort([n1, n2], objectives)
    assert len(fronts) == 1
    assert {n.id for n in fronts[0]} == {1, 2}


def test_non_dominated_sort_no_threshold_simple_dominates_complex():
    n1 = make_node(1, score=0.1, zip_complexity=5000.0)
    n2 = make_node(2, score=0.9, zip_complexity=10.0)
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


def test_larger_tournament_biases_harder_toward_score():
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
                    _random.random() * 5000,
                    content=f"n{i}-{_random.random()}",
                    node_complexity=_random.random() * 5000,
                )
                for i in range(20)
            ]
            median = sorted(n.score for n in nodes)[10]
            pid, _secondary = strategy.select_parent(nodes)
            if next(n for n in nodes if n.id == pid).score <= median:
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
