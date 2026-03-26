from svgizer.search import ChainState, Result, SearchNode
from svgizer.search.nsga import (
    NsgaStrategy,
    _dominates,
    crowding_distance,
    non_dominated_sort,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_node(node_id: int, score: float, complexity: float = 100.0) -> SearchNode:
    state = ChainState(score=score, payload=None)
    return SearchNode(
        score=score, id=node_id, parent_id=0, state=state, complexity=complexity
    )


# ---------------------------------------------------------------------------
# _dominates
# ---------------------------------------------------------------------------


def test_dominates_strictly_better():
    assert _dominates((0.1, 0.2), (0.3, 0.4))


def test_dominates_equal_is_not_dominance():
    assert not _dominates((0.3, 0.4), (0.3, 0.4))


def test_dominates_better_in_one_equal_in_other():
    assert _dominates((0.1, 0.4), (0.3, 0.4))
    assert _dominates((0.3, 0.2), (0.3, 0.4))


def test_dominates_incomparable():
    assert not _dominates((0.1, 0.5), (0.3, 0.2))
    assert not _dominates((0.3, 0.2), (0.1, 0.5))


# ---------------------------------------------------------------------------
# non_dominated_sort
# ---------------------------------------------------------------------------


def test_non_dominated_sort_all_pareto():
    nodes = [make_node(1, 0.1), make_node(2, 0.5), make_node(3, 0.9)]
    objectives = {1: (0.1, 0.9), 2: (0.5, 0.5), 3: (0.9, 0.1)}
    fronts = non_dominated_sort(nodes, objectives)
    assert len(fronts) == 1
    assert {n.id for n in fronts[0]} == {1, 2, 3}


def test_non_dominated_sort_chain():
    nodes = [make_node(i, float(i)) for i in range(1, 4)]
    objectives = {1: (0.1, 0.1), 2: (0.5, 0.5), 3: (0.9, 0.9)}
    fronts = non_dominated_sort(nodes, objectives)
    assert len(fronts) == 3
    assert fronts[0][0].id == 1
    assert fronts[1][0].id == 2
    assert fronts[2][0].id == 3


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


# ---------------------------------------------------------------------------
# crowding_distance
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# NsgaStrategy
# ---------------------------------------------------------------------------


def test_select_parent_returns_valid_node_id():
    strategy = NsgaStrategy(pool_size=5, crossover_prob=0.0)
    nodes = [make_node(i, i * 0.1, i * 100.0) for i in range(1, 6)]
    pid, secondary = strategy.select_parent(nodes, progress=0.5)
    assert pid in {n.id for n in nodes}
    assert secondary is None


def test_select_parent_crossover_returns_two_distinct_parents():
    strategy = NsgaStrategy(pool_size=5, crossover_prob=1.0)
    nodes = [make_node(i, i * 0.1, i * 100.0) for i in range(1, 6)]
    results = set()
    for _ in range(20):
        pid, secondary = strategy.select_parent(nodes, progress=0.5)
        if secondary is not None:
            results.add((pid, secondary))
    assert all(len(pair) == 2 and pair[0] != pair[1] for pair in results)


def test_select_parent_skips_invalid_nodes():
    strategy = NsgaStrategy(pool_size=10, crossover_prob=0.0)
    sentinel = SearchNode(
        score=float("inf"),
        id=0,
        parent_id=0,
        state=ChainState(score=float("inf"), payload=None),
        complexity=0.0,
    )
    valid = make_node(1, 0.3, 200.0)
    pid, _ = strategy.select_parent([sentinel, valid], progress=0.0)
    assert pid == 1


def test_select_parent_only_invalid_falls_back():
    strategy = NsgaStrategy(pool_size=5, crossover_prob=0.0)
    sentinel = SearchNode(
        score=float("inf"),
        id=0,
        parent_id=0,
        state=ChainState(score=float("inf"), payload=None),
        complexity=0.0,
    )
    pid, secondary = strategy.select_parent([sentinel], progress=0.0)
    assert pid == 0
    assert secondary is None


def test_create_new_state_propagates_score_and_payload():
    strategy = NsgaStrategy()
    result = Result(
        task_id=1,
        parent_id=1,
        worker_slot=0,
        valid=True,
        score=0.42,
        payload="<svg/>",
        complexity=500.0,
    )
    state = strategy.create_new_state(result)
    assert state.score == 0.42
    assert state.payload == "<svg/>"


def test_pareto_front_prefers_simpler_for_equal_quality():
    """Node with same visual score but lower complexity must be on front 0."""
    n_simple = make_node(1, 0.3, complexity=100.0)
    n_complex = make_node(2, 0.3, complexity=5000.0)
    max_c = 5000.0
    objectives = {
        1: (0.3, 100.0 / max_c),
        2: (0.3, 1.0),
    }
    fronts = non_dominated_sort([n_simple, n_complex], objectives)
    assert fronts[0][0].id == 1
