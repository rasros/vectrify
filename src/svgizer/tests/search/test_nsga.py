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


def make_node(
    node_id: int,
    score: float,
    complexity: float = 100.0,
    content: str | None = None,
) -> SearchNode:
    state = ChainState(score=score, payload=None)
    return SearchNode(
        score=score,
        id=node_id,
        parent_id=0,
        state=state,
        complexity=complexity,
        content=content,
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


# ---------------------------------------------------------------------------
# Diversity filtering
# ---------------------------------------------------------------------------


def test_diversity_admits_distinct_nodes():
    """All nodes with different content should fill the pool normally."""
    strategy = NsgaStrategy(pool_size=3, crossover_prob=0.0, diversity_threshold=0.97)
    nodes = [
        make_node(1, 0.1, content="<svg><circle/></svg>"),
        make_node(2, 0.2, content="<svg><rect/></svg>"),
        make_node(3, 0.3, content="<svg><line/></svg>"),
    ]
    # All three are distinct — any can be selected
    for _ in range(10):
        pid, _ = strategy.select_parent(nodes, 0.0)
        assert pid in {1, 2, 3}


def test_diversity_rejects_near_duplicate_with_worse_score():
    """A node nearly identical to a better node must be excluded from the pool."""
    base = "<svg>" + "x" * 500 + "</svg>"
    near_dup = "<svg>" + "x" * 499 + "y</svg>"  # one char differs — very similar

    strategy = NsgaStrategy(pool_size=5, crossover_prob=0.0, diversity_threshold=0.97)
    good = make_node(1, 0.1, content=base)
    near = make_node(2, 0.9, content=near_dup)  # worse score
    different = make_node(3, 0.5, content="<svg><completely different/></svg>")

    # Run many times; node 2 (near-dup of better node 1) should never be selected
    selected = set()
    for _ in range(50):
        pid, _ = strategy.select_parent([good, near, different], 0.0)
        selected.add(pid)
    assert 2 not in selected, "near-duplicate with worse score leaked into pool"


def test_diversity_admits_node_with_no_content():
    """Nodes without content are always admitted (can't evaluate similarity)."""
    strategy = NsgaStrategy(pool_size=3, crossover_prob=0.0, diversity_threshold=0.97)
    nodes = [
        make_node(1, 0.1, content=None),
        make_node(2, 0.2, content=None),
    ]
    for _ in range(10):
        pid, _ = strategy.select_parent(nodes, 0.0)
        assert pid in {1, 2}


def test_diversity_disabled_at_threshold_one():
    """Setting diversity_threshold=1.0 disables filtering entirely."""
    base = "<svg>" + "x" * 500 + "</svg>"
    strategy = NsgaStrategy(pool_size=5, crossover_prob=0.0, diversity_threshold=1.0)
    nodes = [make_node(i, i * 0.1, content=base) for i in range(1, 4)]
    # All identical content but filtering is disabled — no crash, any id valid
    for _ in range(10):
        pid, _ = strategy.select_parent(nodes, 0.0)
        assert pid in {1, 2, 3}


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


# ---------------------------------------------------------------------------
# Binary tournament
# ---------------------------------------------------------------------------


def test_tournament_prefers_lower_rank():
    """Node on front 0 must consistently beat node on front 1 in tournament."""
    strategy = NsgaStrategy(pool_size=10, crossover_prob=0.0)
    # Node 1: better score (front 0), Node 2: worse score (front 1)
    nodes = [make_node(1, 0.1, complexity=100.0), make_node(2, 0.9, complexity=100.0)]
    selected = {strategy.select_parent(nodes, 0.0)[0] for _ in range(30)}
    # Node 1 dominates node 2 — should always win the tournament
    assert 1 in selected
    assert 2 not in selected


def test_tournament_single_pool_candidate_returns_it():
    """When only one candidate exists in the pool the tournament still returns it."""
    strategy = NsgaStrategy(pool_size=10, crossover_prob=0.0)
    nodes = [make_node(1, 0.3, complexity=200.0)]
    pid, secondary = strategy.select_parent(nodes, 0.0)
    assert pid == 1
    assert secondary is None


def test_tournament_two_equal_rank_nodes_both_selectable():
    """Two Pareto-equivalent nodes (incomparable) should both be selectable."""
    strategy = NsgaStrategy(pool_size=10, crossover_prob=0.0)
    # Incomparable: node 1 has better score, node 2 has lower complexity
    nodes = [make_node(1, 0.1, complexity=500.0), make_node(2, 0.9, complexity=10.0)]
    selected = {strategy.select_parent(nodes, 0.0)[0] for _ in range(50)}
    assert 1 in selected
    assert 2 in selected


# ---------------------------------------------------------------------------
# Pool size boundary
# ---------------------------------------------------------------------------


def test_pool_size_limits_candidate_set():
    """With pool_size=2 and 5 nodes, only the top 2 (by rank) can be selected."""
    strategy = NsgaStrategy(pool_size=2, crossover_prob=0.0)
    # Nodes 1 and 2 dominate the rest (clearly better score AND complexity)
    nodes = [
        make_node(1, 0.1, complexity=10.0),
        make_node(2, 0.2, complexity=20.0),
        make_node(3, 0.8, complexity=800.0),
        make_node(4, 0.9, complexity=900.0),
        make_node(5, 1.0, complexity=1000.0),
    ]
    selected = {strategy.select_parent(nodes, 0.0)[0] for _ in range(50)}
    assert selected <= {1, 2}, f"Nodes outside pool were selected: {selected - {1, 2}}"


def test_pool_size_one_always_returns_same_node():
    """pool_size=1 means the single best node is always chosen."""
    strategy = NsgaStrategy(pool_size=1, crossover_prob=0.0)
    nodes = [make_node(i, i * 0.1, complexity=float(i * 100)) for i in range(1, 6)]
    selected = {strategy.select_parent(nodes, 0.0)[0] for _ in range(20)}
    assert selected == {1}


# ---------------------------------------------------------------------------
# should_diversify tests
# ---------------------------------------------------------------------------


def test_should_diversify_small_pool_needs_boost():
    """A small homogenous pool triggers diversity boost exhaustively."""
    strategy = NsgaStrategy(diversity_boost_threshold=0.5)
    nodes = [make_node(i, 0.1, content="<svg><circle/></svg>") for i in range(1, 5)]
    assert strategy.should_diversify(nodes) is True


def test_should_diversify_large_pool_needs_boost():
    """A large homogenous pool triggers diversity boost via sampling."""
    strategy = NsgaStrategy(diversity_boost_threshold=0.5)
    nodes = [make_node(i, 0.1, content="<svg><circle/></svg>") for i in range(1, 21)]
    assert strategy.should_diversify(nodes) is True


def test_should_not_diversify_diverse_pool():
    """A highly diverse pool does not trigger the boost."""
    strategy = NsgaStrategy(diversity_boost_threshold=0.01)
    nodes = [
        make_node(i, 0.1, content=f"<svg><circle r='{i}'/></svg>") for i in range(1, 5)
    ]
    assert strategy.should_diversify(nodes) is False


def test_should_not_diversify_too_few_nodes():
    """A pool that is too small shouldn't be prematurely checked."""
    strategy = NsgaStrategy(diversity_boost_threshold=0.99)
    nodes = [make_node(i, 0.1, content="<svg><circle/></svg>") for i in range(1, 4)]
    # < 4 nodes returns False early
    assert strategy.should_diversify(nodes) is False
