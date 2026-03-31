import pytest

from svgizer.search import ChainState, GreedyHillClimbingStrategy, Result, SearchNode


@pytest.fixture
def strategy():
    return GreedyHillClimbingStrategy(beams=4, cull_keep=1.0)


def test_select_parent_picks_randomly_across_beams(strategy):
    """All beams should be selected, not just the best one."""
    dummy_state = ChainState(score=0.0, payload=None)
    nodes = [
        SearchNode(score=0.8, id=1, parent_id=0, state=dummy_state),
        SearchNode(score=0.2, id=2, parent_id=0, state=dummy_state),
        SearchNode(score=0.5, id=3, parent_id=0, state=dummy_state),
    ]
    selected_ids = {strategy.select_parent(nodes, progress=0.5)[0] for _ in range(100)}
    # All three beams should be selected at some point
    assert selected_ids == {1, 2, 3}


def test_select_parent_never_returns_secondary():
    strategy = GreedyHillClimbingStrategy(beams=2)
    dummy_state = ChainState(score=0.0, payload=None)
    nodes = [
        SearchNode(score=0.8, id=1, parent_id=0, state=dummy_state),
        SearchNode(score=0.2, id=2, parent_id=0, state=dummy_state),
    ]
    for _ in range(20):
        _, secondary = strategy.select_parent(nodes, progress=0.5)
        assert secondary is None


def test_select_parent_empty_list_returns_zero():
    strategy = GreedyHillClimbingStrategy(beams=1)
    selected_id, secondary = strategy.select_parent([], progress=0.0)
    assert selected_id == 0
    assert secondary is None


def test_top_k_count_matches_beams():
    assert GreedyHillClimbingStrategy(beams=6).top_k_count == 6


def test_cull_keep_restricts_selection_to_top_beams():
    """With cull_keep=0.5 and 4 beams only the top 2 should ever be selected."""
    strategy = GreedyHillClimbingStrategy(beams=4, cull_keep=0.5)
    dummy_state = ChainState(score=0.0, payload=None)
    nodes = [
        SearchNode(score=0.1, id=1, parent_id=0, state=dummy_state),
        SearchNode(score=0.3, id=2, parent_id=0, state=dummy_state),
        SearchNode(score=0.7, id=3, parent_id=0, state=dummy_state),
        SearchNode(score=0.9, id=4, parent_id=0, state=dummy_state),
    ]
    selected_ids = {strategy.select_parent(nodes, progress=0.0)[0] for _ in range(100)}
    assert selected_ids <= {1, 2}
    assert 3 not in selected_ids
    assert 4 not in selected_ids


def test_should_diversify_always_false(strategy):
    dummy_state = ChainState(score=0.0, payload=None)
    nodes = [
        SearchNode(score=0.3, id=i, parent_id=0, state=dummy_state) for i in range(5)
    ]
    triggered, diversity = strategy.should_diversify(nodes)
    assert triggered is False
    assert diversity == 0.0


def test_epoch_seeds_returns_empty_for_full_restart(strategy):
    dummy_state = ChainState(score=0.0, payload=None)
    nodes = [
        SearchNode(score=0.3, id=i, parent_id=0, state=dummy_state) for i in range(4)
    ]
    assert strategy.epoch_seeds(nodes, max_seeds=4) == []


def test_create_new_state_applies_score(strategy):
    res = Result(
        task_id=1,
        parent_id=1,
        worker_slot=0,
        valid=True,
        score=0.4,
        payload="new",
    )
    new_state = strategy.create_new_state(res)
    assert new_state.score == 0.4
    assert new_state.payload == "new"
