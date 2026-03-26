import pytest

from svgizer.search import ChainState, GreedyHillClimbingStrategy, Result, SearchNode


@pytest.fixture
def strategy():
    return GreedyHillClimbingStrategy()


def test_select_parent_always_picks_best(strategy):
    dummy_state = ChainState(score=0.0, payload=None)
    nodes = [
        SearchNode(score=0.8, id=1, parent_id=0, state=dummy_state),
        SearchNode(score=0.2, id=2, parent_id=0, state=dummy_state),
        SearchNode(score=0.5, id=3, parent_id=0, state=dummy_state),
    ]
    selected_id, _ = strategy.select_parent(nodes, progress=0.5)
    assert selected_id == 2


def test_select_parent_empty_list_returns_zero(strategy):
    selected_id, secondary = strategy.select_parent([], progress=0.0)
    assert selected_id == 0
    assert secondary is None


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
