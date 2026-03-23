import pytest

from svgizer.search import ChainState, GreedyHillClimbingStrategy, Result, SearchNode


@pytest.fixture
def strategy():
    return GreedyHillClimbingStrategy(patience=2, temp_step=0.5, max_temp=2.0)


def test_select_parent_always_picks_best(strategy):
    dummy_state = ChainState(
        score=0.0, model_temperature=0.0, stale_hits=0, payload=None
    )
    nodes = [
        SearchNode(score=0.8, id=1, parent_id=0, state=dummy_state),
        SearchNode(score=0.2, id=2, parent_id=0, state=dummy_state),
        SearchNode(score=0.5, id=3, parent_id=0, state=dummy_state),
    ]

    selected_id, secondary = strategy.select_parent(nodes, progress=0.5)

    assert selected_id == 2
    assert secondary is None


def test_create_new_state_cools_on_improvement(strategy):
    parent_state = ChainState(
        score=0.5,
        model_temperature=0.5,
        stale_hits=1,
        payload="old",
    )

    res = Result(
        task_id=1,
        parent_id=1,
        worker_slot=0,
        valid=True,
        score=0.4,
        used_temperature=0.5,
        payload="new",
    )

    new_state = strategy.create_new_state(parent_state, res)

    assert new_state.stale_hits == 0
    assert new_state.model_temperature == pytest.approx(0.45)


def test_create_new_state_bumps_temp_on_patience_reached(strategy):
    parent_state = ChainState(
        score=0.5,
        model_temperature=0.5,
        stale_hits=1,
        payload="old",
    )

    res = Result(
        task_id=1,
        parent_id=1,
        worker_slot=0,
        valid=True,
        score=0.6,
        used_temperature=0.5,
        payload="new",
    )

    new_state = strategy.create_new_state(parent_state, res)

    assert new_state.stale_hits == 0
    assert new_state.model_temperature == pytest.approx(1.0)


def test_create_new_state_respects_max_temp(strategy):
    parent_state = ChainState(
        score=0.5,
        model_temperature=1.8,
        stale_hits=1,
        payload="old",
    )

    res = Result(
        task_id=1,
        parent_id=1,
        worker_slot=0,
        valid=True,
        score=0.5,
        used_temperature=1.8,
        payload="new",
    )

    new_state = strategy.create_new_state(parent_state, res)

    assert new_state.model_temperature == pytest.approx(2.0)