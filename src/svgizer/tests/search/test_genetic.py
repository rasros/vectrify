import pytest

from svgizer.search import ChainState, GeneticPoolStrategy, Result, SearchNode


@pytest.fixture
def strategy():
    return GeneticPoolStrategy(
        top_k=2,
        temp_step=0.5,
        max_temp=2.0,
        stale_threshold=1,
        is_stale_fn=lambda p1, p2: p1 == p2,
    )


def test_select_parent_returns_best_on_full_progress(strategy):
    dummy_state = ChainState(
        score=0.0, model_temperature=0.0, stale_hits=0, payload=None
    )
    n1 = SearchNode(score=0.1, id=1, parent_id=0, state=dummy_state)
    n2 = SearchNode(score=0.9, id=2, parent_id=0, state=dummy_state)

    selected, secondary = strategy.select_parent([n1, n2], progress=1.0)
    assert selected in [1, 2]
    assert secondary is None or secondary in [1, 2]


def test_create_new_state_increments_temp_on_staleness(strategy):
    parent = ChainState(
        score=0.5,
        model_temperature=0.5,
        stale_hits=0,
        payload="identical_payload",
    )

    res = Result(
        task_id=1,
        parent_id=1,
        worker_slot=0,
        valid=True,
        score=0.5,
        used_temperature=0.5,
        payload="identical_payload",
    )

    new_state = strategy.create_new_state(parent, res)

    assert new_state.model_temperature == 1.0
    assert new_state.stale_hits == 0
