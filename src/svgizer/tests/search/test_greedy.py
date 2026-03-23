import pytest

from svgizer.models import ChainState, Result, SearchNode
from svgizer.search.greedy import GreedyHillClimbingStrategy


@pytest.fixture
def strategy():
    return GreedyHillClimbingStrategy(patience=2, temp_step=0.5, max_temp=2.0)


def test_select_parent_always_picks_best(strategy):
    # Node 2 has the lowest (best) score
    nodes = [
        SearchNode(score=0.8, id=1, parent_id=0, state=None),
        SearchNode(score=0.2, id=2, parent_id=0, state=None),
        SearchNode(score=0.5, id=3, parent_id=0, state=None),
    ]

    selected_id, secondary = strategy.select_parent(nodes, progress=0.5)

    assert selected_id == 2
    assert secondary is None


def test_create_new_state_resets_staleness_on_improvement(strategy):
    parent_state = ChainState(
        svg="<svg/>",
        model_temperature=0.5,
        stale_hits=1,
        score=0.5,
        raster_data_url=None,
        raster_preview_data_url=None,
        invalid_msg=None,
    )

    # The result score (0.4) is better than the parent score (0.5)
    res = Result(
        task_id=1,
        parent_id=1,
        worker_slot=0,
        svg="<svg/>",
        valid=True,
        invalid_msg=None,
        raster_png=b"",
        score=0.4,
        used_temperature=0.5,
        change_summary=None,
    )

    new_state = strategy.create_new_state(parent_state, res)

    assert new_state.stale_hits == 0
    assert new_state.model_temperature == 0.5  # Temp stays the same


def test_create_new_state_bumps_temp_on_patience_reached(strategy):
    parent_state = ChainState(
        svg="<svg/>",
        model_temperature=0.5,
        stale_hits=1,
        score=0.5,
        raster_data_url=None,
        raster_preview_data_url=None,
        invalid_msg=None,
    )

    # The result score (0.6) is WORSE than the parent score (0.5)
    res = Result(
        task_id=1,
        parent_id=1,
        worker_slot=0,
        svg="<svg/>",
        valid=True,
        invalid_msg=None,
        raster_png=b"",
        score=0.6,
        used_temperature=0.5,
        change_summary=None,
    )

    new_state = strategy.create_new_state(parent_state, res)

    # stale_hits was 1, result didn't improve, so hits threshold of 2.
    assert new_state.stale_hits == 0  # Resets after bump
    assert new_state.model_temperature == 1.0  # 0.5 base + 0.5 step


def test_create_new_state_respects_max_temp(strategy):
    parent_state = ChainState(
        svg="<svg/>",
        model_temperature=1.8,
        stale_hits=1,
        score=0.5,
        raster_data_url=None,
        raster_preview_data_url=None,
        invalid_msg=None,
    )

    res = Result(
        task_id=1,
        parent_id=1,
        worker_slot=0,
        svg="<svg/>",
        valid=True,
        invalid_msg=None,
        raster_png=b"",
        score=0.5,
        used_temperature=1.8,
        change_summary=None,
    )

    new_state = strategy.create_new_state(parent_state, res)

    # 1.8 + 0.5 = 2.3, which is above max_temp (2.0)
    assert new_state.model_temperature == 2.0
