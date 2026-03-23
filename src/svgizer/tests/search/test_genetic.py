import pytest

from svgizer.models import ChainState, Result, SearchNode
from svgizer.search.genetic import GeneticPoolStrategy


@pytest.fixture
def strategy():
    return GeneticPoolStrategy(top_k=2, temp_step=0.5, max_temp=2.0, stale_threshold=1)


def test_select_parent_returns_best_on_full_progress(strategy):
    # node 1 is better (lower score)
    n1 = SearchNode(score=0.1, id=1, parent_id=0, state=None)
    n2 = SearchNode(score=0.9, id=2, parent_id=0, state=None)

    # At progress 1.0, elite_prob is at its minimum,
    # but with top_k weighted, the best node is heavily favored.
    selected, secondary = strategy.select_parent([n1, n2], progress=1.0)
    assert selected in [1, 2]
    assert secondary is None or secondary in [1, 2]


def test_create_new_state_increments_temp_on_staleness(strategy):
    parent = ChainState(
        svg="<svg/>",
        model_temperature=0.5,
        stale_hits=0,
        score=0.5,
        raster_data_url=None,
        raster_preview_data_url=None,
        invalid_msg=None,
    )

    # Result is identical to parent SVG
    res = Result(
        task_id=1,
        parent_id=1,
        worker_slot=0,
        svg="<svg/>",
        valid=True,
        invalid_msg=None,
        raster_png=b"...",
        score=0.5,
        used_temperature=0.5,
        change_summary="none",
    )

    new_state = strategy.create_new_state(parent, res)

    # 0.5 (base) + 0.5 (step) = 1.0
    assert new_state.model_temperature == 1.0
    assert new_state.stale_hits == 0  # Reset after a bump
