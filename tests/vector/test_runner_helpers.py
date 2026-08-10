"""Seed budgeting: how many LLM calls a run opens an epoch with."""

from vectrify.formats.models import VectorStatePayload
from vectrify.search.models import ChainState, SearchNode
from vectrify.vector.runner import initial_seed_tasks, resolve_seeds


def _node(content: str | None) -> SearchNode:
    payload = VectorStatePayload(
        content=content,
        raster_data_url=None,
        raster_preview_data_url=None,
        origin=None,
    )
    return SearchNode(
        score=0.5, id=1, parent_id=0, state=ChainState(score=0.5, payload=payload)
    )


def test_seeds_default_to_a_tenth_of_the_pool():
    assert resolve_seeds(None, 40) == 4


def test_an_explicit_seed_count_wins_over_the_pool_size():
    assert resolve_seeds(3, 40) == 3


def test_zero_seeds_is_allowed():
    assert resolve_seeds(0, 40) == 0


def test_a_negative_seed_count_is_floored_at_zero():
    assert resolve_seeds(-5, 40) == 0


def test_a_small_pool_still_derives_a_seed_count():
    assert resolve_seeds(None, 5) == 0


def test_resumed_nodes_pay_for_themselves():
    assert initial_seed_tasks(4, [_node("<svg/>"), _node("<svg/>")]) == 2


def test_a_full_resumed_batch_spends_nothing():
    assert initial_seed_tasks(2, [_node("<svg/>"), _node("<svg/>")]) == 0


def test_more_resumed_nodes_than_seeds_never_goes_negative():
    assert initial_seed_tasks(1, [_node("<svg/>") for _ in range(5)]) == 0


def test_contentless_nodes_do_not_count_as_seeds():
    assert initial_seed_tasks(3, [_node(None), _node("")]) == 3


def test_no_initial_nodes_means_the_full_batch():
    assert initial_seed_tasks(3, []) == 3
