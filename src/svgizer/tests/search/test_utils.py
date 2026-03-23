import pytest

from svgizer.search.models import ChainState, SearchNode
from svgizer.search.utils import calculate_elite_prob, choose_from_top_k_weighted


def test_calculate_elite_prob_interpolation():
    assert calculate_elite_prob(0.0, 0.7, 0.1) == pytest.approx(0.7)
    assert calculate_elite_prob(1.0, 0.7, 0.1) == pytest.approx(0.1)
    assert calculate_elite_prob(0.5, 0.7, 0.1) == pytest.approx(0.4)


def test_calculate_elite_prob_clamping():
    assert calculate_elite_prob(-1.0, 0.8, 0.2) == pytest.approx(0.8)
    assert calculate_elite_prob(5.0, 0.8, 0.2) == pytest.approx(0.2)


def test_choose_from_top_k_weighted_distribution():
    dummy_state = ChainState(
        score=0.0, model_temperature=0.0, stale_hits=0, payload=None
    )
    nodes = [
        SearchNode(score=0.1, id=10, parent_id=0, state=dummy_state),
        SearchNode(score=0.2, id=20, parent_id=0, state=dummy_state),
        SearchNode(score=0.3, id=30, parent_id=0, state=dummy_state),
    ]

    seen_ids = set()
    for _ in range(100):
        seen_ids.add(choose_from_top_k_weighted(nodes))

    assert {10, 20, 30}.issubset(seen_ids)


def test_choose_from_top_k_weighted_single_node():
    dummy_state = ChainState(
        score=0.0, model_temperature=0.0, stale_hits=0, payload=None
    )
    nodes = [SearchNode(score=0.1, id=42, parent_id=0, state=dummy_state)]
    assert choose_from_top_k_weighted(nodes) == 42


def test_choose_from_top_k_weighted_empty():
    assert choose_from_top_k_weighted([]) == 0
