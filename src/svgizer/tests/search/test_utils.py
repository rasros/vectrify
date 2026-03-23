import pytest

from svgizer.search.models import SearchNode
from svgizer.search.utils import calculate_elite_prob, choose_from_top_k_weighted


def test_calculate_elite_prob_interpolation():
    assert calculate_elite_prob(0.0, 0.7, 0.1) == pytest.approx(0.7)
    assert calculate_elite_prob(1.0, 0.7, 0.1) == pytest.approx(0.1)
    assert calculate_elite_prob(0.5, 0.7, 0.1) == pytest.approx(0.4)


def test_calculate_elite_prob_clamping():
    assert calculate_elite_prob(-1.0, 0.8, 0.2) == pytest.approx(0.8)
    assert calculate_elite_prob(5.0, 0.8, 0.2) == pytest.approx(0.2)


def test_choose_from_top_k_weighted_distribution():
    nodes = [
        SearchNode(score=0.1, id=10, parent_id=0, state=None),
        SearchNode(score=0.2, id=20, parent_id=0, state=None),
        SearchNode(score=0.3, id=30, parent_id=0, state=None),
    ]

    seen_ids = set()
    for _ in range(100):
        seen_ids.add(choose_from_top_k_weighted(nodes))

    assert {10, 20, 30}.issubset(seen_ids)


def test_choose_from_top_k_weighted_single_node():
    nodes = [SearchNode(score=0.1, id=42, parent_id=0, state=None)]
    assert choose_from_top_k_weighted(nodes) == 42


def test_choose_from_top_k_weighted_empty():
    assert choose_from_top_k_weighted([]) == 0
