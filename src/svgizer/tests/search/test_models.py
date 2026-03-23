from svgizer.search.base import StrategyType
from svgizer.search.models import INVALID_SCORE, SearchNode


def test_search_node_sorting_by_score():
    n_best = SearchNode(score=0.1, id=10, parent_id=0, state=None)
    n_mid = SearchNode(score=0.5, id=5, parent_id=0, state=None)
    n_worst = SearchNode(score=0.9, id=1, parent_id=0, state=None)

    nodes = [n_mid, n_worst, n_best]
    nodes.sort()

    assert nodes[0].id == 10
    assert nodes[1].id == 5
    assert nodes[2].id == 1


def test_search_node_comparison_ignores_metadata():
    n1 = SearchNode(score=0.2, id=99, parent_id=99, state=None)
    n2 = SearchNode(score=0.8, id=1, parent_id=1, state=None)

    assert n1 < n2


def test_search_node_equality_with_identical_scores():
    n1 = SearchNode(score=0.5, id=1, parent_id=0, state=None)
    n2 = SearchNode(score=0.5, id=2, parent_id=0, state=None)

    assert not (n1 < n2)
    assert not (n2 < n1)


def test_strategy_type_enum():
    assert StrategyType.GENETIC == "genetic"
    assert StrategyType.GREEDY == "greedy"
    assert "genetic" in [e.value for e in StrategyType]


def test_invalid_score_constant():
    assert INVALID_SCORE > 1.0
    assert INVALID_SCORE > 0.0
