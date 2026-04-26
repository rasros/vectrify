from vectrify.search.models import ChainState, SearchNode


def test_search_node_sorting_by_score():
    dummy_state = ChainState(score=0.0, payload=None)
    n_best = SearchNode(score=0.1, id=10, parent_id=0, state=dummy_state)
    n_mid = SearchNode(score=0.5, id=5, parent_id=0, state=dummy_state)
    n_worst = SearchNode(score=0.9, id=1, parent_id=0, state=dummy_state)

    nodes = [n_mid, n_worst, n_best]
    nodes.sort()

    assert nodes[0].id == 10
    assert nodes[1].id == 5
    assert nodes[2].id == 1


def test_search_node_comparison_ignores_metadata():
    dummy_state = ChainState(score=0.0, payload=None)
    n1 = SearchNode(score=0.2, id=99, parent_id=99, state=dummy_state)
    n2 = SearchNode(score=0.8, id=1, parent_id=1, state=dummy_state)

    assert n1 < n2


def test_search_node_equality_with_identical_scores():
    dummy_state = ChainState(score=0.0, payload=None)
    n1 = SearchNode(score=0.5, id=1, parent_id=0, state=dummy_state)
    n2 = SearchNode(score=0.5, id=2, parent_id=0, state=dummy_state)

    assert not (n1 < n2)
    assert not (n2 < n1)
