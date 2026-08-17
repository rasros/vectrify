import pytest

from vectrify.search.models import ChainState, SearchNode


def test_a_node_records_whether_it_was_measured():
    """`valid` says the candidate's markup parsed, it rasterized and the
    measures computed -- nothing about how good it is."""
    node = SearchNode(valid=True, id=10, parent_id=0, state=ChainState(payload=None))

    assert node.valid is True
    assert node.metrics == {}


def test_nodes_are_not_orderable():
    """They used to be, by a float called `score` holding one of two sentinel
    values. Anything that sorted by it was sorting by a constant, and three
    separate defects came from exactly that. Candidates are ordered by
    dominance over their measures, which needs a population, so a single node
    has no position of its own.
    """
    a = SearchNode(valid=True, id=1, parent_id=0, state=ChainState(payload=None))
    b = SearchNode(valid=True, id=2, parent_id=0, state=ChainState(payload=None))

    with pytest.raises(TypeError):
        _ = a < b  # type: ignore[unsupported-operation]

    with pytest.raises(TypeError):
        sorted([a, b])  # type: ignore[bad-specialization]


def test_an_unmeasured_node_says_so():
    node = SearchNode(valid=False, id=3, parent_id=0, state=ChainState(payload=None))

    assert node.valid is False
