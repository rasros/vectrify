import pytest

from vectrify.search import ChainState, SearchNode
from vectrify.search.diversity import hamming_distance, pool_diversity, simhash


def make_node(node_id: int, sig: int | None = None, valid: bool = True) -> SearchNode:
    return SearchNode(
        valid=valid,
        id=node_id,
        parent_id=0,
        state=ChainState(payload=None),
        signature=sig,
    )


def test_simhash_none_returns_none():
    assert simhash(None) is None


def test_simhash_empty_returns_none():
    assert simhash("") is None


def test_simhash_deterministic():
    text = "<svg><rect width='100'/></svg>"
    assert simhash(text) == simhash(text)


def test_simhash_different_texts_differ():
    assert simhash("<svg><rect/></svg>") != simhash("<svg><circle/></svg>")


def test_simhash_short_text_below_ngram_size():
    assert isinstance(simhash("ab"), int)


def test_hamming_distance_identical():
    assert hamming_distance(0b1010, 0b1010) == 0


def test_hamming_distance_single_bit():
    assert hamming_distance(0b1010, 0b1011) == 1


def test_hamming_distance_all_bits():
    assert hamming_distance(0, (1 << 64) - 1) == 64


def test_pool_diversity_all_identical_returns_low():
    h = simhash("<svg><rect/></svg>")
    nodes = [make_node(i, sig=h) for i in range(5)]
    assert pool_diversity(nodes) == pytest.approx(0.0)


def test_pool_diversity_all_unique_returns_high():
    texts = [
        "<svg><rect width='100'/></svg>",
        "<svg><circle r='50' cx='200' cy='300'/></svg>",
        "<svg><polygon points='0,0 100,0 50,86'/></svg>",
        "<svg><text x='10' y='20'>Hello world</text></svg>",
        "<svg><path d='M10 10 L90 90 Z' stroke='red'/></svg>",
    ]
    nodes = [make_node(i + 1, sig=simhash(t)) for i, t in enumerate(texts)]
    diversity = pool_diversity(nodes)
    assert diversity > 0.1


def test_pool_diversity_ignores_none_signatures():
    nodes = [make_node(i, sig=None) for i in range(5)]
    assert pool_diversity(nodes) == 1.0


def test_pool_diversity_ignores_unmeasured_nodes():
    """A node that never rasterized has no drawing to be diverse from."""
    h = simhash("<svg/>")
    nodes = [make_node(i, sig=h, valid=False) for i in range(5)]
    assert pool_diversity(nodes) == 1.0


def test_pool_diversity_too_few_nodes_returns_one():
    assert pool_diversity([]) == 1.0
    assert pool_diversity([make_node(1, sig=simhash("<svg/>"))]) == 1.0
