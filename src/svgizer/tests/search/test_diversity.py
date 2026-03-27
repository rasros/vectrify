import pytest

from svgizer.search import ChainState, SearchNode
from svgizer.search.diversity import hamming_distance, pool_diversity, simhash


def make_node(node_id: int, score: float, sig: int | None = None) -> SearchNode:
    return SearchNode(
        score=score,
        id=node_id,
        parent_id=0,
        state=ChainState(score=score, payload=None),
        signature=sig,
    )


# ── simhash ────────────────────────────────────────────────────────────────────


def test_simhash_none_returns_none():
    assert simhash(None) is None


def test_simhash_empty_returns_none():
    assert simhash("") is None


def test_simhash_returns_int():
    assert isinstance(simhash("<svg/>"), int)


def test_simhash_deterministic():
    text = "<svg><rect width='100'/></svg>"
    assert simhash(text) == simhash(text)


def test_simhash_identical_texts_same_hash():
    text = "<svg><circle r='50'/></svg>"
    assert simhash(text) == simhash(text)


def test_simhash_different_texts_differ():
    assert simhash("<svg><rect/></svg>") != simhash("<svg><circle/></svg>")


def test_simhash_short_text_below_ngram_size():
    assert isinstance(simhash("ab"), int)


# ── hamming_distance ───────────────────────────────────────────────────────────


def test_hamming_distance_identical():
    assert hamming_distance(0b1010, 0b1010) == 0


def test_hamming_distance_single_bit():
    assert hamming_distance(0b1010, 0b1011) == 1


def test_hamming_distance_all_bits():
    assert hamming_distance(0, (1 << 64) - 1) == 64


def test_hamming_distance_symmetric():
    a, b = 0xDEADBEEF, 0xCAFEBABE
    assert hamming_distance(a, b) == hamming_distance(b, a)


# ── pool_diversity ─────────────────────────────────────────────────────────────


def test_pool_diversity_all_identical_returns_low():
    h = simhash("<svg><rect/></svg>")
    nodes = [make_node(i, 0.1, sig=h) for i in range(5)]
    assert pool_diversity(nodes) == pytest.approx(0.0)


def test_pool_diversity_all_unique_returns_high():
    # Use very different texts so SimHash distances are large
    texts = [
        "<svg><rect width='100'/></svg>",
        "<svg><circle r='50' cx='200' cy='300'/></svg>",
        "<svg><polygon points='0,0 100,0 50,86'/></svg>",
        "<svg><text x='10' y='20'>Hello world</text></svg>",
        "<svg><path d='M10 10 L90 90 Z' stroke='red'/></svg>",
    ]
    nodes = [make_node(i + 1, 0.1, sig=simhash(t)) for i, t in enumerate(texts)]
    diversity = pool_diversity(nodes)
    assert diversity > 0.1  # These differ in structure, so Hamming distance > 0


def test_pool_diversity_ignores_none_signatures():
    nodes = [make_node(i, 0.1, sig=None) for i in range(5)]
    assert pool_diversity(nodes) == 1.0


def test_pool_diversity_ignores_inf_score():
    h = simhash("<svg/>")
    nodes = [make_node(i, float("inf"), sig=h) for i in range(5)]
    assert pool_diversity(nodes) == 1.0


def test_pool_diversity_too_few_nodes_returns_one():
    assert pool_diversity([]) == 1.0
    assert pool_diversity([make_node(1, 0.1, sig=simhash("<svg/>"))]) == 1.0


def test_pool_diversity_between_zero_and_one():
    nodes = [
        make_node(i, 0.1, sig=simhash(f"<svg><rect id='{i}'/></svg>"))
        for i in range(10)
    ]
    diversity = pool_diversity(nodes)
    assert 0.0 <= diversity <= 1.0
