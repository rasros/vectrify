import io
from unittest.mock import MagicMock

import pytest
from PIL import Image

from svgizer.formats.models import VectorStatePayload
from svgizer.search import INVALID_SCORE, ChainState, SearchNode, StrategyType
from svgizer.vector.resume import filter_to_pool_size, prefilter_nodes, resume_nodes

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_png(color: str = "red", size: int = 16) -> bytes:
    img = Image.new("RGB", (size, size), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_node(
    node_id: int,
    score: float = 0.5,
    complexity: float = 100.0,
    content: str = "<svg/>",
) -> SearchNode:
    return SearchNode(
        score=score,
        id=node_id,
        parent_id=0,
        complexity=complexity,
        state=ChainState(
            score=score,
            payload=VectorStatePayload(
                content=content,
                raster_data_url=None,
                raster_preview_data_url=None,
                origin=None,
                invalid_msg=None,
            ),
        ),
    )


def _make_prepped(
    old_id: int = 1,
    complexity: float = 100.0,
    png: bytes | None = None,
) -> tuple:
    """Build a prepped-node tuple as produced during resume rasterization."""
    return (
        old_id,
        f"<svg id='{old_id}'/>",
        png or _make_png(),
        "data:image/png;base64,PREVIEW",
        complexity,
        None,  # signature
    )


# ── prefilter_nodes ────────────────────────────────────────────────────────────


def test_prefilter_returns_all_when_under_limit():
    nodes = [_make_prepped(i) for i in range(3)]
    ref_img = Image.new("RGB", (16, 16), color="white")
    result = prefilter_nodes(nodes, ref_img, max_keep=10)
    assert len(result) == 3


def test_prefilter_caps_at_max_keep():
    nodes = [_make_prepped(i, complexity=float(i * 10)) for i in range(20)]
    ref_img = Image.new("RGB", (16, 16), color="white")
    result = prefilter_nodes(nodes, ref_img, max_keep=5)
    assert len(result) <= 5


def test_prefilter_empty_input():
    ref_img = Image.new("RGB", (16, 16), color="white")
    result = prefilter_nodes([], ref_img, max_keep=10)
    assert result == []


def test_prefilter_returns_original_items():
    """Returned items must be the same tuples, not copies."""
    nodes = [_make_prepped(i) for i in range(3)]
    ref_img = Image.new("RGB", (16, 16), color="white")
    result = prefilter_nodes(nodes, ref_img, max_keep=10)
    for item in result:
        assert item in nodes


# ── filter_to_pool_size ────────────────────────────────────────────────────────


def test_filter_no_op_when_within_pool():
    nodes = [_make_node(i, score=float(i) * 0.1) for i in range(3)]
    result = filter_to_pool_size(nodes, pool_size=5, strategy_type=StrategyType.NSGA)
    assert result == nodes


def test_filter_greedy_sorts_by_score():
    nodes = [_make_node(i, score=1.0 - i * 0.1) for i in range(5)]
    result = filter_to_pool_size(nodes, pool_size=3, strategy_type=StrategyType.GREEDY)
    assert len(result) == 3
    scores = [n.score for n in result]
    assert scores == sorted(scores)


def test_filter_nsga_returns_pool_size():
    nodes = [
        _make_node(i, score=float(i) * 0.1, complexity=float(i) * 50) for i in range(10)
    ]
    result = filter_to_pool_size(nodes, pool_size=4, strategy_type=StrategyType.NSGA)
    assert len(result) == 4


def test_filter_nsga_prefers_pareto_front():
    """A node that dominates on both objectives should survive."""
    best = _make_node(1, score=0.1, complexity=10.0)  # dominates all others
    worse = [_make_node(i + 2, score=0.9, complexity=900.0) for i in range(9)]
    result = filter_to_pool_size(
        [best, *worse], pool_size=3, strategy_type=StrategyType.NSGA
    )
    assert best in result


def test_filter_handles_invalid_scores():
    nodes = [
        _make_node(1, score=INVALID_SCORE),
        _make_node(2, score=0.3),
        _make_node(3, score=0.5),
    ]
    result = filter_to_pool_size(nodes, pool_size=2, strategy_type=StrategyType.GREEDY)
    assert len(result) == 2
    # INVALID_SCORE sorts last, so the two valid nodes should be kept
    assert all(n.score < INVALID_SCORE for n in result)


# ── resume_nodes ───────────────────────────────────────────────────────────────


def _make_mock_plugin(png: bytes | None = None) -> MagicMock:
    plugin = MagicMock()
    plugin.rasterize.return_value = png or _make_png()
    return plugin


def _make_mock_scorer(score: float = 0.4) -> tuple[MagicMock, MagicMock]:
    scorer = MagicMock()
    ref = MagicMock()
    scorer.score.return_value = score
    return scorer, ref


def _make_mock_storage() -> MagicMock:
    storage = MagicMock()
    storage.save_node = MagicMock()
    return storage


def test_resume_nodes_returns_one_node_per_item():
    plugin = _make_mock_plugin()
    scorer, ref = _make_mock_scorer(0.3)
    storage = _make_mock_storage()
    ref_img = Image.new("RGB", (32, 32), color="blue")

    items = [(1, "<svg id='1'/>"), (2, "<svg id='2'/>")]
    result = resume_nodes(
        resumed_items=items,
        format_plugin=plugin,
        original_img=ref_img,
        original_w=32,
        original_h=32,
        image_long_side=16,
        pool_size=10,
        workers=1,
        scorer=scorer,
        scoring_ref=ref,
        storage=storage,
    )

    assert len(result) == 2
    assert storage.save_node.call_count == 2


def test_resume_nodes_assigns_sequential_ids():
    plugin = _make_mock_plugin()
    scorer, ref = _make_mock_scorer()
    storage = _make_mock_storage()
    ref_img = Image.new("RGB", (32, 32))

    items = [(10, "<svg id='A'/>"), (20, "<svg id='B'/>"), (30, "<svg id='C'/>")]
    result = resume_nodes(
        resumed_items=items,
        format_plugin=plugin,
        original_img=ref_img,
        original_w=32,
        original_h=32,
        image_long_side=16,
        pool_size=10,
        workers=1,
        scorer=scorer,
        scoring_ref=ref,
        storage=storage,
    )

    ids = sorted(n.id for n in result)
    assert ids == [1, 2, 3]


def test_resume_nodes_deduplicates_identical_content():
    plugin = _make_mock_plugin()
    scorer, ref = _make_mock_scorer()
    storage = _make_mock_storage()
    ref_img = Image.new("RGB", (32, 32))

    same = "<svg><rect width='10' height='10'/></svg>"
    items = [(1, same), (2, same)]  # same content → same simhash
    result = resume_nodes(
        resumed_items=items,
        format_plugin=plugin,
        original_img=ref_img,
        original_w=32,
        original_h=32,
        image_long_side=16,
        pool_size=10,
        workers=1,
        scorer=scorer,
        scoring_ref=ref,
        storage=storage,
    )

    assert len(result) == 1


def test_resume_nodes_stores_origin_with_old_id():
    plugin = _make_mock_plugin()
    scorer, ref = _make_mock_scorer()
    storage = _make_mock_storage()
    ref_img = Image.new("RGB", (32, 32))

    items = [(99, "<svg id='x'/>")]
    result = resume_nodes(
        resumed_items=items,
        format_plugin=plugin,
        original_img=ref_img,
        original_w=32,
        original_h=32,
        image_long_side=16,
        pool_size=10,
        workers=1,
        scorer=scorer,
        scoring_ref=ref,
        storage=storage,
    )

    assert result[0].state.payload.origin == "Imported from Node 99"


def test_resume_nodes_skips_failed_scoring():
    plugin = _make_mock_plugin()
    scorer = MagicMock()
    scorer.score.side_effect = [ValueError("bad"), 0.5]
    ref = MagicMock()
    storage = _make_mock_storage()
    ref_img = Image.new("RGB", (32, 32))

    items = [(1, "<svg id='fail'/>"), (2, "<svg id='ok'/>")]
    result = resume_nodes(
        resumed_items=items,
        format_plugin=plugin,
        original_img=ref_img,
        original_w=32,
        original_h=32,
        image_long_side=16,
        pool_size=10,
        workers=1,
        scorer=scorer,
        scoring_ref=ref,
        storage=storage,
    )

    assert len(result) == 1
    assert result[0].score == pytest.approx(0.5)


def test_resume_nodes_triggers_prefilter_when_many_items():
    """When resumed items exceed 2*pool_size, prefilter should reduce them."""
    pool_size = 3
    # 7 items > 2*3=6 → triggers prefilter
    n_items = 7
    plugin = _make_mock_plugin()
    scorer, ref = _make_mock_scorer(0.2)
    storage = _make_mock_storage()
    ref_img = Image.new("RGB", (32, 32))

    items = [(i, f"<svg id='{i}'/>") for i in range(1, n_items + 1)]
    result = resume_nodes(
        resumed_items=items,
        format_plugin=plugin,
        original_img=ref_img,
        original_w=32,
        original_h=32,
        image_long_side=16,
        pool_size=pool_size,
        workers=2,
        scorer=scorer,
        scoring_ref=ref,
        storage=storage,
    )

    assert len(result) <= 2 * pool_size
