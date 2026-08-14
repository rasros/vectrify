import io
from unittest.mock import MagicMock

from PIL import Image

from vectrify.formats.models import VectorStatePayload
from vectrify.score.compare import Reference, prepare
from vectrify.search import INVALID_SCORE, ChainState, SearchNode
from vectrify.vector.resume import (
    PreppedNode,
    filter_to_pool_size,
    prefilter_nodes,
    resume_nodes,
)


def _make_png(color: str = "red", size: int = 16) -> bytes:
    img = Image.new("RGB", (size, size), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_node(
    node_id: int,
    score: float = 0.5,
    edge: float = 100.0,
    content: str = "<svg/>",
    colour: float = 0.0,
) -> SearchNode:
    return SearchNode(
        score=score,
        id=node_id,
        parent_id=0,
        metrics={
            "edge": edge,
            "colour": colour,
        },
        state=ChainState(
            score=score,
            payload=VectorStatePayload(
                content=content,
                raster_data_url=None,
                raster_preview_data_url=None,
                origin=None,
            ),
        ),
    )


def _make_prepped(
    old_id: int = 1,
    edge: float = 100.0,
    png: bytes | None = None,
    colour: float = 0.0,
) -> PreppedNode:
    return PreppedNode(
        old_id=old_id,
        content=f"<svg id='{old_id}'/>",
        png=png or _make_png(),
        preview_data_url="data:image/png;base64,PREVIEW",
        metrics={
            "edge": edge,
            "colour": colour,
        },
        signature=None,
    )


def test_prefilter_returns_all_when_under_limit():
    nodes = [_make_prepped(i) for i in range(3)]
    ref_img = Image.new("RGB", (16, 16), color="white")
    result = prefilter_nodes(nodes, ref_img, max_keep=10)
    assert len(result) == 3


def test_prefilter_caps_at_max_keep():
    nodes = [_make_prepped(i, edge=float(i * 10)) for i in range(20)]
    ref_img = Image.new("RGB", (16, 16), color="white")
    result = prefilter_nodes(nodes, ref_img, max_keep=5)
    assert len(result) <= 5


def test_prefilter_empty_input():
    ref_img = Image.new("RGB", (16, 16), color="white")
    result = prefilter_nodes([], ref_img, max_keep=10)
    assert result == []


def test_prefilter_returns_original_items():
    nodes = [_make_prepped(i) for i in range(3)]
    ref_img = Image.new("RGB", (16, 16), color="white")
    result = prefilter_nodes(nodes, ref_img, max_keep=10)
    for item in result:
        assert item in nodes


def test_filter_no_op_when_within_pool():
    nodes = [_make_node(i, score=float(i) * 0.1) for i in range(3)]
    result = filter_to_pool_size(nodes, pool_size=5)
    assert result == nodes


def test_filter_nsga_returns_pool_size():
    nodes = [
        _make_node(i, score=float(i) * 0.1, edge=float(i) * 50)
        for i in range(10)
    ]
    result = filter_to_pool_size(nodes, pool_size=4)
    assert len(result) == 4


def test_filter_nsga_prefers_pareto_front():
    best = _make_node(1, score=0.1, edge=10.0)  # dominates all others
    worse = [_make_node(i + 2, score=0.9, edge=900.0) for i in range(9)]
    result = filter_to_pool_size([best, *worse], pool_size=3)
    assert best in result


def test_filter_handles_invalid_scores():
    nodes = [
        _make_node(1, score=INVALID_SCORE),
        _make_node(2, score=0.3),
        _make_node(3, score=0.5),
    ]
    result = filter_to_pool_size(nodes, pool_size=2)
    assert len(result) == 2
    assert all(n.score < INVALID_SCORE for n in result)


def _make_mock_plugin(png: bytes | None = None) -> MagicMock:
    plugin = MagicMock()
    plugin.rasterize.return_value = png or _make_png()
    return plugin


class _FakeEmbeddingScorer:
    """Stands in for the encoder: resume only needs a number back."""

    def score(self, reference, candidate_png: bytes) -> float:
        _ = reference, candidate_png
        return 0.25


def _make_reference() -> Reference:
    """A real reference: resume reduces one comparison into the score and the
    region metrics, so there is nothing left to mock usefully."""
    return prepare(Image.new("RGB", (16, 16), color="blue"))


def _make_mock_storage() -> MagicMock:
    storage = MagicMock()
    storage.save_node = MagicMock()
    return storage


def test_resume_nodes_returns_one_node_per_item():
    plugin = _make_mock_plugin()
    ref = _make_reference()
    storage = _make_mock_storage()
    ref_img = Image.new("RGB", (32, 32), color="blue")

    items = [(1, "<svg id='1'/>"), (2, "<svg id='2'/>")]
    result = resume_nodes(
        resumed_items=items,
        format_plugin=plugin,
        original_img=ref_img,
        original_w=32,
        original_h=32,
        resolution_llm=16,
        pool_size=10,
        workers=1,
        scoring_ref=ref,
        scorer=_FakeEmbeddingScorer(),
        embedding_ref=object(),
        storage=storage,
    )

    assert len(result) == 2
    assert storage.save_node.call_count == 2


def test_resume_nodes_assigns_sequential_ids():
    plugin = _make_mock_plugin()
    ref = _make_reference()
    storage = _make_mock_storage()
    ref_img = Image.new("RGB", (32, 32))

    items = [(10, "<svg id='A'/>"), (20, "<svg id='B'/>"), (30, "<svg id='C'/>")]
    result = resume_nodes(
        resumed_items=items,
        format_plugin=plugin,
        original_img=ref_img,
        original_w=32,
        original_h=32,
        resolution_llm=16,
        pool_size=10,
        workers=1,
        scoring_ref=ref,
        scorer=_FakeEmbeddingScorer(),
        embedding_ref=object(),
        storage=storage,
    )

    ids = sorted(n.id for n in result)
    assert ids == [1, 2, 3]


def test_resume_nodes_deduplicates_identical_content():
    plugin = _make_mock_plugin()
    ref = _make_reference()
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
        resolution_llm=16,
        pool_size=10,
        workers=1,
        scoring_ref=ref,
        scorer=_FakeEmbeddingScorer(),
        embedding_ref=object(),
        storage=storage,
    )

    assert len(result) == 1


def test_resume_nodes_stores_origin_with_old_id():
    plugin = _make_mock_plugin()
    ref = _make_reference()
    storage = _make_mock_storage()
    ref_img = Image.new("RGB", (32, 32))

    items = [(99, "<svg id='x'/>")]
    result = resume_nodes(
        resumed_items=items,
        format_plugin=plugin,
        original_img=ref_img,
        original_w=32,
        original_h=32,
        resolution_llm=16,
        pool_size=10,
        workers=1,
        scoring_ref=ref,
        scorer=_FakeEmbeddingScorer(),
        embedding_ref=object(),
        storage=storage,
    )

    assert result[0].state.payload.origin == "Imported from Node 99"


def test_resume_nodes_skips_failed_scoring(monkeypatch):
    """A resumed node that cannot be scored is dropped, not carried in with a
    missing score that would read as best-possible on every objective."""
    import vectrify.vector.resume as resume_module

    real_compare = resume_module.compare
    calls = {"n": 0}

    def flaky(reference, png):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("bad")
        return real_compare(reference, png)

    monkeypatch.setattr(resume_module, "compare", flaky)

    plugin = _make_mock_plugin()
    ref = _make_reference()
    storage = _make_mock_storage()
    ref_img = Image.new("RGB", (32, 32))

    items = [(1, "<svg id='fail'/>"), (2, "<svg id='ok'/>")]
    result = resume_nodes(
        resumed_items=items,
        format_plugin=plugin,
        original_img=ref_img,
        original_w=32,
        original_h=32,
        resolution_llm=16,
        pool_size=10,
        workers=1,
        scoring_ref=ref,
        scorer=_FakeEmbeddingScorer(),
        embedding_ref=object(),
        storage=storage,
    )

    assert len(result) == 1
    assert result[0].score < INVALID_SCORE


def test_resume_nodes_triggers_prefilter_when_many_items(monkeypatch):
    import vectrify.vector.resume as resume_module

    pool_size = 3
    n_items = 7

    prefilter_sizes: list[int] = []
    real_prefilter = resume_module.prefilter_nodes

    def spy_prefilter(prepped, original_img, max_keep):
        prefilter_sizes.append(len(prepped))
        return real_prefilter(prepped, original_img, max_keep)

    monkeypatch.setattr(resume_module, "prefilter_nodes", spy_prefilter)
    plugin = _make_mock_plugin()
    ref = _make_reference()
    storage = _make_mock_storage()
    ref_img = Image.new("RGB", (32, 32))

    items = [(i, f"<svg id='{i}'/>") for i in range(1, n_items + 1)]
    result = resume_nodes(
        resumed_items=items,
        format_plugin=plugin,
        original_img=ref_img,
        original_w=32,
        original_h=32,
        resolution_llm=16,
        pool_size=pool_size,
        workers=2,
        scoring_ref=ref,
        scorer=_FakeEmbeddingScorer(),
        embedding_ref=object(),
        storage=storage,
    )

    # 7 items > 2 * pool_size, so prefiltering must run over all 7 items and
    # the surviving pool must not exceed the prefilter cap.
    assert prefilter_sizes == [n_items]
    assert len(result) <= 2 * pool_size
