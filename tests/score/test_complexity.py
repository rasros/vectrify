import io

from PIL import Image

from tests.helpers import make_png as _make_png
from vectrify.score.complexity import node_complexity, zip_complexity


def _noise_png(size: int = 64) -> bytes:
    import random

    img = Image.new("RGB", (size, size))
    img.putdata(
        [
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for _ in range(size * size)
        ]
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_flat_image_lower_than_noisy():
    flat = _make_png("blue", size=64)
    noisy = _noise_png(size=64)
    assert zip_complexity(flat) < zip_complexity(noisy)


def test_larger_image_higher_than_smaller():
    small = _make_png("red", size=32)
    large = _make_png("red", size=128)
    assert zip_complexity(large) > zip_complexity(small)


def test_different_flat_colors_similar_complexity():
    red = _make_png("red", size=64)
    blue = _make_png("blue", size=64)
    ratio = zip_complexity(red) / zip_complexity(blue)
    assert 0.5 < ratio < 2.0


# ── node_complexity ─────────────────────────────────────────────────────

_SIMPLE_SVG = '<svg><rect fill="red" width="100" height="100"/></svg>'

_DOT = 'digraph G {\n  node [shape=box, fillcolor="lightblue"];\n  a -> b;\n}'

_TYPST = "#set page(width: 200pt)\n#place(circle(radius: 40pt, fill: blue))\n"


def test_node_complexity_empty_source_is_zero():
    assert node_complexity("") == 0.0
    assert node_complexity("   \n\t  ") == 0.0


def test_node_complexity_is_nonzero_for_every_format():
    """Regression: the old SVG-regex measure scored DOT and Typst as exactly
    0.0, silently removing the structural objective for two of three backends.
    """
    for source in (_SIMPLE_SVG, _DOT, _TYPST):
        assert node_complexity(source) > 0.0


def test_node_complexity_grows_with_element_count():
    few = "<svg>" + '<path d="M0 0 L10 10"/>' * 3 + "</svg>"
    many = "<svg>" + '<path d="M0 0 L10 10"/>' * 20 + "</svg>"
    assert node_complexity(many) > node_complexity(few)


def test_node_complexity_ignores_indentation():
    minified = '<svg><rect fill="red"/><circle r="5"/></svg>'
    pretty = '<svg>\n    <rect fill="red"/>\n    <circle r="5"/>\n</svg>\n'
    assert node_complexity(minified) == node_complexity(pretty)


def test_node_complexity_does_not_discount_repetition():
    """Why this is source length and not gzip: every crossover operator injects
    elements from a related parent, so near-duplicate elements accumulate. A
    compressed measure discounts that by ~80% and would leave the bloat this
    objective exists to charge for effectively free.
    """
    n = 60
    diverse = "".join(
        f'<circle cx="{i * 3}" cy="{i * 7 % 251}" r="{i % 13 + 2}"/>' for i in range(n)
    )
    identical = '<circle cx="40" cy="40" r="9"/>' * n
    ratio = node_complexity(f"<svg>{identical}</svg>") / node_complexity(
        f"<svg>{diverse}</svg>"
    )
    assert ratio > 0.9, f"repetition discounted by {(1 - ratio) * 100:.0f}%"


# ── metric registry ───────────────────────────────────────────────────────────


def test_registry_covers_the_declared_metrics():
    from vectrify.score.complexity import METRIC_NAMES, METRICS, SCORER_METRICS

    assert tuple(METRICS) + SCORER_METRICS == METRIC_NAMES
    assert set(METRIC_NAMES) == {
        "zip_complexity",
        "node_complexity",
        "worst_region_4",
        "worst_region_16",
        "zip_ratio",
        "node_ratio",
    }


def test_worker_metrics_precede_scorer_metrics():
    from vectrify.score.complexity import METRIC_NAMES, METRICS

    assert METRIC_NAMES[: len(METRICS)] == tuple(METRICS)


def test_measure_all_evaluates_every_worker_metric():
    from vectrify.score.complexity import METRICS, measure_all

    metrics = measure_all(_make_png("red", size=32), _SIMPLE_SVG)
    assert set(metrics) == set(METRICS)
    assert all(v > 0.0 for v in metrics.values())


def test_measure_all_omits_scorer_metrics():
    from vectrify.score.complexity import SCORER_METRICS, measure_all

    metrics = measure_all(_make_png("red", size=32), _SIMPLE_SVG)
    assert not set(metrics) & set(SCORER_METRICS)


def test_lineage_columns_derive_from_the_registry():
    from vectrify.score.complexity import METRIC_NAMES
    from vectrify.vector.storage import LINEAGE_COLUMNS

    for name in METRIC_NAMES:
        assert name in LINEAGE_COLUMNS


def test_objective_arity_follows_the_registry():
    from vectrify.score.complexity import METRIC_NAMES
    from vectrify.search import ChainState, SearchNode
    from vectrify.search.nsga import build_objectives

    nodes = [
        SearchNode(
            score=0.1 * i,
            id=i,
            parent_id=0,
            state=ChainState(score=0.1 * i, payload=None),
            metrics=dict.fromkeys(METRIC_NAMES, float(i)),
        )
        for i in (1, 2)
    ]
    objectives = build_objectives(nodes)
    from vectrify.score.complexity import OBJECTIVE_NAMES

    assert all(len(v) == len(OBJECTIVE_NAMES) + 1 for v in objectives.values())


def test_read_metrics_round_trips_a_written_row():
    from vectrify.score.complexity import METRIC_NAMES, read_metrics

    row = dict.fromkeys(METRIC_NAMES, "250")
    assert read_metrics(row) == dict.fromkeys(METRIC_NAMES, 250.0)


def test_read_metrics_defaults_absent_columns_to_zero():
    from vectrify.score.complexity import METRIC_NAMES, read_metrics

    metrics = read_metrics({"zip_complexity": "100"})
    assert set(metrics) == set(METRIC_NAMES)
    assert metrics["zip_complexity"] == 100.0
    assert metrics["node_complexity"] == 0.0


def test_row_has_metrics_rejects_sparse_eviction_rows():
    from vectrify.score.complexity import row_has_metrics

    assert row_has_metrics({"zip_complexity": "100"}) is True
    assert row_has_metrics({"id": "7", "evicted": "42"}) is False
    assert row_has_metrics({"zip_complexity": "", "node_complexity": ""}) is False
