import io

from PIL import Image

from vectrify.score.complexity import structural_complexity, visual_complexity
from vectrify.tests.helpers import make_png as _make_png


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


def test_returns_positive():
    assert visual_complexity(_make_png("white")) > 0.0


def test_flat_image_lower_than_noisy():
    flat = _make_png("blue", size=64)
    noisy = _noise_png(size=64)
    assert visual_complexity(flat) < visual_complexity(noisy)


def test_larger_image_higher_than_smaller():
    small = _make_png("red", size=32)
    large = _make_png("red", size=128)
    assert visual_complexity(large) > visual_complexity(small)


def test_different_flat_colors_similar_complexity():
    red = _make_png("red", size=64)
    blue = _make_png("blue", size=64)
    ratio = visual_complexity(red) / visual_complexity(blue)
    assert 0.5 < ratio < 2.0


# ── structural_complexity ─────────────────────────────────────────────────────

_SIMPLE_SVG = '<svg><rect fill="red" width="100" height="100"/></svg>'

_DOT = 'digraph G {\n  node [shape=box, fillcolor="lightblue"];\n  a -> b;\n}'

_TYPST = "#set page(width: 200pt)\n#place(circle(radius: 40pt, fill: blue))\n"


def test_structural_complexity_positive():
    assert structural_complexity(_SIMPLE_SVG) > 0.0


def test_structural_complexity_empty_source_is_zero():
    assert structural_complexity("") == 0.0
    assert structural_complexity("   \n\t  ") == 0.0


def test_structural_complexity_is_nonzero_for_every_format():
    """Regression: the old SVG-regex measure scored DOT and Typst as exactly
    0.0, silently removing the structural objective for two of three backends.
    """
    for source in (_SIMPLE_SVG, _DOT, _TYPST):
        assert structural_complexity(source) > 0.0


def test_structural_complexity_grows_with_element_count():
    few = "<svg>" + '<path d="M0 0 L10 10"/>' * 3 + "</svg>"
    many = "<svg>" + '<path d="M0 0 L10 10"/>' * 20 + "</svg>"
    assert structural_complexity(many) > structural_complexity(few)


def test_structural_complexity_ignores_indentation():
    """Pretty-printing varies with whatever the model emitted and carries no
    complexity information, so it must not shift the objective."""
    minified = '<svg><rect fill="red"/><circle r="5"/></svg>'
    pretty = '<svg>\n    <rect fill="red"/>\n    <circle r="5"/>\n</svg>\n'
    assert structural_complexity(minified) == structural_complexity(pretty)


def test_structural_complexity_does_not_discount_repetition():
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
    ratio = structural_complexity(f"<svg>{identical}</svg>") / structural_complexity(
        f"<svg>{diverse}</svg>"
    )
    assert ratio > 0.9, f"repetition discounted by {(1 - ratio) * 100:.0f}%"


# ── metric registry ───────────────────────────────────────────────────────────


def test_registry_covers_the_declared_metrics():
    from vectrify.score.complexity import METRIC_NAMES, METRICS

    assert tuple(METRICS) == METRIC_NAMES
    assert set(METRIC_NAMES) == {"visual_complexity", "structural_complexity"}


def test_measure_all_evaluates_every_registered_metric():
    from vectrify.score.complexity import METRIC_NAMES, measure_all

    metrics = measure_all(_make_png("red", size=32), _SIMPLE_SVG)
    assert set(metrics) == set(METRIC_NAMES)
    assert all(v > 0.0 for v in metrics.values())


def test_lineage_columns_derive_from_the_registry():
    """A registered metric must become a lineage.csv column with no edit there."""
    from vectrify.score.complexity import METRIC_NAMES
    from vectrify.vector.storage import LINEAGE_COLUMNS

    for name in METRIC_NAMES:
        assert name in LINEAGE_COLUMNS


def test_objective_arity_follows_the_registry():
    """The objective vector is score plus one entry per registered metric."""
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
    assert all(len(v) == len(METRIC_NAMES) + 1 for v in objectives.values())


def test_read_metrics_round_trips_a_written_row():
    from vectrify.score.complexity import METRIC_NAMES, read_metrics

    row = dict.fromkeys(METRIC_NAMES, "250")
    assert read_metrics(row) == dict.fromkeys(METRIC_NAMES, 250.0)


def test_read_metrics_defaults_absent_columns_to_zero():
    """A row written before a metric was registered must stay readable."""
    from vectrify.score.complexity import METRIC_NAMES, read_metrics

    metrics = read_metrics({"visual_complexity": "100"})
    assert set(metrics) == set(METRIC_NAMES)
    assert metrics["visual_complexity"] == 100.0
    assert metrics["structural_complexity"] == 0.0


def test_read_metrics_maps_the_legacy_blended_column():
    from vectrify.score.complexity import read_metrics

    metrics = read_metrics({"complexity": "1500"})
    assert metrics["visual_complexity"] == 1500.0
    assert metrics["structural_complexity"] == 0.0


def test_row_has_metrics_rejects_sparse_eviction_rows():
    """An eviction row sets only id and evicted; reading it as metrics would
    overwrite the node's real values with zeros."""
    from vectrify.score.complexity import row_has_metrics

    assert row_has_metrics({"visual_complexity": "100"}) is True
    assert row_has_metrics({"complexity": "1500"}) is True
    assert row_has_metrics({"id": "7", "evicted": "42"}) is False
    assert (
        row_has_metrics({"visual_complexity": "", "structural_complexity": ""}) is False
    )
