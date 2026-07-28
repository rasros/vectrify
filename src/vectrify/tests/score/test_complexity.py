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
