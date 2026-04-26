import io

from PIL import Image

from vectrify.score.complexity import complexity, content_complexity, visual_complexity


def _make_png(color: str | tuple, size: int = 64) -> bytes:
    img = Image.new("RGB", (size, size), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


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


def test_returns_float():
    assert isinstance(visual_complexity(_make_png("red")), float)


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


# ── content_complexity ────────────────────────────────────────────────────────

_SIMPLE_SVG = '<svg><rect fill="red" width="100" height="100"/></svg>'

_COMPLEX_SVG = """
<svg>
  <defs>
    <linearGradient id="g">
      <stop offset="0%" stop-color="#ff0000"/>
      <stop offset="100%" stop-color="#0000ff"/>
    </linearGradient>
    <filter id="f"><feGaussianBlur stdDeviation="2"/></filter>
  </defs>
  <path fill="#123456" d="M0 0 L100 0 C50 50 50 50 100 100 Z"/>
  <path fill="#abcdef" d="M10 10 L90 10 L90 90 L10 90 Z"/>
  <circle fill="green" cx="50" cy="50" r="40"/>
  <rect fill="url(#g)" x="0" y="0" width="50" height="50" filter="url(#f)"/>
</svg>
"""


def test_content_complexity_returns_float():
    assert isinstance(content_complexity(_SIMPLE_SVG), float)


def test_content_complexity_positive():
    assert content_complexity(_SIMPLE_SVG) > 0.0


def test_content_complexity_empty_svg():
    assert content_complexity("<svg></svg>") == 0.0


def test_content_complexity_complex_greater_than_simple():
    assert content_complexity(_COMPLEX_SVG) > content_complexity(_SIMPLE_SVG)


def test_content_complexity_more_paths_higher():
    few = "<svg>" + '<path d="M0 0 L10 10"/>' * 3 + "</svg>"
    many = "<svg>" + '<path d="M0 0 L10 10"/>' * 20 + "</svg>"
    assert content_complexity(many) > content_complexity(few)


def test_content_complexity_path_commands_contribute():
    short_path = '<svg><path fill="red" d="M0 0 L10 10"/></svg>'
    long_path = (
        '<svg><path fill="red"'
        ' d="M0 0 C10 10 20 20 30 30 C40 40 50 50 60 60 L70 70 Z"/></svg>'
    )
    assert content_complexity(long_path) > content_complexity(short_path)


def test_content_complexity_gradients_and_filters_contribute():
    no_extras = '<svg><path fill="red" d="M0 0 L10 10"/></svg>'
    with_extras = _COMPLEX_SVG
    assert content_complexity(with_extras) > content_complexity(no_extras)


# ── complexity (blended) ──────────────────────────────────────────────────────


def test_blended_complexity_returns_float():
    png = _make_png("red")
    assert isinstance(complexity(png, _SIMPLE_SVG), float)


def test_blended_complexity_positive():
    png = _make_png("white")
    assert complexity(png, _SIMPLE_SVG) > 0.0


def test_blended_complexity_weighted_combination():
    from vectrify.score.complexity import _VISUAL_WEIGHT

    png = _make_png("red")
    v = visual_complexity(png)
    c = content_complexity(_SIMPLE_SVG)
    expected = _VISUAL_WEIGHT * v + (1.0 - _VISUAL_WEIGHT) * c
    assert complexity(png, _SIMPLE_SVG) == expected


def test_blended_complexity_higher_for_complex_svg():
    png = _make_png("red")
    simple = complexity(png, _SIMPLE_SVG)
    complex_ = complexity(png, _COMPLEX_SVG)
    assert complex_ > simple
