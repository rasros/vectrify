import io

from PIL import Image

from svgizer.score.complexity import visual_complexity


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
