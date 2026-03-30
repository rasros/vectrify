import io

from PIL import Image

from svgizer.score.simple import SimpleFallbackScorer


def _make_png(color: str = "red", size: int = 32) -> bytes:
    img = Image.new("RGB", (size, size), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_base_diff_heatmap_returns_none_when_reference_has_no_image():
    scorer = SimpleFallbackScorer()
    result = scorer.diff_heatmap(object(), _make_png(), long_side=32)
    assert result is None


def test_base_diff_heatmap_returns_valid_png():
    scorer = SimpleFallbackScorer()
    ref = scorer.prepare_reference(Image.new("RGB", (32, 32), color="red"))
    result = scorer.diff_heatmap(ref, _make_png("blue"), long_side=32)
    assert result is not None
    img = Image.open(io.BytesIO(result))
    assert img.mode == "RGB"


def test_base_diff_heatmap_identical_images_are_black():
    scorer = SimpleFallbackScorer()
    ref = scorer.prepare_reference(Image.new("RGB", (32, 32), color="green"))
    result = scorer.diff_heatmap(ref, _make_png("green"), long_side=32)
    assert result is not None
    img = Image.open(io.BytesIO(result)).convert("RGB")
    assert all(p == (0, 0, 0) for p in img.get_flattened_data())


def test_base_diff_heatmap_respects_long_side():
    scorer = SimpleFallbackScorer()
    ref = scorer.prepare_reference(Image.new("RGB", (128, 128), color="red"))
    result = scorer.diff_heatmap(ref, _make_png("blue", size=128), long_side=32)
    assert result is not None
    img = Image.open(io.BytesIO(result))
    assert max(img.size) <= 32
