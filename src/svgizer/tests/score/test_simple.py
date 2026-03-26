import io

from PIL import Image

from svgizer.score.simple import SimpleFallbackScorer


def test_simple_fallback_scorer_identical():
    scorer = SimpleFallbackScorer(target_long_side=64)

    img_red = Image.new("RGB", (100, 100), color="red")

    ref = scorer.prepare_reference(img_red)
    buf_red = io.BytesIO()
    img_red.save(buf_red, format="PNG")
    cand_red_bytes = buf_red.getvalue()

    score_identical = scorer.score(ref, cand_red_bytes)

    assert score_identical == 0.0


def test_simple_fallback_scorer_different():
    scorer = SimpleFallbackScorer(target_long_side=64)

    img_red = Image.new("RGB", (100, 100), color="red")
    ref = scorer.prepare_reference(img_red)

    img_blue = Image.new("RGB", (100, 100), color="blue")
    buf_blue = io.BytesIO()
    img_blue.save(buf_blue, format="PNG")
    cand_blue_bytes = buf_blue.getvalue()

    score_diff = scorer.score(ref, cand_blue_bytes)

    assert score_diff > 0.0
    assert score_diff <= 1.0


def test_simple_fallback_scorer_handles_size_mismatch():
    scorer = SimpleFallbackScorer(target_long_side=64)

    img_ref = Image.new("RGB", (200, 200), color="green")
    ref = scorer.prepare_reference(img_ref)

    img_cand = Image.new("RGB", (50, 80), color="green")
    buf_cand = io.BytesIO()
    img_cand.save(buf_cand, format="PNG")
    cand_bytes = buf_cand.getvalue()

    score = scorer.score(ref, cand_bytes)

    assert score == 0.0


def test_simple_fallback_scorer_invalid_data_returns_max_diff():
    scorer = SimpleFallbackScorer(target_long_side=64)
    img_red = Image.new("RGB", (10, 10), color="red")
    ref = scorer.prepare_reference(img_red)

    score = scorer.score(ref, b"not a png")

    assert score == 1.0
