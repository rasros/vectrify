import io

from PIL import Image

from svgizer.score.simple import SimpleFallbackScorer


def test_simple_fallback_scorer_identical():
    scorer = SimpleFallbackScorer()
    img_red = Image.new("RGB", (100, 100), color="red")

    ref = scorer.prepare_reference(img_red)
    buf_red = io.BytesIO()
    img_red.save(buf_red, format="PNG")
    cand_red_bytes = buf_red.getvalue()

    score_identical = scorer.score(ref, cand_red_bytes)
    assert score_identical == 0.0


def test_simple_fallback_scorer_different():
    scorer = SimpleFallbackScorer()
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
    scorer = SimpleFallbackScorer()
    img_ref = Image.new("RGB", (200, 200), color="green")
    ref = scorer.prepare_reference(img_ref)

    img_cand = Image.new("RGB", (50, 80), color="green")
    buf_cand = io.BytesIO()
    img_cand.save(buf_cand, format="PNG")
    cand_bytes = buf_cand.getvalue()

    score = scorer.score(ref, cand_bytes)
    assert score == 0.0


def test_simple_fallback_scorer_invalid_data_returns_max_diff():
    scorer = SimpleFallbackScorer()
    img_red = Image.new("RGB", (10, 10), color="red")
    ref = scorer.prepare_reference(img_red)

    score = scorer.score(ref, b"not a png")
    assert score == 1.0


def test_lab_l1_identical_images_zero():
    from svgizer.score.utils import lab_l1

    img = Image.new("RGB", (32, 32), color="green")
    assert lab_l1(img, img) == 0.0


def test_lab_l1_different_images_nonzero():
    from svgizer.score.utils import lab_l1

    red = Image.new("RGB", (32, 32), color="red")
    blue = Image.new("RGB", (32, 32), color="blue")
    assert lab_l1(red, blue) > 0.0


def test_simple_diff_heatmap_returns_valid_png():
    scorer = SimpleFallbackScorer()
    ref_img = Image.new("RGB", (32, 32), color="red")
    ref = scorer.prepare_reference(ref_img)

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color="blue").save(buf, format="PNG")
    result = scorer.diff_heatmap(ref, buf.getvalue(), long_side=32)

    assert result is not None
    img = Image.open(io.BytesIO(result))
    assert img.mode == "RGB"


def test_simple_diff_heatmap_identical_images_are_black():
    scorer = SimpleFallbackScorer()
    ref_img = Image.new("RGB", (32, 32), color="green")
    ref = scorer.prepare_reference(ref_img)

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color="green").save(buf, format="PNG")
    result = scorer.diff_heatmap(ref, buf.getvalue(), long_side=32)

    assert result is not None
    img = Image.open(io.BytesIO(result)).convert("RGB")
    assert all(p == (0, 0, 0) for p in img.get_flattened_data())
