import io

from PIL import Image

from vectrify.score.simple import SimpleFallbackScorer


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
    from vectrify.score.utils import lab_l1

    img = Image.new("RGB", (32, 32), color="green")
    assert lab_l1(img, img) == 0.0


def test_lab_l1_different_images_nonzero():
    from vectrify.score.utils import lab_l1

    red = Image.new("RGB", (32, 32), color="red")
    blue = Image.new("RGB", (32, 32), color="blue")
    assert lab_l1(red, blue) > 0.0


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


def test_diff_heatmap_returns_none_when_reference_has_no_image():
    scorer = SimpleFallbackScorer()
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color="red").save(buf, format="PNG")
    result = scorer.diff_heatmap(object(), buf.getvalue(), long_side=32)
    assert result is None


def test_diff_heatmap_respects_long_side():
    scorer = SimpleFallbackScorer()
    ref = scorer.prepare_reference(Image.new("RGB", (128, 128), color="red"))
    buf = io.BytesIO()
    Image.new("RGB", (128, 128), color="blue").save(buf, format="PNG")
    result = scorer.diff_heatmap(ref, buf.getvalue(), long_side=32)
    assert result is not None
    img = Image.open(io.BytesIO(result))
    assert max(img.size) <= 32


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_erasing_fine_detail_no_longer_beats_drawing_it_slightly_wrong():
    """Colour distance is an average over pixels, so thin strokes are cheap to
    delete and expensive to get slightly wrong: on this figure colour alone
    scores erasing every stroke at 0.037 against 0.074 for keeping them one
    pixel off. That is the wordmark and connect-the-dots failure -- a search
    optimising colour alone drove their vision scores several times worse over
    4000 tasks. Mixing structure in reverses the ordering."""
    from PIL import ImageDraw

    reference = Image.new("RGB", (64, 64), "white")
    draw = ImageDraw.Draw(reference)
    for x in range(4, 64, 8):
        draw.line((x, 4, x, 60), fill="black", width=1)

    one_pixel_off = Image.new("RGB", (64, 64), "white")
    draw = ImageDraw.Draw(one_pixel_off)
    for x in range(5, 64, 8):
        draw.line((x, 4, x, 60), fill="black", width=1)

    erased = Image.new("RGB", (64, 64), "white")

    scorer = SimpleFallbackScorer()
    ref = scorer.prepare_reference(reference)
    assert scorer.score(ref, _png(one_pixel_off)) < scorer.score(ref, _png(erased))
