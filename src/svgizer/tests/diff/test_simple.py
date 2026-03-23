import io

from PIL import Image

# Updated import to match the project structure
from svgizer.diff.simple import SimpleFallbackScorer


def test_simple_fallback_scorer_identical():
    scorer = SimpleFallbackScorer(target_long_side=64)

    # Create a solid red image
    img_red = Image.new("RGB", (100, 100), color="red")

    # Prepare reference
    ref = scorer.prepare_reference(img_red)

    # Save the red image to bytes to act as the candidate
    buf_red = io.BytesIO()
    img_red.save(buf_red, format="PNG")
    cand_red_bytes = buf_red.getvalue()

    # Score identical images
    score_identical = scorer.score(ref, cand_red_bytes)

    # Identical images should have a difference of 0.0
    assert score_identical == 0.0


def test_simple_fallback_scorer_different():
    scorer = SimpleFallbackScorer(target_long_side=64)

    img_red = Image.new("RGB", (100, 100), color="red")
    ref = scorer.prepare_reference(img_red)

    # Create a completely different blue image
    img_blue = Image.new("RGB", (100, 100), color="blue")
    buf_blue = io.BytesIO()
    img_blue.save(buf_blue, format="PNG")
    cand_blue_bytes = buf_blue.getvalue()

    # Score totally different images
    score_diff = scorer.score(ref, cand_blue_bytes)

    # Different images should have a score greater than 0
    assert score_diff > 0.0
    # And must be bounded by 1.0
    assert score_diff <= 1.0


def test_simple_fallback_scorer_handles_size_mismatch():
    scorer = SimpleFallbackScorer(target_long_side=64)

    img_ref = Image.new("RGB", (200, 200), color="green")
    ref = scorer.prepare_reference(img_ref)

    # Candidate has a completely different aspect ratio and size
    img_cand = Image.new("RGB", (50, 80), color="green")
    buf_cand = io.BytesIO()
    img_cand.save(buf_cand, format="PNG")
    cand_bytes = buf_cand.getvalue()

    # The scorer should automatically resize the candidate to match the reference
    score = scorer.score(ref, cand_bytes)

    # They are the same color, so score should still be 0.0 despite the initial size mismatch
    assert score == 0.0


def test_simple_fallback_scorer_invalid_data_returns_max_diff():
    """Verify that corrupt bytes return 1.0 rather than crashing."""
    scorer = SimpleFallbackScorer(target_long_side=64)
    img_red = Image.new("RGB", (10, 10), color="red")
    ref = scorer.prepare_reference(img_red)

    # Passing garbage bytes instead of a valid PNG
    score = scorer.score(ref, b"not a png")

    assert score == 1.0
