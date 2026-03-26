import io

import pytest
from PIL import Image

from svgizer.score.dreamsim import DreamSimScorer


@pytest.fixture(scope="module")
def scorer():
    s = DreamSimScorer(device="cpu", dreamsim_type="dino_vitb16")
    try:
        s.validate_environment()
    except ImportError as e:
        pytest.skip(f"DreamSim unavailable: {e}")
    return s


def _png(color: str, size: int = 8) -> bytes:
    img = Image.new("RGB", (size, size), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_identical_image_scores_zero(scorer):
    ref_img = Image.new("RGB", (8, 8), color="red")
    ref = scorer.prepare_reference(ref_img)
    score = scorer.score(ref, _png("red"))
    assert score == pytest.approx(0.0, abs=0.05)


def test_different_image_scores_higher(scorer):
    ref_img = Image.new("RGB", (8, 8), color="red")
    ref = scorer.prepare_reference(ref_img)
    score_same = scorer.score(ref, _png("red"))
    score_diff = scorer.score(ref, _png("blue"))
    assert score_diff > score_same


def test_score_is_in_unit_range(scorer):
    ref_img = Image.new("RGB", (8, 8), color="green")
    ref = scorer.prepare_reference(ref_img)
    for color in ("green", "red", "blue", "white", "black"):
        score = scorer.score(ref, _png(color))
        assert 0.0 <= score <= 1.0, f"Score out of range for {color}: {score}"


def test_score_handles_size_mismatch(scorer):
    ref_img = Image.new("RGB", (16, 16), color="red")
    ref = scorer.prepare_reference(ref_img)
    score = scorer.score(ref, _png("red", size=4))
    assert 0.0 <= score <= 1.0


def test_validate_environment_does_not_raise(scorer):
    scorer.validate_environment()


def test_load_is_idempotent(scorer):
    ref_img = Image.new("RGB", (8, 8), color="white")
    ref1 = scorer.prepare_reference(ref_img)
    ref2 = scorer.prepare_reference(ref_img)
    s1 = scorer.score(ref1, _png("white"))
    s2 = scorer.score(ref2, _png("white"))
    assert s1 == pytest.approx(s2, abs=1e-6)
