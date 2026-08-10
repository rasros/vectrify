import pytest
from PIL import Image

from tests.helpers import make_png
from vectrify.score.base import safe_score
from vectrify.score.utils import MAX_SCORE, clamp01, color_score


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-1.0, 0.0), (0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (2.5, 1.0)],
)
def test_clamp01(value, expected):
    assert clamp01(value) == expected


def test_clamp01_returns_float():
    assert isinstance(clamp01(1), float)


def test_color_score_identical_is_zero():
    ref = Image.new("RGB", (32, 32), color="red")
    assert color_score(ref, make_png("red", 32)) == pytest.approx(0.0)


def test_color_score_different_is_positive():
    ref = Image.new("RGB", (32, 32), color="red")
    assert color_score(ref, make_png("blue", 32)) > 0.0


def test_color_score_resizes_mismatched_candidate():
    ref = Image.new("RGB", (32, 32), color="red")
    assert color_score(ref, make_png("red", 64)) == pytest.approx(0.0)


def test_color_score_is_clamped():
    ref = Image.new("RGB", (16, 16), color="white")
    assert 0.0 <= color_score(ref, make_png("black", 16)) <= 1.0


class _Scorer:
    @safe_score
    def score(self, fail: bool) -> float:
        if fail:
            raise RuntimeError("scorer exploded")
        return 0.25


def test_safe_score_passes_through_success():
    assert _Scorer().score(fail=False) == 0.25


def test_safe_score_returns_worst_score_on_failure():
    assert _Scorer().score(fail=True) == MAX_SCORE
