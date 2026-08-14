import pytest
from PIL import Image

from tests.helpers import make_png
from vectrify.score.base import Scorer, safe_score
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


def test_lab_l1_does_not_let_chroma_errors_cancel():
    """Regression: PIL stores Lab's a/b channels offset-encoded, and reducing
    them through ImageChops + ImageStat gives a signed mean, so a candidate too
    blue in one place and too yellow in another scored as if both were right.
    Colour distance was mostly a lightness distance, under-reporting by ~4x."""
    import numpy as np
    from PIL import Image

    from vectrify.score.utils import lab_array, lab_l1

    reference = Image.new("RGB", (64, 64), (128, 128, 128))
    mixed = Image.new("RGB", (64, 64), (128, 128, 128))
    mixed.paste(Image.new("RGB", (32, 64), (128, 128, 200)), (0, 0))
    mixed.paste(Image.new("RGB", (32, 64), (128, 128, 40)), (32, 0))

    truth = float(np.abs(lab_array(reference) - lab_array(mixed)).mean()) / 255.0
    assert lab_l1(reference, mixed) == pytest.approx(truth)
    assert lab_l1(reference, mixed) > 0.3


def test_lab_l1_separates_a_hue_error_from_a_lightness_error():
    """Two candidates equally far off in Lab, one in lightness and one in hue.
    The old reduction scored the hue one as nearly perfect."""
    from PIL import Image

    from vectrify.score.utils import lab_l1

    reference = Image.new("RGB", (32, 32), (128, 128, 128))
    hue_shift = Image.new("RGB", (32, 32), (128, 128, 220))

    assert lab_l1(reference, hue_shift) > 0.05


def test_score_many_defaults_to_scoring_one_by_one():
    """A scorer with no per-call overhead gains nothing from a batch, so the
    protocol's default must keep working without an override."""

    class CountingScorer(Scorer):
        def __init__(self):
            self.calls = 0

        def prepare_reference(self, original_rgb):
            return original_rgb

        def score(self, reference, candidate_png):
            _ = reference
            self.calls += 1
            return len(candidate_png) / 100.0

    scorer = CountingScorer()
    assert scorer.score_many(None, [b"a", b"bb", b"ccc"]) == [0.01, 0.02, 0.03]
    assert scorer.calls == 3
