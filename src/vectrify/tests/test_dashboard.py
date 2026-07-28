import pytest

from vectrify.dashboard import _variance_fraction


@pytest.mark.parametrize(
    ("epoch_variance", "pool_std", "expected"),
    [
        (0.0, 0.5, 0.0),  # criterion disabled
        (0.1, 1.0, 0.1),  # far from the stop
        (0.1, 0.2, 0.5),  # halfway
        (0.1, 0.1, 1.0),  # exactly at the threshold
        (0.1, 0.05, 1.0),  # past it, clamped
    ],
)
def test_variance_fraction(epoch_variance, pool_std, expected):
    assert _variance_fraction(epoch_variance, pool_std) == pytest.approx(expected)


def test_variance_fraction_is_full_at_zero_spread():
    """Regression: zero spread returned 0.0, so the bar read empty at exactly
    the moment the criterion was most satisfied. A pool whose scores are all
    identical is the collapsed state --epoch-variance exists to detect.
    """
    assert _variance_fraction(0.1, 0.0) == 1.0


def test_variance_fraction_stays_zero_when_disabled_even_at_zero_spread():
    assert _variance_fraction(0.0, 0.0) == 0.0
