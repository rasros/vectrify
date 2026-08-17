import pytest

from vectrify.dashboard import _distinct_fraction


@pytest.mark.parametrize(
    ("epoch_distinct", "pool_distinct", "expected"),
    [
        (0.0, 0.5, 0.0),  # criterion disabled
        (0.1, 1.0, 0.1),  # far from the stop
        (0.1, 0.2, 0.5),  # halfway
        (0.1, 0.1, 1.0),  # exactly at the threshold
        (0.1, 0.05, 1.0),  # past it, clamped
    ],
)
def test_distinct_fraction(epoch_distinct, pool_distinct, expected):
    assert _distinct_fraction(epoch_distinct, pool_distinct) == pytest.approx(expected)


def test_distinct_fraction_is_full_when_nothing_is_distinct():
    """A pool with no distinct member is the collapsed state the criterion
    exists to detect, so the bar reads full rather than empty there."""
    assert _distinct_fraction(0.1, 0.0) == 1.0


def test_distinct_fraction_stays_zero_when_disabled():
    assert _distinct_fraction(0.0, 0.0) == 0.0
