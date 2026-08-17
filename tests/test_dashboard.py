import pytest

from vectrify.dashboard import _dominated_fraction


@pytest.mark.parametrize(
    ("epoch_dominated", "pool_dominated", "expected"),
    [
        (0.0, 0.5, 0.0),  # criterion disabled
        (0.1, 1.0, 0.1),  # far from the stop
        (0.1, 0.2, 0.5),  # halfway
        (0.1, 0.1, 1.0),  # exactly at the threshold
        (0.1, 0.05, 1.0),  # past it, clamped
    ],
)
def test_dominated_fraction(epoch_dominated, pool_dominated, expected):
    assert _dominated_fraction(epoch_dominated, pool_dominated) == pytest.approx(
        expected
    )


def test_dominated_fraction_is_full_when_nothing_is_dominated():
    """A pool where nothing outranks anything is the collapsed state the
    criterion exists to detect, so the bar reads full rather than empty."""
    assert _dominated_fraction(0.1, 0.0) == 1.0


def test_dominated_fraction_stays_zero_when_disabled():
    assert _dominated_fraction(0.0, 0.0) == 0.0
