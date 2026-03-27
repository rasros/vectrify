"""Smoke-test that the search base module imports cleanly."""

from svgizer.search.base import StrategyType


def test_strategy_type_values():
    assert StrategyType.GREEDY == "greedy"
    assert StrategyType.NSGA == "nsga"
