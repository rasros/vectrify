"""The metric registry."""

from vectrify.score.metrics import FRONT_SCORE, SCORER_METRICS


def test_the_evaluator_verdict_is_recorded_but_never_an_objective():
    """It exists on a handful of nodes per epoch, and a metric absent elsewhere
    reads as 0.0 -- best possible for a minimised objective -- which would let
    every unevaluated candidate dominate every evaluated one."""
    assert FRONT_SCORE == "front_score"
    assert FRONT_SCORE not in SCORER_METRICS
