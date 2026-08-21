"""What a run got, versus what it asked for.

A silent fall back to the simple scorer changes what every number in a run
means -- the artifact, the lineage and the final "evaluator X" line all look
identical either way. One comparison of four arms was ruined by three of them
scoring with the fallback because their environment lacked torch, and nothing
downstream could tell.
"""

import pytest

from vectrify.score import ScorerType, choose_scorer


def test_an_explicit_request_that_cannot_be_met_fails_immediately(monkeypatch):
    """Asking for the panel and quietly getting something else is the failure
    this exists to stop, so a named scorer is validated before the run starts
    rather than at the first epoch boundary."""
    import vectrify.score as score

    class Broken:
        def validate_environment(self):
            raise RuntimeError("no torch here")

    monkeypatch.setattr(score, "EnsembleScorer", Broken)
    with pytest.raises(RuntimeError, match="no torch"):
        choose_scorer(ScorerType.PANEL)


def test_auto_may_degrade_but_says_so(monkeypatch):
    """AUTO answering on a machine without CUDA is what it is for. Recording
    that it did is the part that was missing."""
    import vectrify.score as score

    class Broken:
        def validate_environment(self):
            raise RuntimeError("no torch here")

    monkeypatch.setattr(score, "EnsembleScorer", Broken)
    choice = choose_scorer(ScorerType.AUTO)
    assert choice.degraded
    assert choice.name == "simple"
    assert choice.requested == "auto"
    assert "no torch" in (choice.reason or "")
    assert "DEGRADED" in choice.summary()


def test_the_record_is_machine_readable_enough_to_assert_on():
    """A comparison of two runs has to be able to check that both used the same
    judge without parsing a log."""
    choice = choose_scorer(ScorerType.SIMPLE)
    record = choice.as_record()
    assert record.splitlines()[0] == "simple"
    assert "degraded=false" in record


def test_a_scorer_with_no_dependencies_needs_no_environment():
    choice = choose_scorer(ScorerType.SIMPLE)
    assert not choice.degraded
    assert choice.name == "simple"
