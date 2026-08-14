from unittest.mock import patch

from vectrify.score import ScorerType, get_scorer
from vectrify.score.ensemble import EnsembleScorer
from vectrify.score.simple import SimpleFallbackScorer
from vectrify.score.vision import VisionScorer


def test_get_scorer_simple_returns_simple_fallback_scorer():
    scorer = get_scorer(ScorerType.SIMPLE)
    assert isinstance(scorer, SimpleFallbackScorer)


def test_get_scorer_string_input_simple():
    scorer = get_scorer("simple")
    assert isinstance(scorer, SimpleFallbackScorer)


def test_get_scorer_vision_returns_vision_scorer():
    scorer = get_scorer(ScorerType.VISION)
    assert isinstance(scorer, VisionScorer)


def test_get_scorer_panel_returns_the_evaluator_panel():
    scorer = get_scorer(ScorerType.PANEL)
    assert isinstance(scorer, EnsembleScorer)


def test_get_scorer_auto_falls_back_to_simple_when_the_panel_is_unavailable():
    with patch.object(
        EnsembleScorer,
        "validate_environment",
        side_effect=ImportError("torch not installed"),
    ):
        scorer = get_scorer(ScorerType.AUTO)
    assert isinstance(scorer, SimpleFallbackScorer)


def test_get_scorer_auto_returns_the_panel_when_available():
    with patch.object(EnsembleScorer, "validate_environment", return_value=None):
        scorer = get_scorer(ScorerType.AUTO)
    assert isinstance(scorer, EnsembleScorer)
