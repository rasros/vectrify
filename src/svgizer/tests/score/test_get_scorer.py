from unittest.mock import patch

from svgizer.score import ScorerType, get_scorer
from svgizer.score.simple import SimpleFallbackScorer
from svgizer.score.vision import VisionScorer


def test_get_scorer_simple_returns_simple_fallback_scorer():
    scorer = get_scorer(ScorerType.SIMPLE)
    assert isinstance(scorer, SimpleFallbackScorer)


def test_get_scorer_string_input_simple():
    scorer = get_scorer("simple")
    assert isinstance(scorer, SimpleFallbackScorer)


def test_get_scorer_vision_returns_vision_scorer():
    scorer = get_scorer(ScorerType.VISION)
    assert isinstance(scorer, VisionScorer)


def test_get_scorer_auto_falls_back_to_simple_when_vision_unavailable():
    with patch.object(
        VisionScorer,
        "validate_environment",
        side_effect=ImportError("torch not installed"),
    ):
        scorer = get_scorer(ScorerType.AUTO)
    assert isinstance(scorer, SimpleFallbackScorer)


def test_get_scorer_auto_returns_vision_when_available():
    with patch.object(VisionScorer, "validate_environment", return_value=None):
        scorer = get_scorer(ScorerType.AUTO)
    assert isinstance(scorer, VisionScorer)
