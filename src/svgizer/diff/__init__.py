import logging
from enum import Enum

from .base import DiffScorer, ScoreConfig
from .dreamsim import DreamSimScorer, get_dreamsim_models
from .simple import SimpleFallbackScorer
from .llm_judge import LLMJudgeScorer

log = logging.getLogger(__name__)


class ScorerType(str, Enum):
    AUTO = "auto"
    DREAMSIM = "dreamsim"
    SIMPLE = "simple"
    LLM = "llm"


__all__ = ["DiffScorer", "ScoreConfig", "ScorerType", "get_scorer"]


def get_scorer(scorer_type: ScorerType | str = ScorerType.AUTO) -> DiffScorer:
    """Factory that returns the explicitly requested scorer, or falls back gracefully if AUTO."""
    if isinstance(scorer_type, str):
        scorer_type = ScorerType(scorer_type.lower())

    if scorer_type == ScorerType.LLM:
        log.info("Using LLMJudgeScorer.")
        return LLMJudgeScorer()

    if scorer_type == ScorerType.SIMPLE:
        log.info("Using SimpleFallbackScorer.")
        return SimpleFallbackScorer()

    if scorer_type == ScorerType.DREAMSIM:
        try:
            get_dreamsim_models()
            log.info("DreamSim models loaded successfully. Using DreamSimScorer.")
            return DreamSimScorer()
        except Exception as e:
            log.error(f"Failed to load DreamSim: {e}")
            raise

    # AUTO behavior
    try:
        get_dreamsim_models()
        log.info("AUTO: DreamSim models loaded successfully. Using DreamSimScorer.")
        return DreamSimScorer()
    except Exception as e:
        log.warning(
            f"AUTO: DreamSim unavailable ({e}). Falling back to SimpleFallbackScorer."
        )
        return SimpleFallbackScorer()
