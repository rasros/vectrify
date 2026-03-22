import logging
from enum import Enum
from typing import Dict, Type

from .base import DiffScorer, ScoreConfig
from .dreamsim import DreamSimScorer
from .simple import SimpleFallbackScorer
from .llm_judge import LLMJudgeScorer

log = logging.getLogger(__name__)

class ScorerType(str, Enum):
    AUTO = "auto"
    DREAMSIM = "dreamsim"
    SIMPLE = "simple"
    LLM = "llm"

# Registry for mapping types to classes
SCORER_REGISTRY: Dict[ScorerType, Type[DiffScorer]] = {
    ScorerType.DREAMSIM: DreamSimScorer,
    ScorerType.SIMPLE: SimpleFallbackScorer,
    ScorerType.LLM: LLMJudgeScorer,
}

__all__ = ["DiffScorer", "ScoreConfig", "ScorerType", "get_scorer"]

def get_scorer(scorer_type: ScorerType | str = ScorerType.AUTO) -> DiffScorer:
    """Factory that returns the requested scorer with lazy-loading behavior."""
    if isinstance(scorer_type, str):
        scorer_type = ScorerType(scorer_type.lower())

    # Handle Explicit Requests
    if scorer_type in SCORER_REGISTRY and scorer_type != ScorerType.AUTO:
        log.info(f"Using {scorer_type.value} scorer.")
        return SCORER_REGISTRY[scorer_type]()

    # Handle AUTO Logic
    log.info("AUTO mode: Attempting to initialize DreamSim...")
    try:
        scorer = DreamSimScorer()
        scorer.validate_environment() # Check if models/torch are available
        log.info("AUTO: DreamSim initialized successfully.")
        return scorer
    except Exception as e:
        log.warning(f"AUTO: DreamSim unavailable ({e}). Falling back to SimpleFallbackScorer.")
        return SimpleFallbackScorer()