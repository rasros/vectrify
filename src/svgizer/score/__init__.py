import logging
from enum import Enum

from svgizer.score.base import ScoreConfig, Scorer
from svgizer.score.dreamsim import DreamSimScorer
from svgizer.score.llm_judge import LLMJudgeScorer
from svgizer.score.simple import SimpleFallbackScorer

log = logging.getLogger(__name__)


class ScorerType(str, Enum):
    AUTO = "auto"
    DREAMSIM = "dreamsim"
    SIMPLE = "simple"
    LLM = "llm"


SCORER_REGISTRY: dict[ScorerType, type[Scorer]] = {
    ScorerType.DREAMSIM: DreamSimScorer,
    ScorerType.SIMPLE: SimpleFallbackScorer,
    ScorerType.LLM: LLMJudgeScorer,
}

__all__ = ["ScoreConfig", "Scorer", "ScorerType", "get_scorer"]


def get_scorer(
    scorer_type: ScorerType | str = ScorerType.AUTO,
    provider_name: str = "openai",
    api_key: str | None = None,
    dreamsim_type: str = "ensemble",
) -> Scorer:
    if isinstance(scorer_type, str):
        scorer_type = ScorerType(scorer_type.lower())

    if scorer_type in SCORER_REGISTRY and scorer_type != ScorerType.AUTO:
        log.info(f"Using {scorer_type.value} scorer.")
        if scorer_type == ScorerType.LLM:
            return LLMJudgeScorer(provider_name=provider_name, api_key=api_key)
        if scorer_type == ScorerType.DREAMSIM:
            return DreamSimScorer(dreamsim_type=dreamsim_type)
        return SCORER_REGISTRY[scorer_type]()

    log.info("AUTO mode: Attempting to initialize DreamSim...")
    try:
        scorer = DreamSimScorer(dreamsim_type=dreamsim_type)
        scorer.validate_environment()
        log.info("AUTO: DreamSim initialized successfully.")
        return scorer
    except Exception as e:
        log.warning(
            f"AUTO: DreamSim unavailable ({e}). Falling back to SimpleFallbackScorer."
        )
        return SimpleFallbackScorer()
