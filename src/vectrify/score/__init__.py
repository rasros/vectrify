import logging
from enum import Enum

from vectrify.score.base import ScoreConfig, Scorer
from vectrify.score.llm_judge import LLMJudgeScorer
from vectrify.score.simple import SimpleFallbackScorer
from vectrify.score.vision import VisionScorer

log = logging.getLogger(__name__)


class ScorerType(str, Enum):
    AUTO = "auto"
    VISION = "vision"
    SIMPLE = "simple"
    LLM = "llm"


SCORER_REGISTRY: dict[ScorerType, type[Scorer]] = {
    ScorerType.VISION: VisionScorer,
    ScorerType.SIMPLE: SimpleFallbackScorer,
    ScorerType.LLM: LLMJudgeScorer,
}

__all__ = ["ScoreConfig", "Scorer", "ScorerType", "get_scorer"]


def get_scorer(
    scorer_type: ScorerType | str = ScorerType.AUTO,
    provider_name: str = "openai",
    api_key: str | None = None,
    vision_model: str = "google/siglip-so400m-patch14-384",
) -> Scorer:
    if isinstance(scorer_type, str):
        scorer_type = ScorerType(scorer_type.lower())

    if scorer_type in SCORER_REGISTRY and scorer_type != ScorerType.AUTO:
        log.info(f"Using {scorer_type.value} scorer.")
        if scorer_type == ScorerType.LLM:
            return LLMJudgeScorer(provider_name=provider_name, api_key=api_key)
        if scorer_type == ScorerType.VISION:
            return VisionScorer(model_name=vision_model)
        return SCORER_REGISTRY[scorer_type]()

    log.info("AUTO mode: Attempting to initialize vision scorer...")
    try:
        scorer = VisionScorer(model_name=vision_model)
        scorer.validate_environment()
        log.info("AUTO: Vision scorer initialized successfully.")
        return scorer
    except Exception as e:
        log.warning(
            f"AUTO: Vision scorer unavailable ({e}). "
            "Falling back to SimpleFallbackScorer."
        )
        return SimpleFallbackScorer()
