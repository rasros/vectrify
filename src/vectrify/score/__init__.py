import logging
from collections.abc import Callable
from enum import Enum

from vectrify.score.base import ScoreConfig, Scorer
from vectrify.score.llm_judge import LLMJudgeScorer
from vectrify.score.simple import SimpleFallbackScorer
from vectrify.score.vision import DEFAULT_VISION_MODEL, VisionScorer

log = logging.getLogger(__name__)


class ScorerType(str, Enum):
    AUTO = "auto"
    VISION = "vision"
    SIMPLE = "simple"
    LLM = "llm"


__all__ = ["ScoreConfig", "Scorer", "ScorerType", "get_scorer"]


def get_scorer(
    scorer_type: ScorerType | str = ScorerType.AUTO,
    provider_name: str = "openai",
    api_key: str | None = None,
    vision_model: str = DEFAULT_VISION_MODEL,
) -> Scorer:
    if isinstance(scorer_type, str):
        scorer_type = ScorerType(scorer_type.lower())

    # Every scorer takes different construction arguments, so the registry
    # holds thunks rather than classes.
    builders: dict[ScorerType, Callable[[], Scorer]] = {
        ScorerType.VISION: lambda: VisionScorer(model_name=vision_model),
        ScorerType.SIMPLE: SimpleFallbackScorer,
        ScorerType.LLM: lambda: LLMJudgeScorer(
            provider_name=provider_name, api_key=api_key
        ),
    }

    if scorer_type in builders:
        log.info(f"Using {scorer_type.value} scorer.")
        return builders[scorer_type]()

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
