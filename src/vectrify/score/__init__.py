import logging
from collections.abc import Callable
from enum import Enum

from vectrify.score.base import ScoreConfig, Scorer
from vectrify.score.ensemble import GRID, EnsembleScorer
from vectrify.score.simple import SimpleFallbackScorer
from vectrify.score.vision import DEFAULT_VISION_MODEL, VisionScorer

log = logging.getLogger(__name__)


class ScorerType(str, Enum):
    AUTO = "auto"
    PANEL = "panel"
    VISION = "vision"
    SIMPLE = "simple"


__all__ = ["ScoreConfig", "Scorer", "ScorerType", "get_scorer"]


def get_scorer(
    scorer_type: ScorerType | str = ScorerType.AUTO,
    vision_model: str = DEFAULT_VISION_MODEL,
    panel_grid: int = GRID,
) -> Scorer:
    if isinstance(scorer_type, str):
        scorer_type = ScorerType(scorer_type.lower())

    # Every scorer takes different construction arguments, so the registry
    # holds thunks rather than classes.
    builders: dict[ScorerType, Callable[[], Scorer]] = {
        ScorerType.PANEL: lambda: EnsembleScorer(grid=panel_grid),
        ScorerType.VISION: lambda: VisionScorer(model_name=vision_model),
        ScorerType.SIMPLE: SimpleFallbackScorer,
    }

    if scorer_type in builders:
        log.info(f"Using {scorer_type.value} scorer.")
        return builders[scorer_type]()

    log.info("AUTO mode: Attempting to initialize the evaluator panel...")
    try:
        scorer = EnsembleScorer(grid=panel_grid)
        scorer.validate_environment()
        log.info("AUTO: Evaluator panel initialized successfully.")
        return scorer
    except Exception as e:
        log.warning(
            f"AUTO: Vision scorer unavailable ({e}). "
            "Falling back to SimpleFallbackScorer."
        )
        return SimpleFallbackScorer()
