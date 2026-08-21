import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from vectrify.score.base import ScoreConfig, Scorer
from vectrify.score.ensemble import EnsembleScorer
from vectrify.score.simple import SimpleFallbackScorer
from vectrify.score.vision import DEFAULT_VISION_MODEL, VisionScorer

log = logging.getLogger(__name__)


class ScorerType(str, Enum):
    AUTO = "auto"
    PANEL = "panel"
    VISION = "vision"
    SIMPLE = "simple"


__all__ = [
    "ScoreConfig",
    "Scorer",
    "ScorerChoice",
    "ScorerType",
    "choose_scorer",
    "get_scorer",
]


@dataclass(frozen=True)
class ScorerChoice:
    """Which scorer a run actually got, and whether that is what it asked for.

    The distinction has to be reportable rather than only logged. AUTO falling
    back changes what every number in a run means -- the artifact, the lineage
    and the final "evaluator X" line all look identical either way -- and one
    comparison of four arms was ruined by three of them silently scoring with
    the simple fallback because their environment lacked torch. Nothing
    downstream could tell.
    """

    scorer: Scorer
    name: str
    requested: str
    degraded: bool
    reason: str | None = None

    def as_record(self) -> str:
        """The choice as a file, so a comparison can assert two runs match.

        A log line is exactly what gets skimmed; this sits in the run directory
        next to the numbers it qualifies.
        """
        lines = [
            self.name,
            f"requested={self.requested}",
            f"degraded={str(self.degraded).lower()}",
        ]
        if self.reason:
            lines.append(f"reason={self.reason}")
        return "\n".join(lines) + "\n"

    def summary(self) -> str:
        if not self.degraded:
            return f"scorer={self.name} (requested {self.requested})"
        return f"scorer={self.name} DEGRADED from {self.requested}: {self.reason}"


def choose_scorer(
    scorer_type: ScorerType | str = ScorerType.AUTO,
    vision_model: str = DEFAULT_VISION_MODEL,
) -> ScorerChoice:
    """Pick the scorer for a run and say plainly which one it is.

    A named scorer is validated here rather than at first use, so a run asking
    for a model it cannot load fails before it spends anything. Only AUTO may
    fall back, since answering on a machine without CUDA is what it is for --
    and when it does, the choice says so.
    """
    if isinstance(scorer_type, str):
        scorer_type = ScorerType(scorer_type.lower())
    requested = scorer_type.value

    # Every scorer takes different construction arguments, so the registry
    # holds thunks rather than classes.
    builders: dict[ScorerType, Callable[[], Scorer]] = {
        ScorerType.PANEL: EnsembleScorer,
        ScorerType.VISION: lambda: VisionScorer(model_name=vision_model),
        ScorerType.SIMPLE: SimpleFallbackScorer,
    }

    if scorer_type in builders:
        scorer = builders[scorer_type]()
        # Fail fast on an explicit request: asking for the panel and getting
        # something else without noticing is the failure this exists to stop.
        scorer.validate_environment()
        log.info(f"Using {requested} scorer.")
        return ScorerChoice(scorer, requested, requested, degraded=False)

    log.info("AUTO mode: Attempting to initialize the evaluator panel...")
    try:
        scorer = EnsembleScorer()
        scorer.validate_environment()
        log.info("AUTO: Evaluator panel initialized successfully.")
        return ScorerChoice(scorer, ScorerType.PANEL.value, requested, degraded=False)
    except Exception as e:
        reason = str(e)
        log.error(
            "=" * 72
            + f"\nSCORER DEGRADED: the evaluator panel is unavailable ({reason})."
            + "\nFalling back to the simple scorer. Every score this run reports"
            + "\nis on a different scale and is NOT comparable with a panel run."
            + "\nPass --scorer panel to fail instead, or --scorer simple to mean it.\n"
            + "=" * 72
        )
        return ScorerChoice(
            SimpleFallbackScorer(),
            ScorerType.SIMPLE.value,
            requested,
            degraded=True,
            reason=reason,
        )


def get_scorer(
    scorer_type: ScorerType | str = ScorerType.AUTO,
    vision_model: str = DEFAULT_VISION_MODEL,
) -> Scorer:
    """The scorer alone, for callers with nothing to report it to."""
    return choose_scorer(scorer_type, vision_model=vision_model).scorer
