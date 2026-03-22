from dataclasses import dataclass
from typing import Any, Protocol

from PIL import Image


class DiffScorer(Protocol):
    """Protocol for algorithms that score image differences."""

    def prepare_reference(self, original_rgb: Image.Image) -> Any:
        """Pre-processes the reference image into the format required by the scorer."""
        ...

    def score(self, reference: Any, candidate_png: bytes) -> float:
        """Returns a difference score between 0.0 (identical) and 1.0 (completely different)."""
        ...


@dataclass(frozen=True)
class ScoreConfig:
    target_long_side: int = 256
    w_dreamsim: float = 0.85
    w_color: float = 0.15


DEFAULT_CONFIG = ScoreConfig()
