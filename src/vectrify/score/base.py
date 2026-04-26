from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from PIL import Image

from vectrify.image_utils import pixel_diff_png


class Scorer(ABC):
    @abstractmethod
    def prepare_reference(self, original_rgb: Image.Image) -> Any: ...

    @abstractmethod
    def score(self, reference: Any, candidate_png: bytes) -> float: ...

    def diff_heatmap(
        self, reference: Any, candidate_png: bytes, long_side: int
    ) -> bytes | None:
        """Pixel-based diff heatmap (brightness-boosted RGB difference).

        Subclasses may override this with a perceptually richer implementation.
        Returns PNG bytes, or None if the reference lacks an ``image`` attribute.
        """
        ref_img = getattr(reference, "image", None)
        if ref_img is None:
            return None
        return pixel_diff_png(ref_img, candidate_png, long_side)


@dataclass(frozen=True)
class ScoreConfig:
    target_long_side: int = 256
    w_vision: float = 0.85
    w_color: float = 0.15


DEFAULT_CONFIG = ScoreConfig()
