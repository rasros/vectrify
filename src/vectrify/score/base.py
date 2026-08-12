import functools
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PIL import Image

from vectrify.image_utils import pixel_diff_png
from vectrify.score.utils import MAX_SCORE

log = logging.getLogger(__name__)


def safe_score(fn: Callable[..., float]) -> Callable[..., float]:
    """Return the worst possible score instead of raising.

    A scorer failure must not kill the search: the candidate simply loses.
    """

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs) -> float:
        try:
            return fn(self, *args, **kwargs)
        except Exception as e:
            log.error(f"{type(self).__name__} failed to score candidate: {e}")
            return MAX_SCORE

    return wrapper


class Scorer(ABC):
    @abstractmethod
    def prepare_reference(self, original_rgb: Image.Image) -> Any: ...

    @abstractmethod
    def score(self, reference: Any, candidate_png: bytes) -> float: ...

    def diff_heatmap(
        self,
        reference: Any,
        candidate_png: bytes,
        long_side: int,
    ) -> bytes | None:
        """Pixel diff heatmap for the --save-heatmap sidecar.

        Returns PNG bytes, or None if the reference lacks an ``image``.
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
    # The cheap scorer's split between structure and colour. Measured against
    # the vision model over mutation chains on the bench corpus: colour alone
    # ranks candidates at rho 0.48, structure alone at 0.83, and every mix from
    # 0.5 to 0.85 lands within noise of each other. Half and half because pure
    # structure is colour-blind -- it scores a recoloured drawing as perfect.
    w_structure: float = 0.5


DEFAULT_CONFIG = ScoreConfig()
