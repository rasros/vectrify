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
    @property
    def comparability(self) -> str:
        """What has to match for two runs' scores to mean the same thing.

        A score is only ever a distance in some scorer's own space. Change the
        members, or how a picture is fed to them, and the numbers move without
        any drawing getting better or worse -- dropping the 5x5 lattice shifted
        one run's recorded best from 0.296 to 0.081. Nothing in a run's output
        recorded that, so a plot spanning the change compared two different
        rulers and read as a 3.7x improvement.
        """
        return type(self).__name__

    @abstractmethod
    def prepare_reference(self, original_rgb: Image.Image) -> Any: ...

    @abstractmethod
    def score(self, reference: Any, candidate_png: bytes) -> float: ...

    def score_many(self, reference: Any, candidate_pngs: list[bytes]) -> list[float]:
        """Score several candidates at once.

        Defaults to scoring them one by one, which is what a scorer with no
        fixed per-call overhead wants. A model-backed scorer overrides this:
        its cost is dominated by the forward pass, so one pass over a batch is
        far cheaper than a pass each.
        """
        return [self.score(reference, png) for png in candidate_pngs]

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
