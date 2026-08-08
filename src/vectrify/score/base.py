import functools
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from vectrify.image_utils import pixel_diff_png
from vectrify.score.regions import block_distance_grid
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

    def region_distance_grid(
        self, reference: Any, candidate_png: bytes
    ) -> np.ndarray | None:
        """Per-region distances laid out over the canvas, or None if unavailable.

        Feeds the ``worst_region`` objective. The default is a block-wise Lab
        L1 grid, which needs nothing beyond Pillow, so the objective survives
        ``--scorer simple`` and any environment without torch. Subclasses with
        a spatial model override this with something perceptually richer.

        Returns None only when the reference carries no image to compare
        against; callers must treat that as "no measurement" rather than as a
        distance of zero.
        """
        ref_img = getattr(reference, "image", None)
        if ref_img is None:
            return None
        return block_distance_grid(ref_img, candidate_png)

    def diff_heatmap(
        self,
        reference: Any,
        candidate_png: bytes,
        long_side: int,
        grid: np.ndarray | None = None,  # noqa: ARG002 - part of the override contract
    ) -> bytes | None:
        """Pixel-based diff heatmap (brightness-boosted RGB difference).

        Subclasses may override this with a perceptually richer implementation.
        Returns PNG bytes, or None if the reference lacks an ``image`` attribute.

        *grid* is an already-computed ``region_distance_grid`` the caller is
        offering to save recomputing; this implementation derives its heatmap
        per-pixel and ignores it.
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

    # Scoring resolution. One whole-image pass sees a 700px drawing at 384px,
    # so a 7px numeral lands under SigLIP's 14px patch and cannot be told from
    # a smudge. Scoring a grid of crops instead keeps that detail above the
    # patch size. tiles=1 is the untiled scorer.
    #
    # Crops are cut at exactly this size and fed in unresampled, so scaling
    # introduces no artifact the scorer could mistake for candidate error.
    # None means the model's own input size, which is the only size that needs
    # no resampling at all. The crop count follows from the image.
    tile_size: int | None = None
    tile_overlap: float = 0.5

    # Fraction of tiles averaged into the primary score, worst-first. Averaging
    # every tile re-dilutes a localised defect across the mostly-blank ones,
    # which is the problem tiling was meant to fix; scoring only the worst
    # share keeps the pressure where the error is.
    score_tile_fraction: float = 0.5


DEFAULT_CONFIG = ScoreConfig()
