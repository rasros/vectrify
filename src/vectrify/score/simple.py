import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image

from vectrify.image_utils import resize_long_side
from vectrify.score.base import DEFAULT_CONFIG, Scorer, safe_score
from vectrify.score.edges import edge_score
from vectrify.score.utils import MAX_SCORE, clamp01, color_score

log = logging.getLogger(__name__)


@dataclass
class SimpleReference:
    image: Image.Image


class SimpleFallbackScorer(Scorer):
    def prepare_reference(self, original_rgb: Image.Image) -> SimpleReference:
        ref_small = resize_long_side(original_rgb, DEFAULT_CONFIG.target_long_side)
        return SimpleReference(image=ref_small)

    @safe_score
    def score(self, reference: SimpleReference, candidate_png: bytes) -> float:
        """Structure and colour, mixed the way the vision score mixes them.

        Colour distance alone is dominated by area: a background off by 3%
        outweighs a shape that is missing entirely. Measured on the bench
        corpus it ranks candidates at rho 0.48 against the vision model, and
        on the two cases where it ranks worst -- a wordmark at 0.14 and a
        connect-the-dots at 0.25 -- a search optimising it drove the vision
        score several times *worse* over 4000 tasks. Adding structure takes
        the whole corpus to 0.83.
        """
        structure = edge_score(reference.image, candidate_png)
        colour = color_score(reference.image, candidate_png)
        score = (
            DEFAULT_CONFIG.w_structure * structure
            + (1.0 - DEFAULT_CONFIG.w_structure) * colour
        )
        if not np.isfinite(score):
            return MAX_SCORE
        return clamp01(score)
