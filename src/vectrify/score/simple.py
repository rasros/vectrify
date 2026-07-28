import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image

from vectrify.image_utils import resize_long_side
from vectrify.score.base import DEFAULT_CONFIG, Scorer, safe_score
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
        score = color_score(reference.image, candidate_png)
        if not np.isfinite(score):
            return MAX_SCORE
        return clamp01(score)
