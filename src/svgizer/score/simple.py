import io
import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image
from PIL.Image import Resampling

from svgizer.image_utils import resize_long_side
from svgizer.score.base import Scorer
from svgizer.score.utils import lab_l1

log = logging.getLogger(__name__)


@dataclass
class SimpleReference:
    image: Image.Image


class SimpleFallbackScorer(Scorer):
    def __init__(self, target_long_side: int = 256):
        self.target_long_side = target_long_side

    def prepare_reference(self, original_rgb: Image.Image) -> SimpleReference:
        ref_small = resize_long_side(original_rgb, self.target_long_side)
        return SimpleReference(image=ref_small)

    def score(self, reference: SimpleReference, candidate_png: bytes) -> float:
        try:
            cand = Image.open(io.BytesIO(candidate_png)).convert("RGB")

            if cand.size != reference.image.size:
                cand = cand.resize(reference.image.size, resample=Resampling.BILINEAR)

            color_score = lab_l1(reference.image, cand)

            if not np.isfinite(color_score):
                return 1.0

            return float(max(0.0, min(1.0, color_score)))

        except Exception as e:
            log.error(f"SimpleFallbackScorer failed to process candidate: {e}")
            return 1.0
