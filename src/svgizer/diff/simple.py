import io
import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .base import DiffScorer
from .utils import lab_l1, resize_long_side

log = logging.getLogger(__name__)

@dataclass
class SimpleReference:
    image: Image.Image

class SimpleFallbackScorer(DiffScorer):
    """
    A fast, CPU-bound fallback relying solely on LAB color distance.
    This is the default 'safe' scorer that requires no ML dependencies or API keys.
    """

    def __init__(self, target_long_side: int = 256):
        self.target_long_side = target_long_side

    def prepare_reference(self, original_rgb: Image.Image) -> SimpleReference:
        """Resizes the reference to a manageable size for pixel-wise comparison."""
        ref_small = resize_long_side(original_rgb, self.target_long_side)
        return SimpleReference(image=ref_small)

    def score(self, reference: SimpleReference, candidate_png: bytes) -> float:
        """
        Calculates the mean absolute error in LAB color space.
        Returns 0.0 for identical images and 1.0 for maximum difference.
        """
        try:
            cand = Image.open(io.BytesIO(candidate_png)).convert("RGB")

            # Ensure dimensions match the reference for a valid L1 comparison
            if cand.size != reference.image.size:
                cand = cand.resize(reference.image.size, resample=Image.BILINEAR)

            color_score = lab_l1(reference.image, cand)

            if not np.isfinite(color_score):
                return 1.0

            return float(max(0.0, min(1.0, color_score)))

        except Exception as e:
            log.error(f"SimpleFallbackScorer failed to process candidate: {e}")
            return 1.0