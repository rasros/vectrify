import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .utils import lab_l1, resize_long_side


@dataclass
class SimpleReference:
    image: Image.Image


class SimpleFallbackScorer:
    """A fast, CPU-bound fallback relying solely on LAB color distance."""

    def __init__(self, target_long_side: int = 256):
        self.target_long_side = target_long_side

    def prepare_reference(self, original_rgb: Image.Image) -> SimpleReference:
        ref_small = resize_long_side(original_rgb, self.target_long_side)
        return SimpleReference(image=ref_small)

    def score(self, reference: SimpleReference, candidate_png: bytes) -> float:
        cand = Image.open(io.BytesIO(candidate_png)).convert("RGB")

        if cand.size != reference.image.size:
            cand = cand.resize(reference.image.size, resample=Image.BILINEAR)

        color_score = lab_l1(reference.image, cand)

        if not np.isfinite(color_score):
            return 1.0
        return float(max(0.0, min(1.0, color_score)))
