import io
import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from PIL import Image

from .base import DEFAULT_CONFIG, DiffScorer
from .utils import get_device, lab_l1, resize_long_side

log = logging.getLogger(__name__)

@dataclass
class DreamSimReference:
    image: Image.Image
    tensor: Any  # torch.Tensor

class DreamSimScorer(DiffScorer):
    """ML-based layout/semantic scorer with lazy-loaded dependencies."""

    def __init__(self):
        self._model = None
        self._preprocess = None

    def _load_dependencies(self):
        """Internal helper to load torch and models only when needed."""
        if self._model is None:
            try:
                import torch
                from dreamsim import dreamsim

                device = get_device()
                model, preprocess = dreamsim(pretrained=True, device=device)
                model.eval()

                self._model = model
                self._preprocess = preprocess
                self._torch = torch
            except ImportError as e:
                raise ImportError("dreamsim or torch not installed. Run 'pip install .[ml]'") from e

    def validate_environment(self):
        """Used by the factory to check viability without a full score pass."""
        self._load_dependencies()

    def prepare_reference(self, original_rgb: Image.Image) -> DreamSimReference:
        self._load_dependencies()
        device = get_device()

        ref_small = resize_long_side(original_rgb, DEFAULT_CONFIG.target_long_side)
        ref_tensor = self._preprocess(ref_small).to(device)

        if ref_tensor.ndim == 3:
            ref_tensor = ref_tensor.unsqueeze(0)

        return DreamSimReference(image=ref_small, tensor=ref_tensor)

    def score(self, reference: DreamSimReference, candidate_png: bytes) -> float:
        self._load_dependencies()

        cand = Image.open(io.BytesIO(candidate_png)).convert("RGB")
        if cand.size != reference.image.size:
            cand = cand.resize(reference.image.size, resample=Image.BILINEAR)

        device = get_device()
        cand_t = self._preprocess(cand).to(device)
        if cand_t.ndim == 3:
            cand_t = cand_t.unsqueeze(0)

        with self._torch.no_grad():
            dist_tensor = self._model(reference.tensor, cand_t)
            dreamsim_dist = float(dist_tensor.item())

        struct_score = max(0.0, min(1.0, dreamsim_dist))
        color_score = float(max(0.0, min(1.0, lab_l1(reference.image, cand))))

        score = (DEFAULT_CONFIG.w_dreamsim * struct_score) + (
                DEFAULT_CONFIG.w_color * color_score
        )

        if not np.isfinite(score):
            return 1.0
        return float(max(0.0, min(1.0, score)))