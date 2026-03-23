import io
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from svgizer.diff.base import DEFAULT_CONFIG, DiffScorer
from svgizer.diff.utils import get_device, lab_l1
from svgizer.image_utils import resize_long_side

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)


@dataclass
class DreamSimReference:
    image: Image.Image
    tensor: "torch.Tensor"


class DreamSimScorer(DiffScorer):
    def __init__(self):
        self._model: Any | None = None
        self._preprocess: Callable | None = None
        self._torch: Any | None = None

    def _load_dependencies(self):
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
                raise ImportError(
                    "dreamsim or torch not installed. Run 'pip install .[ml]'"
                ) from e

    def validate_environment(self):
        self._load_dependencies()

    def prepare_reference(self, original_rgb: Image.Image) -> DreamSimReference:
        self._load_dependencies()
        assert self._preprocess is not None

        device = get_device()
        ref_small = resize_long_side(original_rgb, DEFAULT_CONFIG.target_long_side)
        ref_tensor = self._preprocess(ref_small).to(device)

        if ref_tensor.ndim == 3:
            ref_tensor = ref_tensor.unsqueeze(0)

        return DreamSimReference(image=ref_small, tensor=ref_tensor)

    def score(self, reference: DreamSimReference, candidate_png: bytes) -> float:
        self._load_dependencies()
        assert self._preprocess is not None
        assert self._model is not None
        assert self._torch is not None

        cand = Image.open(io.BytesIO(candidate_png)).convert("RGB")
        if cand.size != reference.image.size:
            cand = cand.resize(reference.image.size, resample=Image.Resampling.BILINEAR)

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
