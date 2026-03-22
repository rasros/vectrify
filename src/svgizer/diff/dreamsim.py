import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .base import DEFAULT_CONFIG
from .utils import get_device, lab_l1, resize_long_side

_DREAMSIM_MODEL = None
_DREAMSIM_PREPROCESS = None


def get_dreamsim_models():
    global _DREAMSIM_MODEL, _DREAMSIM_PREPROCESS
    if _DREAMSIM_MODEL is None:
        from dreamsim import dreamsim

        device = get_device()
        _DREAMSIM_MODEL, _DREAMSIM_PREPROCESS = dreamsim(pretrained=True, device=device)
        _DREAMSIM_MODEL.eval()
    return _DREAMSIM_MODEL, _DREAMSIM_PREPROCESS


@dataclass
class DreamSimReference:
    image: Image.Image
    tensor: Any  # torch.Tensor


class DreamSimScorer:
    """The original ML-based layout/semantic scorer."""

    def prepare_reference(self, original_rgb: Image.Image) -> DreamSimReference:
        _, preprocess = get_dreamsim_models()
        device = get_device()

        ref_small = resize_long_side(original_rgb, DEFAULT_CONFIG.target_long_side)
        ref_tensor = preprocess(ref_small).to(device)

        if ref_tensor.ndim == 3:
            ref_tensor = ref_tensor.unsqueeze(0)

        return DreamSimReference(image=ref_small, tensor=ref_tensor)

    def score(self, reference: DreamSimReference, candidate_png: bytes) -> float:
        import torch

        cand = Image.open(io.BytesIO(candidate_png)).convert("RGB")

        if cand.size != reference.image.size:
            cand = cand.resize(reference.image.size, resample=Image.BILINEAR)

        model, preprocess = get_dreamsim_models()
        device = get_device()

        cand_t = preprocess(cand).to(device)
        if cand_t.ndim == 3:
            cand_t = cand_t.unsqueeze(0)

        with torch.no_grad():
            dist_tensor = model(reference.tensor, cand_t)
            dreamsim_dist = float(dist_tensor.item())

        struct_score = max(0.0, min(1.0, dreamsim_dist))
        color_score = float(max(0.0, min(1.0, lab_l1(reference.image, cand))))

        score = (DEFAULT_CONFIG.w_dreamsim * struct_score) + (
            DEFAULT_CONFIG.w_color * color_score
        )

        if not np.isfinite(score):
            return 1.0
        return float(max(0.0, min(1.0, score)))
