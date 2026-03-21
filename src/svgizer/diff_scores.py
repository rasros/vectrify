import io
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageChops, ImageStat

@dataclass(frozen=True)
class ScoreConfig:
    target_long_side: int = 256  # DreamSim handles specific resizing implicitly via preprocess
    w_dreamsim: float = 0.85     # Heavily weight semantic/structural layout
    w_color: float = 0.15        # Keep a small fallback for palette accuracy


_CFG = ScoreConfig()
_DREAMSIM_MODEL = None
_DREAMSIM_PREPROCESS = None


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_dreamsim_models():
    """Lazily load the DreamSim model and preprocessor per worker process."""
    global _DREAMSIM_MODEL, _DREAMSIM_PREPROCESS
    if _DREAMSIM_MODEL is None:
        from dreamsim import dreamsim

        device = _get_device()
        # Initialize the ensemble model (CLIP, DINO, OpenCLIP)
        _DREAMSIM_MODEL, _DREAMSIM_PREPROCESS = dreamsim(pretrained=True, device=device)
        _DREAMSIM_MODEL.eval()

    return _DREAMSIM_MODEL, _DREAMSIM_PREPROCESS


def _resize_long_side(im: Image.Image, long_side: int) -> Image.Image:
    w, h = im.size
    if max(w, h) <= long_side:
        return im
    if w >= h:
        new_w = long_side
        new_h = int(round(h * (long_side / float(w))))
    else:
        new_h = long_side
        new_w = int(round(w * (long_side / float(h))))
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    return im.resize((new_w, new_h), resample=Image.BILINEAR)


def _lab_l1(a_rgb: Image.Image, b_rgb: Image.Image) -> float:
    a_lab = a_rgb.convert("LAB")
    b_lab = b_rgb.convert("LAB")
    diff = ImageChops.difference(a_lab, b_lab)
    stat = ImageStat.Stat(diff)
    mean_abs = float(sum(stat.mean) / 3.0)  # [0..255]
    return mean_abs / 255.0


@dataclass
class ScoringReference:
    """Holds both the PIL image (for color checks) and the pre-computed GPU tensor."""
    image: Image.Image
    tensor: torch.Tensor


def get_scoring_reference(original_rgb: Image.Image) -> ScoringReference:
    """
    Pre-process the original image to the target scoring resolution
    and convert it to a PyTorch tensor ONCE using DreamSim's preprocessor.
    """
    # Ensure model is loaded to fetch the preprocessor
    _, preprocess = _get_dreamsim_models()
    device = _get_device()

    ref_small = _resize_long_side(original_rgb, _CFG.target_long_side)

    # DreamSim's preprocessor returns a normalized tensor.
    ref_tensor = preprocess(ref_small).to(device)

    # Ensure it has a batch dimension
    if ref_tensor.ndim == 3:
        ref_tensor = ref_tensor.unsqueeze(0)

    return ScoringReference(image=ref_small, tensor=ref_tensor)


def pixel_diff_score(reference: ScoringReference, candidate_png: bytes) -> float:
    cand = Image.open(io.BytesIO(candidate_png)).convert("RGB")

    # Resize candidate to match the pre-computed reference
    if cand.size != reference.image.size:
        cand = cand.resize(reference.image.size, resample=Image.BILINEAR)

    # 1. Structural/Perceptual term via DreamSim
    model, preprocess = _get_dreamsim_models()
    device = _get_device()

    cand_t = preprocess(cand).to(device)
    if cand_t.ndim == 3:
        cand_t = cand_t.unsqueeze(0)

    with torch.no_grad():
        dist_tensor = model(reference.tensor, cand_t)
        dreamsim_dist = float(dist_tensor.item())

    struct_score = max(0.0, min(1.0, dreamsim_dist))

    # 2. Perceptual-ish color term in Lab
    color_score = float(max(0.0, min(1.0, _lab_l1(reference.image, cand))))

    score = (_CFG.w_dreamsim * struct_score) + (_CFG.w_color * color_score)

    if not np.isfinite(score):
        return 1.0
    return float(max(0.0, min(1.0, score)))