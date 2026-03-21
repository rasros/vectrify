import io
from dataclasses import dataclass

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageChops, ImageStat

@dataclass(frozen=True)
class ScoreConfig:
    target_long_side: int = 256  # scoring resolution
    w_lpips: float = 0.85        # heavily weight structural similarity
    w_color: float = 0.15        # keep a small color check for palette accuracy


_CFG = ScoreConfig()
_LPIPS_MODEL = None


def _get_lpips_model():
    """Lazily load the LPIPS model per worker process."""
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        import lpips

        # Initialize AlexNet backbone and set to evaluation mode
        _LPIPS_MODEL = lpips.LPIPS(net='alex', spatial=False).eval()

        # Push to hardware acceleration if available
        if torch.cuda.is_available():
            _LPIPS_MODEL = _LPIPS_MODEL.to("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _LPIPS_MODEL = _LPIPS_MODEL.to("mps")

    return _LPIPS_MODEL


def _image_to_lpips_tensor(im: Image.Image) -> torch.Tensor:
    """Convert PIL Image to a normalized tensor suitable for LPIPS."""
    t = TF.to_tensor(im)  # [0, 1]
    t = t * 2.0 - 1.0     # Normalize to [-1, 1]
    t = t.unsqueeze(0)    # Add batch dimension -> [1, C, H, W]

    if torch.cuda.is_available():
        t = t.to("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        t = t.to("mps")

    return t


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
    and convert it to a PyTorch tensor ONCE.
    """
    # Ensure model is loaded so hardware device is detected
    _get_lpips_model()

    ref_small = _resize_long_side(original_rgb, _CFG.target_long_side)
    ref_tensor = _image_to_lpips_tensor(ref_small)

    return ScoringReference(image=ref_small, tensor=ref_tensor)


def pixel_diff_score(reference: ScoringReference, candidate_png: bytes) -> float:
    cand = Image.open(io.BytesIO(candidate_png)).convert("RGB")

    # Resize candidate to match the pre-computed reference
    if cand.size != reference.image.size:
        cand = cand.resize(reference.image.size, resample=Image.BILINEAR)

    # 1. Structural/Perceptual term via LPIPS
    model = _get_lpips_model()
    cand_t = _image_to_lpips_tensor(cand)

    with torch.no_grad():
        # reference.tensor is already on the GPU/MPS if available
        dist_tensor = model(reference.tensor, cand_t)
        lpips_dist = float(dist_tensor.item())

    struct_score = max(0.0, min(1.0, lpips_dist))

    # 2. Perceptual-ish color term in Lab
    color_score = float(max(0.0, min(1.0, _lab_l1(reference.image, cand))))

    score = (_CFG.w_lpips * struct_score) + (_CFG.w_color * color_score)

    if not np.isfinite(score):
        return 1.0
    return float(max(0.0, min(1.0, score)))