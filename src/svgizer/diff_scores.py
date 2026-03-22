import io
import logging
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageChops, ImageStat

log = logging.getLogger(__name__)


class DiffScorer(Protocol):
    """Protocol for algorithms that score image differences."""

    def prepare_reference(self, original_rgb: Image.Image) -> Any:
        """Pre-processes the reference image into the format required by the scorer."""
        ...

    def score(self, reference: Any, candidate_png: bytes) -> float:
        """Returns a difference score between 0.0 (identical) and 1.0 (completely different)."""
        ...


@dataclass(frozen=True)
class ScoreConfig:
    target_long_side: int = 256
    w_dreamsim: float = 0.85
    w_color: float = 0.15


_CFG = ScoreConfig()
_DREAMSIM_MODEL = None
_DREAMSIM_PREPROCESS = None


def _get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_dreamsim_models():
    global _DREAMSIM_MODEL, _DREAMSIM_PREPROCESS
    if _DREAMSIM_MODEL is None:
        from dreamsim import dreamsim

        device = _get_device()
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
    return im.resize((max(1, new_w), max(1, new_h)), resample=Image.BILINEAR)


def _lab_l1(a_rgb: Image.Image, b_rgb: Image.Image) -> float:
    a_lab = a_rgb.convert("LAB")
    b_lab = b_rgb.convert("LAB")
    diff = ImageChops.difference(a_lab, b_lab)
    stat = ImageStat.Stat(diff)
    mean_abs = float(sum(stat.mean) / 3.0)  # [0..255]
    return mean_abs / 255.0


@dataclass
class DreamSimReference:
    image: Image.Image
    tensor: Any  # torch.Tensor


class DreamSimScorer:
    """The original ML-based layout/semantic scorer."""

    def prepare_reference(self, original_rgb: Image.Image) -> DreamSimReference:
        _, preprocess = _get_dreamsim_models()
        device = _get_device()

        ref_small = _resize_long_side(original_rgb, _CFG.target_long_side)
        ref_tensor = preprocess(ref_small).to(device)

        if ref_tensor.ndim == 3:
            ref_tensor = ref_tensor.unsqueeze(0)

        return DreamSimReference(image=ref_small, tensor=ref_tensor)

    def score(self, reference: DreamSimReference, candidate_png: bytes) -> float:
        import torch

        cand = Image.open(io.BytesIO(candidate_png)).convert("RGB")

        if cand.size != reference.image.size:
            cand = cand.resize(reference.image.size, resample=Image.BILINEAR)

        model, preprocess = _get_dreamsim_models()
        device = _get_device()

        cand_t = preprocess(cand).to(device)
        if cand_t.ndim == 3:
            cand_t = cand_t.unsqueeze(0)

        with torch.no_grad():
            dist_tensor = model(reference.tensor, cand_t)
            dreamsim_dist = float(dist_tensor.item())

        struct_score = max(0.0, min(1.0, dreamsim_dist))
        color_score = float(max(0.0, min(1.0, _lab_l1(reference.image, cand))))

        score = (_CFG.w_dreamsim * struct_score) + (_CFG.w_color * color_score)

        if not np.isfinite(score):
            return 1.0
        return float(max(0.0, min(1.0, score)))


@dataclass
class SimpleReference:
    image: Image.Image


class SimpleFallbackScorer:
    """A fast, CPU-bound fallback relying solely on LAB color distance."""

    def __init__(self, target_long_side: int = 256):
        self.target_long_side = target_long_side

    def prepare_reference(self, original_rgb: Image.Image) -> SimpleReference:
        ref_small = _resize_long_side(original_rgb, self.target_long_side)
        return SimpleReference(image=ref_small)

    def score(self, reference: SimpleReference, candidate_png: bytes) -> float:
        cand = Image.open(io.BytesIO(candidate_png)).convert("RGB")

        if cand.size != reference.image.size:
            cand = cand.resize(reference.image.size, resample=Image.BILINEAR)

        color_score = _lab_l1(reference.image, cand)

        if not np.isfinite(color_score):
            return 1.0
        return float(max(0.0, min(1.0, color_score)))


def get_default_scorer() -> DiffScorer:
    """Factory that attempts to use ML models but falls back gracefully."""
    try:
        # Test loading the model to catch missing dependencies or GPU errors early
        _get_dreamsim_models()
        log.info("DreamSim models loaded successfully. Using DreamSimScorer.")
        return DreamSimScorer()
    except Exception as e:
        log.warning(
            f"DreamSim unavailable ({e}). Falling back to SimpleFallbackScorer."
        )
        return SimpleFallbackScorer()
