import io
from functools import cache

import numpy as np
from PIL import Image, ImageCms


def get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@cache
def _rgb_to_lab_transform() -> ImageCms.ImageCmsTransform:
    srgb = ImageCms.createProfile("sRGB")
    lab = ImageCms.createProfile("LAB")
    return ImageCms.buildTransformFromOpenProfiles(srgb, lab, "RGB", "LAB")


MAX_SCORE = 1.0


def lab_array(img_rgb: Image.Image) -> np.ndarray:
    """RGB image as a float32 Lab array, for per-pixel arithmetic.

    ``lab_l1`` collapses straight to a single mean; callers that need the
    spatial layout preserved (per-region distances) use this instead.
    """
    lab = ImageCms.applyTransform(img_rgb, _rgb_to_lab_transform())
    if lab is None:
        raise RuntimeError("ImageCms.applyTransform returned None")
    return np.asarray(lab, dtype=np.float32)


def clamp01(x: float) -> float:
    """Clamp to the [0, 1] score range."""
    return float(max(0.0, min(1.0, x)))


def color_score(reference_rgb: Image.Image, candidate_png: bytes) -> float:
    """Perceptual color distance between the reference and a candidate render.

    The candidate is resized to the reference's size, which lab_l1 requires.
    """
    candidate = Image.open(io.BytesIO(candidate_png)).convert("RGB")
    if candidate.size != reference_rgb.size:
        candidate = candidate.resize(
            reference_rgb.size, resample=Image.Resampling.BILINEAR
        )
    return clamp01(lab_l1(reference_rgb, candidate))


def lab_l1(a_rgb: Image.Image, b_rgb: Image.Image) -> float:
    """Mean absolute Lab difference, normalised to [0, 1].

    Computed in numpy rather than through ImageChops and ImageStat. PIL stores
    Lab's a and b channels offset-encoded, and that path reduces them to a
    *signed* mean, so opposite-sign chroma errors cancel between pixels and
    only lightness survives intact: a candidate half too blue and half too
    yellow used to read 0.0739 against a true 0.3484, barely distinguishable
    from being uniformly too blue. Colour distance was mostly a lightness
    distance.
    """
    return float(np.abs(lab_array(a_rgb) - lab_array(b_rgb)).mean()) / 255.0
