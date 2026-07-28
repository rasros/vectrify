import io
from functools import cache

from PIL import Image, ImageChops, ImageCms, ImageStat


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
    t = _rgb_to_lab_transform()
    a_lab = ImageCms.applyTransform(a_rgb, t)
    b_lab = ImageCms.applyTransform(b_rgb, t)
    if a_lab is None or b_lab is None:
        raise RuntimeError("ImageCms.applyTransform returned None")
    diff = ImageChops.difference(a_lab, b_lab)
    stat = ImageStat.Stat(diff)
    mean_abs = float(sum(stat.mean) / 3.0)
    return mean_abs / 255.0
