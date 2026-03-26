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


def lab_l1(a_rgb: Image.Image, b_rgb: Image.Image) -> float:
    t = _rgb_to_lab_transform()
    a_lab = ImageCms.applyTransform(a_rgb, t)
    b_lab = ImageCms.applyTransform(b_rgb, t)
    if a_lab is None or b_lab is None:
        raise RuntimeError("ImageCms.applyTransform returned None")
    diff = ImageChops.difference(a_lab, b_lab)
    stat = ImageStat.Stat(diff)
    mean_abs = float(sum(stat.mean) / 3.0)  # [0..255]
    return mean_abs / 255.0
