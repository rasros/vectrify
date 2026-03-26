from PIL import Image, ImageChops, ImageStat


def get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def lab_l1(a_rgb: Image.Image, b_rgb: Image.Image) -> float:
    a_lab = a_rgb.convert("LAB")
    b_lab = b_rgb.convert("LAB")
    diff = ImageChops.difference(a_lab, b_lab)
    stat = ImageStat.Stat(diff)
    mean_abs = float(sum(stat.mean) / 3.0)  # [0..255]
    return mean_abs / 255.0
