from __future__ import annotations

import io
from PIL import Image, ImageChops, ImageStat


def pixel_diff_score(original_rgb: Image.Image, candidate_png: bytes) -> float:
    """
    Mean absolute RGB diff normalized to [0, 1]. Lower is better.
    Uses PIL ops for speed/clarity (no numpy dependency).
    """
    cand = Image.open(io.BytesIO(candidate_png)).convert("RGB")
    if cand.size != original_rgb.size:
        cand = cand.resize(original_rgb.size, resample=Image.BILINEAR)

    diff = ImageChops.difference(original_rgb, cand)
    stat = ImageStat.Stat(diff)
    # stat.mean is per-channel mean absolute difference in [0..255]
    mean_abs = (stat.mean[0] + stat.mean[1] + stat.mean[2]) / 3.0
    return float(mean_abs / 255.0)
