"""Structural distance over gradient magnitude maps.

Pixel distance is dominated by area: a background off by 3% outweighs a shape
that is missing entirely, because the background is most of the pixels and the
shape is not. The vision score is mostly an embedding cosine, and the model
keys on structure, so pixel distance is weakest exactly where the objective is
strongest.

An edge map throws away flat regions and keeps the boundaries, which makes this
close to orthogonal to colour error rather than a second opinion on it.

The comparison is per pixel, so it saturates: once two boundaries are more than
about an edge width apart their maps no longer overlap and moving further costs
nothing more. It reports whether the structure lines up, not how far away it
is, which is why it is a companion to a distance that does degrade smoothly and
not a replacement for one.
"""

import io

import numpy as np
from PIL import Image

from vectrify.score.utils import clamp01, lab_array

# Gradient magnitudes of a Lab L channel in [0, 255] run to a few tens at a hard
# edge. Scaling by this before clamping keeps a fully-wrong edge map near 1.0
# instead of saturating on the first strong boundary.
_EDGE_SCALE = 32.0


def edge_map(image_rgb: Image.Image) -> np.ndarray:
    """Gradient magnitude of the L channel, normalised to roughly [0, 1].

    Central differences rather than a Sobel convolution: one numpy call per
    axis, no kernel, and the extra smoothing a Sobel gives is not worth the
    cost when the map is about to be compared cell by cell.
    """
    lightness = lab_array(image_rgb)[:, :, 0]
    gy, gx = np.gradient(lightness)
    return np.clip(np.hypot(gx, gy) / _EDGE_SCALE, 0.0, 1.0)


def edge_score(reference_rgb: Image.Image, candidate_png: bytes) -> float:
    """Mean absolute difference between the two images' edge maps.

    Zero when the structure matches, however wrong the colours are.
    """
    candidate = Image.open(io.BytesIO(candidate_png)).convert("RGB")
    if candidate.size != reference_rgb.size:
        candidate = candidate.resize(
            reference_rgb.size, resample=Image.Resampling.BILINEAR
        )
    difference = np.abs(edge_map(reference_rgb) - edge_map(candidate))
    return clamp01(float(difference.mean()))
