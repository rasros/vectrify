"""Structural distance over gradient magnitude maps.

Pixel distance is dominated by area: a background off by 3% outweighs a shape
that is missing entirely, because the background is most of the pixels and the
shape is not. The vision score is mostly an embedding cosine, and the model
keys on structure, so pixel distance is weakest exactly where the objective is
strongest.

An edge map throws away flat regions and keeps the boundaries, which makes this
close to orthogonal to colour error rather than a second opinion on it.

The maps are compared by overlap rather than by difference. A plain per-pixel
difference makes erasing structure cheaper than getting it slightly wrong -- a
blank canvas has no boundaries to mismatch, so it scores better than a shape
that is present and two pixels off. Measured, that inverted the metric: it
correlated with the vision score at -0.50, actively rewarding the search for
deleting things. Overlap has no such hole, because a missing boundary overlaps
nothing and costs the maximum.

The comparison is still per pixel, so it saturates: once two boundaries are
more than about an edge width apart their maps no longer overlap and moving
further costs nothing more. It reports whether the structure lines up, not how
far away it is, which is why it is a companion to a distance that degrades
smoothly and not a replacement for one.
"""

import io

import numpy as np
from PIL import Image, ImageFilter

from vectrify.score.utils import clamp01, lab_array

# Gradient magnitudes of a Lab L channel in [0, 255] run to a few tens at a hard
# edge. Scaling by this before clamping keeps a fully-wrong edge map near 1.0
# instead of saturating on the first strong boundary.
_EDGE_SCALE = 32.0

# How far a boundary may be off before it stops counting as the same boundary.
# Edges are a pixel or two wide, so without this the overlap is all-or-nothing
# and a candidate two pixels out scores the same as one drawn somewhere else.
_TOLERANCE_PX = 2.0


def edge_map(image_rgb: Image.Image) -> np.ndarray:
    """Gradient magnitude of the L channel, normalised to roughly [0, 1].

    Central differences rather than a Sobel convolution: one numpy call per
    axis, no kernel, and the extra smoothing a Sobel gives is not worth the
    cost when the map is about to be compared cell by cell.
    """
    lightness = lab_array(image_rgb)[:, :, 0]
    gy, gx = np.gradient(lightness)
    magnitude = np.clip(np.hypot(gx, gy) / _EDGE_SCALE, 0.0, 1.0)

    blurred = Image.fromarray((magnitude * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(_TOLERANCE_PX)
    )
    return np.asarray(blurred, dtype=np.float32) / 255.0


def edge_score(reference_rgb: Image.Image, candidate_png: bytes) -> float:
    """One minus the overlap between the two images' edge maps.

    Zero when the structure matches, however wrong the colours are; one when
    the candidate shares no boundary with the target, which is what a blank
    canvas gets.
    """
    candidate = Image.open(io.BytesIO(candidate_png)).convert("RGB")
    if candidate.size != reference_rgb.size:
        candidate = candidate.resize(
            reference_rgb.size, resample=Image.Resampling.BILINEAR
        )
    reference_edges = edge_map(reference_rgb)
    candidate_edges = edge_map(candidate)

    total = float(reference_edges.sum() + candidate_edges.sum())
    if total == 0.0:
        # Neither image has a boundary anywhere: two flat canvases do match.
        return 0.0
    shared = float(np.minimum(reference_edges, candidate_edges).sum())
    return clamp01(1.0 - 2.0 * shared / total)
