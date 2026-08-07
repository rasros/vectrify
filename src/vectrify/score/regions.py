"""Region-level fidelity: *where* a candidate is wrong, not just how wrong overall.

The primary score averages error across the whole canvas, so a small defect is
diluted until it cannot be selected for. On a 700x700 drawing a misdrawn bill
covers roughly 900 dark pixels out of 490,000; closing its contours moves the
global score less than a coordinate nudge somewhere else does, so every
candidate that fixes it is dominated and discarded. The search is not ignoring
the defect -- it is correctly optimising an objective that barely mentions it.

``worst_region`` restores that pressure by collapsing a grid of per-region
distances into the mean of its worst *k*. Nothing has to name a region up
front: whichever tiles are worst *are* the region, recomputed per candidate, so
the objective follows the defect around instead of being pinned to a box. Fix
the worst area and this improves; polish an already-good area and it does not.

Two grids feed it, with the same shape either way so the objective means the
same thing across scorers:

- SigLIP patch cosine distances, when a vision scorer is loaded. This is the
  granularity the model itself reasons at and it is already being computed for
  the diff heatmap, so it costs nothing extra.
- Block-wise Lab L1 otherwise, so ``--scorer simple`` (and any run without
  torch installed) still produces the objective. A metric that silently
  vanished on the fallback path would read as 0.0 -- the *best* possible value
  for a minimised objective -- and would quietly hand untested candidates a
  perfect score on it.
"""

import io

import numpy as np
from PIL import Image

from vectrify.score.utils import lab_array

# Matches the 27x27 patch grid of the default SigLIP model, so the fallback and
# the vision path divide the canvas the same way and their values stay
# comparable across a resume that switches scorers.
REGION_GRID: tuple[int, int] = (27, 27)

# Fraction of regions averaged into the metric. At the default grid this is 7 of
# 729 tiles, about the footprint of one small drawn element. Too large and the
# defect is diluted again; too small and the objective chases single-tile noise.
WORST_FRACTION = 0.01
MIN_WORST_REGIONS = 4


def worst_k(n_regions: int) -> int:
    """How many of the worst regions to average, given a grid size."""
    return max(MIN_WORST_REGIONS, round(n_regions * WORST_FRACTION))


def worst_region_score(grid: np.ndarray) -> float:
    """Mean distance over the worst *k* regions of *grid*.

    Deliberately not the single maximum: one tile is noisy enough that the
    objective would reward luck over improvement, and a defect worth fixing
    spans several tiles anyway.
    """
    flat = np.asarray(grid, dtype=np.float64).ravel()
    if flat.size == 0:
        return 0.0
    k = min(worst_k(flat.size), flat.size)
    # argpartition beats a full sort: only the top k need to be correct.
    worst = np.partition(flat, -k)[-k:]
    value = float(worst.mean())
    return value if np.isfinite(value) else 0.0


def block_distance_grid(
    reference_rgb: Image.Image,
    candidate_png: bytes,
    grid_hw: tuple[int, int] = REGION_GRID,
) -> np.ndarray:
    """Per-block Lab L1 distance, the no-torch stand-in for patch distances.

    Lab rather than RGB so the blocks are weighted the way the primary colour
    term already is, keeping the fallback consistent with the score it sits
    beside.
    """
    candidate = Image.open(io.BytesIO(candidate_png)).convert("RGB")
    if candidate.size != reference_rgb.size:
        candidate = candidate.resize(
            reference_rgb.size, resample=Image.Resampling.BILINEAR
        )

    diff = np.abs(lab_array(reference_rgb) - lab_array(candidate)).mean(axis=2) / 255.0

    # array_split rather than a reshape: the reference is resized to a long side
    # that rarely divides by 27, and uneven blocks are better than cropping the
    # remainder away, which would blind the metric to one edge of the canvas.
    h, w = grid_hw
    return np.array(
        [
            [block.mean() for block in np.array_split(row, w, axis=1)]
            for row in np.array_split(diff, h, axis=0)
        ],
        dtype=np.float64,
    )
