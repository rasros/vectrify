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

import hashlib
import io
import math

import numpy as np
from PIL import Image

from vectrify.score.utils import lab_array

# Matches the 27x27 patch grid of the default SigLIP model, so the fallback and
# the vision path divide the canvas the same way and their values stay
# comparable across a resume that switches scorers.
REGION_GRID: tuple[int, int] = (27, 27)


def _spaced(extent: int, box: int, count: int) -> tuple[list[int], int]:
    """*count* boxes of *box* px spaced evenly across *extent*.

    The one splitting rule. Whatever is left over after the boxes are placed
    becomes extra overlap between them, so every box keeps its exact size and
    the last one lands flush against the far edge -- no box is stretched and no
    strip of canvas goes unmeasured.
    """
    box = max(1, min(box, extent))
    if count <= 1:
        return [0], box
    span = extent - box
    return [round(i * span / (count - 1)) for i in range(count)], box


def tile_boxes(
    size: tuple[int, int], tile_size: int, overlap: float
) -> list[tuple[int, int, int, int]]:
    """Native-resolution crops of exactly *tile_size*, covering the whole image.

    Crops are cut at the size the model consumes, so nothing is ever resampled
    on the way in -- no interpolated pixels, no resampling ringing along the
    thin strokes this is meant to measure. An artifact introduced by scaling is
    error the candidate did not commit, and the scorer cannot tell the two
    apart.

    The count follows from the image rather than being chosen: as many crops as
    it takes to cover the canvas at no less than *overlap*. When the size does
    not divide evenly the remainder is absorbed by overlapping more, never by
    stretching a crop or by cropping the edge away.

    The single geometry both consumers use -- the scorer embeds these boxes and
    the region objective reads the distances they produce -- so "the worst
    region" always names something the score separately measured.
    """
    if tile_size < 1:
        raise ValueError(f"tile_size must be >= 1, got {tile_size}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    def axis(extent: int) -> tuple[list[int], int]:
        # Smaller than one crop: there is nothing to cut, and padding out to
        # the model size would invent content. The caller resizes this case.
        if extent <= tile_size:
            return [0], extent
        stride = max(1, round(tile_size * (1.0 - overlap)))
        count = math.ceil((extent - tile_size) / stride) + 1
        return _spaced(extent, tile_size, count)

    xs, step_x = axis(size[0])
    ys, step_y = axis(size[1])
    return [(x, y, x + step_x, y + step_y) for y in ys for x in xs]


def grid_boxes(size: tuple[int, int], cells: int) -> list[tuple[int, int, int, int]]:
    """*cells* x *cells* boxes over the image, via the same geometry.

    For measures with no model resolution to respect (the pixel fallback), the
    box size is chosen from a cell count instead of from the model input. Same
    splitting rule underneath, so there is still only one way the canvas gets
    divided. Overlap is not a parameter here: with the count fixed and the box
    derived from it, the boxes tile the canvas exactly.
    """
    if cells < 1:
        raise ValueError(f"cells must be >= 1, got {cells}")

    def axis(extent: int) -> tuple[list[int], int]:
        return _spaced(extent, max(1, round(extent / cells)), cells)

    xs, step_x = axis(size[0])
    ys, step_y = axis(size[1])
    return [(x, y, x + step_x, y + step_y) for y in ys for x in xs]


def tile_key(index: int, tile: Image.Image) -> bytes:
    """Cache key for one tile's distance.

    The reference is fixed for a run, so a tile's distance depends only on its
    own pixels -- which makes the distance a cacheable scalar rather than
    something that has to be recomputed per candidate. The index is part of the
    key because identical pixels at a different position are compared against a
    different reference tile.
    """
    return hashlib.blake2b(
        index.to_bytes(4, "little") + tile.tobytes(), digest_size=16
    ).digest()


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

    # Same tile_boxes geometry the vision path scores on, at overlap 0 -- there
    # is no model resolution limit on a pixel measure, so this can afford a
    # finer grid than the scored tiles without needing a second way to divide
    # the canvas.
    h, _ = grid_hw
    boxes = grid_boxes(reference_rgb.size, h)
    values = [float(diff[y0:y1, x0:x1].mean()) for x0, y0, x1, y1 in boxes]
    side = math.isqrt(len(values))
    grid = np.array(values, dtype=np.float64)
    return grid.reshape(side, side) if side * side == len(values) else grid
