"""Region geometry and the derived objectives.

The canvas is split into an even grid and each cell scored on its own, which
is what lets a small defect in a mostly-correct render cost something: the
whole-image score is an average, and averages forgive exactly the localised
errors worth fixing.
"""

import io

import numpy as np
from PIL import Image

from vectrify.score.utils import lab_array


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


# A defect worth fixing spans several cells, and a single cell is noisy enough
# that taking the maximum would reward luck over improvement.
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


# Quarters catch a whole area being wrong; sixteenths catch a localised defect.
# They disagree in exactly the cases a single scale handled badly, so both are
# objectives.
REGION_SCALES: tuple[int, ...] = (2, 4)


def region_worst_scores(
    reference_rgb: Image.Image,
    candidate_png: bytes,
    scales: tuple[int, ...] = REGION_SCALES,
) -> dict[int, float]:
    """worst_region at each scale, from one Lab difference.

    The per-pixel difference is the expensive part, so it is computed once and
    reduced at every scale rather than re-read per grid.
    """
    candidate = Image.open(io.BytesIO(candidate_png)).convert("RGB")
    if candidate.size != reference_rgb.size:
        candidate = candidate.resize(
            reference_rgb.size, resample=Image.Resampling.BILINEAR
        )
    diff = np.abs(lab_array(reference_rgb) - lab_array(candidate)).mean(axis=2) / 255.0

    out: dict[int, float] = {}
    for cells in scales:
        boxes = grid_boxes(reference_rgb.size, cells)
        values = np.array(
            [float(diff[y0:y1, x0:x1].mean()) for x0, y0, x1, y1 in boxes],
            dtype=np.float64,
        )
        out[cells] = worst_region_score(values)
    return out


def complexity_ratio(
    complexity: float,
    score: float,
    blank_error: float,
    min_gain_fraction: float = 0.5,
    ceiling: float = 1e6,
) -> float:
    """Complexity charged against the error it removes.

    Raw complexity cannot be an objective on its own: an empty canvas beats
    everything on it and so is never dominated, and it holds a pool slot and
    gets picked as a parent. Multiplying by quality does not help either --
    any complexity * f(score) tends to zero as complexity does, so the blank
    canvas wins by the largest margin. Only a denominator that vanishes for the
    blank canvas excludes it.

    Below *min_gain_fraction* of the available error the ratio is pinned to
    *ceiling*, which also rules out a single flat rectangle of the average
    colour: it earns a real gain and a fine ratio, but it is not raw material
    the search can build on, and it would otherwise absorb parent selections
    for a whole round.

    Never returns infinity: build_objectives normalises by the population
    maximum, so one infinite value would drive every other candidate's
    normalised value to zero and silently destroy the objective.
    """
    gain = blank_error - score
    if blank_error <= 0.0 or gain < min_gain_fraction * blank_error:
        return ceiling
    return min(complexity / gain, ceiling)
