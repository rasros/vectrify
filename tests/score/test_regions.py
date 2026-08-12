import io
import math

import numpy as np
import pytest
from PIL import Image, ImageDraw

from vectrify.score.regions import (
    grid_boxes,
    worst_k,
    worst_region_score,
)


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _white(size: int = 128) -> Image.Image:
    return Image.new("RGB", (size, size), "white")


def _white_with_blot(size: int = 128, blot: int = 12) -> Image.Image:
    img = _white(size)
    img.paste(Image.new("RGB", (blot, blot), "black"), (4, 4))
    return img


# ── worst_region_score ────────────────────────────────────────────────────────


def test_worst_region_averages_only_the_worst_k():
    grid = np.zeros((27, 27))
    k = worst_k(grid.size)
    flat = grid.ravel()
    flat[:k] = 1.0
    grid = flat.reshape(27, 27)

    # Exactly k regions at 1.0 and the rest at 0.0 -> the mean of the worst k is 1.0.
    assert worst_region_score(grid) == pytest.approx(1.0)


def test_worst_region_ignores_a_good_global_average():
    localised = np.zeros((27, 27))
    localised.ravel()[: worst_k(729)] = 1.0

    spread = np.full((27, 27), localised.mean())

    # Both grids carry identical total error, so a global mean cannot tell them
    # apart; worst_region must rank the concentrated defect as far worse.
    assert spread.mean() == pytest.approx(localised.mean())
    assert worst_region_score(localised) > worst_region_score(spread) * 10


def test_worst_region_of_a_perfect_match_is_zero():
    assert worst_region_score(np.zeros((27, 27))) == 0.0


def test_worst_region_handles_an_empty_grid():
    assert worst_region_score(np.array([])) == 0.0


def test_worst_region_survives_non_finite_values():
    grid = np.zeros((27, 27))
    grid[0, 0] = np.nan
    assert np.isfinite(worst_region_score(grid))


def test_worst_k_has_a_floor_on_tiny_grids():
    assert worst_k(9) == 4
    assert worst_k(729) == 7


# ── scorer integration ────────────────────────────────────────────────────────


# ── tiling geometry ───────────────────────────────────────────────────────────


def test_grid_boxes_honours_the_requested_cell_count():
    assert len(grid_boxes((256, 256), 27)) == 27 * 27
    assert len(grid_boxes((100, 61), 27)) == 27 * 27


# ── raster snapping ───────────────────────────────────────────────────────────


# ── non-square rasters ────────────────────────────────────────────────────────


# ── objective metrics ─────────────────────────────────────────────────────────


def test_region_worst_scores_returns_every_requested_scale():
    from vectrify.score.compare import compare, prepare
    from vectrify.score.regions import region_worst_scores

    ref = Image.new("RGB", (64, 64), "white")
    cand = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(cand).rectangle([0, 0, 15, 15], fill="black")
    buf = io.BytesIO()
    cand.save(buf, format="PNG")

    scores = region_worst_scores(compare(prepare(ref), buf.getvalue()), ref.size)
    assert set(scores) == {2, 4}
    # A defect filling one sixteenth of the canvas is a whole cell at 4x4 and a
    # quarter of one at 2x2, so the finer scale must report it as worse. That
    # difference is the reason both scales are objectives.
    assert scores[4] > scores[2]


def test_complexity_ratio_excludes_the_blank_canvas():
    """Raw complexity puts an empty canvas on the front permanently: nothing
    beats it, so it is never dominated and it eats a pool slot."""
    from vectrify.score.regions import complexity_ratio

    blank_error = 0.2
    assert complexity_ratio(0.0, blank_error, blank_error) >= 1e6
    assert complexity_ratio(0.0, 0.5, blank_error) >= 1e6  # worse than blank


def test_complexity_ratio_pins_candidates_below_the_gain_floor():
    """A flat rectangle of the average colour earns a real gain and a fine
    ratio, but it is not raw material the search can build on."""
    from vectrify.score.regions import complexity_ratio

    blank_error = 0.2
    barely = complexity_ratio(1.0, blank_error * 0.6, blank_error)
    useful = complexity_ratio(1.0, blank_error * 0.2, blank_error)
    assert barely >= 1e6
    assert useful < 1e6


def test_complexity_ratio_orders_by_complexity_once_past_the_floor():
    from vectrify.score.regions import complexity_ratio

    blank_error, score = 0.2, 0.02
    lean = complexity_ratio(100.0, score, blank_error)
    bloated = complexity_ratio(900.0, score, blank_error)
    assert lean < bloated


def test_complexity_ratio_never_returns_infinity():
    """build_objectives normalises by the population maximum, so one infinite
    value would drive every other candidate's normalised value to zero and
    silently destroy the objective."""
    from vectrify.score.regions import complexity_ratio

    for score in (0.0, 0.2, 1.0):
        for blank in (0.0, 0.2):
            value = complexity_ratio(1e12, score, blank)
            assert math.isfinite(value)


def test_region_scores_read_colour_only():
    """The whole-canvas score blends structure in; the region metrics do not.
    Measured, structure-aware cells were neutral on the vision score and
    significantly worse on the round score -- a cell is small enough that edge
    overlap inside it is close to binary. This pins the decision so the two
    do not drift back together by accident."""
    import io

    from PIL import Image, ImageDraw

    from vectrify.score.compare import compare, prepare
    from vectrify.score.regions import region_worst_scores

    def png(image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    # Same colours everywhere, different structure: a recoloured-free redraw
    # that moves every boundary. Colour-only cells cannot tell these apart by
    # much, and that is the documented behaviour.
    reference = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(reference).rectangle((8, 8, 24, 24), fill="black")

    shifted = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(shifted).rectangle((10, 10, 26, 26), fill="black")

    prepared = prepare(reference)
    scores = region_worst_scores(compare(prepared, png(shifted)), (64, 64))

    # A colour-only cell reads a two-pixel shift as a small colour difference.
    assert 0.0 < scores[4] < 0.5
