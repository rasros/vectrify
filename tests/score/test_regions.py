import io
import math
from itertools import pairwise

import numpy as np
import pytest
from PIL import Image, ImageDraw

from vectrify.score.base import Scorer
from vectrify.score.regions import (
    REGION_GRID,
    block_distance_grid,
    crop_tile,
    grid_boxes,
    snap_raster,
    tile_boxes,
    tile_key,
    worst_k,
    worst_region_score,
)
from vectrify.score.simple import SimpleFallbackScorer


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


# ── block_distance_grid ───────────────────────────────────────────────────────


def test_block_grid_has_the_requested_shape():
    grid = block_distance_grid(_white(), _png(_white()))
    assert grid.shape == REGION_GRID


def test_block_grid_shape_holds_when_size_does_not_divide_evenly():
    grid = block_distance_grid(Image.new("RGB", (100, 61), "white"), _png(_white(100)))
    assert grid.shape == REGION_GRID


def test_block_grid_is_zero_for_an_identical_image():
    grid = block_distance_grid(_white(), _png(_white()))
    assert grid.max() == pytest.approx(0.0, abs=1e-6)


def test_block_grid_localises_a_defect():
    ref, cand = _white(), _white_with_blot()
    grid = block_distance_grid(ref, _png(cand))

    # The blot sits in the top-left corner, so that block must be the worst and
    # the far corner must be untouched.
    assert grid[0, 0] > 0.0
    assert grid[-1, -1] == pytest.approx(0.0, abs=1e-6)
    assert np.unravel_index(grid.argmax(), grid.shape)[0] < 5


def test_block_grid_resizes_a_mismatched_candidate():
    grid = block_distance_grid(_white(128), _png(_white(64)))
    assert grid.shape == REGION_GRID
    assert grid.max() == pytest.approx(0.0, abs=1e-6)


# ── scorer integration ────────────────────────────────────────────────────────


def test_simple_scorer_produces_a_region_grid_without_torch():
    scorer = SimpleFallbackScorer()
    ref = scorer.prepare_reference(_white())
    grid = scorer.region_distance_grid(ref, _png(_white_with_blot()))

    assert grid is not None
    assert grid.shape == REGION_GRID
    assert worst_region_score(grid) > 0.0


def test_region_grid_is_none_without_a_reference_image():

    class _Refless(Scorer):
        def prepare_reference(self, original_rgb):  # noqa: ARG002 - override contract
            return object()

        def score(self, reference, candidate_png):  # noqa: ARG002 - override contract
            return 0.0

    assert _Refless().region_distance_grid(object(), _png(_white())) is None


def test_localised_defect_scores_worse_than_a_faint_global_one():
    scorer = SimpleFallbackScorer()
    ref = scorer.prepare_reference(_white())

    local_grid = scorer.region_distance_grid(ref, _png(_white_with_blot()))
    washed_grid = scorer.region_distance_grid(
        ref, _png(Image.new("RGB", (128, 128), (247, 247, 247)))
    )
    assert local_grid is not None
    assert washed_grid is not None

    assert worst_region_score(local_grid) > worst_region_score(washed_grid)


# ── tiling geometry ───────────────────────────────────────────────────────────


def test_tiles_are_exactly_the_requested_size():
    for w, h in ((700, 700), (1024, 768), (801, 399)):
        for box in tile_boxes((w, h), 384, 0.5):
            assert box[2] - box[0] == 384
            assert box[3] - box[1] == 384


def test_tiles_cover_the_whole_canvas():
    boxes = tile_boxes((700, 700), 384, 0.5)
    assert min(b[0] for b in boxes) == 0
    assert min(b[1] for b in boxes) == 0
    assert max(b[2] for b in boxes) >= 700
    assert max(b[3] for b in boxes) >= 700


def test_uneven_sizes_become_extra_overlap_not_stretched_tiles():
    boxes = tile_boxes((701, 701), 384, 0.5)
    xs = sorted({b[0] for b in boxes})
    gaps = [b - a for a, b in pairwise(xs)]
    assert all(g <= 384 * 0.5 for g in gaps)  # at least the requested overlap
    assert all(b[2] - b[0] == 384 for b in boxes)


def test_image_smaller_than_a_tile_still_yields_a_full_size_box():
    assert tile_boxes((300, 300), 384, 0.5) == [(0, 0, 384, 384)]


def test_tile_count_follows_from_raster_size():
    counts = [len(tile_boxes((px, px), 384, 0.5)) for px in (384, 700, 1024, 1400)]
    assert counts == sorted(counts)
    assert counts[0] == 1
    assert len(set(counts)) > 1


def test_grid_boxes_honours_the_requested_cell_count():
    assert len(grid_boxes((256, 256), 27)) == 27 * 27
    assert len(grid_boxes((100, 61), 27)) == 27 * 27


def test_overlap_must_be_a_fraction():
    with pytest.raises(ValueError, match="overlap"):
        tile_boxes((700, 700), 384, 1.0)
    with pytest.raises(ValueError, match="overlap"):
        tile_boxes((700, 700), 384, -0.1)


def test_tile_key_distinguishes_position():
    tile = _white(64)
    assert tile_key(0, tile) != tile_key(1, tile)
    assert tile_key(0, tile) == tile_key(0, _white(64))


def test_tile_key_distinguishes_content():
    assert tile_key(0, _white(64)) != tile_key(0, _white_with_blot(64, 8))


# ── raster snapping ───────────────────────────────────────────────────────────


def test_snap_raster_rounds_up_to_whole_crops():
    assert snap_raster(512, 384) == 768
    assert snap_raster(700, 384) == 768
    assert snap_raster(768, 384) == 768
    assert snap_raster(1000, 384) == 1152


def test_snap_raster_never_goes_below_one_crop():
    assert snap_raster(100, 384) == 384
    assert snap_raster(1, 384) == 384


def test_snapped_rasters_tile_with_uniform_coverage():
    """The reason for snapping: position must carry no weight.

    Overlapping crops count a centre pixel up to 9x a corner one, which biases
    against exactly the edge detail this is meant to resolve.
    """
    for request in (512, 700, 1000, 1500):
        px = snap_raster(request, 384)
        coverage = np.zeros((px, px), dtype=int)
        for x0, y0, x1, y1 in tile_boxes((px, px), 384, 0.0):
            coverage[y0:y1, x0:x1] += 1
        assert coverage.min() == 1
        assert coverage.max() == 1


def test_overlapping_tiles_are_the_biased_case_snapping_avoids():
    px = 700
    coverage = np.zeros((px, px), dtype=int)
    for x0, y0, x1, y1 in tile_boxes((px, px), 384, 0.5):
        coverage[y0:y1, x0:x1] += 1
    assert coverage[0, 0] == 1
    assert coverage[px // 2, px // 2] > coverage[0, 0]


# ── non-square rasters ────────────────────────────────────────────────────────


def test_every_crop_is_tile_sized_at_any_aspect_ratio():
    for size in ((768, 480), (768, 576), (768, 269), (768, 128), (768, 768)):
        for box in tile_boxes(size, 384, 0.0):
            assert box[2] - box[0] == 384
            assert box[3] - box[1] == 384


def test_coverage_is_uniform_over_real_pixels_at_any_aspect_ratio():
    for w, h in ((768, 480), (768, 576), (768, 269), (1152, 864)):
        coverage = np.zeros((h, w), dtype=int)
        for x0, y0, x1, y1 in tile_boxes((w, h), 384, 0.0):
            coverage[y0 : min(y1, h), x0 : min(x1, w)] += 1
        assert coverage.min() == 1
        assert coverage.max() == 1


def test_crop_tile_pads_the_overhang_instead_of_returning_a_short_tile():
    img = _white(500)
    tile = crop_tile(img, (384, 384, 768, 768))
    assert tile.size == (384, 384)
    # The real pixels are white and so is the pad, so a padded crop compares
    # equal between reference and candidate and contributes nothing.
    assert tile.getpixel((0, 0)) == (255, 255, 255)
    assert tile.getpixel((383, 383)) == (255, 255, 255)


def test_crop_tile_leaves_in_bounds_boxes_untouched():
    img = _white_with_blot(500)
    box = (0, 0, 384, 384)
    assert crop_tile(img, box).tobytes() == img.crop(box).tobytes()


# ── objective metrics ─────────────────────────────────────────────────────────


def test_region_worst_scores_returns_every_requested_scale():
    from vectrify.score.regions import region_worst_scores

    ref = Image.new("RGB", (64, 64), "white")
    cand = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(cand).rectangle([0, 0, 15, 15], fill="black")
    buf = io.BytesIO()
    cand.save(buf, format="PNG")

    scores = region_worst_scores(ref, buf.getvalue())
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
