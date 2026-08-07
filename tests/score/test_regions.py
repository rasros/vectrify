import io

import numpy as np
import pytest
from PIL import Image

from vectrify.score.base import Scorer
from vectrify.score.regions import (
    REGION_GRID,
    block_distance_grid,
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
    """A white canvas with one small dark square — a localised defect."""
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
    """The whole point: a tiny defect must not be averaged away."""
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
    """The reference is resized to a long side that rarely divides by 27."""
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
    """The fallback path must still yield the objective."""
    scorer = SimpleFallbackScorer()
    ref = scorer.prepare_reference(_white())
    grid = scorer.region_distance_grid(ref, _png(_white_with_blot()))

    assert grid is not None
    assert grid.shape == REGION_GRID
    assert worst_region_score(grid) > 0.0


def test_region_grid_is_none_without_a_reference_image():
    """None means 'not measured' — callers must not read it as a zero distance."""

    class _Refless(Scorer):
        def prepare_reference(self, original_rgb):  # noqa: ARG002 - override contract
            return object()

        def score(self, reference, candidate_png):  # noqa: ARG002 - override contract
            return 0.0

    assert _Refless().region_distance_grid(object(), _png(_white())) is None


def test_localised_defect_scores_worse_than_a_faint_global_one():
    """End-to-end: the case the metric exists to catch.

    A small hard defect and a faint wash carry comparable total error, so the
    primary colour score barely separates them. worst_region must.
    """
    scorer = SimpleFallbackScorer()
    ref = scorer.prepare_reference(_white())

    local_grid = scorer.region_distance_grid(ref, _png(_white_with_blot()))
    washed_grid = scorer.region_distance_grid(
        ref, _png(Image.new("RGB", (128, 128), (247, 247, 247)))
    )
    assert local_grid is not None
    assert washed_grid is not None

    assert worst_region_score(local_grid) > worst_region_score(washed_grid)
