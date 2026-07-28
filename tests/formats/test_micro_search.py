"""Tests for the shared micro-search loop.

Uses a fake rasterizer that maps content strings to flat-color renders, so the
selection logic is exercised without any real renderer.
"""

import pytest
from PIL import Image

from tests.helpers import make_png
from vectrify.formats.micro_search import MAX_DISTANCE, fast_lab_l1, with_micro_search

TARGET = Image.new("RGB", (32, 32), color="blue")


def _fake_rasterize(content: str) -> bytes | None:
    """Render "<color>" to a flat image; anything else fails to render."""
    if not content.startswith("color:"):
        return None
    return make_png(content.removeprefix("color:"), 32)


def test_picks_closest_candidate():
    yields = [("color:red", "bad"), ("color:blue", "good")]
    best, summary = with_micro_search(
        lambda: yields.pop(0),
        fallback="color:red",
        rasterize=_fake_rasterize,
        orig_img_fast=TARGET,
        num_trials=2,
    )
    assert best == "color:blue"
    assert summary == "good"


def test_candidates_equal_to_fallback_are_skipped():
    best, summary = with_micro_search(
        lambda: ("color:blue", "same"),
        fallback="color:blue",
        rasterize=_fake_rasterize,
        orig_img_fast=TARGET,
        num_trials=3,
        default_summary="No change",
    )
    assert best == "color:blue"
    assert summary == "No change"


def test_returns_best_candidate_even_if_worse_than_fallback():
    best, summary = with_micro_search(
        lambda: ("color:red", "worse"),
        fallback="color:blue",
        rasterize=_fake_rasterize,
        orig_img_fast=TARGET,
        num_trials=2,
        default_summary="No change",
    )
    assert best == "color:red"
    assert summary == "worse"


def test_unrenderable_candidates_are_ignored():
    best, summary = with_micro_search(
        lambda: ("not renderable", "invalid"),
        fallback="color:red",
        rasterize=_fake_rasterize,
        orig_img_fast=TARGET,
        num_trials=3,
        default_summary="No change",
    )
    assert best == "color:red"
    assert summary == "No change"


def test_num_trials_bounds_the_generator():
    calls = []

    def _op():
        calls.append(1)
        return "color:red", "x"

    with_micro_search(
        _op,
        fallback="color:blue",
        rasterize=_fake_rasterize,
        orig_img_fast=TARGET,
        num_trials=4,
    )
    assert len(calls) == 4


def test_fast_lab_l1_identical_is_zero():
    assert fast_lab_l1(TARGET, make_png("blue", 32)) == pytest.approx(0.0)


def test_fast_lab_l1_unreadable_png_is_max_distance():
    assert fast_lab_l1(TARGET, b"not a png") == MAX_DISTANCE


def test_fast_lab_l1_compares_whole_image_despite_aspect_mismatch():
    """A render whose aspect differs must still be compared in full.

    PIL crops to the overlap when sizes differ, so without resizing, a
    candidate that only covers part of the target would have the rest of
    the target silently excluded from the score.
    """
    # Target: top half blue, bottom half red.
    target = Image.new("RGB", (32, 32), color="blue")
    for y in range(16, 32):
        for x in range(32):
            target.putpixel((x, y), (255, 0, 0))

    # Candidate is all blue and half as tall — it matches the target's top
    # half exactly but gets the bottom half completely wrong.
    wrong_bottom = make_png("blue", (32, 16))
    assert fast_lab_l1(target, wrong_bottom) > 0.1
