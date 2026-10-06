"""Representation and alpha metrics used by the structural planner."""

import numpy as np
import pytest

from vectrify.document import import_svg
from vectrify.refine.cel_plan.score import (
    foreground_mask,
    measurements,
    representation,
)


def test_compound_paths_cannot_hide_contour_cost():
    document = import_svg('<svg><path d="M0 0 L3 0 L3 3 Z M6 0 L9 0 L9 3 Z"/></svg>')
    stats = representation(document)
    assert stats.paths == 1
    assert stats.contours == 2
    assert stats.nodes == 6
    assert stats.cost == 16


def test_unused_definitions_do_not_count_as_visible_paths():
    document = import_svg(
        '<svg><defs><path id="unused" d="M0 0 L3 0 L3 3 Z"/></defs>'
        '<path d="M0 0 L2 0 L2 2 Z"/></svg>'
    )
    assert representation(document).paths == 1


def test_rgb_and_alpha_measurements_do_not_lose_transparent_spill():
    truth = np.zeros((16, 16, 4), dtype=np.float32)
    truth[5:11, 5:11] = (1, 1, 1, 1)
    actual = truth.copy()
    actual[3:5, 5:11] = (1, 1, 1, 1)
    measured = measurements(actual, truth, foreground_mask(truth))
    assert measured["mse"] == 0  # White spill is invisible on a white backdrop.
    assert measured["alpha_spill_pixels"] == 12
    assert measured["alpha_iou"] == pytest.approx(0.75)
    assert measured["premultiplied_error"] > 0


def test_local_feature_error_cannot_be_diluted_by_the_canvas():
    truth = np.ones((100, 100, 4), dtype=np.float32)
    actual = truth.copy()
    actual[20:22, 20:22, :3] = 0
    score = measurements(
        actual,
        truth,
        foreground_mask(truth),
        features={"mark": (20, 20, 2, 2)},
    )
    assert score["features"]["mark"] == pytest.approx(255**2)
    assert score["mse"] == pytest.approx(255**2 * 4 / 10_000)


def test_empty_reference_is_measurable():
    image = np.zeros((5, 5, 4), dtype=np.float32)
    score = measurements(image, image, foreground_mask(image))
    assert score["mse"] == 0
    assert score["alpha_iou"] == 1
