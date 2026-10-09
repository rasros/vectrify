"""Stroke support must come from ink within frozen raw-positive windows."""

from dataclasses import replace

import numpy as np
import pytest

from vectrify.refine.cel_plan.line_fidelity import SourceLineGuard, SourceProfile
from vectrify.refine.cel_plan.model import StageInterruptedError, Work
from vectrify.refine.cel_plan.score import render


def fixture():
    source = render(
        '<svg width="96" height="64"><path d="M0 0H96V64H0Z" fill="white"/>'
        '<path d="M12.5 30.5H80.5" fill="none" stroke="black" '
        'stroke-width="1"/></svg>',
        (96, 64),
    )
    profile = SourceProfile.at(((14.5, 30.5), (78.5, 30.5)), 3)
    guard = SourceLineGuard(source, (profile,))
    body = guard.fitting_body(profile, (0, 0, 96, 64))
    assert body is not None
    return source, profile, guard, body


def test_background_and_raw_bright_queries_cannot_supply_stroke_support():
    _, _, _, body = fixture()
    absent = np.zeros((64, 96))
    missing = body.observe(absent)
    assert missing["qualified_samples"] > 100
    assert missing["missing_samples"] == missing["qualified_samples"]
    assert missing["penalty"] > 0
    # These pixels are in the normal search window, but the raw source is white
    # there. A nearby stroke cannot borrow support from this bright region.
    wrong = absent.copy()
    wrong[32, :] = 1
    assert body.observe(wrong) == missing


def test_supported_displacement_and_actual_stroke_opacity_are_measured():
    _, _, _, body = fixture()
    alpha = np.zeros((64, 96))
    alpha[31, :] = 1
    # The raw-positive probes permit a small displacement even though literal
    # source centres lie in row 30 and this body supplies nothing at those cells.
    report = body.observe(alpha)
    assert report["missing_samples"] == 0
    assert report["penalty"] == 0
    faded = body.observe(alpha * 0.04)
    assert faded["missing_samples"] == faded["qualified_samples"]


def test_source_and_profile_mutation_cannot_change_frozen_body_queries():
    source, profile, guard, body = fixture()
    alpha = np.zeros((64, 96))
    alpha[31, :] = 1
    expected = body.observe(alpha)
    source[:] = 0
    profile.points.flags.writeable = True
    profile.points[:] += 20
    profile.direction.flags.writeable = True
    profile.direction[:] = (1, 0)
    assert body.observe(alpha) == expected
    fresh = guard.fitting_body(profile, (0, 0, 96, 64))
    assert fresh is not None
    assert fresh.observe(alpha) == expected
    assert guard.fitting_body(replace(profile), (0, 0, 96, 64)) is None


def test_native_crop_preserves_every_qualified_query_and_excludes_missing_edges():
    _, profile, guard, body = fixture()
    cropped = guard.fitting_body(profile, (20, 20, 70, 40))
    assert cropped is not None
    alpha = np.zeros((64, 96))
    alpha[30, :] = 1
    assert body.observe(alpha)["missing_samples"] == 0
    report = cropped.observe(alpha[20:40, 20:70])
    assert 0 < report["missing_samples"] < report["qualified_samples"]
    assert report["qualified_samples"] == body.observe(alpha)["qualified_samples"]


@pytest.mark.parametrize("bounds", [(0, 0, 97, 64), (0, 0, 96.5, 64), (1, 0, 1, 64)])
def test_invalid_native_crops_are_rejected(bounds):
    _, profile, guard, _ = fixture()
    with pytest.raises(ValueError, match="integer native crop"):
        guard.fitting_body(profile, bounds)


def test_bad_rasters_and_interruption_cannot_report_partial_support():
    _, profile, guard, body = fixture()
    for alpha in (np.zeros((4, 4)), np.full((64, 96), np.nan), np.ones((64, 96)) * 2):
        with pytest.raises(ValueError, match="finite aligned native alpha"):
            body.observe(alpha)
    with pytest.raises(StageInterruptedError):
        body.observe(np.ones((64, 96)), work=Work.start(0))
    with pytest.raises(StageInterruptedError):
        guard.fitting_body(profile, (0, 0, 96, 64), work=Work.start(0))
