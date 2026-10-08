"""Source-ended carrier fits retain paint, exact footprints and bounded work."""

from dataclasses import replace

import numpy as np
import pathops
import pytest

from tests.refine.test_cel_plan_ink import exterior_profile, line
from tests.refine.test_cel_plan_ink_models import drawing
from vectrify.document import Geometry
from vectrify.document.join import curve_path, transformed_geometry
from vectrify.document.svg import parse_path
from vectrify.refine.cel_plan import ink_carrier
from vectrify.refine.cel_plan.ink import Ink, measure
from vectrify.refine.cel_plan.ink_carrier import CarrierFit
from vectrify.refine.cel_plan.ink_models import carried, footprint
from vectrify.refine.cel_plan.model import Options, Work
from vectrify.refine.cel_plan.score import render


@pytest.mark.parametrize("frame", [(1, 1, 0, 0), (1.5, 0.75, 7.25, -3.5)])
def test_carrier_raster_preserves_native_phase_holes_and_tile_equivalence(
    frame, monkeypatch
):
    evidence, _ = drawing()
    sx, sy, ox, oy = frame
    evidence = replace(evidence, scale=(sx, sy), offset=(ox, oy))
    geometry = transformed_geometry(
        parse_path("M8 8H88V88H8Z M40 38H56V60H40Z"), (1 / sx, 0, 0, 1 / sy, ox, oy)
    )
    carrier = curve_path(geometry, "evenodd")
    fit = CarrierFit(evidence, carrier)
    actual = fit.raster(Work.start(10))
    expected = render(
        '<svg width="96" height="96" '
        f'viewBox="{ox} {oy} {96 / sx} {96 / sy}" preserveAspectRatio="none">'
        f'<path d="{geometry.path_data()}" fill="black" fill-rule="evenodd"/></svg>',
        (96, 96),
    )[..., 3]
    np.testing.assert_array_equal(actual, expected)
    assert not actual.flags.writeable
    assert actual[40:58, 42:54].max() == 0
    monkeypatch.setattr(ink_carrier, "MAX_CROP_PIXELS", 512)
    tiled = CarrierFit(evidence, carrier)
    np.testing.assert_array_equal(tiled.raster(Work.start(10)), actual)
    assert tiled.diagnostics["coverage_tiles"] > 1
    assert tiled.diagnostics["coverage_bytes"] == 96 * 96 * 4
    assert fit.raster(Work.start(10)) is actual


@pytest.mark.parametrize("bound", ["MAX_PIXELS", "MAX_MASK_BYTES", "MAX_TILES"])
def test_carrier_raster_bounds_precede_allocation_and_publish_no_mask(
    monkeypatch, bound
):
    evidence, _ = drawing()
    fit = CarrierFit(evidence, curve_path(parse_path("M8 8H88V88H8Z")))
    monkeypatch.setattr(ink_carrier, bound, 1)
    if bound == "MAX_TILES":
        monkeypatch.setattr(ink_carrier, "MAX_CROP_PIXELS", 512)
    monkeypatch.setattr(
        ink_carrier, "render", lambda *_: pytest.fail("Bounds reached render")
    )
    assert fit.raster(Work.start(10)) is None
    assert fit.coverage is None
    assert fit.diagnostics["bounds_exclusions"] == 1


def test_mid_tile_cancellation_does_not_cache_partial_carrier(monkeypatch):
    evidence, _ = drawing()
    fit = CarrierFit(evidence, curve_path(parse_path("M8 8H88V88H8Z")))
    original = ink_carrier.render
    work = Work.start(10)

    def cancelled(*args):
        pixels = original(*args)
        work.stop.set()
        return pixels

    monkeypatch.setattr(ink_carrier, "render", cancelled)
    assert fit.raster(work) is None
    assert fit.coverage is None
    assert fit.diagnostics["coverage_bytes"] == 0


@pytest.mark.parametrize("hole", [False, True])
def test_original_ended_butt_fit_needs_exact_unchanged_carrier_proof(hole):
    evidence, _ = drawing()
    run = np.array([[8.0, 40.0], [28.0, 40.0], [68.0, 40.0], [88.0, 40.0]])
    proof = Ink(run.copy(), 3, np.full(3, 32), 1, 0)
    carrier = curve_path(
        parse_path("M8 8H88V88H8Z" + (" M40 38H56V44H40Z" if hole else "")), "evenodd"
    )
    fit = CarrierFit(evidence, carrier)
    args = (
        run,
        proof,
        evidence,
        Options(),
        carrier,
        Work.start(10),
        False,
        {"runs": 0, "points": 0},
        None,
        ~evidence.empty,
    )
    assert carried(*args) == []
    offered = carried(*args, carrier_fit=fit)
    if hole:
        assert not offered
        return
    assert len(offered) == 1
    part, supported, model, ceiling, cap = offered[0]
    np.testing.assert_array_equal(part, run)
    np.testing.assert_array_equal(supported.points[[0, -1]], run[[0, -1]])
    np.testing.assert_array_equal(supported.paint, proof.paint)
    assert cap == "butt"
    assert supported.width == proof.width
    assert ceiling >= supported.width
    outside = pathops.op(
        curve_path(
            footprint(Geometry("run", (model.contour,)), supported.width, cap=cap)
        ),
        carrier,
        pathops.PathOp.DIFFERENCE,
    )
    assert abs(outside.area) <= 1e-8
    assert fit.diagnostics["recovered"] == 1
    assert fit.coverage is None  # Precision succeeds before allocating a mask.
    assert carried(*args, cap="round", carrier_fit=fit) == []
    assert carried(*args[:3], Options(line_width=3), *args[4:], carrier_fit=fit) == []


@pytest.mark.parametrize("alpha", [1, 0.5, 0.25])
def test_conditional_profile_uses_real_source_ink_and_keeps_source_ends(alpha):
    points, target, opacity = exterior_profile(alpha)
    visible = opacity > 8 / 255
    native = measure(points, target, 4, visible=visible, opacity=opacity)
    coverage = np.ones(target.shape[:2], np.float32)
    coverage[40] = 0
    changed = measure(
        points, target, 4, visible=visible, opacity=opacity, coverage=coverage
    )
    assert native is not None
    assert changed is not None
    assert changed.width < native.width
    np.testing.assert_array_equal(changed.points[[0, -1]], points[[0, -1]])
    np.testing.assert_array_equal(changed.paint, native.paint)
    assert (
        measure(
            points,
            target,
            4,
            visible=visible,
            opacity=opacity,
            coverage=np.zeros_like(coverage),
        )
        is None
    )
    target[40:44, 38:62] = 180
    assert (
        measure(points, target, 4, visible=visible, opacity=opacity, coverage=coverage)
        is None
    )
    target[:] = 180
    target[40:] = 30
    assert measure(line(), target, 4, visible=visible, coverage=coverage) is None


@pytest.mark.parametrize(
    "invalid", ["shape", "visibility", "nonfinite", "negative", "too-large"]
)
def test_carrier_profile_rejects_invalid_coverage(invalid):
    points, target, opacity = exterior_profile()
    visible = opacity > 8 / 255
    coverage = np.ones(target.shape[:2], np.float32)
    if invalid == "shape":
        coverage = coverage[:-1]
    elif invalid == "visibility":
        visible = None
    elif invalid == "nonfinite":
        coverage[:] = np.nan
    elif invalid == "negative":
        coverage[:] = -1
    else:
        coverage[:] = 2
    with pytest.raises(ValueError, match="coverage"):
        measure(points, target, 4, visible=visible, coverage=coverage)
