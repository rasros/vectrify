"""Source-only body width fits preserve discovery, anchors and original alternatives."""

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from tests.refine.test_cel_plan_ink_models import drawing
from tests.refine.test_cel_plan_source_absence import observed_gap
from tests.refine.test_cel_plan_source_links import drawing as junction
from vectrify.document import Geometry
from vectrify.document.svg import parse_path
from vectrify.refine.cel_plan import ink_models, source_widths
from vectrify.refine.cel_plan.model import Options, StageInterruptedError, Work
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.source_absence import SourceAbsence
from vectrify.refine.cel_plan.source_widths import SourceWidths


def group(path, cap="round"):
    return {"contours": parse_path(path).subpaths, "cap": cap}


@pytest.mark.parametrize("cap", ["round", "butt"])
def test_native_neighbour_absence_recovery_keeps_complete_geometry_and_caps(cap):
    evidence, profiles = observed_gap()
    absence = SourceAbsence(evidence, profiles, Work.start(10))
    source = group("M20.25 30.5H76.25", cap)
    geometry = Geometry("complete-source", source["contours"])
    assert not absence.permits(geometry, 4, cap, Work.start(10))
    fit = SourceWidths(absence)
    widths = fit.fit((source,), (4,), 1, Work.start(10))
    assert widths == pytest.approx((2.8,))
    assert geometry.subpaths == source["contours"]
    assert source["cap"] == cap
    assert fit.diagnostics["changed_styles"] == 1
    alpha = render(
        f'<svg width="96" height="96"><path d="{geometry.path_data()}" '
        f'fill="none" stroke="black" stroke-width="{widths[0]}" '
        f'stroke-linecap="{cap}"/></svg>',
        evidence.source_size,
    )[..., 3]
    queried = map_coordinates(
        alpha,
        [absence.points[:, 1] - 0.5, absence.points[:, 0] - 0.5],
        order=1,
        mode="constant",
        cval=0,
    )
    assert queried.max(initial=0) <= 1 / 255 + 1e-7
    assert geometry.subpaths[0].nodes[0].endpoint == (20.25, 30.5)
    assert geometry.subpaths[0].nodes[-1].endpoint == (76.25, 30.5)


def test_actual_own_gap_cannot_be_repaired_by_narrowing_a_full_chain():
    evidence, profiles = observed_gap()
    absence = SourceAbsence(evidence, profiles, Work.start(10))
    fit = SourceWidths(absence)
    assert fit.fit((group("M20.25 28.25H76.25"),), (3,), 1, Work.start(10)) == (3,)
    assert fit.diagnostics["changed_styles"] == 0
    assert fit.diagnostics["styles"][0]["fitted_complete_bodies"] == 0


class Absence:
    points = np.array([[48.0, 28.0]])

    def __init__(self, limit=2.5):
        self.limit, self.calls = limit, []

    def permits(self, geometry, width, cap, _work):
        self.calls.append((geometry, width, cap))
        return width <= self.limit


@pytest.mark.parametrize("limit", [0, 2.5, 10])
def test_widest_maximum_retention_wins_and_no_gain_keeps_original(limit):
    absence = Absence(limit)
    fit = SourceWidths(absence)
    expected = 2.4 if limit == 2.5 else 3
    assert fit.fit((group("M20 28H76"),), (3,), 1, Work.start(10)) == pytest.approx(
        (expected,)
    )
    assert fit.diagnostics["fits"] == 5


@pytest.mark.parametrize("scale", [0.5, 1, 2])
def test_intrinsic_analysis_floor_deduplicates_trials_without_widening(scale):
    absence = Absence(0)
    fit = SourceWidths(absence)
    width = 0.8 / scale
    assert fit.fit((group("M20 28H76"),), (width,), scale, Work.start(10)) == (width,)
    assert [c[1] for c in absence.calls] == [width]


def test_capacity_failure_after_a_successful_style_returns_all_original_widths(
    monkeypatch,
):
    monkeypatch.setattr(source_widths, "MAX_FITS", 6)
    fit = SourceWidths(Absence())
    groups = (group("M20 28H76"), group("M20 36H76"))
    assert fit.fit(groups, (3, 4), 1, Work.start(10)) == (3, 4)
    assert fit.diagnostics["bounded"] == 1
    assert fit.diagnostics["changed_styles"] == 0
    assert fit.diagnostics["styles"] == []


@pytest.mark.parametrize("bound", ["MAX_STYLES", "MAX_RUNS", "MAX_NODES"])
def test_preflight_bounds_skip_native_fits(monkeypatch, bound):
    monkeypatch.setattr(source_widths, bound, 0)
    absence = Absence()
    fit = SourceWidths(absence)
    assert fit.fit((group("M20 28H76"),), (3,), 1, Work.start(10)) == (3,)
    assert absence.calls == []
    assert fit.diagnostics["bounded"] == 1


def test_mid_native_call_stop_does_not_publish_fitted_styles():
    work = Work.start(10)
    absence = Absence()

    def stopped(geometry, width, cap, _work):
        del geometry, width, cap
        work.stop.set()
        return True

    absence.permits = stopped
    fit = SourceWidths(absence)
    with pytest.raises(StageInterruptedError):
        fit.fit((group("M20 28H76"),), (3,), 1, work)
    assert fit.diagnostics["styles"] == []


def test_empty_absence_and_reuse_leave_no_stale_width_selection():
    absence = Absence()
    fit = SourceWidths(absence)
    fit.fit((group("M20 28H76"),), (3,), 1, Work.start(10))
    assert fit.diagnostics["changed_styles"] == 1
    absence.points = np.empty((0, 2))
    assert fit.fit((group("M20 28H76"),), (3,), 1, Work.start(10)) == (3,)
    assert fit.diagnostics["styles"] == []
    assert fit.diagnostics["changed_styles"] == 0


@pytest.mark.parametrize(
    ("scale", "widths"),
    [(0, (3,)), (float("nan"), (3,)), (1, (0,)), (1, (float("inf"),)), (1, ())],
)
def test_invalid_width_frame_rejected(scale, widths):
    with pytest.raises(ValueError, match="finite aligned"):
        SourceWidths(Absence()).fit(
            (group("M20 28H76"),), widths, scale, Work.start(10)
        )


@pytest.mark.parametrize("alpha", [255, 128, 64])
def test_width_override_junctions_and_full_source_observations_stay_exact(
    alpha, monkeypatch
):
    evidence, mask = junction(alpha)
    options = Options(line_width=2)
    old_bank, new_bank = [], []
    kwargs = {"prune_spurs": True, "source_absence": True, "source_intervals": True}
    old = ink_models.models(
        mask, evidence, options, Work.start(10), source_profiles=old_bank, **kwargs
    )
    monkeypatch.setattr(
        ink_models,
        "SourceWidths",
        lambda *_a: pytest.fail("Explicit width reached fitter"),
    )
    new = ink_models.models(
        mask,
        evidence,
        options,
        Work.start(10),
        fit_widths=True,
        source_profiles=new_bank,
        **kwargs,
    )
    assert sum(m.details["junction_links"] for m in new) == 1
    for a, b in zip(old, new, strict=True):
        assert a.geometry == b.geometry
        assert a.details == b.details
        np.testing.assert_array_equal(a.paint, b.paint)
        np.testing.assert_array_equal(a.selected, b.selected)
    for a, b in zip(old_bank, new_bank, strict=True):
        for name in ("points", "sides", "direction", "tolerance"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


@pytest.mark.parametrize("alpha", [255, 128, 64])
def test_real_gap_intervals_and_bank_do_not_become_a_connected_repair(alpha):
    evidence, mask = drawing(alpha, gap=True)
    mask[27:30, 40:56] = True
    before = mask.copy()
    bank = []
    models = ink_models.models(
        mask,
        evidence,
        Options(),
        Work.start(10),
        source_absence=True,
        source_intervals=True,
        fit_widths=True,
        source_profiles=bank,
    )
    assert len(models) == 1
    assert len(models[0].geometry.subpaths) == 2
    assert models[0].details["source_width_fit"]["changed_styles"] == 0
    assert len(bank) == 1
    assert not models[0].selected[28, 48]
    np.testing.assert_array_equal(mask, before)


def test_mid_fitting_cancellation_publishes_neither_models_nor_partial_bank(
    monkeypatch,
):
    evidence, mask = drawing(gap=True)
    mask[27:30, 40:56] = True
    work, bank = Work.start(10), []

    def stop(_self, *_args):
        work.stop.set()
        raise StageInterruptedError("stopped width fit")

    monkeypatch.setattr(SourceWidths, "fit", stop)
    assert (
        ink_models.models(
            mask,
            evidence,
            Options(),
            work,
            source_absence=True,
            source_intervals=True,
            fit_widths=True,
            source_profiles=bank,
        )
        == ()
    )
    assert bank == []


def test_fitting_requires_retained_source_interval_constraints():
    evidence, mask = drawing()
    with pytest.raises(ValueError, match="interval constraints"):
        ink_models.models(mask, evidence, Options(), Work.start(10), fit_widths=True)


def test_fit_budget_restores_original_models_and_preserves_complete_bank(monkeypatch):
    evidence, mask = drawing(gap=True)
    mask[27:30, 40:56] = True
    old_bank, new_bank = [], []
    kwargs = {"source_absence": True, "source_intervals": True}
    old = ink_models.models(
        mask, evidence, Options(), Work.start(10), source_profiles=old_bank, **kwargs
    )
    monkeypatch.setattr(source_widths, "MAX_FITS", 0)
    new = ink_models.models(
        mask,
        evidence,
        Options(),
        Work.start(10),
        fit_widths=True,
        source_profiles=new_bank,
        **kwargs,
    )
    assert len(new) == len(old) == 1
    assert new[0].geometry == old[0].geometry
    assert new[0].details["width"] == old[0].details["width"]
    assert new[0].details["source_width_fit"]["bounded"] == 1
    assert new[0].details["source_width_fit"]["changed_styles"] == 0
    np.testing.assert_array_equal(new[0].paint, old[0].paint)
    np.testing.assert_array_equal(new[0].selected, old[0].selected)
    assert len(new_bank) == len(old_bank)
    np.testing.assert_array_equal(new_bank[0].points, old_bank[0].points)
