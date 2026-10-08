"""Raw gaps authorize bounded editable intervals, never unsupported repairs."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from tests.refine.test_cel_plan_ink_models import drawing
from tests.refine.test_cel_plan_source_absence import observed_gap
from tests.refine.test_cel_plan_source_links import drawing as junction
from vectrify.document import Geometry
from vectrify.document.svg import parse_path
from vectrify.refine.cel_plan import ink_models, source_intervals
from vectrify.refine.cel_plan.line_fidelity import SourceLineGuard, SourceProfile
from vectrify.refine.cel_plan.model import Options, StageInterruptedError, Work
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.source_absence import SourceAbsence
from vectrify.refine.cel_plan.source_intervals import SourceIntervals


def source_case():
    evidence, profiles = observed_gap()
    original = parse_path("M20.25 28.25H76.25").subpaths[0]
    absence = SourceAbsence(evidence, profiles, Work.start(10), intervals=True)
    return evidence, profiles[0], original, absence


def test_breaks_are_bound_to_original_physical_profile_and_copied_native_evidence():
    evidence, profiles = observed_gap()
    original = profiles[0]
    guard = SourceLineGuard(evidence.rgba, profiles)
    observed = guard.source_breaks(original)
    assert observed is not None
    assert observed.gaps.any()
    assert not np.shares_memory(observed.points, original.points)
    for value in (observed.points, observed.qualified, observed.gaps):
        assert not value.flags.writeable
    assert guard.source_breaks(replace(original)) is None
    assert guard.source_breaks(SourceProfile.at([[43, 28], [53, 28]], 3)) is None
    with pytest.raises(StageInterruptedError):
        guard.source_breaks(original, work=Work.start(0))


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("cap", ["round", "butt"])
def test_source_gaps_retain_two_editable_intervals_with_exact_surviving_terminals(
    width, cap
):
    evidence, profile, original, absence = source_case()
    recovery = SourceIntervals(absence)
    parts = recovery.recover(
        original,
        profile,
        width,
        cap,
        Options(line_width=width),
        Work.start(10),
        lambda _g: True,
    )
    assert len(parts) == 2
    assert parts[0].nodes[0].endpoint == original.nodes[0].endpoint
    assert parts[-1].nodes[-1].endpoint == original.nodes[-1].endpoint
    assert parts[0].nodes[-1].endpoint[0] < 43
    assert parts[-1].nodes[0].endpoint[0] > 53
    assert recovery.diagnostics["recovered_runs"] == 2
    # Full native rasterization independently checks actual body/cap absence.
    geometry = Geometry("intervals", parts)
    alpha = render(
        f'<svg width="96" height="96"><path d="{geometry.path_data()}" '
        f'fill="none" stroke="black" stroke-width="{width}" '
        f'stroke-linecap="{cap}" stroke-linejoin="round"/></svg>',
        evidence.source_size,
    )[..., 3]
    sampled = map_coordinates(
        alpha,
        [absence.points[:, 1] - 0.5, absence.points[:, 0] - 0.5],
        order=1,
        mode="constant",
        cval=0,
    )
    assert sampled.max(initial=0) <= 1 / 255 + 1e-7
    observed = absence.source_breaks(profile)
    assert observed is not None
    for point in (parts[0].nodes[-1].endpoint, parts[-1].nodes[0].endpoint):
        idx = int(np.argmin(np.linalg.norm(observed.points - point, axis=1)))
        assert observed.qualified[idx]
        np.testing.assert_array_equal(observed.points[idx], point)


def test_missing_own_gaps_or_mismatched_ends_cannot_authorize_a_split():
    _, profile, original, absence = source_case()
    recovery = SourceIntervals(absence)
    assert (
        recovery.recover(
            original,
            replace(profile),
            3,
            "round",
            Options(),
            Work.start(10),
            lambda _g: True,
        )
        == ()
    )
    changed = parse_path("M21.25 28.25H76.25").subpaths[0]
    assert (
        recovery.recover(
            changed, profile, 3, "round", Options(), Work.start(10), lambda _g: True
        )
        == ()
    )
    complete, _ = drawing()
    supported = SourceProfile.at([[22, 29], [73, 58]], 3)
    same = parse_path("M22 29L73 58").subpaths[0]
    proof = SourceAbsence(complete, [supported], Work.start(10), intervals=True)
    assert (
        SourceIntervals(proof).recover(
            same, supported, 3, "round", Options(), Work.start(10), lambda _g: True
        )
        == ()
    )


@pytest.mark.parametrize(
    "bound", ["MAX_INTERVALS", "MAX_FITS", "MAX_RUNS", "MAX_NODES"]
)
def test_independent_bounds_never_publish_partial_intervals(monkeypatch, bound):
    _, profile, original, absence = source_case()
    monkeypatch.setattr(
        source_intervals, bound, 1 if bound in {"MAX_INTERVALS", "MAX_NODES"} else 0
    )
    with pytest.raises(ValueError, match="bound"):
        SourceIntervals(absence).recover(
            original, profile, 3, "round", Options(), Work.start(10), lambda _g: True
        )


def test_rejected_carrier_and_mid_recovery_stop_return_no_partial_geometry():
    _, profile, original, absence = source_case()
    assert (
        SourceIntervals(absence).recover(
            original, profile, 3, "round", Options(), Work.start(10), lambda _g: False
        )
        == ()
    )
    work = Work.start(10)

    def stop(_geometry):
        work.stop.set()
        return True

    with pytest.raises(StageInterruptedError):
        SourceIntervals(absence).recover(
            original, profile, 3, "round", Options(), work, stop
        )


@pytest.mark.parametrize("alpha", [255, 128, 64])
def test_coarse_detection_bridge_recovers_actual_source_intervals_and_complete_bank(
    alpha,
):
    evidence, mask = drawing(alpha, gap=True)
    mask[27:30, 40:56] = True
    before = mask.copy()
    bank, constrained_bank = [], []
    constrained = ink_models.models(
        mask,
        evidence,
        Options(),
        Work.start(10),
        source_absence=True,
        source_profiles=constrained_bank,
    )
    recovered = ink_models.models(
        mask,
        evidence,
        Options(),
        Work.start(10),
        source_absence=True,
        source_intervals=True,
        source_profiles=bank,
    )
    assert constrained == ()
    assert len(recovered) == 1
    assert len(recovered[0].geometry.subpaths) == 2
    assert recovered[0].details["source_interval_reconstructed_runs"] == 1
    assert recovered[0].details["source_interval_recovered_contours"] == 2
    assert recovered[0].details["source_absence_excluded_runs"] == 0
    assert len(bank) == len(constrained_bank) == 1
    np.testing.assert_array_equal(bank[0].points, constrained_bank[0].points)
    np.testing.assert_array_equal(mask, before)
    assert np.any(mask & ~recovered[0].selected)
    assert not recovered[0].selected[28, 48]


@pytest.mark.parametrize("alpha", [255, 128, 64])
def test_valid_junctions_remain_exact_with_width_override(alpha):
    evidence, mask = junction(alpha)
    options = Options(line_width=2)
    previous = ink_models.models(
        mask, evidence, options, Work.start(10), prune_spurs=True, source_absence=True
    )
    found = ink_models.models(
        mask,
        evidence,
        options,
        Work.start(10),
        prune_spurs=True,
        source_absence=True,
        source_intervals=True,
    )
    assert sum(m.details["junction_links"] for m in found) == 1
    for model, baseline in zip(found, previous, strict=True):
        assert model.geometry == baseline.geometry
        assert model.details["width"] == baseline.details["width"] == 2
        np.testing.assert_array_equal(model.paint, baseline.paint)
        np.testing.assert_array_equal(model.selected, baseline.selected)


def test_recovered_host_without_original_port_cannot_authorize_a_short_link(
    monkeypatch,
):
    evidence, mask = junction()

    class Absence:
        points = np.empty((0, 2))
        source_breaks = True

        def __init__(self, *_args, **_kwargs):
            pass

        def permits(self, geometry, *_args):
            if geometry.id.startswith("source-ink-"):
                return True
            return all(
                sub.id == "recovered"
                or not any(n.endpoint[0] < 48 for n in sub.nodes)
                or np.linalg.norm(
                    np.subtract(sub.nodes[-1].endpoint, sub.nodes[0].endpoint)
                )
                < 8
                for sub in geometry.subpaths
            )

    class Recovery:
        def __init__(self, _absence):
            self.diagnostics = {}

        def recover(self, *_args):
            return (replace(parse_path("M20 20L32 32").subpaths[0], id="recovered"),)

    monkeypatch.setattr(ink_models, "SourceAbsence", Absence)
    monkeypatch.setattr(ink_models, "SourceIntervals", Recovery)
    found = ink_models.models(
        mask,
        evidence,
        Options(),
        Work.start(10),
        prune_spurs=True,
        source_absence=True,
        source_intervals=True,
    )
    assert found
    assert sum(m.details["junction_links"] for m in found) == 0
    assert sum(m.details["source_interval_reconstructed_runs"] for m in found) == 2


def test_interval_stop_preserves_source_and_withholds_diagnostic_bank(monkeypatch):
    evidence, mask = drawing(gap=True)
    mask[27:30, 40:56] = True
    before = mask.copy()
    bank, work = [], Work.start(10)
    original = source_intervals.fitted

    def stop(*args, **kwargs):
        result = original(*args, **kwargs)
        work.stop.set()
        return result

    monkeypatch.setattr(source_intervals, "fitted", stop)
    found = ink_models.models(
        mask,
        evidence,
        Options(),
        work,
        source_absence=True,
        source_intervals=True,
        source_profiles=bank,
    )
    assert found == ()
    assert bank == []
    np.testing.assert_array_equal(mask, before)


@pytest.mark.parametrize("width", [0, -1, float("nan"), float("inf")])
def test_invalid_styles_cannot_enter_interval_fitting(width):
    _, profile, original, absence = source_case()
    with pytest.raises(ValueError, match="finite supported stroke"):
        SourceIntervals(absence).recover(
            original,
            profile,
            width,
            "round",
            Options(),
            Work.start(10),
            lambda _g: True,
        )


def test_output_limit_after_first_recovery_withholds_all_models_but_keeps_complete_bank(
    monkeypatch,
):
    evidence, mask = drawing(gap=True)
    mask[27:30, 40:56] = True
    before, bank = mask.copy(), []
    monkeypatch.setattr(source_intervals, "MAX_RUNS", 1)
    found = ink_models.models(
        mask,
        evidence,
        Options(),
        Work.start(10),
        source_absence=True,
        source_intervals=True,
        source_profiles=bank,
    )
    assert found == ()
    assert len(bank) == 1
    np.testing.assert_array_equal(mask, before)


def test_closed_source_interpretation_is_retained_as_fill():
    _, profile, original, absence = source_case()
    recovery = SourceIntervals(absence)
    assert (
        recovery.recover(
            replace(original, closed=True),
            profile,
            3,
            "round",
            Options(),
            Work.start(10),
            lambda _g: True,
        )
        == ()
    )
    assert recovery.diagnostics["closed_exclusions"] == 1


def test_native_interval_geometry_is_independent_of_analysis_scale_and_offset():
    evidence, profile, original, absence = source_case()
    options = Options()
    first = SourceIntervals(absence).recover(
        original, profile, 3, "round", options, Work.start(10), lambda _g: True
    )
    changed = replace(evidence, scale=(1.5, 0.75), offset=(7.25, -3.5))
    second_proof = SourceAbsence(changed, [profile], Work.start(10), intervals=True)
    second = SourceIntervals(second_proof).recover(
        original, profile, 3, "round", options, Work.start(10), lambda _g: True
    )
    assert first == second
