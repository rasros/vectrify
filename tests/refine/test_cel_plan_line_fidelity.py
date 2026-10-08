"""Source-only native line support rejects loss, floods and completed gaps."""

from dataclasses import replace

import numpy as np
import pathops
import pytest

from tests.refine.test_cel_plan_ink_models import drawing
from vectrify.refine.cel_plan import ink_models, line_fidelity
from vectrify.refine.cel_plan.ink import Ink
from vectrify.refine.cel_plan.line_fidelity import SourceLineGuard, SourceProfile
from vectrify.refine.cel_plan.model import Options, StageInterruptedError, Work
from vectrify.refine.cel_plan.score import render


def source(*, alpha=1, gap=False, shifted=0, blank=False):
    path = "M10 32H46 M50 32H86" if gap else f"M10 {32 + shifted}H86"
    return render(
        '<svg width="96" height="64">'
        f'<g opacity="{alpha}"><path d="M0 0H96V64H0Z" fill="#c4b79c"/>'
        + (
            ""
            if blank
            else f'<path d="{path}" fill="none" stroke="#202020" '
            'stroke-width="3" stroke-linecap="butt"/>'
        )
        + "</g></svg>",
        (96, 64),
    )


def profile():
    return SourceProfile.at(
        np.column_stack((np.arange(12.5, 84), np.full(72, 32.5))), 3
    )


@pytest.mark.parametrize("alpha", [1, 0.5, 0.25])
def test_native_source_contract_detects_a_removed_interval_and_accepts_small_shift(
    alpha,
):
    truth = source(alpha=alpha)
    guard = SourceLineGuard(truth, [profile()])
    guard.establish(truth)
    assert guard.metrics(truth)["missing_samples"] == 0
    assert guard.metrics(source(alpha=alpha, shifted=0.75))["rejections"] == []
    cut = truth.copy()
    cut[:, 40:60] = source(alpha=alpha, blank=True)[:, 40:60]
    lost = guard.metrics(cut)
    assert lost["missing_samples"] > 20
    assert lost["profiles"][0]["maximum_missing_span"] > 15
    assert lost["rejections"] == [{"profile": 0, "reason": "source-line-lost"}]


def test_dark_surface_flood_cannot_impersonate_a_retained_line():
    truth = source()
    guard = SourceLineGuard(truth, [profile()])
    guard.establish(truth)
    flooded = truth.copy()
    flooded[..., :3] = 32 / 255
    assert guard.metrics(flooded)["missing_share"] == 1
    assert guard.metrics(flooded)["rejections"]


@pytest.mark.parametrize("alpha", [1, 0.5, 0.25])
def test_equal_gap_totals_cannot_exchange_original_protected_positions(alpha):
    def drawing(path):
        return render(
            '<svg width="96" height="64">'
            f'<g opacity="{alpha}"><path d="M0 0H96V64H0Z" fill="#c4b79c"/>'
            f'<path d="{path}" fill="none" stroke="#202020" '
            'stroke-width="3" stroke-linecap="butt"/></g></svg>',
            (96, 64),
        )

    truth = drawing("M10 32H30 M34 32H54 M58 32H86")
    before = drawing("M10 32H54 M58 32H86")
    after = drawing("M10 32H30 M34 32H86")
    guard = SourceLineGuard(truth, [profile()])
    guard.establish(before)
    initial = guard.metrics(before)
    current = guard.metrics(after)
    assert initial["gap_completed"] == current["gap_completed"] > 0
    assert current["new_gap_completed"] > 0
    assert current["rejections"] == [
        {"profile": 0, "reason": "source-line-gap-completed"}
    ]
    assert (
        guard.compare(before, after)["new_gap_completed"]
        == current["new_gap_completed"]
    )
    assert guard.metrics(before) == initial
    states = guard.gap_states(after)
    assert all(not a.flags.writeable for a in states)


def test_a_nearby_source_trough_is_not_a_false_negative_gap():
    truth = source()
    p = profile()
    points = p.points.copy()
    # A measured centre can stray just outside the source's antialiased body.
    # Positive line matching allows this normal displacement. The same source
    # support must prevent classifying the bright centre as an absent line.
    points[36, 1] += 2
    p = replace(p, points=points)
    guard = SourceLineGuard(truth, [p])
    guard.establish(truth)
    assert guard.metrics(source(shifted=1))["rejections"] == []


@pytest.mark.parametrize("transparent", [False, True])
def test_sparse_source_ports_cannot_hide_a_bright_or_transparent_gap(transparent):
    truth = source(gap=True)
    if transparent:
        truth[:, 46:50, 3] = 0
        truth[:, 46:50, :3] = (0, 1, 0)  # Invisible RGB supplies no line evidence.
    sparse = SourceProfile.at(np.array(((12.5, 32.5), (83.5, 32.5))), 3)
    guard = SourceLineGuard(truth, [sparse])
    guard.establish(truth)
    baseline = guard.metrics(truth)
    assert baseline["qualified_samples"] > 100
    assert baseline["gap_samples"] >= 4
    assert baseline["gap_completed"] == 0
    bridge = guard.metrics(source())
    assert bridge["gap_completed"] >= 4
    assert {r["reason"] for r in bridge["rejections"]} == {"source-line-gap-completed"}


def test_painted_interior_proves_exterior_ink_without_transparent_rgb():
    svg = (
        '<svg width="96" height="64"><path d="M8 8H88V56H8Z" fill="#c4b79c"/>'
        '<path d="M8.5 12V52" stroke="#202020" fill="none" stroke-width="2"/>'
        "</svg>"
    )
    truth = render(svg, (96, 64))
    positions = np.column_stack((np.full(38, 8.5), np.arange(13.5, 51)))
    a = SourceLineGuard(truth, [SourceProfile.at(positions, 2)])
    hidden = truth.copy()
    hidden[hidden[..., 3] == 0, :3] = (0.9, 0.1, 0.7)
    b = SourceLineGuard(hidden, [SourceProfile.at(positions, 2)])
    a.establish(truth)
    b.establish(hidden)
    assert a.metrics(truth) == b.metrics(hidden)
    assert a.metrics(truth)["qualified_samples"] > 40
    empty = truth.copy()
    empty[..., 3] = 0
    assert a.metrics(empty)["missing_share"] == 1


def test_a_shade_discontinuity_without_a_trough_is_not_a_source_line():
    truth = source(blank=True)
    truth[32:, :, :3] = 32 / 255
    guard = SourceLineGuard(truth, [profile()])
    guard.establish(truth)
    assert guard.metrics(truth)["qualified_samples"] == 0
    assert guard.metrics(source(blank=True))["rejections"] == []


def test_anisotropic_source_mapping_and_copied_profiles_keep_their_native_phase():
    evidence, _mask = drawing()
    evidence = replace(evidence, scale=(0.5, 1), offset=(8, 5))
    points = np.column_stack((np.full(22, 14.5), np.arange(13.5, 35)))
    ink = Ink(points.copy(), 2, np.full(3, 32), 1, 0)
    observed = SourceProfile.from_ink(points, ink, evidence)
    native = SourceProfile.at(points / evidence.scale + evidence.offset, 4)
    for field in ("points", "sides", "direction", "tolerance"):
        np.testing.assert_array_equal(getattr(observed, field), getattr(native, field))
        assert not getattr(observed, field).flags.writeable
    ink.points[0] = 0
    assert observed.points[0, 0] == 37


def test_baseline_is_fixed_and_rasters_must_match_the_source():
    truth = source()
    guard = SourceLineGuard(truth, [profile()])
    with pytest.raises(ValueError, match="established"):
        guard.metrics(truth)
    guard.establish(truth)
    with pytest.raises(ValueError, match="cannot change"):
        guard.establish(source(blank=True))
    for invalid in (truth[:-1], truth * np.nan, truth + 1):
        with pytest.raises(ValueError, match="finite aligned"):
            guard.metrics(invalid)


@pytest.mark.parametrize("alpha", [1, 0.5, 0.25])
def test_reused_painted_observation_keeps_exact_source_checks_and_fixed_baseline(alpha):
    truth = source(alpha=alpha, gap=True)
    guard = SourceLineGuard(truth, [profile()])
    guard.establish(truth)
    before = truth.copy()
    observed = guard.observe(before)
    baseline = guard.metrics(truth)
    for after in (
        source(alpha=alpha),
        source(alpha=alpha, gap=True),
        source(alpha=alpha, blank=True),
    ):
        assert guard.compare_observed(observed, after) == guard.compare(before, after)
    # The snapshot owns sampled observations, never views into caller pixels.
    before[:] = source(alpha=alpha)
    result = guard.compare_observed(observed, before)
    assert result["new_gap_completed"] > 0
    assert result["rejections"]
    assert all(not gaps.flags.writeable for gaps in observed.gaps)
    assert guard.metrics(truth) == baseline


def test_observation_from_identical_but_different_source_bank_is_rejected():
    truth = source(gap=True)
    a = SourceLineGuard(truth, [profile()])
    b = SourceLineGuard(truth, [profile()])
    with pytest.raises(ValueError, match="another source bank"):
        b.compare_observed(a.observe(truth), truth)


def test_interrupted_parent_observation_does_not_publish_or_change_baseline(
    monkeypatch,
):
    truth = source(gap=True)
    guard = SourceLineGuard(truth, [profile(), profile()])
    guard.establish(truth)
    baseline = guard.metrics(truth)
    work = Work.start(10)
    original = line_fidelity._gap_state

    def interrupted(*args):
        result = original(*args)
        work.stop.set()
        return result

    monkeypatch.setattr(line_fidelity, "_gap_state", interrupted)
    with pytest.raises(StageInterruptedError, match="line fidelity"):
        guard.observe(truth, work=work)
    assert guard.metrics(truth) == baseline


def test_sample_limit_and_cancellation_never_publish_partial_fidelity(monkeypatch):
    truth = source()
    p = profile()
    monkeypatch.setattr(line_fidelity, "MAX_SAMPLES", 100)
    with pytest.raises(ValueError, match="sample limit"):
        SourceLineGuard(truth, [p])
    monkeypatch.setattr(line_fidelity, "MAX_SAMPLES", 131_072)
    guard = SourceLineGuard(truth, [p, p])
    guard.establish(truth)
    work = Work.start(10)
    original = line_fidelity._error

    def interrupted(*args):
        result = original(*args)
        work.stop.set()
        return result

    monkeypatch.setattr(line_fidelity, "_error", interrupted)
    with pytest.raises(StageInterruptedError, match="line fidelity"):
        guard.metrics(source(blank=True), work=work)
    assert guard._baseline == guard.assess(truth)


def test_unexportable_source_profiles_are_still_published_after_complete_discovery():
    evidence, mask = drawing()
    profiles = []
    # A genuine source trough may not fit a fixed requested width/carrier. The
    # diagnostic must retain its original measured evidence, not erase it.
    found = ink_models.models(
        mask,
        evidence,
        Options(line_width=30),
        Work.start(10),
        carrier=pathops.Path(),
        source_profiles=profiles,
    )
    assert found == ()
    assert profiles
    assert all(not p.points.flags.writeable for p in profiles)


def test_observing_source_profiles_does_not_change_physical_stroke_models():
    evidence, mask = drawing()
    control = ink_models.models(mask, evidence, Options(), Work.start(10))
    profiles = []
    observed = ink_models.models(
        mask, evidence, Options(), Work.start(10), source_profiles=profiles
    )
    assert profiles
    assert control
    assert len(observed) == len(control)
    for actual, original in zip(observed, control, strict=True):
        assert actual.geometry == original.geometry
        assert actual.footprint.path_data() == original.footprint.path_data()
        assert actual.details == original.details
        np.testing.assert_array_equal(actual.paint, original.paint)
        np.testing.assert_array_equal(actual.selected, original.selected)
    assert all(
        not array.flags.writeable
        for p in profiles
        for array in (p.points, p.sides, p.direction, p.tolerance)
    )


def test_cancelled_discovery_cannot_publish_a_prefix_of_source_profiles(monkeypatch):
    evidence, mask = drawing()
    profiles = []
    work = Work.start(10)
    original = ink_models.carried

    def cancelled(*args, **kwargs):
        result = original(*args, **kwargs)
        work.stop.set()
        return result

    monkeypatch.setattr(ink_models, "carried", cancelled)
    assert (
        ink_models.models(mask, evidence, Options(), work, source_profiles=profiles)
        == ()
    )
    assert profiles == []


def test_an_empty_renderer_cannot_establish_a_source_line_baseline():
    truth = source()
    guard = SourceLineGuard(truth, [profile()])
    with pytest.raises(ValueError, match="lose every"):
        guard.establish(source(blank=True))
    guard.establish(truth)
    assert guard.metrics(truth)["rejections"] == []


def test_distinct_facing_source_components_keep_their_gap_in_the_render():
    truth = source(gap=True)
    left = SourceProfile.at(
        np.column_stack((np.arange(12.5, 46), np.full(34, 32.5))), 3
    )
    right = SourceProfile.at(
        np.column_stack((np.arange(50.5, 84), np.full(34, 32.5))), 3
    )
    guard = SourceLineGuard(truth, [left, right])
    guard.establish(truth)
    assert guard.metrics(truth)["endpoint_gap_profiles"] == 1
    assert guard.metrics(truth)["gap_completed"] == 0
    assert guard.metrics(source())["gap_completed"] > 0
    assert guard.metrics(source())["rejections"] == [
        {"profile": 2, "reason": "source-line-gap-completed"}
    ]


def test_source_component_identity_and_gap_resource_limits_are_respected(monkeypatch):
    truth = source(gap=True)
    left = SourceProfile.at(
        np.column_stack((np.arange(12.5, 46), np.full(34, 32.5))), 3
    )
    right = SourceProfile.at(
        np.column_stack((np.arange(50.5, 84), np.full(34, 32.5))), 3
    )
    guard = SourceLineGuard(
        truth,
        [
            replace(left, component=("source", 1)),
            replace(right, component=("source", 1)),
        ],
    )
    guard.establish(truth)
    # Another branch of one existing physical component is not evidence for
    # a new endpoint-to-endpoint connection or an independently proved gap.
    assert guard.metrics(truth)["endpoint_gap_profiles"] == 0
    monkeypatch.setattr(line_fidelity, "MAX_GAP_PROFILES", 0)
    with pytest.raises(ValueError, match="Source gaps exceed"):
        SourceLineGuard(truth, [left, right])
    monkeypatch.setattr(line_fidelity, "MAX_PROFILES", 1)
    with pytest.raises(ValueError, match="bounded profile"):
        SourceLineGuard(truth, [left, right])
