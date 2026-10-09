"""Automatic source seeds and fitting keep physical ports and real gaps."""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from tests.helpers import required
from tests.refine.test_cel_plan_source_absence import observed_gap
from vectrify.document import export_svg, import_svg, load_project, save_project
from vectrify.document.join import transformed_geometry
from vectrify.document.redraw import root_matrix
from vectrify.document.topology import inverse_matrix
from vectrify.refine.cel_plan import band_fit, source_bands
from vectrify.refine.cel_plan.band_fit import BandFit, opaque_core, painted_context
from vectrify.refine.cel_plan.band_plans import BandPlans
from vectrify.refine.cel_plan.evidence import collect
from vectrify.refine.cel_plan.line_fidelity import SourceLineGuard, SourceProfile
from vectrify.refine.cel_plan.local import Box, _native_raster
from vectrify.refine.cel_plan.model import Options, StageInterruptedError, Work
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.source_bands import SourceBands


def fixture(*, alpha=1, diagonal=False, larger=False):
    side = 240 if larger else 96
    frame = (
        'transform="translate(70.12345 60.23456)"'
        if larger
        else 'transform="matrix(.8 .6 -.6 .8 20 -4)"'
        if diagonal
        else ""
    )
    source = (
        f'<svg width="{side}" height="{side}"><g opacity="{alpha}">'
        f'<path id="bg" d="M0 0H{side}V{side}H0Z" fill="#ad8665"/>'
        f'<path id="line" {frame} d="M20.25 28.25H76.25" fill="none" '
        'stroke="#121008" stroke-width="3" stroke-linecap="butt"/></g></svg>'
    )
    rgba = render(source, (side, side))
    evidence = collect(
        Image.fromarray(np.rint(rgba * 255).astype(np.uint8)),
        None,
        Options(),
        Work.start(10),
    )
    document = import_svg(
        source.replace('id="line"', 'id="ink"')
        .replace(
            'd="M20.25 28.25H76.25" fill="none"',
            'd="M20.25 25.75H76.25V30.75H20.25Z M70 70H72V72H70Z" fill="#121008"',
        )
        .replace('stroke="#121008"', 'stroke="none"')
    )
    anchors = np.column_stack((np.linspace(20.25, 76.25, 20), np.full(20, 28.25)))
    if diagonal or larger:
        a, b, c, d, e, f = root_matrix(document, "ink")
        anchors = anchors @ np.array([[a, b], [c, d]]) + (e, f)
    profile = SourceProfile.at(anchors, 3)
    guard = SourceLineGuard(evidence.rgba, (profile,))
    geometry = document.geometry_for("ink")
    main = replace(geometry, subpaths=geometry.subpaths[:1])
    marks = replace(geometry, subpaths=geometry.subpaths[1:])
    return evidence, guard, document, main, marks


@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("alpha", [1, 0.6])
def test_complete_seed_uses_copied_source_anchors_and_maintains_rotated_ports(
    alpha, diagonal
):
    evidence, guard, document, main, _ = fixture(alpha=alpha, diagonal=diagonal)
    seed = SourceBands(evidence, guard).seed(
        document, "ink", main, "nonzero", Work.start(10)
    )
    assert seed is not None
    assert len(seed.band.geometry.subpaths[0].nodes) == 2
    observed = guard.source_breaks(guard.original_profiles()[0])
    assert observed is not None
    assert observed.anchors is not None
    assert seed.band.geometry.subpaths[0].nodes[0].endpoint == tuple(
        observed.anchors[0]
    )
    assert seed.band.geometry.subpaths[0].nodes[-1].endpoint == tuple(
        observed.anchors[-1]
    )
    assert abs(seed.band.width - 3) < 0.6
    assert not seed.anchors.flags.writeable
    original = guard.original_profiles()[0]
    original.points.flags.writeable = True
    original.points[:] += 10
    repeated = SourceBands(evidence, guard).seed(
        document, "ink", main, "nonzero", Work.start(10)
    )
    assert repeated is not None
    assert repeated.band.geometry == seed.band.geometry


def test_gap_ambiguity_and_short_fragment_do_not_define_a_complete_stroke():
    evidence, guard, document, main, _ = fixture()
    profile = guard.original_profiles()[0]
    duplicate = SourceLineGuard(evidence.rgba, (profile, replace(profile)))
    assert (
        SourceBands(evidence, duplicate).seed(
            document, "ink", main, "nonzero", Work.start(10)
        )
        is None
    )
    fragment = SourceProfile.at(profile.points[:7], 3)
    short = SourceLineGuard(evidence.rgba, (fragment,))
    assert (
        SourceBands(evidence, short).seed(
            document, "ink", main, "nonzero", Work.start(10)
        )
        is None
    )
    gap_evidence, profiles = observed_gap()
    gap_guard = SourceLineGuard(gap_evidence.rgba, profiles)
    assert (
        SourceBands(gap_evidence, gap_guard).seed(
            document, "ink", main, "nonzero", Work.start(10)
        )
        is None
    )


@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
def test_atomic_fit_preserves_materials_ports_explicit_width_and_native_crop(
    fixed, diagonal
):
    evidence, guard, document, main, marks = fixture(alpha=0.6, diagonal=diagonal)
    work = Work.start(20)
    seed = SourceBands(evidence, guard).seed(
        document, "ink", main, "nonzero", work, width=3 if fixed else 0
    )
    frame = root_matrix(document, "ink")
    assert seed is not None
    model = replace(
        seed.band,
        geometry=transformed_geometry(seed.band.geometry, inverse_matrix(frame)),
    )
    assembled = BandPlans.document(
        document,
        "ink",
        "ink-marks",
        model,
        marks,
        {"fill": seed.paint, "fill-opacity": "1"},
    )
    fitted = BandFit(evidence, guard).fit(
        document, assembled, "ink", seed, work, width_fixed=fixed
    )
    assert fitted is not None
    candidate, details = fitted
    assert details["native_body_absence"]
    assert details["evaluations"] <= band_fit.MAX_EVALUATIONS
    assert details["fitted_loss"] <= details["initial_loss"]
    if fixed:
        assert details["width"] == 3
        assert details["evaluations"] == 1
    assert candidate.geometry_for("bg") == document.geometry_for("bg")
    assert candidate.element("bg") == document.element("bg")
    assert candidate.geometry_for("ink-marks") == assembled.geometry_for("ink-marks")
    np.testing.assert_array_equal(
        details["source_ports"], (seed.anchors[0], seed.anchors[-1])
    )
    actual = render(export_svg(candidate), evidence.source_size)
    context = painted_context(candidate, seed.bounds, work, keep=("ink",))
    np.testing.assert_array_equal(
        _native_raster(context, evidence.source_size).crop(Box(*seed.bounds)),
        actual[seed.bounds[1] : seed.bounds[3], seed.bounds[0] : seed.bounds[2]],
    )
    reloaded, _ = load_project(save_project(candidate))
    np.testing.assert_array_equal(
        actual, render(export_svg(reloaded), evidence.source_size)
    )


def test_fitting_crop_retains_original_qualification_gaps_and_observation_authority():
    evidence, profiles = observed_gap()
    guard = SourceLineGuard(evidence.rgba, profiles)
    local = guard.fitting_crop((0, 0, 96, 96), work=Work.start(10))
    assert local is not None
    before = local.observe(evidence.rgba)
    assert before.profiles == guard.assess(evidence.rgba)
    actual = evidence.rgba.copy()
    actual[27:30, 42:55, :3] = 0.02
    assert local.penalty(before, actual) > 0
    other = guard.fitting_crop((0, 0, 96, 96))
    assert other is not None
    with pytest.raises(ValueError, match="another crop"):
        other.penalty(before, evidence.rgba)
    with pytest.raises(ValueError, match="integer native crop"):
        guard.fitting_crop((0.5, 0, 96, 96))
    with pytest.raises(ValueError, match="finite native crop"):
        local.observe(evidence.rgba[:90])


def test_core_removes_the_replacement_stroke_and_respects_parent_opacity():
    evidence, guard, document, main, marks = fixture(alpha=0.6)
    work = Work.start(10)
    seed = SourceBands(evidence, guard).seed(document, "ink", main, "nonzero", work)
    assert seed is not None
    assembled = BandPlans.document(
        document,
        "ink",
        "ink-marks",
        seed.band,
        marks,
        {"fill": seed.paint, "fill-opacity": "1"},
    )
    core = opaque_core(assembled, "ink", main, evidence.source_size, work)
    assert core is not None
    assert core.subpaths


def test_nonzero_native_crop_preserves_fractional_frame_and_viewport():
    evidence, guard, document, main, marks = fixture(larger=True)
    work = Work.start(20)
    seed = SourceBands(evidence, guard).seed(
        document, "ink", main, "nonzero", work, width=3
    )
    assert seed is not None
    assert seed.bounds[0] > 0
    assert seed.bounds[1] > 0
    frame = root_matrix(document, "ink")
    model = replace(
        seed.band,
        geometry=transformed_geometry(seed.band.geometry, inverse_matrix(frame)),
    )
    assembled = BandPlans.document(
        document,
        "ink",
        "ink-marks",
        model,
        marks,
        {"fill": seed.paint, "fill-opacity": "1"},
    )
    candidate, details = required(
        BandFit(evidence, guard).fit(
            document, assembled, "ink", seed, work, width_fixed=True
        )
    )
    assert details["evaluations"] == 1
    actual = render(export_svg(candidate), evidence.source_size)
    context = painted_context(candidate, seed.bounds, work, keep=("ink",))
    assert float(required(context.get("width"))) == 240
    np.testing.assert_array_equal(
        _native_raster(context, evidence.source_size).crop(Box(*seed.bounds)),
        actual[seed.bounds[1] : seed.bounds[3], seed.bounds[0] : seed.bounds[2]],
    )


def test_stop_inside_optimizer_discards_fit_and_retains_document(monkeypatch):
    evidence, guard, document, main, marks = fixture()
    work = Work.start(20)
    seed = SourceBands(evidence, guard).seed(document, "ink", main, "nonzero", work)
    assert seed is not None
    assembled = BandPlans.document(
        document,
        "ink",
        "ink-marks",
        seed.band,
        marks,
        {"fill": seed.paint, "fill-opacity": "1"},
    )
    saved = save_project(assembled)
    fitter = BandFit(evidence, guard)

    def stopped(score, values, **_kwargs):
        work.stop.set()
        score(values)

    monkeypatch.setattr(band_fit, "minimize", stopped)
    with pytest.raises(StageInterruptedError):
        fitter.fit(document, assembled, "ink", seed, work)
    assert save_project(assembled) == saved
    assert fitter._cached is None


def test_fit_cache_reuses_only_identical_native_context_and_rechecks_complete_result():
    evidence, guard, document, main, marks = fixture(alpha=0.6)
    work = Work.start(20)
    seed = SourceBands(evidence, guard).seed(document, "ink", main, "nonzero", work)
    assert seed is not None
    assembled = BandPlans.document(
        document,
        "ink",
        "ink-marks",
        seed.band,
        marks,
        {"fill": seed.paint, "fill-opacity": "1"},
    )
    fitter = BandFit(evidence, guard)
    first, original = required(fitter.fit(document, assembled, "ink", seed, work))
    repeated, reused = required(fitter.fit(document, assembled, "ink", seed, work))
    assert original["evaluations"] > 0
    assert reused["evaluations"] == 0
    assert reused["reused_fit"]
    assert first == repeated
    from vectrify.document import Editor, Selection

    changed = []
    for current in (document, assembled):
        editor = Editor(current, selection=Selection(whole_document=True))
        with editor.transaction("Different fit context") as tx:
            tx.set_fill("bg", "#ae8766")
        changed.append(editor.snapshot.document)
    result = fitter.fit(changed[0], changed[1], "ink", seed, work)
    assert result is not None
    assert not result[1]["reused_fit"]
    assert result[1]["evaluations"] > 0


def test_alpha_constrained_fit_does_not_reuse_an_infeasible_appearance_fit():
    from vectrify.document import Editor, Selection
    from vectrify.document.svg import parse_path

    evidence, guard, document, main, marks = fixture()
    work = Work.start(20)
    seed = required(
        SourceBands(evidence, guard).seed(document, "ink", main, "nonzero", work)
    )
    # The existing fill contributes to the silhouette above this underpaint.
    # A thinner stroke matches the source color but loses native coverage there.
    editor = Editor(document, selection=Selection(whole_document=True))
    with editor.transaction("Partial underpaint") as tx:
        tx.replace_geometry("bg", parse_path("M0 29H96V96H0Z"))
    document = editor.snapshot.document
    assembled = BandPlans.document(
        document,
        "ink",
        "ink-marks",
        seed.band,
        marks,
        {"fill": seed.paint, "fill-opacity": "1"},
    )
    fitter = BandFit(evidence, guard)
    appearance, _ = required(
        fitter.fit(document, assembled, "ink", seed, work, width_fixed=True)
    )
    assert not np.array_equal(
        render(export_svg(document), evidence.source_size)[..., 3],
        render(export_svg(appearance), evidence.source_size)[..., 3],
    )
    saved = save_project(assembled)
    assert (
        fitter.fit(
            document,
            assembled,
            "ink",
            seed,
            work,
            width_fixed=True,
            preserve_alpha=True,
        )
        is None
    )
    assert save_project(assembled) == saved


def test_alpha_constrained_fit_retains_a_feasible_width_over_a_thinner_source_match(
    monkeypatch,
):
    from vectrify.document import Editor, Selection
    from vectrify.document.svg import parse_path

    evidence, guard, document, main, marks = fixture()
    work = Work.start(20)
    seed = required(
        SourceBands(evidence, guard).seed(
            document, "ink", main, "nonzero", work, width=3
        )
    )
    editor = Editor(document, selection=Selection(whole_document=True))
    with editor.transaction("Partially exposed outline") as tx:
        tx.replace_geometry("bg", parse_path("M0 0H21V29H76V0H96V96H0Z"))
        tx.replace_geometry(
            "ink", parse_path("M20.25 26.46875H76.25V30H20.25Z M70 70H72V72H70Z")
        )
    document = editor.snapshot.document
    assembled = BandPlans.document(
        document,
        "ink",
        "ink-marks",
        seed.band,
        marks,
        {"fill": seed.paint, "fill-opacity": "1"},
    )

    def candidates(score, values, **_kwargs):
        score(np.array([3.6]))
        score(values)

    monkeypatch.setattr(band_fit, "minimize", candidates)
    candidate, details = required(
        BandFit(evidence, guard).fit(
            document, assembled, "ink", seed, work, preserve_alpha=True
        )
    )
    assert details["width"] == 3.6
    assert details["native_alpha_exact"]
    np.testing.assert_array_equal(
        render(export_svg(document), evidence.source_size)[..., 3],
        render(export_svg(candidate), evidence.source_size)[..., 3],
    )


def test_bounds_and_interruption_discard_seed_and_fit_without_mutating_input(
    monkeypatch,
):
    evidence, guard, document, main, _ = fixture()
    monkeypatch.setattr(source_bands, "MAX_PROFILES", 0)
    assert (
        SourceBands(evidence, guard).seed(
            document, "ink", main, "nonzero", Work.start(10)
        )
        is None
    )
    with pytest.raises(StageInterruptedError):
        SourceBands(evidence, guard).seed(
            document, "ink", main, "nonzero", Work.start(0)
        )
    with pytest.raises(StageInterruptedError):
        BandFit(evidence, guard).fit(document, document, "ink", None, Work.start(0))
