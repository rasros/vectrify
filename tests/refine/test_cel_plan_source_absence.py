"""Raw absence constrains actual caps, native phase and retained junctions."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from tests.refine.test_cel_plan_ink_models import drawing
from tests.refine.test_cel_plan_source_links import drawing as junction
from vectrify.document.svg import parse_path
from vectrify.refine.cel_plan import ink_models, source_absence
from vectrify.refine.cel_plan.line_fidelity import SourceLineGuard, SourceProfile
from vectrify.refine.cel_plan.model import Options, StageInterruptedError, Work
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.source_absence import SourceAbsence


def observed_gap():
    evidence, _mask = drawing(gap=True)
    points = np.column_stack((np.arange(20.25, 76.26, 4), np.full(15, 28.25)))
    return evidence, [SourceProfile.at(points, 3)]


def test_absence_positions_are_independent_copied_source_observations():
    evidence, profiles = observed_gap()
    guard = SourceLineGuard(evidence.rgba, profiles)
    points = guard.gap_centres()
    assert len(points) > 0
    assert not points.flags.writeable
    assert not np.shares_memory(points, profiles[0].points)
    assert ((points[:, 0] > 43) & (points[:, 0] < 53)).all()
    with pytest.raises(ValueError, match="point limit"):
        guard.gap_centres(limit=1)
    complete, _ = drawing()
    empty = SourceLineGuard(complete.rgba, []).gap_centres()
    assert empty.shape == (0, 2)
    assert not empty.flags.writeable


def test_reused_guard_retains_frozen_absence_after_raw_identity_changes():
    evidence, profiles = observed_gap()
    guard = SourceLineGuard(evidence.rgba, profiles)
    expected = guard.gap_centres()
    profiles[0].points.flags.writeable = True
    profiles[0].points[:] += 20
    proof = SourceAbsence(evidence, (), Work.start(10), intervals=True, guard=guard)
    np.testing.assert_array_equal(proof.points, expected)
    assert proof.source_breaks is not None
    assert proof.source_breaks(profiles[0]) is guard.source_breaks(profiles[0])
    assert not proof.permits(parse_path("M20 28H76"), 3, "butt", Work.start(10))
    with pytest.raises(ValueError, match="guard must match"):
        SourceAbsence(
            evidence,
            (),
            Work.start(10),
            guard=SourceLineGuard(evidence.rgba[:95], ()),
        )


@pytest.mark.parametrize("tile", [8, 16, 64])
@pytest.mark.parametrize("cap", ["round", "butt"])
@pytest.mark.parametrize(
    "path",
    ["M20 28H76", "M20 28H42 M54 28H76", "M20 28H46", "M20 32H76V28"],
)
def test_native_query_tiles_match_full_stroke_including_seams_and_caps(
    monkeypatch, tile, cap, path
):
    evidence, profiles = observed_gap()
    monkeypatch.setattr(source_absence, "TILE_SIDE", tile)
    proof = SourceAbsence(evidence, profiles, Work.start(10))
    alpha = render(
        f'<svg width="96" height="96"><path d="{path}" fill="none" '
        f'stroke="black" stroke-width="3" stroke-linecap="{cap}" '
        'stroke-linejoin="round"/></svg>',
        (96, 96),
    )[..., 3]
    actual = map_coordinates(
        alpha,
        [proof.points[:, 1] - 0.5, proof.points[:, 0] - 0.5],
        order=1,
        mode="constant",
        cval=0,
    )
    expected = bool(np.all(actual <= source_absence.ALPHA_TOLERANCE))
    assert proof.permits(parse_path(path), 3, cap, Work.start(10)) == expected
    assert expected == (path != "M20 28H76" and path != "M20 28H46")


@pytest.mark.parametrize(
    "bound",
    [
        "MAX_NATIVE_PIXELS",
        "MAX_PROFILE_SAMPLES",
        "MAX_POINTS",
        "MAX_TILES",
        "MAX_MASK_BYTES",
    ],
)
def test_independent_proof_bounds_precede_rendering(monkeypatch, bound):
    evidence, profiles = observed_gap()
    monkeypatch.setattr(source_absence, bound, 1)
    if bound == "MAX_TILES":
        monkeypatch.setattr(source_absence, "TILE_SIDE", 8)
    monkeypatch.setattr(
        source_absence, "render", lambda *_: pytest.fail("Bound reached rasterizer")
    )
    with pytest.raises(ValueError, match=r"bound|limit"):
        SourceAbsence(evidence, profiles, Work.start(10))


def test_frame_mismatch_and_mid_tile_stop_cannot_publish_a_positive_proof(monkeypatch):
    evidence, profiles = observed_gap()
    with pytest.raises(ValueError, match="frame"):
        SourceAbsence(replace(evidence, source_size=(95, 96)), profiles, Work.start(10))
    proof = SourceAbsence(evidence, profiles, Work.start(10))
    work = Work.start(10)
    original = source_absence.render

    def stopped(*args):
        pixels = original(*args)
        work.stop.set()
        return pixels

    monkeypatch.setattr(source_absence, "render", stopped)
    with pytest.raises(StageInterruptedError):
        proof.permits(parse_path("M20 28H42 M54 28H76"), 3, "round", work)


@pytest.mark.parametrize("frame", [(1, 1, 0, 0), (1.5, 0.75, 7.25, -3.5)])
def test_constraint_uses_native_observations_independently_of_analysis_frame(frame):
    evidence, profiles = observed_gap()
    sx, sy, ox, oy = frame
    evidence = replace(evidence, scale=(sx, sy), offset=(ox, oy))
    proof = SourceAbsence(evidence, profiles, Work.start(10))
    assert not proof.permits(parse_path("M20 28H76"), 3, "round", Work.start(10))
    assert proof.permits(parse_path("M20 28H42 M54 28H76"), 3, "round", Work.start(10))
    assert not proof.indices.flags.writeable
    assert all(not a.flags.writeable for query in proof.queries for a in query)


@pytest.mark.parametrize("alpha", [255, 128, 64])
def test_source_absence_rejects_a_coarse_bridge_without_inventing_new_ends(alpha):
    evidence, mask = drawing(alpha, gap=True)
    mask[26:30, 20:76] = True  # Detection's connected mask crosses raw source absence.
    before = mask.copy()
    baseline = ink_models.models(mask, evidence, Options(), Work.start(10))
    assert baseline
    assert any(m.selected[28, 48] for m in baseline)
    profiles = []
    assert (
        ink_models.models(
            mask,
            evidence,
            Options(),
            Work.start(10),
            source_absence=True,
            source_profiles=profiles,
        )
        == ()
    )
    assert profiles  # Complete inspected bank includes the rejected whole chain.
    np.testing.assert_array_equal(mask, before)


@pytest.mark.parametrize("alpha", [255, 128, 64])
def test_supported_disconnected_chains_keep_geometry_paint_width_and_real_gap(alpha):
    evidence, mask = drawing(alpha, gap=True)
    baseline = ink_models.models(mask, evidence, Options(), Work.start(10))
    constrained = ink_models.models(
        mask, evidence, Options(), Work.start(10), source_absence=True
    )
    assert len(constrained) == len(baseline)
    for model, previous in zip(constrained, baseline, strict=True):
        assert model.geometry == previous.geometry
        assert model.details["width"] == previous.details["width"]
        np.testing.assert_array_equal(model.paint, previous.paint)
        np.testing.assert_array_equal(model.selected, previous.selected)


def test_absent_incident_hosts_cannot_leave_an_orphan_short_link(monkeypatch):
    evidence, mask = junction()

    class RejectLeftHosts:
        points = np.empty((0, 2))

        def __init__(self, *_args):
            pass

        def permits(self, geometry, _width, _cap, _work):
            return all(
                not any(node.endpoint[0] < 48 for node in sub.nodes)
                or np.linalg.norm(
                    np.subtract(sub.nodes[-1].endpoint, sub.nodes[0].endpoint)
                )
                < 8
                for sub in geometry.subpaths
            )

    baseline = ink_models.models(
        mask, evidence, Options(), Work.start(10), prune_spurs=True
    )
    assert sum(m.details["junction_links"] for m in baseline) == 1
    monkeypatch.setattr(ink_models, "SourceAbsence", RejectLeftHosts)
    found = ink_models.models(
        mask, evidence, Options(), Work.start(10), prune_spurs=True, source_absence=True
    )
    assert found
    assert sum(m.details["junction_links"] for m in found) == 0
    assert sum(len(m.geometry.subpaths) for m in found) == 2
    assert all(m.details["source_absence_excluded_runs"] == 3 for m in found)


@pytest.mark.parametrize("alpha", [255, 128, 64])
def test_supported_source_junctions_and_explicit_width_survive_the_constraint(alpha):
    evidence, mask = junction(alpha)
    options = Options(line_width=2)
    baseline = ink_models.models(
        mask, evidence, options, Work.start(10), prune_spurs=True
    )
    constrained = ink_models.models(
        mask, evidence, options, Work.start(10), prune_spurs=True, source_absence=True
    )
    assert len(constrained) == len(baseline)
    assert sum(m.details["junction_links"] for m in constrained) == 1
    for model, previous in zip(constrained, baseline, strict=True):
        assert model.geometry == previous.geometry
        assert model.details["width"] == 2
        np.testing.assert_array_equal(model.paint, previous.paint)
        np.testing.assert_array_equal(model.selected, previous.selected)


def test_failed_compound_style_cannot_authorize_another_styles_link(monkeypatch):
    evidence, mask = junction()
    target, rgba = evidence.target.copy(), evidence.rgba.copy()
    left = mask & (np.indices(mask.shape)[1] < 48)
    target[left], rgba[left, :3] = 40, 40 / 255
    target[47:49, 44:50], rgba[47:49, 44:50, :3] = 32, 32 / 255
    evidence = replace(evidence, target=target, rgba=rgba)
    normal = ink_models.carried

    def distinct_caps(*args, **kwargs):
        result = normal(*args, **kwargs)
        if kwargs.get("cap") is None:
            result = [
                (*item[:-1], "butt" if item[0][:, 0].mean() < 48 else "round")
                for item in result
            ]
        return result

    class RejectCompoundLeft:
        points = np.empty((0, 2))

        def __init__(self, *_args):
            pass

        def permits(self, geometry, _width, cap, _work):
            return cap != "butt" or len(geometry.subpaths) == 1

    monkeypatch.setattr(ink_models, "carried", distinct_caps)
    baseline = ink_models.models(
        mask, evidence, Options(), Work.start(10), prune_spurs=True
    )
    assert len(baseline) == 2
    assert sum(m.details["junction_links"] for m in baseline) == 1
    assert (
        next(m for m in baseline if m.details["linecap"] == "round").details[
            "junction_links"
        ]
        == 1
    )
    monkeypatch.setattr(ink_models, "SourceAbsence", RejectCompoundLeft)
    constrained = ink_models.models(
        mask, evidence, Options(), Work.start(10), prune_spurs=True, source_absence=True
    )
    assert len(constrained) == 1
    assert constrained[0].details["linecap"] == "round"
    assert constrained[0].details["junction_links"] == 0
    assert len(constrained[0].geometry.subpaths) == 2


def test_mid_body_constraint_stop_publishes_no_model_or_partial_source_bank(
    monkeypatch,
):
    evidence, mask = drawing(gap=True)
    mask[26:30, 20:76] = True
    before = mask.copy()
    profiles = []
    work = Work.start(10)
    original = source_absence.render

    def stopped(*args):
        pixels = original(*args)
        work.stop.set()
        return pixels

    monkeypatch.setattr(source_absence, "render", stopped)
    assert (
        ink_models.models(
            mask,
            evidence,
            Options(),
            work,
            source_absence=True,
            source_profiles=profiles,
        )
        == ()
    )
    assert profiles == []
    np.testing.assert_array_equal(mask, before)
