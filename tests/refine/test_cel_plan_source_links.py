"""Short existing junction links do not authorize source-gap completion."""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from tests.helpers import required
from tests.refine.test_cel_plan_families import prepared
from vectrify.document import export_svg, load_project, save_project
from vectrify.document.join import curve_path
from vectrify.refine import cel
from vectrify.refine.cel_plan import ink_models
from vectrify.refine.cel_plan.core_cells import CoreCells
from vectrify.refine.cel_plan.evidence import collect
from vectrify.refine.cel_plan.families import Families
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.ink import measure, measure_link
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Options, Work
from vectrify.refine.cel_plan.proposals import Operators
from vectrify.refine.cel_plan.score import render


def drawing(alpha=255, *, gap=False):
    path = "M20 48H44 M50 48H76 M44 20V48 M50 48V76"
    path += " M44 48H46 M48 48H50" if gap else " M44 48H50"
    svg = (
        '<svg width="96" height="96"><g opacity="'
        + str(alpha / 255)
        + '"><path d="M8 8H88V88H8Z" fill="#ad8665"/>'
        f'<path d="{path}" stroke="#202020" stroke-width="2" '
        'fill="none" stroke-linecap="butt"/></g></svg>'
    )
    rgba = render(svg, (96, 96))
    evidence = collect(
        Image.fromarray((rgba * 255).round().astype(np.uint8)),
        None,
        Options(),
        Work.start(10),
    )
    mask = (~evidence.empty) & (cel.lightness(evidence.target) < 65)
    return evidence, mask


@pytest.mark.parametrize("alpha", [255, 128, 64])
def test_existing_short_link_makes_supported_source_junctions_editably_connected(alpha):
    evidence, mask = drawing(alpha)
    before = mask.copy()
    models = ink_models.models(
        mask, evidence, Options(), Work.start(10), prune_spurs=True
    )
    assert models
    assert sum(m.details["junction_links"] for m in models) == 1
    contours = [sub for m in models for sub in m.geometry.subpaths]
    assert len(contours) == 5
    link = next(
        sub
        for sub in contours
        if np.linalg.norm(np.subtract(sub.nodes[-1].endpoint, sub.nodes[0].endpoint))
        < 8
    )
    for end in (link.nodes[0].endpoint, link.nodes[-1].endpoint):
        assert (
            sum(
                end in (sub.nodes[0].endpoint, sub.nodes[-1].endpoint)
                for sub in contours
            )
            == 3
        )
    assert any(curve_path(m.footprint).contains((47, 48)) for m in models)
    assert any(m.selected[48, 47] for m in models)
    assert all(not m.selected.flags.writeable for m in models)
    np.testing.assert_array_equal(mask, before)
    assert all(m.details["source_runs_scanned"] <= ink_models.MAX_RUNS for m in models)
    assert all(
        m.details["source_points_scanned"] <= ink_models.MAX_POINTS for m in models
    )


def test_a_real_gap_between_source_junctions_is_never_a_link():
    evidence, mask = drawing(gap=True)
    models = ink_models.models(
        mask, evidence, Options(), Work.start(10), prune_spurs=True
    )
    assert models
    assert all(m.details["junction_links"] == 0 for m in models)
    assert all(not m.selected[48, 47] for m in models)
    assert all(not curve_path(m.footprint).contains((47, 48)) for m in models)


def test_incompatible_source_paints_cannot_authorize_a_shared_style_link():
    evidence, mask = drawing()
    target, rgba = evidence.target.copy(), evidence.rgba.copy()
    _y, x = np.indices(mask.shape)
    other = mask & (x >= 48)
    target[other], rgba[other, :3] = 90, 90 / 255
    evidence = replace(evidence, target=target, rgba=rgba)
    models = ink_models.models(
        mask, evidence, Options(), Work.start(10), prune_spurs=True
    )
    assert len(models) == 2
    assert all(m.details["junction_links"] == 0 for m in models)
    assert all(m.details["source_link_candidates"] == 1 for m in models)
    assert (
        np.ptp(np.array([m.paint for m in models]), axis=0).min()
        > ink_models.PAINT_SPREAD
    )


@pytest.mark.parametrize("hidden", [False, True])
def test_sparse_ports_and_smoothed_or_existing_ink_cannot_prove_a_raw_gap(hidden):
    target = np.full((40, 60, 3), 180.0)
    target[19:22, 8:52] = 20
    visible = np.ones((40, 60), bool)
    # The two supplied ports skip the missing source pixel. Even a caller
    # reporting complete incident-body coverage cannot override raw source.
    target[19:22, 27] = 180 if not hidden else (0, 240, 15)
    visible[19:22, 27] = not hidden
    light = np.full((40, 60), 180.0)
    light[19:22, 8:52] = 20
    points = np.array(((24.5, 20.5), (30.5, 20.5)))
    assert (
        measure_link(
            points,
            target,
            2,
            light=light,
            visible=visible,
            junctions=lambda p: np.ones(len(p), bool),
            paint=np.full(3, 20),
        )
        is None
    )


def test_a_short_source_trough_has_a_stricter_proof_than_the_ordinary_length_rule():
    target = np.full((40, 60, 3), 180.0)
    target[19:22, 8:52] = 20
    points = np.column_stack((np.arange(24.5, 31), np.full(7, 20.5)))
    assert measure(points, target, 2) is None
    proof = measure_link(points, target, 2, visible=np.ones((40, 60), bool))
    assert proof is not None
    assert proof.support == 1
    assert proof.peak_gap == 0
    np.testing.assert_array_equal(proof.points[[0, -1]], points[[0, -1]])
    target[20:, :] = 20
    assert measure_link(points, target, 2, visible=np.ones((40, 60), bool)) is None


def test_no_profile_headroom_retains_the_short_link_as_fill(monkeypatch):
    evidence, mask = drawing()
    monkeypatch.setattr(ink_models, "MAX_RUNS", 5)
    models = ink_models.models(
        mask, evidence, Options(), Work.start(10), prune_spurs=True
    )
    assert models
    assert all(m.details["junction_links"] == 0 for m in models)
    assert all(m.details["source_link_profiles_scanned"] == 0 for m in models)
    assert all(m.details["source_runs_scanned"] <= 5 for m in models)


def test_link_cannot_change_incident_long_chain_geometry_width_or_paint(monkeypatch):
    evidence, mask = drawing()
    enabled = ink_models.models(
        mask, evidence, Options(), Work.start(10), prune_spurs=True
    )
    monkeypatch.setattr(ink_models, "measure_link", lambda *_a, **_k: None)
    baseline = ink_models.models(
        mask, evidence, Options(), Work.start(10), prune_spurs=True
    )
    assert len(enabled) == len(baseline)
    for model, previous in zip(enabled, baseline, strict=True):
        assert model.details["width"] == previous.details["width"]
        assert model.details["linecap"] == previous.details["linecap"]
        np.testing.assert_array_equal(model.paint, previous.paint)
        assert (
            model.geometry.subpaths[: len(previous.geometry.subpaths)]
            == previous.geometry.subpaths
        )


def test_cancelled_link_proof_discards_discovery_without_changing_source(monkeypatch):
    evidence, mask = drawing()
    before = mask.copy()
    work = Work.start(10)
    original = ink_models.measure_link

    def stopped(*args, **kwargs):
        result = original(*args, **kwargs)
        work.stop.set()
        return result

    monkeypatch.setattr(ink_models, "measure_link", stopped)
    assert ink_models.models(mask, evidence, Options(), work, prune_spurs=True) == ()
    np.testing.assert_array_equal(mask, before)


def test_held_short_link_leaves_incident_complete_chains_editable():
    evidence, mask = drawing()
    (model,) = ink_models.models(
        mask, evidence, Options(), Work.start(10), prune_spurs=True
    )
    owned = model.selected.copy()
    owned[48, 47] = False
    retained = ink_models.owned_model(model, owned, evidence, Work.start(10))
    assert retained is not None
    assert len(retained.geometry.subpaths) == 4
    assert retained.details["junction_links"] == 0
    assert retained.details["source_style_junction_links"] == 1
    assert retained.details["owner_excluded_runs"] == 1
    assert not retained.selected[48, 47]
    assert not curve_path(retained.footprint).contains((47, 48))
    assert retained.geometry.subpaths == model.geometry.subpaths[:4]


@pytest.mark.parametrize("alpha", [128, 64])
def test_joint_native_link_edit_keeps_alpha_ownership_and_reload(alpha):
    evidence, mask = drawing(alpha)
    evidence = replace(evidence, drawn=mask, line=mask)
    frontier, state, options = prepared(evidence, layers=True)
    graph = build(evidence)
    factory = CoreCells(
        Families(evidence, graph, options),
        options,
        joint=True,
        grouping="ward",
        boundary_fit="anchored",
        ink_support="connected",
    )
    edits = list(factory(state, Work.start(20)))
    assert edits, factory.diagnostics
    linked = [
        e
        for e in edits
        if any(
            m["junction_links"]
            for m in required(e.details)["core_material_cells"]["stroke_models"]
        )
    ]
    assert linked, factory.diagnostics
    for edit in linked:
        assert edit.partition is not None
        Operators(evidence, graph, options).validate_partition(
            edit.partition, Work.start(10)
        )
        assert edit.component is not None
        assert edit.component.validate(
            state.document,
            edit.document,
            state.partition,
            edit.partition,
            edit.ids,
            edit.bounds,
            Work.start(10),
        )
        svg = export_svg(edit.document)
        full = frontier.policy.evaluate(svg)
        assert full.valid, full.rejections
        actual = render(svg, evidence.source_size)
        np.testing.assert_array_equal(
            actual[..., 3], render(state.svg, evidence.source_size)[..., 3]
        )
        assert state.partition is not None
        assert edit.partition.follows(state.partition)
        local = LocalPolicy(frontier.policy).update(
            state.snapshot, svg, edit.bounds, full.structure
        )
        assert local.canvas.matches(actual)
        assert local.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
        restored, _ = load_project(save_project(edit.document))
        np.testing.assert_array_equal(
            render(export_svg(restored), evidence.source_size), actual
        )
