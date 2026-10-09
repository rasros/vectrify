"""Broad gradient hypotheses screen whole owners and preserve native structure."""

from dataclasses import replace

import numpy as np
import pathops
import pytest

from tests.helpers import required
from tests.refine.test_cel_plan_families import prepared
from tests.refine.test_cel_plan_ownership import stripes
from vectrify.document import (
    Editor,
    Selection,
    export_svg,
    load_project,
    save_project,
)
from vectrify.document.join import transformed_geometry
from vectrify.refine.cel_plan import surface_models
from vectrify.refine.cel_plan.families import Families
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Work
from vectrify.refine.cel_plan.ownership import Partition
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.search import search
from vectrify.refine.cel_plan.surface_models import MaterialSurfaces


def ramp(alpha=255, hole=False):
    evidence = stripes(alpha=alpha, gradient=True, hole=hole)
    target = evidence.target.copy()
    shown = ~evidence.empty
    values = np.broadcast_to(16 + 3 * (np.arange(96) - 8), shown.shape)
    target[shown, 0] = values[shown]
    rgba = evidence.rgba.copy()
    rgba[..., :3][shown] = target[shown] / 255
    return replace(evidence, target=target, smooth=target, coarse=target, rgba=rgba)


@pytest.mark.parametrize("alpha", [255, 128])
@pytest.mark.parametrize("hole", [False, True])
def test_broad_material_refits_all_shades_preserving_alpha_holes_and_full_score(
    alpha, hole
):
    evidence = ramp(alpha, hole)
    frontier, state, options = prepared(evidence, layers=alpha != 255)
    factory = Families(evidence, build(evidence), options)
    # The established seed-color distance cannot reach all eight distant shades.
    assert all(len(ids) < 8 for ids, _limit in factory._groups(state, Work.start(10)))
    edits = list(factory.surface_models(state, Work.start(10)))
    whole = [p for p in edits if len(p.ids) == 8]
    assert whole
    edit = whole[0]
    assert edit.parameters[0] == "gradient"
    assert edit.details is not None
    assert edit.details["material_surface"]["removed_paths"] == 7
    assert edit.partition is not None
    edit.partition.validate(edit.document)
    assert state.partition is not None
    assert edit.partition.owners.keys() == state.partition.owners.keys()
    actual = render(export_svg(edit.document), evidence.source_size)
    before = render(state.svg, evidence.source_size)
    np.testing.assert_array_equal(actual[..., 3], before[..., 3])
    full = frontier.policy.evaluate(export_svg(edit.document))
    assert full.valid
    assert full.cost < state.snapshot.evaluation.cost / 2
    updated = LocalPolicy(frontier.policy).update(
        state.snapshot, export_svg(edit.document), edit.bounds, full.structure
    )
    assert updated.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
    assert updated.canvas.matches(actual)
    assert frontier.checkpoint(
        export_svg(edit.document),
        "Broad material",
        edit.details,
        updated.evaluation,
        raster=updated.canvas,
    )
    saved, _ = load_project(save_project(edit.document))
    required(
        Partition.from_metadata(
            edit.details.get("planning_surfaces") or edit.partition.metadata()
        )
    ).validate(saved)
    np.testing.assert_array_equal(
        render(export_svg(saved), evidence.source_size), actual
    )
    assert len(state.partition.surfaces) == (9 if alpha != 255 else 8)


def test_one_native_outlier_retains_its_entire_owner():
    evidence = ramp()
    target, rgba = evidence.target.copy(), evidence.rgba.copy()
    target[32, 23] = (0, 0, 0)
    rgba[32, 23, :3] = 0
    evidence = replace(evidence, target=target, smooth=target, coarse=target, rgba=rgba)
    _frontier, state, options = prepared(evidence)
    factory = MaterialSurfaces(Families(evidence, build(evidence), options), options)
    edits = list(factory(state, Work.start(10)))
    assert edits
    assert state.partition is not None
    protected = state.partition.owners[2]
    for edit in edits:
        assert protected not in edit.ids
        assert edit.partition is not None
        assert edit.partition.owners[2] == protected
        assert edit.document.geometry_for(protected) == state.document.geometry_for(
            protected
        )
        assert (
            edit.document.element(protected).attributes
            == state.document.element(protected).attributes
        )


def test_stable_shade_step_remains_two_distinct_materials():
    evidence = ramp(128)
    target = evidence.target.copy()
    target[evidence.labels > 4, 0] = 244
    target[(evidence.labels > 0) & (evidence.labels <= 4), 0] = 24
    rgba = evidence.rgba.copy()
    rgba[..., :3][~evidence.empty] = target[~evidence.empty] / 255
    evidence = replace(evidence, target=target, smooth=target, coarse=target, rgba=rgba)
    frontier, state, options = prepared(evidence, layers=True)
    factory = MaterialSurfaces(Families(evidence, build(evidence), options), options)
    edits = list(factory(state, Work.start(10)))
    assert edits
    for edit in edits:
        assert edit.details is not None
        members = set(edit.details["material_surface"]["source_members"])
        assert members <= {1, 2, 3, 4} or members <= {5, 6, 7, 8}
        assert frontier.policy.evaluate(export_svg(edit.document)).valid


def test_mild_broad_ramp_offers_flat_and_gradient_on_one_common_frontier():
    evidence = stripes(gradient=True)
    frontier, state, options = prepared(evidence)
    factory = MaterialSurfaces(Families(evidence, build(evidence), options), options)
    edits = list(factory(state, Work.start(10)))
    assert {p.parameters[0] for p in edits if len(p.ids) == 8} == {"flat", "gradient"}
    report = search(frontier, options, Work.start(10), factory)
    assert report["score_disagreements"] == 0
    assert frontier.select(0).metrics["material_surface"]["paint_model"] == "flat"
    assert frontier.select(100).metrics["material_surface"]["paint_model"] == "gradient"
    costs = [frontier.select(c).metrics["representation_cost"] for c in range(101)]
    assert costs == sorted(costs)


def test_new_model_boolean_failure_keeps_established_family_opportunities(monkeypatch):
    evidence = stripes()
    frontier, state, options = prepared(evidence)
    factory = Families(evidence, build(evidence), options)

    def broken(_self, _state, _work):
        raise pathops.PathOpsError("Injected broad material boolean failure")
        yield

    monkeypatch.setattr(MaterialSurfaces, "__call__", broken)
    edits = list(factory(state, Work.start(10)))
    assert edits
    assert factory.diagnostics["family_boolean_failures"] == 1
    assert any(frontier.policy.evaluate(export_svg(p.document)).valid for p in edits)


def test_distinct_ink_owner_is_retained_between_coherent_gradient_parts():
    evidence = ramp(128)
    labels, target, rgba = (
        evidence.labels.copy(),
        evidence.target.copy(),
        evidence.rgba.copy(),
    )
    marked = np.zeros(labels.shape, bool)
    marked[31:33, 8:88] = True
    labels[marked] = 9
    target[marked] = 16
    rgba[marked, :3] = 16 / 255
    evidence = replace(
        evidence,
        labels=labels,
        target=target,
        smooth=target,
        coarse=target,
        rgba=rgba,
        drawn=marked | np.isin(labels, (2, 4, 6, 8)),
    )
    frontier, state, options = prepared(evidence, layers=True)
    factory = MaterialSurfaces(Families(evidence, build(evidence), options), options)
    edits = list(factory(state, Work.start(10)))
    assert any(len(p.ids) == 8 for p in edits)
    assert state.partition is not None
    oid = state.partition.owners[9]
    for edit in edits:
        assert oid not in edit.ids
        assert edit.document.geometry_for(oid) == state.document.geometry_for(oid)
        assert (
            edit.document.element(oid).attributes
            == state.document.element(oid).attributes
        )
        assert frontier.policy.evaluate(export_svg(edit.document)).valid
        np.testing.assert_array_equal(
            render(export_svg(edit.document), evidence.source_size)[marked],
            render(state.svg, evidence.source_size)[marked],
        )


def test_broad_rgba_material_requires_actual_core_geometry():
    evidence = ramp(128)
    _frontier, state, options = prepared(evidence, layers=True)
    base = next(s for s in required(state.partition).surfaces if s.role == "underlay")
    editor = Editor(state.document, selection=Selection(whole_document=True))
    with editor.transaction("Remove actual alpha core") as transaction:
        transaction.delete_objects(frozenset({base.id}))
    state = replace(
        state,
        document=editor.snapshot.document,
        partition=Partition(
            tuple(s for s in required(state.partition).surfaces if s.id != base.id)
        ),
    )
    factory = MaterialSurfaces(Families(evidence, build(evidence), options), options)
    assert list(factory(state, Work.start(10))) == []
    assert factory.diagnostics["core_exclusions"] > 0


def test_independent_current_frames_and_scaled_evidence_keep_native_agreement():
    evidence = stripes(alpha=128)
    native = np.zeros((96, 128, 4), dtype=np.float32)
    native[24:56, 20:68] = evidence.rgba[::2, ::2]
    evidence = replace(
        evidence, rgba=native, source_size=(128, 96), offset=(20, 24), scale=(2, 2)
    )
    frontier, state, options = prepared(evidence, layers=True)
    editor = Editor(state.document, selection=Selection(whole_document=True))
    assert state.partition is not None
    for surface in state.partition.surfaces:
        if surface.role != "surface":
            continue
        with editor.transaction("Equivalent child frame") as transaction:
            transaction.set_attributes(surface.id, {"transform": "translate(5 -3)"})
            transaction.replace_geometry(
                surface.id,
                transformed_geometry(
                    state.document.geometry_for(surface.id), (1, 0, 0, 1, -5, 3)
                ),
            )
    document = editor.snapshot.document
    svg = export_svg(document)
    np.testing.assert_array_equal(
        render(svg, evidence.source_size), render(state.svg, evidence.source_size)
    )
    state = replace(
        state,
        document=document,
        svg=svg,
        snapshot=LocalPolicy(frontier.policy).start(svg, frontier.policy.evaluate(svg)),
    )
    factory = MaterialSurfaces(Families(evidence, build(evidence), options), options)
    edits = list(factory(state, Work.start(10)))
    assert edits
    for edit in edits:
        svg = export_svg(edit.document)
        full = frontier.policy.evaluate(svg)
        assert full.valid
        updated = LocalPolicy(frontier.policy).update(
            state.snapshot, svg, edit.bounds, full.structure
        )
        assert updated.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
        assert updated.canvas.matches(render(svg, evidence.source_size))


@pytest.mark.parametrize(("shift", "safe"), [(-30, False), (40, True)])
def test_order_proof_uses_only_the_moved_fragment_prefix(shift, safe):
    evidence = ramp()
    _frontier, state, options = prepared(evidence)
    assert state.partition is not None
    oid = state.partition.owners[4]
    editor = Editor(state.document, selection=Selection(whole_document=True))
    with editor.transaction("Intervening painted overlap") as transaction:
        transaction.replace_geometry(
            oid,
            transformed_geometry(
                state.document.geometry_for(oid), (1, 0, 0, 1, shift, 0)
            ),
        )
    changed = replace(state, document=editor.snapshot.document)
    factory = MaterialSurfaces(Families(evidence, build(evidence), options), options)
    # Owner four paints over the earlier fragment; moving that fragment above
    # it is a real overlap, even though the immutable source labels are disjoint.
    ids = tuple(required(state.partition).owners[i] for i in (1, 2, 3, 5, 6, 7, 8))
    assert factory._order_safe(changed, ids, Work.start(10)) is safe
    assert factory.diagnostics["order_proofs"] > 0


@pytest.mark.parametrize("overlap", [False, True])
def test_many_intervening_siblings_share_one_exact_prefix_proof(overlap, monkeypatch):
    evidence = ramp()
    _frontier, state, options = prepared(evidence)
    if overlap:
        assert state.partition is not None
        oid = state.partition.owners[4]
        editor = Editor(state.document, selection=Selection(whole_document=True))
        with editor.transaction(
            "One overlapping sibling among disjoint ones"
        ) as transaction:
            transaction.replace_geometry(
                oid,
                transformed_geometry(
                    state.document.geometry_for(oid), (1, 0, 0, 1, -30, 0)
                ),
            )
        state = replace(state, document=editor.snapshot.document)
    factory = MaterialSurfaces(Families(evidence, build(evidence), options), options)
    monkeypatch.setattr(surface_models, "MAX_ORDER_PROOFS", 1)
    ids = tuple(required(state.partition).owners[i] for i in (1, 8))
    assert factory._order_safe(state, ids, Work.start(10)) is not overlap
    assert factory.diagnostics["order_proofs"] == 1
    assert factory.diagnostics["order_limits"] == 0


@pytest.mark.parametrize("chunk", [47, 113])
def test_complete_streamed_screen_and_stop_do_not_publish_partial_ownership(
    chunk, monkeypatch
):
    evidence = ramp()
    frontier, state, options = prepared(evidence)
    monkeypatch.setattr(surface_models, "CHUNK_PIXELS", chunk)
    factory = MaterialSurfaces(Families(evidence, build(evidence), options), options)
    report = search(frontier, options, Work.start(10), factory)
    assert report["accepted"] > 0
    assert report["score_disagreements"] == 0
    assert (
        frontier.select(50).metrics["nodes"]
        < state.snapshot.evaluation.structure["nodes"]
    )
    work = Work.start(10)
    original = surface_models.prediction

    def stop(*args, **kwargs):
        value = original(*args, **kwargs)
        work.stop.set()
        return value

    monkeypatch.setattr(surface_models, "prediction", stop)
    assert list(factory(state, work)) == []
    assert frontier.baseline is not None
    assert state.svg == frontier.baseline.svg
    assert state.snapshot.canvas.matches(render(state.svg, evidence.source_size))


def test_bounded_analysis_does_not_replace_the_safe_state(monkeypatch):
    evidence = ramp()
    frontier, state, options = prepared(evidence)
    monkeypatch.setattr(surface_models, "MAX_PIXELS", 1)
    factory = MaterialSurfaces(Families(evidence, build(evidence), options), options)
    assert list(factory(state, Work.start(10))) == []
    assert factory.diagnostics["bounded_groups"] > 0
    assert frontier.baseline is not None
    assert frontier.baseline.svg == state.svg
