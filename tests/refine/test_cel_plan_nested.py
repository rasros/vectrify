"""RGB cavities can continue under retained marks; alpha holes cannot."""

from dataclasses import replace

import numpy as np
import pytest

from tests.refine.test_cel_plan_families import prepared
from tests.refine.test_cel_plan_overlays import source
from vectrify.document import (
    Editor,
    Selection,
    export_svg,
    import_svg,
    load_project,
    save_project,
)
from vectrify.document.paint import GradientStop, LinearGradient
from vectrify.refine.cel_plan import families, nested
from vectrify.refine.cel_plan.families import Families
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Work
from vectrify.refine.cel_plan.overlays import ClosedOverlays
from vectrify.refine.cel_plan.ownership import Partition
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.search import search


def marked(*, alpha=128, inner_alpha=None, early=False):
    evidence = source(alpha=alpha)
    labels, target, opacity = (
        evidence.labels.copy(),
        evidence.target.copy(),
        evidence.opacity.copy(),
    )
    labels[54:60, 55:61] = 6
    target[54:60, 55:61] = (245, 255, 220)
    if inner_alpha is not None:
        opacity[54:60, 55:61] = inner_alpha / 255
    if early:
        labels = np.array([6, 1, 2, 3, 4, 5, 0], dtype=np.int32)[labels]
    return replace(
        evidence,
        labels=labels,
        target=target,
        smooth=target,
        coarse=target,
        opacity=opacity,
        empty=opacity == 0,
        rgba=np.concatenate((target / 255, opacity[..., None]), axis=-1),
    )


@pytest.mark.parametrize("alpha", [253, 128, 64])
@pytest.mark.parametrize("early", [False, True])
def test_a_continuing_closed_surface_keeps_the_owned_opaque_mark_above_it(alpha, early):
    evidence = marked(alpha=alpha, early=early)
    frontier, state, options = prepared(evidence, layers=True)
    factory = ClosedOverlays(evidence, build(evidence), options)
    edits = [
        p for p in factory(state, Work.start(10)) if p.parameters[2] == (2, 3, 4, 5)
    ]
    assert {p.parameters[0] for p in edits} == {"ellipse", "contour"}
    oid = "cel-fill-0" if early else "cel-fill-6"
    old = render(state.svg, evidence.source_size)
    for edit in edits:
        assert edit.document.geometry_for(oid) == state.document.geometry_for(oid)
        assert (
            edit.document.element(oid).attributes
            == state.document.element(oid).attributes
        )
        overlay = next(s for s in edit.partition.surfaces if s.role == "overlay")
        assert overlay.covered == ((0,) if early else (6,))
        parent = edit.document.ancestry(oid)[-2]
        order = [c.id for c in parent.children]
        assert order.index(overlay.id) < order.index(oid)
        assert edit.partition.owners.keys() == state.partition.owners.keys()
        actual = render(export_svg(edit.document), evidence.source_size)
        # The opaque interior stays the same. Mixed edge pixels can change
        # when the original mark is drawn over the newly continuing material.
        np.testing.assert_array_equal(actual[56:58, 57:59], old[56:58, 57:59])
        np.testing.assert_array_equal(actual[..., 3], evidence.rgba[..., 3])
        full = frontier.policy.evaluate(export_svg(edit.document))
        assert full.valid
        assert full.cost < state.snapshot.evaluation.cost
        local = LocalPolicy(frontier.policy).update(
            state.snapshot, export_svg(edit.document), edit.bounds, full.structure
        )
        assert local.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
        assert frontier.checkpoint(
            export_svg(edit.document),
            "Nested closed surface",
            edit.details,
            local.evaluation,
            raster=local.canvas,
        )
    assert factory.diagnostics["nested_proposals"] >= 2


@pytest.mark.parametrize("inner_alpha", [0, 1, 64])
def test_alpha_holes_and_translucent_marks_are_not_reclassified_as_opaque_cavities(
    inner_alpha,
):
    evidence = marked(inner_alpha=inner_alpha)
    _frontier, state, options = prepared(evidence, layers=True)
    edits = list(
        ClosedOverlays(evidence, build(evidence), options)(state, Work.start(10))
    )
    assert not any(p.parameters[2] == (2, 3, 4, 5) for p in edits)
    old = render(state.svg, evidence.source_size)
    for edit in edits:
        np.testing.assert_array_equal(
            render(export_svg(edit.document), evidence.source_size)[54:60, 55:61],
            old[54:60, 55:61],
        )


def test_partially_transparent_current_paint_is_not_a_proof_of_an_opaque_mark():
    evidence = marked()
    _frontier, state, options = prepared(evidence, layers=True)
    editor = Editor(state.document, selection=Selection(whole_document=True))
    with editor.transaction("Make current mark translucent") as transaction:
        transaction.set_attributes("cel-fill-6", {"fill-opacity": "0.5"})
    state = replace(state, document=editor.snapshot.document)
    assert not any(
        p.parameters[2] == (2, 3, 4, 5)
        for p in ClosedOverlays(evidence, build(evidence), options)(
            state, Work.start(10)
        )
    )


def test_enclosure_cannot_move_a_distant_mark_with_the_same_primary_owner():
    evidence = marked()
    labels, target = evidence.labels.copy(), evidence.target.copy()
    labels[12:14, 12:14] = 6
    target[12:14, 12:14] = (245, 255, 220)
    evidence = replace(
        evidence,
        labels=labels,
        target=target,
        smooth=target,
        rgba=np.concatenate((target / 255, evidence.opacity[..., None]), axis=-1),
    )
    _frontier, state, options = prepared(evidence, layers=True)
    assert not any(
        p.parameters[2] == (2, 3, 4, 5)
        for p in ClosedOverlays(evidence, build(evidence), options)(
            state, Work.start(10)
        )
    )


def test_a_nested_gradient_keeps_its_geometry_stops_and_rendered_color():
    evidence = marked()
    _frontier, state, options = prepared(evidence, layers=True)
    editor = Editor(state.document, selection=Selection(whole_document=True))
    gradient = LinearGradient(
        (55, 54), (61, 60), (GradientStop(0, "#c0ffd0"), GradientStop(1, "#f8ffdd"))
    )
    with editor.transaction("Nested gradient") as transaction:
        transaction.set_fill("cel-fill-6", gradient)
    document = editor.snapshot.document
    state = replace(state, document=document, svg=export_svg(document))
    edits = [
        p
        for p in ClosedOverlays(evidence, build(evidence), options)(
            state, Work.start(10)
        )
        if p.parameters[2] == (2, 3, 4, 5)
    ]
    assert edits
    for edit in edits:
        assert edit.document.geometry_for("cel-fill-6") == document.geometry_for(
            "cel-fill-6"
        )
        assert edit.document.element("cel-fill-6") == document.element("cel-fill-6")
        server = document.element("cel-fill-6").get("fill")[5:-1]
        assert edit.document.element(server) == document.element(server)
        np.testing.assert_array_equal(
            render(export_svg(edit.document), evidence.source_size)[56:58, 57:59],
            render(state.svg, evidence.source_size)[56:58, 57:59],
        )


@pytest.mark.parametrize("factory_type", [ClosedOverlays, Families])
def test_nested_ownership_and_mark_pixels_survive_search_scaled_scope_and_reload(
    factory_type,
):
    evidence = marked(alpha=253, early=True)
    native = np.zeros((128, 160, 4), dtype=np.float32)
    native[30:90, 20:80] = evidence.rgba[::2, ::2]
    evidence = replace(
        evidence, rgba=native, source_size=(160, 128), offset=(20, 30), scale=(2, 2)
    )
    frontier, _state, options = prepared(evidence, layers=True)
    result = search(
        frontier,
        options,
        Work.start(10),
        factory_type(evidence, build(evidence), options),
    )
    assert result["accepted"] > 0
    assert result["score_disagreements"] == 0
    selected = frontier.select(50)
    partition = Partition.from_metadata(selected.metrics["planning_surfaces"])
    assert any(s.covered for s in partition.surfaces)
    document, _ = load_project(save_project(import_svg(selected.svg)))
    partition.validate(document)
    assert Partition.from_metadata(partition.metadata()) == partition
    np.testing.assert_array_equal(
        render(export_svg(document), evidence.source_size),
        render(selected.svg, evidence.source_size),
    )


@pytest.mark.parametrize("alpha", [253, 128, 64])
@pytest.mark.parametrize("early", [False, True])
def test_coherent_family_competes_as_a_base_beneath_the_preserved_mark(alpha, early):
    evidence = marked(alpha=alpha, early=early)
    frontier, state, options = prepared(evidence, layers=True)
    factory = Families(evidence, build(evidence), options)
    edits = [
        p for p in factory(state, Work.start(10)) if p.parameters[2] == (2, 3, 4, 5)
    ]
    adjacent = next(p for p in edits if not p.details.get("nested_surface"))
    continued = next(p for p in edits if p.details.get("nested_surface"))
    mark = "cel-fill-0" if early else "cel-fill-6"
    assert continued.partition.owners == adjacent.partition.owners
    assert continued.document.geometry_for(mark) == state.document.geometry_for(mark)
    assert continued.document.element(mark) == state.document.element(mark)
    base = next(s for s in continued.partition.surfaces if s.covered)
    assert base.covered == ((0,) if early else (6,))
    assert len(continued.document.geometry_for(base.id).subpaths) == 1
    assert len(adjacent.document.geometry_for(base.id).subpaths) == 2
    svg = export_svg(continued.document)
    full = frontier.policy.evaluate(svg)
    assert full.valid
    assert full.cost < frontier.policy.evaluate(export_svg(adjacent.document)).cost
    local = LocalPolicy(frontier.policy).update(
        state.snapshot, svg, continued.bounds, full.structure
    )
    assert local.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
    assert local.canvas.matches(render(svg, evidence.source_size))
    assert frontier.checkpoint(
        svg,
        "Continuing family",
        continued.details,
        local.evaluation,
        raster=local.canvas,
    )
    # Hiding the retained mark exposes material, rather than a hole or the
    # previous surrounding material. This proves the new base's actual fill.
    editor = Editor(continued.document, selection=Selection(whole_document=True))
    with editor.transaction("Inspect hidden family coverage") as transaction:
        transaction.set_fill(mark, "none")
    actual = render(export_svg(editor.snapshot.document), evidence.source_size)
    np.testing.assert_array_equal(actual[56:58, 57:59], actual[60:62, 57:59])
    assert not any(s.covered for s in state.partition.surfaces)


@pytest.mark.parametrize("inner_alpha", [0, 1, 64])
def test_family_can_merge_around_a_true_hole_without_continuing_beneath_it(inner_alpha):
    evidence = marked(inner_alpha=inner_alpha)
    _frontier, state, options = prepared(evidence, layers=True)
    factory = Families(evidence, build(evidence), options)
    edits = [
        p for p in factory(state, Work.start(10)) if p.parameters[2] == (2, 3, 4, 5)
    ]
    assert edits
    assert not any(p.details.get("nested_surface") for p in edits)
    for edit in edits:
        np.testing.assert_array_equal(
            render(export_svg(edit.document), evidence.source_size)[..., 3],
            render(state.svg, evidence.source_size)[..., 3],
        )
    assert factory.nesting_rejections


@pytest.mark.parametrize("limit", ["crop", "marks", "core"])
def test_nested_discovery_limits_preserve_the_adjacent_family_competitor(
    limit, monkeypatch
):
    evidence = marked()
    _frontier, state, options = prepared(evidence, layers=True)
    if limit == "crop":
        monkeypatch.setattr(families, "MAX_CROP_PIXELS", 4)
    elif limit == "marks":
        monkeypatch.setattr(nested, "MAX_MARKS", 0)
    else:
        editor = Editor(state.document, selection=Selection(whole_document=True))
        core = next(s.id for s in state.partition.surfaces if s.role == "underlay")
        with editor.transaction(
            "A current core no longer covers the cavity"
        ) as transaction:
            transaction.set_attributes(core, {"transform": "translate(200 0)"})
        state = replace(state, document=editor.snapshot.document)
    factory = Families(evidence, build(evidence), options)
    edits = [
        p for p in factory(state, Work.start(10)) if p.parameters[2] == (2, 3, 4, 5)
    ]
    assert edits
    assert not any(p.details.get("nested_surface") for p in edits)
    assert not any(s.covered for s in state.partition.surfaces)


@pytest.mark.parametrize("factory_type", [Families, ClosedOverlays])
def test_moving_a_nested_mark_cannot_cross_unrelated_covering_paint(factory_type):
    evidence = marked(early=True)
    _frontier, state, options = prepared(evidence, layers=True)
    parent = state.document.ancestry("cel-fill-0")[-2]
    extra = import_svg('<svg><path id="cover" fill="#f00" d="M56 55H59V58H56Z"/></svg>')
    editor = Editor(state.document, selection=Selection(whole_document=True))
    with editor.transaction(
        "Interpose unrelated paint above the existing mark"
    ) as transaction:
        transaction.insert_object(
            parent.id,
            extra.element("cover"),
            index=2,
            geometries=(extra.geometry_for("cover"),),
        )
    state = replace(state, document=editor.snapshot.document)
    factory = factory_type(evidence, build(evidence), options)
    edits = [
        p for p in factory(state, Work.start(10)) if p.parameters[2] == (2, 3, 4, 5)
    ]
    assert not any(
        p.details.get("nested_surface") or p.details.get("closed_overlay")
        for p in edits
    )
    assert (
        factory.diagnostics[
            "nested_order_exclusions"
            if factory_type is Families
            else "order_exclusions"
        ]
        > 0
    )


@pytest.mark.parametrize("factory_type", [Families, ClosedOverlays])
@pytest.mark.parametrize("core_present", [False, True])
def test_opaque_current_marks_with_source_alpha_variation_require_an_actual_core(
    factory_type, core_present
):
    evidence = marked(alpha=252, inner_alpha=253)
    frontier, state, options = prepared(evidence, layers=True)
    mark = state.document.element("cel-fill-6")
    assert mark.get("fill-opacity") == "1"
    if not core_present:
        editor = Editor(state.document, selection=Selection(whole_document=True))
        core = next(s.id for s in state.partition.surfaces if s.role == "underlay")
        with editor.transaction(
            "Core no longer contains the actual marks"
        ) as transaction:
            transaction.set_attributes(core, {"transform": "translate(200 0)"})
        state = replace(state, document=editor.snapshot.document)
    factory = factory_type(evidence, build(evidence), options)
    edits = [
        p
        for p in factory(state, Work.start(10))
        if p.parameters[2] == (2, 3, 4, 5)
        and (p.details.get("nested_surface") or p.details.get("closed_overlay"))
    ]
    assert bool(edits) == core_present
    if not core_present:
        assert factory.nesting_rejections["unproved-source-alpha-variation"] > 0
    for edit in edits:
        svg = export_svg(edit.document)
        assert edit.document.element("cel-fill-6") == mark
        assert frontier.policy.evaluate(svg).valid
        np.testing.assert_array_equal(
            render(svg, evidence.source_size)[..., 3],
            render(state.svg, evidence.source_size)[..., 3],
        )
