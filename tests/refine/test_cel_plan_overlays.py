"""Closed RGBA replacements retain ownership, underpaint and exact validation."""

from dataclasses import replace

import numpy as np
import pytest

from tests.refine.test_cel_plan_families import prepared
from tests.refine.test_cel_plan_layers import evidence as opaque
from vectrify.document import (
    Editor,
    Selection,
    export_svg,
    import_svg,
    load_project,
    save_project,
)
from vectrify.document.join import path_style, union_geometry
from vectrify.document.paint import GradientStop, LinearGradient
from vectrify.refine.cel_plan import overlays
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Work
from vectrify.refine.cel_plan.overlays import ClosedOverlays
from vectrify.refine.cel_plan.ownership import Partition
from vectrify.refine.cel_plan.score import render, representation
from vectrify.refine.cel_plan.search import search


def source(*, alpha=128, hole=False, irregular=False, gradient=False):
    evidence = opaque()
    y, x = np.mgrid[:120, :120]
    own = evidence.labels == 2
    if irregular:
        own = (np.abs(x - 60) + np.abs(y - 60) <= 14) & ~((x < 60) & (y > 63))
    if hole:
        own[57:64, 57:64] = False
    labels = (x >= 60).astype(np.int32)
    labels[own] = 2 + np.clip((x[own] - 48) // 6, 0, 3)
    target = np.array(
        [(200, 150, 100), (140, 100, 65), *[(60, 150, 110)] * 4], dtype=np.float32
    )[labels]
    if gradient:
        target[own, 0] = 40 + 2 * (x[own] - 48)
    opacity = np.full(labels.shape, alpha / 255, dtype=np.float32)
    rgba = np.concatenate((target / 255, opacity[..., None]), axis=-1)
    return replace(
        evidence,
        labels=labels,
        target=target,
        smooth=target,
        coarse=target,
        rgba=rgba,
        opacity=opacity,
    )


@pytest.mark.parametrize("alpha", [253, 128, 64])
def test_closed_family_removes_fragments_with_native_checkpoint_agreement(alpha):
    evidence = source(alpha=alpha)
    frontier, state, options = prepared(evidence, layers=True)
    factory = ClosedOverlays(
        evidence, build(evidence), replace(options, gradients=False)
    )
    edits = [
        p for p in factory(state, Work.start(10)) if p.parameters[2] == (2, 3, 4, 5)
    ]
    assert {p.parameters[0] for p in edits} == {"ellipse", "contour"}
    for edit in edits:
        svg = export_svg(edit.document)
        full = frontier.policy.evaluate(svg)
        updated = LocalPolicy(frontier.policy).update(
            state.snapshot, svg, edit.bounds, full.structure
        )
        assert full.valid
        assert full.cost < state.snapshot.evaluation.cost
        assert updated.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
        assert updated.canvas.matches(render(svg, evidence.source_size))
        assert frontier.checkpoint(
            svg,
            "Closed overlay",
            edit.details,
            updated.evaluation,
            raster=updated.canvas,
        )
        assert edit.partition.owners.keys() == state.partition.owners.keys()
        edit.partition.validate(edit.document)
        overlay = next(s for s in edit.partition.surfaces if s.role == "overlay")
        assert overlay.members == (2, 3, 4, 5)
        assert overlay.id in edit.details["geometry_constraints"]
        assert len([s for s in edit.partition.surfaces if s.role == "surface"]) == 2
        np.testing.assert_array_equal(
            render(svg, evidence.source_size)[..., 3], evidence.rgba[..., 3]
        )
    assert not any(s.role == "overlay" for s in state.partition.surfaces)
    assert state.snapshot.canvas.matches(
        render(export_svg(state.document), evidence.source_size)
    )


def test_closed_rgb_mark_precedes_complex_paint_and_reaches_native_evaluation():
    evidence = source()
    y, x = np.mgrid[:120, :120]
    radius = np.hypot(x - 60, y - 60)
    labels = (x >= 60).astype(np.int32)
    labels[radius <= 14] = 2
    labels[radius <= 11] = 3
    labels[(x >= 10) & (x < 38) & (y >= 70) & (y < 106) & (y % 4 < 2)] = 4
    target = np.array(
        [(200, 150, 100), (140, 100, 65), (32, 32, 32), (60, 150, 110), (160, 105, 70)],
        dtype=np.float32,
    )[labels]
    rgba = np.concatenate((target / 255, evidence.opacity[..., None]), axis=-1)
    evidence = replace(evidence, labels=labels, target=target, rgba=rgba)
    frontier, state, options = prepared(evidence, layers=True)
    factory = ClosedOverlays(evidence, build(evidence), options)
    groups = factory.groups(state, Work.start(10))
    ids, members = next(groups)
    groups.close()
    assert members == (2,)
    assert ids == (state.partition.owners[2],)
    assert factory.diagnostics["enclosed_priorities"] == 1
    edits = list(factory(state, Work.start(10)))
    ellipses = [
        p for p in edits if p.parameters[0] == "ellipse" and p.parameters[2] == (2,)
    ]
    assert ellipses
    for edit in ellipses:
        svg = export_svg(edit.document)
        full = frontier.policy.evaluate(svg)
        updated = LocalPolicy(frontier.policy).update(
            state.snapshot, svg, edit.bounds, full.structure
        )
        assert full.valid
        assert updated.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
        assert edit.partition.owners.keys() == state.partition.owners.keys()
        np.testing.assert_array_equal(
            render(svg, evidence.source_size)[60, 60],
            render(state.svg, evidence.source_size)[60, 60],
        )


def test_underpaint_continues_both_neighbor_shades_in_the_old_footprint():
    evidence = source()
    _frontier, state, options = prepared(evidence, layers=True)
    edit = next(
        p
        for p in ClosedOverlays(evidence, build(evidence), options)(
            state, Work.start(10)
        )
        if p.parameters[2] == (2, 3, 4, 5)
    )
    overlay = next(s for s in edit.partition.surfaces if s.role == "overlay")
    editor = Editor(edit.document, selection=Selection(whole_document=True))
    with editor.transaction("Inspect restored paint") as transaction:
        transaction.set_fill(overlay.id, "none")
    actual = render(export_svg(editor.snapshot.document), evidence.source_size)
    before = render(state.svg, evidence.source_size)
    np.testing.assert_array_equal(actual[60, 55], before[40, 55])
    np.testing.assert_array_equal(actual[60, 65], before[40, 65])


def test_gradient_overlay_and_underpaint_keep_their_original_frames():
    evidence = source(gradient=True)
    frontier, state, options = prepared(evidence, layers=True)
    editor = Editor(state.document, selection=Selection(whole_document=True))
    with editor.transaction("Neighbor gradient") as transaction:
        transaction.set_fill(
            "cel-fill-0",
            LinearGradient(
                (0, 0),
                (120, 0),
                (GradientStop(0, "#c89664"), GradientStop(1, "#cc9664")),
            ),
        )
    document = editor.snapshot.document
    svg = export_svg(document)
    state = replace(
        state,
        document=document,
        svg=svg,
        snapshot=LocalPolicy(frontier.policy).start(svg, frontier.policy.evaluate(svg)),
    )
    edits = [
        p
        for p in ClosedOverlays(evidence, build(evidence), options)(
            state, Work.start(10)
        )
        if p.parameters[1] == "gradient" and p.parameters[2] == (2, 3, 4, 5)
    ]
    assert edits
    edit = edits[0]
    overlay = next(s for s in edit.partition.surfaces if s.role == "overlay")
    assert edit.document.element(overlay.id).get("fill").startswith("url(")
    editor = Editor(edit.document, selection=Selection(whole_document=True))
    with editor.transaction("Inspect gradient continuation") as transaction:
        transaction.set_fill(overlay.id, "none")
    actual = render(export_svg(editor.snapshot.document), evidence.source_size)
    before = render(svg, evidence.source_size)
    np.testing.assert_array_equal(actual[60, 50:59], before[40, 50:59])
    assert frontier.policy.evaluate(export_svg(edit.document)).valid


def test_hole_and_irregular_corners_are_not_completed_as_ellipses():
    for evidence in (source(hole=True), source(irregular=True)):
        _frontier, state, options = prepared(evidence, layers=True)
        edits = list(
            ClosedOverlays(evidence, build(evidence), options)(state, Work.start(10))
        )
        assert not any(p.parameters[0] == "ellipse" for p in edits)
        for edit in edits:
            actual = render(export_svg(edit.document), evidence.source_size)
            before = render(state.svg, evidence.source_size)
            np.testing.assert_array_equal(
                actual[57:64, 57:64, 3], before[57:64, 57:64, 3]
            )


def test_adjacent_translucency_without_a_proved_core_has_no_overlay():
    evidence = source(alpha=64)
    _frontier, state, options = prepared(evidence, layers=False)
    factory = ClosedOverlays(evidence, build(evidence), options)
    assert not list(factory(state, Work.start(10)))
    assert factory.diagnostics["no_restoration"] > 0


def test_source_label_order_cannot_put_the_new_overlay_beneath_restored_bases():
    evidence = source()
    mapping = np.array([5, 1, 2, 3, 4, 0])
    evidence = replace(evidence, labels=mapping[evidence.labels].astype(np.int32))
    frontier, state, options = prepared(evidence, layers=True)
    edits = [
        p
        for p in ClosedOverlays(evidence, build(evidence), options)(
            state, Work.start(10)
        )
        if p.parameters[2] == (0, 2, 3, 4)
    ]
    assert edits
    for edit in edits:
        overlay = next(s for s in edit.partition.surfaces if s.role == "overlay")
        parent = edit.document.ancestry(overlay.id)[-2]
        order = [c.id for c in parent.children]
        assert all(
            order.index(s.id) < order.index(overlay.id)
            for s in edit.partition.surfaces
            if s.covered
        )
        assert frontier.policy.evaluate(export_svg(edit.document)).valid
        actual = render(export_svg(edit.document), evidence.source_size)
        np.testing.assert_array_equal(
            actual[60, 60], render(state.svg, evidence.source_size)[60, 60]
        )


def test_an_unrelated_covering_shape_prevents_an_unproved_order_change():
    evidence = source()
    _frontier, state, options = prepared(evidence, layers=True)
    editor = Editor(state.document, selection=Selection(whole_document=True))
    parent = state.document.ancestry("cel-fill-5")[-2]
    extra = import_svg('<svg><path id="cover" fill="#000" d="M54 54H66V66H54Z"/></svg>')
    with editor.transaction("Cover a local part") as transaction:
        transaction.reorder_object("cel-fill-0", len(parent.children) - 1)
        transaction.insert_object(
            parent.id,
            extra.element("cover"),
            index=len(parent.children) - 1,
            geometries=(extra.geometry_for("cover"),),
        )
    state = replace(state, document=editor.snapshot.document)
    factory = ClosedOverlays(evidence, build(evidence), options)
    edits = list(factory(state, Work.start(10)))
    assert not any(p.parameters[2] == (2, 3, 4, 5) for p in edits)
    assert factory.diagnostics["order_exclusions"] > 0


def test_overlapping_bounds_do_not_block_a_geometrically_disjoint_sibling():
    evidence = source()
    _frontier, state, options = prepared(evidence, layers=True)
    editor = Editor(state.document, selection=Selection(whole_document=True))
    parent = state.document.ancestry("cel-fill-5")[-2]
    extra = import_svg(
        '<svg><path id="ring" fill="#000" fill-rule="evenodd" '
        'd="M40 40H80V80H40Z M44 44H76V76H44Z"/></svg>'
    )
    with editor.transaction(
        "Add disjoint paint with overlapping bounds"
    ) as transaction:
        transaction.reorder_object("cel-fill-0", len(parent.children) - 1)
        transaction.insert_object(
            parent.id,
            extra.element("ring"),
            index=len(parent.children) - 1,
            geometries=(extra.geometry_for("ring"),),
        )
    state = replace(state, document=editor.snapshot.document)
    factory = ClosedOverlays(evidence, build(evidence), options)
    edits = [
        p for p in factory(state, Work.start(10)) if p.parameters[2] == (2, 3, 4, 5)
    ]
    assert edits
    assert factory.diagnostics["order_proofs"] > 0
    for edit in edits:
        assert edit.document.geometry_for("ring") == state.document.geometry_for("ring")


def test_faint_intentional_mark_outside_the_overlay_keeps_its_geometry_and_alpha():
    evidence = source()
    labels, target, opacity = (
        evidence.labels.copy(),
        evidence.target.copy(),
        evidence.opacity.copy(),
    )
    labels[12, 12:15] = 6
    target[12, 12:15] = (32, 44, 55)
    opacity[12, 12:15] = 1 / 255
    evidence = replace(
        evidence,
        labels=labels,
        target=target,
        smooth=target,
        opacity=opacity,
        rgba=np.concatenate((target / 255, opacity[..., None]), axis=-1),
    )
    _frontier, state, options = prepared(evidence, layers=True)
    edits = list(
        ClosedOverlays(evidence, build(evidence), options)(state, Work.start(10))
    )
    assert edits
    before = render(state.svg, evidence.source_size)
    for edit in edits:
        assert edit.document.geometry_for("cel-fill-6") == state.document.geometry_for(
            "cel-fill-6"
        )
        np.testing.assert_array_equal(
            render(export_svg(edit.document), evidence.source_size)[12, 12:15],
            before[12, 12:15],
        )


def test_discovery_budget_does_not_expire_other_operators(monkeypatch):
    evidence = source()
    _frontier, state, options = prepared(evidence, layers=True)
    factory = ClosedOverlays(evidence, build(evidence), options)
    work = Work.start(10)

    def expired(_state, local_work):
        assert local_work is not work
        assert local_work.deadline < work.deadline
        local_work.deadline = -1
        yield ("cel-fill-2", "cel-fill-3"), (2, 3)

    monkeypatch.setattr(factory, "groups", expired)
    assert not list(factory(state, work))
    assert factory.diagnostics["time_bounded"] == 1
    assert not work.interrupted


def test_family_discovery_cannot_consume_every_individual_shape_slot(monkeypatch):
    evidence = source()
    _frontier, state, options = prepared(evidence, layers=True)
    factory = ClosedOverlays(evidence, build(evidence), options)

    def families(_state, _work, **_kwargs):
        yield ("cel-fill-2", "cel-fill-3"), 24
        yield ("cel-fill-3", "cel-fill-4"), 24
        yield ("cel-fill-4", "cel-fill-5"), 24

    monkeypatch.setattr(factory.families, "_groups", families)
    monkeypatch.setattr(overlays, "MAX_GROUPS", 2)
    groups = list(factory.groups(state, Work.start(10)))
    assert len(groups) == 2
    assert len(groups[0][0]) == 2
    assert len(groups[1][0]) == 1
    assert (
        factory.diagnostics["family_groups"]
        == factory.diagnostics["single_groups"]
        == 1
    )


def test_fragmented_neighbor_paint_can_continue_within_the_separate_path_bound():
    evidence = source()
    y, x = np.mgrid[:120, :120]
    own = evidence.labels >= 2
    labels = (y // 4).astype(np.int32)
    labels[own] = 30 + np.clip((x[own] - 48) // 6, 0, 3)
    target = np.full((120, 120, 3), (200, 150, 100), dtype=np.float32)
    target[own] = (60, 150, 110)
    evidence = replace(
        evidence,
        labels=labels,
        target=target,
        smooth=target,
        coarse=target,
        rgba=np.concatenate((target / 255, evidence.opacity[..., None]), axis=-1),
    )
    frontier, state, options = prepared(evidence, layers=True)
    edits = [
        p
        for p in ClosedOverlays(evidence, build(evidence), options)(
            state, Work.start(10)
        )
        if p.parameters[2] == (30, 31, 32, 33)
    ]
    assert edits
    for edit in edits:
        assert 4 < len([s for s in edit.partition.surfaces if s.covered]) <= 16
        assert frontier.policy.evaluate(export_svg(edit.document)).valid
        np.testing.assert_array_equal(
            render(export_svg(edit.document), evidence.source_size)[..., 3],
            evidence.rgba[..., 3],
        )


def test_core_must_cover_new_geometry_as_well_as_the_old_owned_footprint():
    evidence = source()
    _frontier, state, options = prepared(evidence, layers=True)
    document = state.document
    ids = tuple(
        s.id
        for s in state.partition.surfaces
        if s.members[0] >= 2 and s.role == "surface"
    )
    original = union_geometry(
        [document.geometry_for(oid) for oid in ids],
        [path_style(document, document.element(oid)) for oid in ids],
    )
    base = next(s for s in state.partition.surfaces if s.role == "underlay")
    editor = Editor(document, selection=Selection(whole_document=True))
    with editor.transaction("Restrict core to the old footprint") as transaction:
        transaction.replace_geometry(
            base.id, replace(original, id=document.geometry_for(base.id).id)
        )
    state = replace(state, document=editor.snapshot.document)
    factory = ClosedOverlays(evidence, build(evidence), options)
    edits = list(factory(state, Work.start(10)))
    assert not any(p.parameters[0] == "ellipse" for p in edits)
    assert factory.diagnostics["no_restoration"] > 0


def test_stop_during_restoration_discards_the_edit_and_preserves_the_parent(
    monkeypatch,
):
    evidence = source()
    _frontier, state, options = prepared(evidence, layers=True)
    factory = ClosedOverlays(evidence, build(evidence), options)
    work = Work.start(10)
    before = state.document
    original = factory.restoration.restorations

    def stop(*args, **kwargs):
        result = original(*args, **kwargs)
        work.stop.set()
        return result

    monkeypatch.setattr(factory.restoration, "restorations", stop)
    assert not list(factory(state, work))
    assert state.document is before
    assert state.partition == Partition.from_metadata(
        state.details["planning_surfaces"]
    )
    assert not state.snapshot.canvas.patches


def test_offset_scaled_scope_apply_reload_and_search_preserve_owned_primitives():
    evidence = source(alpha=253)
    native = np.zeros((128, 160, 4), dtype=np.float32)
    native[30:90, 20:80] = evidence.rgba[::2, ::2]
    evidence = replace(
        evidence, rgba=native, source_size=(160, 128), offset=(20, 30), scale=(2, 2)
    )
    frontier, state, options = prepared(evidence, layers=True)
    result = search(
        frontier,
        options,
        Work.start(10),
        ClosedOverlays(evidence, build(evidence), options),
    )
    assert result["accepted"] > 0
    assert result["score_disagreements"] == 0
    selected = frontier.select(50)
    assert selected.metrics["nodes"] < representation(state.document).metrics()["nodes"]
    partition = Partition.from_metadata(selected.metrics["planning_surfaces"])
    assert any(s.role == "overlay" for s in partition.surfaces)
    document, _ = load_project(save_project(import_svg(selected.svg)))
    partition.validate(document)
    np.testing.assert_array_equal(
        render(export_svg(document), evidence.source_size),
        render(selected.svg, evidence.source_size),
    )
