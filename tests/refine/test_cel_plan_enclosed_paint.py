"""A closed material proposal compacts paint without swallowing owned marks."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from tests.helpers import required
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
from vectrify.document.join import transformed_geometry
from vectrify.refine.cel_plan import enclosed_paint, overlays
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Work
from vectrify.refine.cel_plan.overlays import ClosedOverlays
from vectrify.refine.cel_plan.ownership import Partition
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.search import search


def source(alpha=128, gradient=False, fringe=False, mark_fringe=False):
    evidence = opaque()
    y, x = np.mgrid[:120, :120]
    radius = np.hypot(x - 60, y - 60)
    labels = (x >= 60).astype(np.int32)
    labels[radius <= 14] = 2
    inner = radius <= 11
    labels[inner] = 3 + np.clip((y[inner] - 49) // 4, 0, 5)
    labels[np.hypot(x - 57, y - 56) <= 2] = 9
    labels[(x >= 62) & (x <= 64) & (y >= 60) & (y <= 64)] = 10
    colors = [(200, 150, 100), (140, 100, 65), (24, 24, 24)]
    colors += [(80 + i, 145 + i, 112 + i) for i in range(6)]
    colors += [(232, 245, 238), (16, 16, 16)]
    target = np.asarray(colors, dtype=np.float32)[labels]
    if gradient:
        target[inner & (labels < 9), 0] = 65 + (x[inner & (labels < 9)] - 49) * 2
    if fringe:
        edge = inner & (np.asarray(distance_transform_edt(inner)) <= 1.5) & (labels < 9)
        target[edge] = (target[edge] + 24) * 0.5
    if mark_fringe:
        edge = (
            inner
            & (labels < 9)
            & (np.asarray(distance_transform_edt(labels != 9)) <= 1.5)
        )
        target[edge] = (target[edge] + np.asarray(colors[9])) * 0.5
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
        drawn=np.isin(labels, (2, 10)),
    )


@pytest.mark.parametrize(
    ("alpha", "gradient"), [(253, False), (128, False), (64, True)]
)
def test_compact_material_retains_highlight_ink_ownership_alpha_and_full_score(
    alpha, gradient
):
    evidence = source(alpha, gradient)
    frontier, state, options = prepared(evidence, layers=True)
    factory = ClosedOverlays(evidence, build(evidence), options)
    edits = [
        p for p in factory(state, Work.start(10)) if p.operator == "closed-material"
    ]
    assert edits
    if gradient:
        assert {
            required(p.details)["enclosed_material"]["paint_model"] for p in edits
        } == {
            "gradient",
            "flat",
        }
    for edit in edits:
        assert edit.details is not None
        details = edit.details["enclosed_material"]
        assert details["removed_paths"] >= 4
        for member in (9, 10):
            assert state.partition is not None
            oid = state.partition.owners[member]
            assert oid in details["retained_marks"]
            assert edit.partition is not None
            assert edit.partition.owners[member] == oid
            assert edit.document.geometry_for(oid) == state.document.geometry_for(oid)
            assert (
                edit.document.element(oid).attributes
                == state.document.element(oid).attributes
            )
        assert edit.partition is not None
        edit.partition.validate(edit.document)
        assert state.partition is not None
        assert edit.partition.owners.keys() == state.partition.owners.keys()
        svg = export_svg(edit.document)
        full = frontier.policy.evaluate(svg)
        assert full.valid
        assert full.cost < state.snapshot.evaluation.cost
        updated = LocalPolicy(frontier.policy).update(
            state.snapshot, svg, edit.bounds, full.structure
        )
        assert updated.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
        assert updated.canvas.matches(render(svg, evidence.source_size))
        assert frontier.checkpoint(
            svg,
            "Coupled closed material",
            edit.details,
            updated.evaluation,
            raster=updated.canvas,
        )
        before, after = (
            render(state.svg, evidence.source_size),
            render(svg, evidence.source_size),
        )
        np.testing.assert_array_equal(after[56, 57], before[56, 57])
        np.testing.assert_array_equal(after[62, 63], before[62, 63])
        np.testing.assert_array_equal(after[..., 3], before[..., 3])
    assert state.snapshot.canvas.matches(render(state.svg, evidence.source_size))


def test_one_native_high_residual_pixel_preserves_its_whole_owner():
    evidence = source()
    target = evidence.target.copy()
    target[evidence.labels == 9] = (82, 147, 114)
    target[56, 57] = (232, 245, 238)
    rgba = evidence.rgba.copy()
    rgba[..., :3] = target / 255
    evidence = replace(evidence, target=target, rgba=rgba)
    _, state, options = prepared(evidence, layers=True)
    factory = ClosedOverlays(evidence, build(evidence), options)
    edits = [
        p for p in factory(state, Work.start(10)) if p.operator == "closed-material"
    ]
    assert edits
    assert state.partition is not None
    oid = state.partition.owners[9]
    for edit in edits:
        assert edit.details is not None
        assert oid in edit.details["enclosed_material"]["retained_marks"]
        assert edit.partition is not None
        assert edit.partition.owners[9] == oid
        assert edit.document.geometry_for(oid) == state.document.geometry_for(oid)


@pytest.mark.parametrize("context", ["rim", "mark"])
def test_boundary_coverage_is_explained_without_ignoring_interior_marks(context):
    evidence = source(fringe=context == "rim", mark_fringe=context == "mark")
    frontier, state, options = prepared(evidence, layers=True)
    factory = ClosedOverlays(evidence, build(evidence), options)
    edits = [
        p for p in factory(state, Work.start(10)) if p.operator == "closed-material"
    ]
    assert edits
    key = (
        "coherent_coverage_samples"
        if context == "rim"
        else "coherent_mark_coverage_samples"
    )
    assert factory.diagnostics[key] > 0
    for edit in edits:
        assert edit.details is not None
        assert edit.details["enclosed_material"]["removed_paths"] >= 3
        assert {required(state.partition).owners[i] for i in (9, 10)}.issubset(
            edit.details["enclosed_material"]["retained_marks"]
        )
        full = frontier.policy.evaluate(export_svg(edit.document))
        assert full.valid
        assert full.cost < state.snapshot.evaluation.cost
        assert edit.partition is not None
        assert state.partition is not None
        assert edit.partition.owners.keys() == state.partition.owners.keys()


def test_coupled_material_search_scaled_scope_and_reload_preserve_owned_marks():
    evidence = source(64, True)
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
    assert any(
        d["operator"] == "closed-material" and d["accepted"]
        for d in result["decisions"]
    )
    assert result["score_disagreements"] == 0
    entries = [e for e in frontier.entries if e.details.get("enclosed_material")]
    assert entries
    for entry in entries:
        partition = Partition.from_metadata(entry.details["planning_surfaces"])
        document, _ = load_project(save_project(import_svg(entry.svg)))
        assert partition is not None
        partition.validate(document)
        assert state.partition is not None
        assert partition.owners.keys() == state.partition.owners.keys()
        np.testing.assert_array_equal(
            render(export_svg(document), evidence.source_size),
            render(entry.svg, evidence.source_size),
        )


def test_material_and_retained_marks_keep_independent_object_frames():
    evidence = source(128, False)
    frontier, state, options = prepared(evidence, layers=True)
    editor = Editor(state.document, selection=Selection(whole_document=True))
    with editor.transaction("Independent material frames") as transaction:
        for member in (3, 4, 5, 6, 7, 8, 9, 10):
            assert state.partition is not None
            oid = state.partition.owners[member]
            transaction.set_attributes(oid, {"transform": "translate(5 -3)"})
            transaction.replace_geometry(
                oid,
                transformed_geometry(
                    state.document.geometry_for(oid), (1, 0, 0, 1, -5, 3)
                ),
            )
    document = editor.snapshot.document
    svg = export_svg(document)
    full = frontier.policy.evaluate(svg)
    np.testing.assert_array_equal(
        render(svg, evidence.source_size), render(state.svg, evidence.source_size)
    )
    state = replace(
        state,
        document=document,
        svg=svg,
        key="independent-frames",
        snapshot=LocalPolicy(frontier.policy).start(svg, full),
    )
    edits = [
        p
        for p in ClosedOverlays(evidence, build(evidence), options)(
            state, Work.start(10)
        )
        if p.operator == "closed-material"
    ]
    assert edits
    for edit in edits:
        svg = export_svg(edit.document)
        full = frontier.policy.evaluate(svg)
        assert full.valid
        updated = LocalPolicy(frontier.policy).update(
            state.snapshot, svg, edit.bounds, full.structure
        )
        assert updated.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
        for member in (9, 10):
            assert state.partition is not None
            oid = state.partition.owners[member]
            assert (
                edit.document.element(oid).attributes
                == state.document.element(oid).attributes
            )
            assert edit.document.geometry_for(oid) == state.document.geometry_for(oid)


@pytest.mark.parametrize("protection", ["fixed", "paint"])
def test_protected_owner_is_retained_even_when_its_color_matches_material(protection):
    evidence = source()
    target = evidence.target.copy()
    target[evidence.labels == 9] = (82, 147, 114)
    rgba = evidence.rgba.copy()
    rgba[..., :3] = target / 255
    evidence = replace(evidence, target=target, rgba=rgba)
    _, state, options = prepared(evidence, layers=True)
    graph = build(evidence)
    assert state.partition is not None
    oid = state.partition.owners[9]
    if protection == "fixed":
        graph = replace(
            graph,
            regions=tuple(
                replace(r, fixed=True) if r.id == 9 else r for r in graph.regions
            ),
        )
    else:
        state = replace(state, details={**state.details, "paint_constraints": [oid]})
    edits = [
        p
        for p in ClosedOverlays(evidence, graph, options)(state, Work.start(10))
        if p.operator == "closed-material"
    ]
    assert edits
    for edit in edits:
        assert edit.details is not None
        assert oid in edit.details["enclosed_material"]["retained_marks"]
        assert edit.partition is not None
        assert edit.partition.owners[9] == oid
        assert edit.document.geometry_for(oid) == state.document.geometry_for(oid)


def test_bounded_inspection_and_stop_preserve_the_isolated_alternative(monkeypatch):
    evidence = source()
    _, state, options = prepared(evidence, layers=True)
    monkeypatch.setattr(enclosed_paint, "MAX_PIXELS", 1)
    factory = ClosedOverlays(evidence, build(evidence), options)
    edits = list(factory(state, Work.start(10)))
    assert any(p.operator == "closed-overlay" for p in edits)
    assert not any(p.operator == "closed-material" for p in edits)
    work = Work.start(10)
    work.stop.set()
    assert list(factory(state, work)) == []


def test_enclosure_hint_repeats_core_proof_after_a_state_change(monkeypatch):
    evidence = source()
    _, state, options = prepared(evidence, layers=True)
    monkeypatch.setattr(overlays, "MAX_ENCLOSURE_HINTS", 1)
    factory = ClosedOverlays(evidence, build(evidence), options)
    assert any(p.operator == "closed-material" for p in factory(state, Work.start(10)))
    assert factory.enclosure_hints == {(2,)}
    groups = factory.groups(state, Work.start(10))
    assert next(groups)[1] == (2,)
    groups.close()
    assert factory.diagnostics["enclosure_revisits"] == 1
    core = next(
        s.id for s in required(state.partition).surfaces if s.role == "underlay"
    )
    editor = Editor(state.document, selection=Selection(whole_document=True))
    with editor.transaction("Invalidate core geometry") as transaction:
        transaction.set_attributes(core, {"transform": "translate(200 0)"})
    changed = replace(state, document=editor.snapshot.document)
    assert not any(p.parameters[2] == (2,) for p in factory(changed, Work.start(10)))
