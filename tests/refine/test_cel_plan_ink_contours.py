"""Closed ink is rebuilt with both boundaries and its neighboring underpaint."""

from dataclasses import replace

import numpy as np
import pytest

from tests.refine.test_cel_plan_families import prepared
from tests.refine.test_cel_plan_ink_replace import source
from vectrify.document import Editor, Selection, export_svg, load_project, save_project
from vectrify.document.join import transformed_geometry
from vectrify.document.paint import GradientStop, LinearGradient
from vectrify.document.topology import inverse_matrix
from vectrify.refine.cel_plan import ink_contours
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.ink_replace import InkReplacement
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Work
from vectrify.refine.cel_plan.ownership import Partition
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.search import search


def ring(*, alpha=128, hole=False, gap=False, irregular=False):
    evidence = source(alpha=alpha)
    y, x = np.mgrid[:64, :96]
    dx, dy = (x + 0.5 - 48) / 24, (y + 0.5 - 32) / 20
    angle = np.arctan2(dy, dx)
    radius = np.hypot(dx, dy)
    if irregular:
        radius /= 1 + 0.14 * np.cos(7 * angle)
    inside = radius < 0.78
    ink = (radius < 1) & ~inside
    if gap:
        ink &= np.abs(angle) > 0.3
    labels = inside.astype(np.int32)
    labels[ink] = (
        2 + np.floor((angle[ink] + np.pi) * 12 / (2 * np.pi)).astype(np.int32) % 12
    )
    palette = np.array(
        [(220, 180, 100), (60, 150, 90), *[(12 + i, 8 + i, 5 + i) for i in range(12)]],
        dtype=np.float32,
    )
    target = palette[labels]
    empty = inside if hole else np.zeros(inside.shape, bool)
    opacity = (~empty).astype(np.float32) * (alpha / 255)
    rgba = np.concatenate((target / 255, opacity[..., None]), axis=-1)
    return replace(
        evidence,
        labels=labels,
        target=target,
        smooth=target,
        coarse=target,
        rgba=rgba,
        opacity=opacity,
        empty=empty,
        foreground=~empty,
        line=ink,
        drawn=ink,
        darkness=ink.astype(float) * 180,
        texture=np.zeros(ink.shape),
    )


def rims(factory, state):
    return [
        p
        for p in factory(state, Work.start(10))
        if p.details["ink_replacement"].get("boundary_model") == "source-rim"
    ]


@pytest.mark.parametrize("alpha", [255, 128, 64])
def test_paired_rim_removes_fragments_and_preserves_alpha_under_native_checkpoints(
    alpha,
):
    evidence = ring(alpha=alpha)
    frontier, state, options = prepared(evidence, layers=True)
    factory = InkReplacement(evidence, build(evidence), options)
    edits = rims(factory, state)
    assert edits, factory.diagnostics
    edit = edits[0]
    info = edit.details["ink_replacement"]
    assert info["boundary_models"] == ("ellipse", "ellipse")
    assert info["fitted_nodes"] == 10
    assert info["underpaint_model"] == "ellipse"
    assert info["intrinsic_opacity"] == 1
    assert info["removed_paths"] == 11
    assert info["continued_neighbors"] == 2
    assert edit.partition.follows(state.partition)
    edit.partition.validate(edit.document)
    svg = export_svg(edit.document)
    full = frontier.policy.evaluate(svg)
    assert full.valid
    assert full.cost < state.snapshot.evaluation.cost
    assert full.structure["nodes"] == 23
    assert full.structure["paths"] == 4
    actual = render(svg, evidence.source_size)
    np.testing.assert_array_equal(
        actual[..., 3], render(state.svg, evidence.source_size)[..., 3]
    )
    # The cavity retains independently owned green paint, not black underpaint.
    np.testing.assert_array_equal(
        actual[26:38, 40:56], render(state.svg, evidence.source_size)[26:38, 40:56]
    )
    local = LocalPolicy(frontier.policy).update(
        state.snapshot, svg, edit.bounds, full.structure
    )
    assert local.canvas.matches(actual)
    assert local.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
    assert frontier.checkpoint(
        svg, "Paired rim", edit.details, local.evaluation, raster=local.canvas
    )
    restored, _ = load_project(save_project(edit.document))
    Partition.from_metadata(edit.partition.metadata()).validate(restored)
    np.testing.assert_array_equal(
        render(export_svg(restored), evidence.source_size), actual
    )
    assert state.partition == Partition.from_metadata(
        state.details["planning_surfaces"]
    )


def test_neighbor_continuations_fill_old_rim_without_changing_primary_ownership():
    evidence = ring()
    _frontier, state, options = prepared(evidence, layers=True)
    edit = rims(InkReplacement(evidence, build(evidence), options), state)[0]
    overlay = next(s for s in edit.partition.surfaces if s.role == "overlay")
    continued = [
        s
        for s in edit.partition.surfaces
        if s.role == "surface" and set(overlay.members).issubset(s.covered)
    ]
    assert len(continued) == 2
    for surface in continued:
        assert surface.members == next(
            s.members for s in state.partition.surfaces if s.id == surface.id
        )
        assert edit.document.element(surface.id).get("fill") == state.document.element(
            surface.id
        ).get("fill")
    editor = Editor(edit.document, selection=Selection(whole_document=True))
    with editor.transaction("Inspect material below the fitted rim") as tx:
        tx.set_fill(overlay.id, "none")
    actual = render(export_svg(editor.snapshot.document), evidence.source_size)
    ink = evidence.drawn
    assert np.min(actual[ink, 3]) == 128 / 255
    assert np.min(np.max(actual[ink, :3], axis=1)) > 0.3


@pytest.mark.parametrize("kind", ["gap", "hole", "no-core"])
def test_unproved_closed_ink_keeps_original_alternative(kind):
    evidence = ring(gap=kind == "gap", hole=kind == "hole")
    _frontier, state, options = prepared(evidence, layers=kind != "no-core")
    factory = InkReplacement(evidence, build(evidence), options)
    assert not rims(factory, state)
    # Discovery never replaces the checkpoint or claims ownership after failure.
    assert state.partition == Partition.from_metadata(
        state.details["planning_surfaces"]
    )
    assert render(state.svg, evidence.source_size)[32, 48, 3] == (
        0 if kind == "hole" else 128 / 255
    )


def test_intentionally_irregular_rim_is_not_replaced_by_ellipses():
    evidence = ring(irregular=True)
    _frontier, state, options = prepared(evidence, layers=True)
    for edit in rims(InkReplacement(evidence, build(evidence), options), state):
        assert edit.details["ink_replacement"]["boundary_models"] == ("curve", "curve")


def test_native_offset_scale_and_neighbor_paint_frames_survive_rim_fit():
    evidence = ring()
    native = np.zeros((128, 160, 4), dtype=np.float32)
    native[30:62, 20:68] = evidence.rgba[::2, ::2]
    evidence = replace(
        evidence, rgba=native, source_size=(160, 128), offset=(20, 30), scale=(2, 2)
    )
    frontier, state, options = prepared(evidence, layers=True)
    edit = rims(InkReplacement(evidence, build(evidence), options), state)[0]
    svg = export_svg(edit.document)
    full = frontier.policy.evaluate(svg)
    assert full.valid
    actual = render(svg, evidence.source_size)
    np.testing.assert_array_equal(
        actual[..., 3], render(state.svg, evidence.source_size)[..., 3]
    )
    local = LocalPolicy(frontier.policy).update(
        state.snapshot, svg, edit.bounds, full.structure
    )
    assert local.canvas.matches(actual)
    assert local.evaluation.terms == pytest.approx(full.terms, abs=2e-7)


def test_perimeter_bound_and_stop_leave_fully_validated_seed(monkeypatch):
    evidence = ring()
    frontier, state, options = prepared(evidence, layers=True)
    monkeypatch.setattr(ink_contours, "MAX_PERIMETER", 4)
    factory = InkReplacement(evidence, build(evidence), options)
    assert not rims(factory, state)
    assert factory.diagnostics["rim_perimeter_limits"] > 0
    monkeypatch.setattr(ink_contours, "MAX_PERIMETER", 4096)
    work = Work.start(10)
    work.stop.set()
    report = search(
        frontier, options, work, InkReplacement(evidence, build(evidence), options)
    )
    assert report["accepted"] == 0
    assert frontier.select(50).svg == state.svg


@pytest.mark.parametrize("location", ["neighbor", "core", "partial-core"])
def test_gradients_keep_paint_frames_and_partial_core_cannot_prove_underpaint(location):
    evidence = ring()
    frontier, state, options = prepared(evidence, layers=True)
    oid = "cel-fill-0" if location == "neighbor" else "cel-base-1"
    opacity = 0.8 if location == "partial-core" else 1
    gradient = LinearGradient(
        (0, 0),
        (96, 0),
        (GradientStop(0, "#d8b464", opacity), GradientStop(1, "#e0b464", opacity)),
    )
    editor = Editor(state.document, selection=Selection(whole_document=True))
    with editor.transaction("Gradient material") as tx:
        tx.set_fill(oid, gradient)
    document = editor.snapshot.document
    svg = export_svg(document)
    state = replace(
        state,
        document=document,
        svg=svg,
        snapshot=LocalPolicy(frontier.policy).start(svg, frontier.policy.evaluate(svg)),
    )
    factory = InkReplacement(evidence, build(evidence), options)
    edits = rims(factory, state)
    if location == "partial-core":
        assert not edits
        assert factory.restoration_rejections["unproved-core-coverage"] > 0
        return
    assert edits
    edit = edits[0]
    server = document.element(oid).get("fill")[5:-1]
    assert edit.document.element(server) == document.element(server)
    assert edit.document.element(oid).get("fill") == document.element(oid).get("fill")
    actual = render(export_svg(edit.document), evidence.source_size)
    before = render(state.svg, evidence.source_size)
    np.testing.assert_array_equal(actual[..., 3], before[..., 3])
    np.testing.assert_array_equal(actual[:10], before[:10])
    full = frontier.policy.evaluate(export_svg(edit.document))
    assert full.valid


def test_only_exact_native_hidden_boundaries_can_replace_neighbor_continuations(
    monkeypatch,
):
    from vectrify.refine.cel_plan import ink_underpaint

    evidence = ring()
    _frontier, state, options = prepared(evidence, layers=True)
    original = ink_underpaint.render
    calls = 0

    def uncovered(*args, **kwargs):
        nonlocal calls
        actual = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            # Simulate a renderer-visible discontinuity in fitted ink coverage.
            actual = actual.copy()
            actual[..., 3] *= 0.99
        return actual

    monkeypatch.setattr(ink_underpaint, "render", uncovered)
    factory = InkReplacement(evidence, build(evidence), options)
    edit = rims(factory, state)[0]
    assert edit.details["ink_replacement"]["underpaint_model"] == "traced"
    assert (
        factory.diagnostics["rim_underpaint_exclusions"]["visible-material-boundary"]
        == 1
    )


def test_joint_rim_is_retained_through_real_search_and_full_checkpoint():
    evidence = ring()
    frontier, state, options = prepared(evidence, layers=True)
    factory = InkReplacement(evidence, build(evidence), options)
    result = search(frontier, options, Work.start(10), factory)
    assert result["accepted"] > 0
    assert result["score_disagreements"] == 0
    chosen = frontier.select(50)
    assert chosen.metrics["nodes"] < state.snapshot.evaluation.structure["nodes"]
    assert any(
        entry.details.get("ink_replacement", {}).get("underpaint_model") == "ellipse"
        for entry in frontier.entries
    )
    np.testing.assert_array_equal(
        render(chosen.svg, evidence.source_size)[..., 3],
        render(state.svg, evidence.source_size)[..., 3],
    )


@pytest.mark.parametrize("kind", ["requested", "source-fixed"])
def test_explicit_width_is_not_reinterpreted_by_independent_rim_contours(kind):
    evidence = ring()
    if kind == "source-fixed":
        evidence = replace(evidence, filled_line_width=3)
    _frontier, state, options = prepared(evidence, layers=True)
    if kind == "requested":
        options = replace(options, line_width=3)
    factory = InkReplacement(evidence, build(evidence), options)
    assert not rims(factory, state)
    assert factory.diagnostics["rim_explicit_width_exclusions"] > 0


def test_stop_during_paired_fitting_cannot_publish_a_partly_changed_sibling(
    monkeypatch,
):
    evidence = ring()
    _frontier, state, options = prepared(evidence, layers=True)
    work = Work.start(10)
    original = ink_contours.fitted

    def stopped(*args, **kwargs):
        model = original(*args, **kwargs)
        work.stop.set()
        return model

    monkeypatch.setattr(ink_contours, "fitted", stopped)
    factory = InkReplacement(evidence, build(evidence), options)
    assert list(factory(state, work)) == []
    assert state.snapshot.canvas.matches(render(state.svg, evidence.source_size))
    assert not state.snapshot.canvas.patches


def test_compact_underpaint_proves_inner_paint_above_surrounding_material():
    evidence = ring()
    frontier, state, options = prepared(evidence, layers=True)
    editor = Editor(state.document, selection=Selection(whole_document=True))
    with editor.transaction("Reverse disjoint material order") as tx:
        tx.reorder_object("cel-fill-1", 1)
    document = editor.snapshot.document
    svg = export_svg(document)
    state = replace(
        state,
        document=document,
        svg=svg,
        snapshot=LocalPolicy(frontier.policy).start(svg, frontier.policy.evaluate(svg)),
    )
    factory = InkReplacement(evidence, build(evidence), options)
    edit = rims(factory, state)[0]
    assert edit.details["ink_replacement"]["underpaint_model"] == "ellipse"
    parent = edit.document.ancestry("cel-fill-0")[-2]
    positions = {c.id: i for i, c in enumerate(parent.children)}
    overlay = next(s.id for s in edit.partition.surfaces if s.role == "overlay")
    assert positions["cel-fill-0"] < positions["cel-fill-1"] < positions[overlay]
    actual = render(export_svg(edit.document), evidence.source_size)
    np.testing.assert_array_equal(
        actual[26:38, 40:56], render(state.svg, evidence.source_size)[26:38, 40:56]
    )


@pytest.mark.parametrize("matrix", [(1.3, 0.2, 0.1, 0.9, 7, 11), (-1, 0, 0, 1, 96, 0)])
def test_rim_and_underpaint_share_native_geometry_in_transformed_frames(matrix):
    evidence = ring()
    frontier, state, options = prepared(evidence, layers=True)
    document = state.document
    parent = document.ancestry("cel-fill-0")[-2]
    # Re-express the same native artwork in a different local coordinate frame.
    for child in parent.children:
        document = document.replace_geometry(
            transformed_geometry(
                document.geometry_for(child.id), inverse_matrix(matrix)
            )
        )
    attributes = dict(parent.attributes)
    attributes["transform"] = "matrix(" + " ".join(str(v) for v in matrix) + ")"
    document = document.replace_element(
        replace(document.element(parent.id), attributes=tuple(attributes.items()))
    )
    svg = export_svg(document)
    state = replace(
        state,
        document=document,
        svg=svg,
        snapshot=LocalPolicy(frontier.policy).start(svg, frontier.policy.evaluate(svg)),
    )
    edit = rims(InkReplacement(evidence, build(evidence), options), state)[0]
    assert edit.details["ink_replacement"]["underpaint_model"] == "ellipse"
    actual = render(export_svg(edit.document), evidence.source_size)
    np.testing.assert_array_equal(
        actual[..., 3], render(state.svg, evidence.source_size)[..., 3]
    )
    full = frontier.policy.evaluate(export_svg(edit.document))
    assert full.valid
    local = LocalPolicy(frontier.policy).update(
        state.snapshot, export_svg(edit.document), edit.bounds, full.structure
    )
    assert local.canvas.matches(actual)
    assert local.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
