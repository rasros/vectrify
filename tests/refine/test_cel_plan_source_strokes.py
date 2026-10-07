"""Ink replacement preserves existing materials and source-only topology."""

from dataclasses import replace

import numpy as np
import pytest

from tests.refine.test_cel_plan_families import prepared
from tests.refine.test_cel_plan_ink_models import drawing
from vectrify.document import Editor, Selection, export_svg, load_project, save_project
from vectrify.document.hit_test import multiply
from vectrify.document.join import path_style, transformed_geometry
from vectrify.document.redraw import root_matrix
from vectrify.document.topology import inverse_matrix
from vectrify.refine.cel_plan.core_cells import CoreCells
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Work
from vectrify.refine.cel_plan.proposals import Operators
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.search import State
from vectrify.refine.cel_plan.source_strokes import SourceStrokes


def fragmented(alpha=128, gap=False):
    evidence, ink = drawing(alpha, gap=gap)
    y, x = np.indices(evidence.labels.shape)
    labels = np.where(evidence.empty, 0, 1 + (y // 8) * 12 + x // 8)
    labels[ink] += 200
    return replace(evidence, labels=labels.astype(np.int32), drawn=ink, line=ink)


def mixed_owner_mark():
    evidence = fragmented()
    labels, target, rgba, drawn = (
        a.copy()
        for a in (evidence.labels, evidence.target, evidence.rgba, evidence.drawn)
    )
    owner = int(np.bincount(labels[drawn]).argmax())
    labels[72:77, 78:83] = owner
    target[72:77, 78:83] = 32
    rgba[72:77, 78:83, :3] = 32 / 255
    drawn[72:77, 78:83] = True
    return replace(
        evidence, labels=labels, target=target, rgba=rgba, drawn=drawn, line=drawn
    )


@pytest.mark.parametrize("alpha", [255, 128, 64])
@pytest.mark.parametrize("gap", [False, True])
def test_native_stroke_replaces_fragments_without_collapsing_existing_paint(alpha, gap):
    evidence = fragmented(alpha, gap)
    frontier, state, options = prepared(evidence, layers=True)
    assert state.partition is not None
    graph = build(evidence)
    factory = SourceStrokes(evidence, graph, options)
    edits = list(factory.proposals(state, Work.start(20)))
    assert edits, (
        factory.diagnostics,
        factory.cutter.diagnostics,
        factory.restoration_rejections,
    )
    original = render(state.svg, evidence.source_size)
    for edit in edits:
        assert edit.partition is not None
        assert edit.component is not None
        actual = render(export_svg(edit.document), evidence.source_size)
        full = frontier.policy.evaluate(export_svg(edit.document))
        assert full.valid, full.rejections
        assert edit.partition.follows(state.partition)
        Operators(evidence, graph, options).validate_partition(
            edit.partition, Work.start(10)
        )
        assert edit.component.validate(
            state.document,
            edit.document,
            state.partition,
            edit.partition,
            edit.ids,
            edit.bounds,
            Work.start(10),
        )
        np.testing.assert_array_equal(actual[..., 3], original[..., 3])
        # Paint away from the source stroke footprint is unchanged. Both source
        # materials remain independently editable, with their original paints.
        np.testing.assert_array_equal(actual[70:80, 12:82], original[70:80, 12:82])
        if gap:
            np.testing.assert_array_equal(actual[26:31, 47:49], original[26:31, 47:49])
        for surface in state.partition.surfaces:
            if surface.id in {e.id for e in edit.document.elements()}:
                before = path_style(state.document, state.document.element(surface.id))
                after = path_style(edit.document, edit.document.element(surface.id))
                if after["stroke"] == "none":
                    assert after["fill"] == before["fill"]
        strokes = [
            e for e in edit.document.elements() if e.get("stroke", "none") != "none"
        ]
        assert strokes
        assert all(e.get("fill") == "none" for e in strokes)
        local = LocalPolicy(frontier.policy).update(
            state.snapshot, export_svg(edit.document), edit.bounds, full.structure
        )
        assert local.canvas.matches(actual)
        assert local.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
        restored, _ = load_project(save_project(edit.document))
        np.testing.assert_array_equal(
            render(export_svg(restored), evidence.source_size), actual
        )


@pytest.mark.parametrize("alpha", [128, 64])
@pytest.mark.parametrize("gap", [False, True])
def test_material_compaction_composes_after_source_strokes(alpha, gap):
    evidence = fragmented(alpha, gap)
    frontier, initial, options = prepared(evidence, layers=True)
    ops = Operators(evidence, build(evidence), options)
    factory = SourceStrokes(evidence, ops.graph, options, resolver=ops.branch)
    ink = next(factory.proposals(initial, Work.start(20)))
    assert ink.details is not None
    svg = export_svg(ink.document)
    full = frontier.policy.evaluate(svg)
    assert full.valid
    state = State(
        ink.document,
        svg,
        LocalPolicy(frontier.policy).start(svg, full),
        "stroke-parent",
        {**initial.details, **ink.details},
        partition=ink.partition,
    )
    assert state.partition is not None
    branch = ops.branch(state.partition, Work.start(10))
    material = CoreCells(
        branch.families, options, joint=True, grouping="ward", boundary_fit="curve"
    )
    edits = list(material(state, Work.start(20)))
    assert edits, material.diagnostics
    strokes = [
        e for e in state.document.elements() if e.get("stroke", "none") != "none"
    ]
    assert strokes
    for edit in edits:
        assert edit.partition is not None
        final_svg = export_svg(edit.document)
        final = frontier.policy.evaluate(final_svg)
        assert final.valid, final.rejections
        assert edit.partition.follows(state.partition)
        ops.validate_partition(edit.partition, Work.start(10))
        for stroke in strokes:
            assert edit.document.element(stroke.id) == stroke
            assert edit.document.geometry_for(stroke.id) == state.document.geometry_for(
                stroke.id
            )
        actual = render(final_svg, evidence.source_size)
        np.testing.assert_array_equal(
            actual[..., 3], render(svg, evidence.source_size)[..., 3]
        )
        local = LocalPolicy(frontier.policy).update(
            state.snapshot, final_svg, edit.bounds, final.structure
        )
        assert local.canvas.matches(actual)
        assert local.evaluation.terms == pytest.approx(final.terms, abs=2e-7)


def test_cancellation_after_source_cut_does_not_publish_a_partial_edit(monkeypatch):
    evidence = mixed_owner_mark()
    _frontier, state, options = prepared(evidence, layers=True)
    ops = Operators(evidence, build(evidence), options)
    factory = SourceStrokes(evidence, ops.graph, options, resolver=ops.branch)
    work = Work.start(10)
    original = factory.cutter.prepare
    before = export_svg(state.document)

    def stopped(*args):
        result = original(*args)
        assert result is not None
        work.stop.set()
        return result

    monkeypatch.setattr(factory.cutter, "prepare", stopped)
    assert list(factory.proposals(state, work)) == []
    assert export_svg(state.document) == before
    assert ops.schedule_diagnostics["source_graph_rebuilds"] > 0


def test_disconnected_mark_sharing_a_source_owner_survives_exact_ink_cut():
    evidence = mixed_owner_mark()
    frontier, state, options = prepared(evidence, layers=True)
    ops = Operators(evidence, build(evidence), options)
    factory = SourceStrokes(evidence, ops.graph, options, resolver=ops.branch)
    edits = list(factory.proposals(state, Work.start(20)))
    assert edits
    original = render(export_svg(state.document), evidence.source_size)
    for edit in edits:
        assert edit.partition is not None
        assert edit.details is not None
        svg = export_svg(edit.document)
        assert frontier.policy.evaluate(svg).valid
        assert edit.details["source_strokes"]["cuts"] > 0
        ops.validate_partition(edit.partition, Work.start(10))
        np.testing.assert_array_equal(
            render(svg, evidence.source_size)[71:78, 77:84], original[71:78, 77:84]
        )


def test_paint_constrained_source_owners_are_not_reinterpreted():
    evidence = fragmented()
    _frontier, state, options = prepared(evidence, layers=True)
    assert state.partition is not None
    owners = state.partition.owners
    protected = {owners[int(i)] for i in np.unique(evidence.labels[evidence.drawn])}
    state = replace(
        state, details={**state.details, "paint_constraints": sorted(protected)}
    )
    factory = SourceStrokes(evidence, build(evidence), options)
    assert list(factory.proposals(state, Work.start(10))) == []
    assert factory.diagnostics["owner_exclusions"] > 0


def test_retained_stroke_centerline_does_not_prove_its_whole_width_is_in_carrier():
    evidence = fragmented()
    frontier, initial, options = prepared(evidence, layers=True)
    ops = Operators(evidence, build(evidence), options)
    ink = next(
        SourceStrokes(evidence, ops.graph, options, resolver=ops.branch).proposals(
            initial, Work.start(20)
        )
    )
    assert ink.details is not None
    stroke = next(
        e for e in ink.document.elements() if e.get("stroke", "none") != "none"
    )
    editor = Editor(ink.document, selection=Selection(whole_document=True))
    with editor.transaction("Force an unsupported wide stroke") as tx:
        tx.set_attributes(stroke.id, {"stroke-width": "200"})
    document = editor.snapshot.document
    svg = export_svg(document)
    state = State(
        document,
        svg,
        LocalPolicy(frontier.policy).start(svg, frontier.policy.evaluate(svg)),
        "wide-stroke",
        {**initial.details, **ink.details},
        partition=ink.partition,
    )
    branch = ops.branch(ink.partition, Work.start(10))
    factory = CoreCells(
        branch.families, options, joint=True, grouping="ward", boundary_fit="curve"
    )
    assert list(factory(state, Work.start(20))) == []
    assert factory.diagnostics["style_exclusions"] > 0


def test_source_replacement_preserves_skewed_neighbor_frames_and_native_stroke_width():
    evidence = fragmented()
    frontier, state, options = prepared(evidence, layers=True)
    assert state.partition is not None
    matrix = (1.2, 0.15, 0.2, 1.0, 6, -10)
    editor = Editor(state.document, selection=Selection(whole_document=True))
    with editor.transaction("Reexpress paint frames") as tx:
        for surface in state.partition.surfaces:
            tx.replace_geometry(
                surface.id,
                transformed_geometry(
                    state.document.geometry_for(surface.id),
                    multiply(
                        inverse_matrix(matrix), root_matrix(state.document, surface.id)
                    ),
                ),
            )
            tx.set_attributes(
                surface.id,
                {"transform": f"matrix({' '.join(str(v) for v in matrix)})"},
            )
    document = editor.snapshot.document
    svg = export_svg(document)
    assert frontier.policy.evaluate(svg).valid
    state = replace(
        state,
        document=document,
        svg=svg,
        snapshot=LocalPolicy(frontier.policy).start(svg, frontier.policy.evaluate(svg)),
    )
    ops = Operators(evidence, build(evidence), options)
    edits = list(
        SourceStrokes(evidence, ops.graph, options, resolver=ops.branch).proposals(
            state, Work.start(20)
        )
    )
    assert edits
    for edit in edits:
        final_svg = export_svg(edit.document)
        full = frontier.policy.evaluate(final_svg)
        assert full.valid, full.rejections
        for element in edit.document.elements():
            if element.get("fill", "none") == "none":
                continue
            try:
                before = document.element(element.id)
            except KeyError:
                continue
            assert element.get("transform") == before.get("transform")
            assert (
                path_style(edit.document, element)["fill"]
                == path_style(document, before)["fill"]
            )
        actual = render(final_svg, evidence.source_size)
        np.testing.assert_array_equal(
            actual[..., 3], render(svg, evidence.source_size)[..., 3]
        )
        local = LocalPolicy(frontier.policy).update(
            state.snapshot, final_svg, edit.bounds, full.structure
        )
        assert local.canvas.matches(actual)
        assert local.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
