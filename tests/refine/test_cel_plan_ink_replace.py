"""Ink replacement compacts owned fills and restores paint beneath strokes."""

from dataclasses import replace

import numpy as np
import pytest

from tests.helpers import required
from tests.refine.test_cel_plan_families import prepared
from tests.refine.test_cel_plan_ownership import stripes
from vectrify.document import Editor, Selection, export_svg, load_project, save_project
from vectrify.document.paint import GradientStop, LinearGradient
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.ink_replace import InkReplacement
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Work
from vectrify.refine.cel_plan.ownership import Partition
from vectrify.refine.cel_plan.score import render, representation
from vectrify.refine.cel_plan.search import search


def source(*, alpha=128, tapered=False, gap=False):
    evidence = stripes(alpha=alpha)
    y, x = np.mgrid[:64, :96]
    labels = (y >= 32).astype(np.int32)
    width = 2 + (x - 8) / 8 if tapered else 4
    ink = (np.abs(y - 31.5) < width / 2) & (x >= 8) & (x < 88)
    if gap:
        ink &= (x < 38) | (x >= 58)
    labels[ink] = 2 + (x[ink] - 8) // 10
    palette = np.array(
        [
            (220, 180, 100),
            (160, 100, 70),
            *[(12 + i, 8 + i, 5 + i) for i in range(8)],
        ],
        dtype=np.float32,
    )
    target = palette[labels]
    opacity = np.full((64, 96), alpha / 255, dtype=np.float32)
    rgba = np.concatenate((target / 255, opacity[..., None]), axis=-1)
    empty = np.zeros(labels.shape, dtype=bool)
    return replace(
        evidence,
        rgba=rgba,
        target=target,
        smooth=target,
        coarse=target,
        labels=labels,
        empty=empty,
        foreground=~empty,
        line=ink,
        drawn=ink,
        darkness=ink.astype(float) * 180,
        texture=empty.astype(float),
        opacity=opacity,
    )


@pytest.mark.parametrize("alpha", [255, 128, 64])
def test_stroke_and_filled_ink_remove_fragments_with_exact_checkpoint_agreement(alpha):
    evidence = source(alpha=alpha)
    frontier, state, options = prepared(evidence, layers=True)
    factory = InkReplacement(evidence, build(evidence), options)
    proposals = list(factory(state, Work.start(10)))
    assert [p.parameters[0] for p in proposals] == ["stroke", "filled"]
    for proposal in proposals:
        svg = export_svg(proposal.document)
        evaluator = LocalPolicy(frontier.policy)
        updated = evaluator.update(
            state.snapshot,
            svg,
            proposal.bounds,
            representation(proposal.document).metrics(),
        )
        complete = frontier.policy.evaluate(svg)
        assert updated.evaluation.terms == pytest.approx(complete.terms, abs=2e-7)
        assert updated.canvas.matches(render(svg, evidence.source_size))
        assert complete.valid
        assert complete.cost < state.snapshot.evaluation.cost
        assert state.partition is not None
        assert proposal.partition is not None
        assert proposal.partition.owners.keys() == state.partition.owners.keys()
        proposal.partition.validate(proposal.document)
        assert (
            len(
                [
                    s
                    for s in required(proposal.partition).surfaces
                    if s.role == "overlay"
                ]
            )
            == 1
        )
        assert (
            len([s for s in required(state.partition).surfaces if s.role == "overlay"])
            == 0
        )
        # Opacity is applied once by the isolated component, including underlays.
        np.testing.assert_array_equal(
            render(svg, evidence.source_size)[..., 3], evidence.rgba[..., 3]
        )
        assert proposal.details is not None
        assert frontier.checkpoint(
            svg,
            "Ink replacement",
            proposal.details,
            updated.evaluation,
            raster=updated.canvas,
        )


def test_restored_neighbor_paints_cover_the_old_ink_footprint_beneath_the_stroke():
    evidence = source()
    frontier, state, options = prepared(evidence, layers=True)
    edit = next(
        InkReplacement(evidence, build(evidence), options)(state, Work.start(10))
    )
    assert edit.parameters[0] == "stroke"
    overlays = [s for s in required(edit.partition).surfaces if s.role == "overlay"]
    overlay = overlays[0]
    editor = Editor(edit.document, selection=Selection(whole_document=True))
    with editor.transaction("Inspect restored paint") as transaction:
        transaction.set_attributes(overlay.id, {"stroke": "none"})
    actual = render(export_svg(editor.snapshot.document), evidence.source_size)
    before = render(state.svg, evidence.source_size)
    # Source ink cut across two differently painted surfaces, neither of which
    # supplied the correct paint inside its missing ink strip before restoration.
    np.testing.assert_array_equal(actual[30, 40, :3], before[20, 40, :3])
    np.testing.assert_array_equal(actual[33, 40, :3], before[40, 40, :3])
    assert (
        len([s for s in required(edit.partition).surfaces if s.role == "underlay"]) == 3
    )
    assert frontier.policy.evaluate(export_svg(edit.document)).valid


def test_variable_width_ink_remains_a_filled_competing_mark():
    evidence = source(tapered=True)
    frontier, state, options = prepared(evidence, layers=True)
    factory = InkReplacement(evidence, build(evidence), options)
    proposals = list(factory(state, Work.start(10)))
    assert proposals
    assert all(p.parameters[0] == "filled" for p in proposals)
    result = search(frontier, options, Work.start(10), factory)
    selected = frontier.select(50)
    assert result["accepted"] > 0
    assert result["score_disagreements"] == 0
    assert selected.metrics["nodes"] < state.snapshot.evaluation.structure["nodes"]
    assert selected.metrics["ink_replacement"]["width_ratio"] > 1.6
    np.testing.assert_array_equal(
        render(selected.svg, evidence.source_size)[..., 3], evidence.rgba[..., 3]
    )


def test_a_blank_gap_cannot_be_completed_by_replacement_of_two_separate_families():
    evidence = source(gap=True)
    _frontier, state, options = prepared(evidence, layers=True)
    factory = InkReplacement(evidence, build(evidence), options)
    proposals = list(factory(state, Work.start(10)))
    assert proposals
    before = render(state.svg, evidence.source_size)
    for proposal in proposals:
        actual = render(export_svg(proposal.document), evidence.source_size)
        np.testing.assert_array_equal(actual[29:35, 44:52], before[29:35, 44:52])
        assert not (
            {2, 3, 4} & set(proposal.parameters[1])
            and {7, 8, 9} & set(proposal.parameters[1])
        )


def test_a_dark_shade_without_bilateral_ridge_evidence_has_no_ink_replacement():
    evidence = source()
    target = evidence.target.copy()
    target[evidence.labels >= 2] = (100, 70, 40)
    target[evidence.labels == 1] = (100, 70, 40)
    rgba = evidence.rgba.copy()
    rgba[..., :3] = target / 255
    evidence = replace(evidence, target=target, smooth=target, rgba=rgba)
    _frontier, state, options = prepared(evidence, layers=True)
    assert (
        list(InkReplacement(evidence, build(evidence), options)(state, Work.start(10)))
        == []
    )


def test_uncovered_translucent_surface_keeps_filled_ink_without_alpha_doubling():
    evidence = source(alpha=64)
    _frontier, state, options = prepared(evidence, layers=False)
    proposals = list(
        InkReplacement(evidence, build(evidence), options)(state, Work.start(10))
    )
    assert proposals
    assert all(p.parameters[0] == "filled" for p in proposals)


def test_native_stroke_width_survives_scaled_offset_group():
    evidence = source(alpha=128)
    native = np.zeros((128, 160, 4), dtype=np.float32)
    native[30:62, 20:68] = evidence.rgba[::2, ::2]
    evidence = replace(
        evidence, rgba=native, source_size=(160, 128), offset=(20, 30), scale=(2, 2)
    )
    frontier, state, options = prepared(evidence, layers=True)
    edit = next(
        InkReplacement(evidence, build(evidence), options)(state, Work.start(10))
    )
    svg = export_svg(edit.document)
    evaluated = frontier.policy.evaluate(svg)
    updated = LocalPolicy(frontier.policy).update(
        state.snapshot, svg, edit.bounds, evaluated.structure
    )
    assert evaluated.valid
    assert edit.details is not None
    assert frontier.checkpoint(
        svg, "Offset ink", edit.details, updated.evaluation, raster=updated.canvas
    )
    restored, _selection = load_project(save_project(edit.document))
    np.testing.assert_array_equal(
        render(export_svg(restored), evidence.source_size),
        render(svg, evidence.source_size),
    )
    assert edit.partition is not None
    assert Partition.from_metadata(edit.partition.metadata()) == edit.partition


def test_gradient_underlay_keeps_the_neighbors_original_paint_frame():
    evidence = source()
    frontier, state, options = prepared(evidence, layers=True)
    editor = Editor(state.document, selection=Selection(whole_document=True))
    gradient = LinearGradient(
        (0, 0),
        (96, 0),
        (
            GradientStop(0, "#d8b464"),
            GradientStop(1, "#e0b464"),
        ),
    )
    with editor.transaction("Add neighboring gradient") as transaction:
        transaction.set_fill("cel-fill-0", gradient)
    document = editor.snapshot.document
    svg = export_svg(document)
    state = replace(
        state,
        document=document,
        svg=svg,
        snapshot=LocalPolicy(frontier.policy).start(svg, frontier.policy.evaluate(svg)),
    )
    edit = next(
        InkReplacement(evidence, build(evidence), options)(state, Work.start(10))
    )
    overlay = next(s for s in required(edit.partition).surfaces if s.role == "overlay")
    editor = Editor(edit.document, selection=Selection(whole_document=True))
    with editor.transaction("Inspect gradient continuation") as transaction:
        transaction.set_attributes(overlay.id, {"stroke": "none"})
    actual = render(export_svg(editor.snapshot.document), evidence.source_size)
    before = render(svg, evidence.source_size)
    np.testing.assert_array_equal(actual[31, 14:82], before[20, 14:82])
    underlays = [s for s in required(edit.partition).surfaces if s.role == "underlay"]
    assert any(
        required(edit.document.element(s.id).get("fill", "")).startswith("url(")
        for s in underlays
    )


def test_explicit_native_width_and_existing_holds_survive_replacement():
    evidence = source(tapered=True)
    _frontier, state, options = prepared(evidence, layers=True)
    constrained = [
        s.id for s in required(state.partition).surfaces if min(s.members) >= 2
    ]
    state = replace(state, details={**state.details, "paint_constraints": constrained})
    options = replace(options, line_width=2)
    edit = next(
        InkReplacement(evidence, build(evidence), options)(state, Work.start(10))
    )
    assert edit.parameters[0] == "stroke"
    overlay = next(s for s in required(edit.partition).surfaces if s.role == "overlay")
    assert edit.document.element(overlay.id).get("stroke-width") == "2"
    assert edit.details is not None
    assert overlay.id in edit.details["geometry_constraints"]
    assert overlay.id in edit.details["paint_constraints"]


def test_connected_dark_family_cannot_join_distinct_source_components():
    evidence = source()
    _frontier, state, options = prepared(evidence, layers=True)
    graph = build(evidence)
    graph = replace(
        graph, regions=tuple(replace(r, component=r.id) for r in graph.regions)
    )
    assert list(InkReplacement(evidence, graph, options)(state, Work.start(10))) == []


def test_stop_during_ink_proof_does_not_mutate_a_beam_sibling(monkeypatch):
    evidence = source()
    _frontier, state, options = prepared(evidence, layers=True)
    factory = InkReplacement(evidence, build(evidence), options)
    work = Work.start(10)
    original = factory.proof

    def stopped(*args):
        result = original(*args)
        work.stop.set()
        return result

    monkeypatch.setattr(factory, "proof", stopped)
    assert list(factory(state, work)) == []
    assert state.snapshot.canvas.matches(render(state.svg, evidence.source_size))
    assert state.partition == Partition.from_metadata(
        state.details["planning_surfaces"]
    )
    assert not state.snapshot.canvas.patches


def test_ink_discovery_deadline_does_not_expire_other_operator_work(monkeypatch):
    evidence = source()
    _frontier, state, options = prepared(evidence, layers=True)
    factory = InkReplacement(evidence, build(evidence), options)
    work = Work.start(10)
    seen = []

    def bounded(_box, _own, local_work):
        assert local_work is not work
        assert local_work.deadline < work.deadline
        local_work.deadline = -1
        seen.append(local_work)
        return

    monkeypatch.setattr(factory, "proof", bounded)
    assert list(factory(state, work)) == []
    assert len(seen) == 1
    assert factory.diagnostics["time_bounded"] == 1
    assert not work.interrupted
