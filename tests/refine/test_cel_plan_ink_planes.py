"""Source ink and material planes share complete ownership without losing lines."""

from dataclasses import replace

import numpy as np
import pytest

from tests.refine.test_cel_plan_core_cells import material
from tests.refine.test_cel_plan_families import prepared
from vectrify.document import Editor, Selection, export_svg, load_project, save_project
from vectrify.refine.cel_plan.core_cells import CoreCells
from vectrify.refine.cel_plan.families import Families
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import StageInterruptedError, Work
from vectrify.refine.cel_plan.ownership import Partition
from vectrify.refine.cel_plan.proposals import Operators
from vectrify.refine.cel_plan.score import render, svg_metrics


def source(alpha, hole, gap):
    evidence = material(alpha, hole)
    y, x = np.indices(evidence.labels.shape)
    ink = (y >= 18) & (y < 21) & (x >= 15) & (x < 81)
    if gap:
        ink &= (x < 42) | (x >= 54)
    else:
        ink |= (x >= 47) & (x < 50) & (y >= 12) & (y < 54)
    ink &= ~evidence.empty
    target, rgba, labels = (
        evidence.target.copy(),
        evidence.rgba.copy(),
        evidence.labels.copy(),
    )
    target[ink], rgba[ink, :3] = 20, 20 / 255
    fragments = 20 + x // 8 + 12 * (y // 8)
    labels[ink] = fragments[ink]
    return replace(
        evidence,
        target=target,
        rgba=rgba,
        labels=labels,
        coarse=target,
        smooth=target,
        drawn=ink,
        line=ink,
    ), ink


def combined(evidence, state, options, work=None):
    factory = CoreCells(
        Families(evidence, build(evidence), options),
        options,
        joint=True,
        grouping="ward",
        boundary_fit="anchored",
        ink_support="connected",
        layout="ink-planes",
    )
    return factory, list(factory(state, work or Work.start(20)))


@pytest.mark.parametrize("alpha", [255, 128])
@pytest.mark.parametrize("hole", [False, True])
@pytest.mark.parametrize("gap", [False, True])
def test_three_material_planes_keep_fragmented_ink_and_actual_source_gaps(
    alpha, hole, gap
):
    evidence, ink = source(alpha, hole, gap)
    frontier, state, options = prepared(evidence, layers=True)
    factory, edits = combined(evidence, state, options)
    faithful = [
        p for p in edits if p.details["core_material_cells"]["material_planes"] == 3
    ]
    assert faithful, factory.diagnostics
    edit = faithful[0]
    model = edit.details["core_material_cells"]
    assert model["ink_cells"] >= 1
    assert model["model_stage"] == "source-ink-and-facet-planes"
    assert edit.partition.follows(state.partition)
    ops = Operators(evidence, build(evidence), options)
    ops.validate_partition(edit.partition, Work.start(10))
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
    assert np.mean(actual[ink, :3].mean(axis=1) < 0.4) > 0.9
    # Flat material observations far from the source edges remain their actual
    # colors. Ink above the paint must not influence the fitted material ramp.
    for x in (20, 55, 80):
        np.testing.assert_allclose(
            actual[42, x, :3] * 255, evidence.target[42, x], atol=2
        )
    if gap:
        np.testing.assert_allclose(
            actual[19, 46:50, :3] * 255, evidence.target[19, 46:50], atol=2
        )
    if hole:
        assert actual[24:40, 40:56, 3].max() == 0
    assert svg_metrics(svg)["nodes"] < svg_metrics(state.svg)["nodes"]
    local = LocalPolicy(frontier.policy).update(
        state.snapshot, svg, edit.bounds, full.structure
    )
    assert local.canvas.matches(actual)
    assert local.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
    document, _ = load_project(save_project(edit.document))
    Partition.from_metadata(edit.partition.metadata()).validate(document)
    np.testing.assert_array_equal(
        render(export_svg(document), evidence.source_size), actual
    )


def test_broad_dark_paint_is_not_promoted_to_an_ink_layer():
    evidence = material(128)
    target, rgba = evidence.target.copy(), evidence.rgba.copy()
    target[~evidence.empty], rgba[~evidence.empty, :3] = 100, 100 / 255
    evidence = replace(
        evidence,
        target=target,
        rgba=rgba,
        coarse=target,
        smooth=target,
        drawn=~evidence.empty,
        line=~evidence.empty,
    )
    frontier, state, options = prepared(evidence, layers=True)
    factory, edits = combined(evidence, state, options)
    assert edits, factory.diagnostics
    assert factory.diagnostics["ink_plane_cells"] == 0
    assert all(p.details["core_material_cells"]["ink_cells"] == 0 for p in edits)
    assert all(frontier.policy.evaluate(export_svg(p.document)).valid for p in edits)


def test_cancellation_keeps_the_complete_original_namespace():
    evidence, _ = source(128, True, False)
    _, state, options = prepared(evidence, layers=True)
    work = Work.start(10)
    work.stop.set()
    _, edits = combined(evidence, state, options, work)
    assert edits == []
    assert state.partition.atoms is None


def test_combined_layout_requires_connected_dynamic_source_ink():
    evidence = material()
    _, _, options = prepared(evidence, layers=True)
    with pytest.raises(ValueError, match="connected dynamic"):
        CoreCells(
            Families(evidence, build(evidence), options),
            options,
            joint=True,
            grouping="ward",
            layout="ink-planes",
        )


def test_disabling_new_ink_reveals_its_material_underpaint_without_black_remnants():
    evidence, _ink = source(128, False, False)
    _, state, options = prepared(evidence, layers=True)
    _, edits = combined(evidence, state, options)
    edit = next(
        p for p in edits if p.details["core_material_cells"]["material_planes"] == 3
    )
    count = edit.details["core_material_cells"]["ink_cells"]
    editor = Editor(edit.document, selection=Selection(whole_document=True))
    with editor.transaction("Inspect plane paint beneath ink") as tx:
        for oid in edit.ids[-count:]:
            tx.set_fill(oid, "none")
            tx.set_attributes(oid, {"stroke": "none"})
    actual = render(export_svg(editor.snapshot.document), evidence.source_size)
    expected = material(128).target
    for x in (20, 55, 80):
        np.testing.assert_allclose(actual[19, x, :3] * 255, expected[19, x], atol=2)
    np.testing.assert_array_equal(
        actual[..., 3], render(state.svg, evidence.source_size)[..., 3]
    )


def test_cancelled_final_component_seal_emits_no_partial_plane_or_ink_edit(monkeypatch):
    from vectrify.refine.cel_plan.component_edits import ComponentEdit

    evidence, _ = source(128, False, True)
    _, state, options = prepared(evidence, layers=True)

    def stop(_cls, _document, _partition, _parent, work):
        work.stop.set()
        raise StageInterruptedError("Stopped final combined seal")

    monkeypatch.setattr(ComponentEdit, "bind", classmethod(stop))
    factory, edits = combined(evidence, state, options)
    assert edits == []
    assert factory.diagnostics["proposals"] == 0
    assert state.partition.atoms is None
    assert state.snapshot.canvas.matches(render(state.svg, evidence.source_size))


def test_seed_deduplication_is_local_to_one_call_and_cannot_discard_a_new_search():
    evidence, _ = source(128, False, True)
    _, state, options = prepared(evidence, layers=True)
    factory, first = combined(evidence, state, options)
    second = list(factory(state, Work.start(20)))
    assert first
    assert second
    assert [p.parameters for p in first] == [p.parameters for p in second]
    assert [p.partition.atoms for p in first] == [p.partition.atoms for p in second]
