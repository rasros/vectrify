"""Source-supported facet planes replace fragments with a complete material tree."""

from dataclasses import replace

import numpy as np
import pytest

from tests.helpers import required
from tests.refine.test_cel_plan_core_cells import material
from tests.refine.test_cel_plan_families import prepared
from tests.refine.test_cel_plan_piecewise_surfaces import marked_step
from vectrify.document import export_svg, load_project, save_project
from vectrify.refine.cel_plan.core_cells import CoreCells
from vectrify.refine.cel_plan.families import Families
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Work
from vectrify.refine.cel_plan.ownership import Partition
from vectrify.refine.cel_plan.proposals import Operators
from vectrify.refine.cel_plan.score import render, svg_metrics


def planes(evidence, state, options, work=None):
    factory = CoreCells(
        Families(evidence, build(evidence), options),
        options,
        joint=True,
        layout="planes",
    )
    return factory, list(factory(state, work or Work.start(20)))


@pytest.mark.parametrize("alpha", [255, 128])
@pytest.mark.parametrize("hole", [False, True])
def test_final_three_plane_prefix_is_offered_with_exact_coverage_and_complete_ownership(
    alpha, hole
):
    evidence = material(alpha, hole)
    frontier, state, options = prepared(evidence, layers=True)
    factory, edits = planes(evidence, state, options)
    exact = [p for p in edits if p.parameters[0] == 3]
    assert exact, factory.diagnostics
    edit = exact[0]
    assert edit.details is not None
    assert edit.details["core_material_cells"]["model_stage"] == "source-facet-planes"
    assert edit.details["core_material_cells"]["source_squared_error"] == 0
    assert edit.partition is not None
    assert state.partition is not None
    assert edit.partition.follows(state.partition)
    assert edit.partition.atoms is not None
    assert edit.partition.atoms.cuts
    ops = Operators(evidence, build(evidence), options)
    ops.validate_partition(edit.partition, Work.start(10))
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
    if hole:
        assert actual[24:40, 40:56, 3].max() == 0
    assert svg_metrics(svg)["nodes"] < svg_metrics(state.svg)["nodes"]
    local = LocalPolicy(frontier.policy).update(
        state.snapshot, svg, edit.bounds, full.structure
    )
    assert local.canvas.matches(actual)
    assert local.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
    document, _ = load_project(save_project(edit.document))
    required(Partition.from_metadata(edit.partition.metadata())).validate(document)
    np.testing.assert_array_equal(
        render(export_svg(document), evidence.source_size), actual
    )


def test_independently_held_mark_keeps_geometry_paint_and_source_owner_above_planes():
    evidence = marked_step(128)
    evidence = replace(evidence, opacity=evidence.rgba[..., 3])
    frontier, state, options = prepared(evidence, layers=True)
    assert state.partition is not None
    mark = state.partition.owners[9]
    state = replace(state, details={**state.details, "paint_constraints": [mark]})
    factory, edits = planes(evidence, state, options)
    assert edits, factory.diagnostics
    for p in edits:
        assert p.partition is not None
        assert p.partition.owners[9] == mark
        assert p.document.element(mark) == state.document.element(mark)
        assert p.document.geometry_for(mark) == state.document.geometry_for(mark)
        assert frontier.policy.evaluate(export_svg(p.document)).valid


def test_cancelled_plan_does_not_publish_a_material_prefix():
    evidence = material()
    _, state, options = prepared(evidence, layers=True)
    work = Work.start(10)
    work.stop.set()
    _, edits = planes(evidence, state, options, work)
    assert edits == []
    assert state.partition is not None
    assert state.partition.atoms is None


def test_plane_layout_requires_an_atomic_joint_component():
    evidence = material()
    _, _state, options = prepared(evidence, layers=True)
    with pytest.raises(ValueError, match="joint"):
        CoreCells(
            Families(evidence, build(evidence), options), options, layout="planes"
        )
