"""Only final combined source/material cells consume the exported-cell budget."""

from dataclasses import replace

import numpy as np
import pytest

from tests.refine.test_cel_plan_core_cells import material
from tests.refine.test_cel_plan_families import prepared
from vectrify.document import export_svg, load_project, save_project
from vectrify.refine.cel_plan import core_cells
from vectrify.refine.cel_plan.core_cells import CoreCells
from vectrify.refine.cel_plan.families import Families
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Work
from vectrify.refine.cel_plan.proposals import Operators
from vectrify.refine.cel_plan.score import render


def fragmented(gap=False):
    evidence = material(128, hole=False)
    y, x = np.indices(evidence.labels.shape)
    ink = (y >= 18) & (y < 24) & (x >= 15) & (x < 81)
    if gap:
        ink &= (x < 46) | (x >= 50)
    rgb, rgba, labels = (
        evidence.target.copy(),
        evidence.rgba.copy(),
        evidence.labels.copy(),
    )
    rgb[ink], rgba[ink, :3] = 20, 20 / 255
    # Each source pixel has independent native ownership. One complete fitted
    # chain can replace those observations together, preserving real gaps.
    labels[ink] = np.arange(9, 9 + ink.sum())
    return replace(
        evidence,
        target=rgb,
        smooth=rgb,
        coarse=rgb,
        rgba=rgba,
        labels=labels,
        drawn=ink,
        line=ink,
    ), ink


def factory_for(evidence, options, monkeypatch, cap):
    grouped = core_cells.grouped

    def unmerged(colors, sizes, edges, _budget, work, **kwargs):
        # Exercise the valid no-material-merge grammar with the real hierarchy;
        # physical source decoding must compact its many ink observations.
        return grouped(colors, sizes, edges, len(sizes), work, **kwargs)

    monkeypatch.setattr(core_cells, "grouped", unmerged)
    monkeypatch.setattr(core_cells, "MAX_REGION_CELLS", cap)
    return CoreCells(
        Families(evidence, build(evidence), options),
        options,
        joint=True,
        grouping="ward",
        boundary_fit="anchored",
        ink_support="connected",
    )


@pytest.mark.parametrize("gap", [False, True])
def test_more_than_byte_sized_virtual_roots_compact_into_nine_final_native_cells(
    gap, monkeypatch
):
    evidence, ink = fragmented(gap)
    frontier, state, options = prepared(evidence, layers=True)
    options = replace(options, line_width=6)
    factory = factory_for(evidence, options, monkeypatch, 9)
    edits = list(factory(state, Work.start(30)))
    assert edits, factory.diagnostics
    assert factory.diagnostics["source_class_roots_peak"] > 256
    assert factory.diagnostics["source_final_cells_peak"] == 9
    ops = Operators(evidence, build(evidence), options)
    for edit in edits:
        assert edit.details["core_material_cells"]["cells"] == 9
        assert len(edit.details["core_material_cells"]["stroke_models"]) == 1
        assert edit.partition.atoms.cuts == ()
        assert edit.partition.follows(state.partition)
        ops.validate_partition(edit.partition, Work.start(20))
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
        # Source roots above 255 cannot wrap into material IDs or hide the line.
        # Opaque line interiors stay dark. Constant-width fitting can change
        # the original rectangular antialias fringe; that is scored natively.
        interior = ink[20, 20:76].copy()
        if gap:
            # Keep this assertion inside the bodies, away from fitted caps.
            interior &= (np.arange(20, 76) < 43) | (np.arange(20, 76) >= 53)
        np.testing.assert_allclose(actual[20, 20:76, :3][interior] * 255, 20, atol=2)
        if gap:
            np.testing.assert_allclose(
                actual[20, 47:49, :3] * 255, evidence.target[20, 47:49], atol=2
            )
        local = LocalPolicy(frontier.policy).update(
            state.snapshot, svg, edit.bounds, full.structure
        )
        assert local.canvas.matches(actual)
        assert local.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
        restored, _ = load_project(save_project(edit.document))
        np.testing.assert_array_equal(
            render(export_svg(restored), evidence.source_size), actual
        )
    assert state.partition.atoms is None


def test_final_over_budget_classification_publishes_no_prefix_or_source_cuts(
    monkeypatch,
):
    evidence, _ink = fragmented()
    _frontier, state, options = prepared(evidence, layers=True)
    options = replace(options, line_width=6)
    before = export_svg(state.document)
    factory = factory_for(evidence, options, monkeypatch, 8)
    assert list(factory(state, Work.start(30))) == []
    assert factory.diagnostics["source_class_roots_peak"] > 256
    assert factory.diagnostics["source_final_cells_peak"] == 9
    assert factory.diagnostics["region_exclusions"] > 0
    assert factory.diagnostics["cells"] == factory.diagnostics["proposals"] == 0
    assert state.partition.atoms is None
    assert export_svg(state.document) == before


def two_styles():
    evidence = material(128, hole=False)
    y, x = np.indices(evidence.labels.shape)
    first = (y >= 18) & (y < 22) & (x >= 15) & (x < 81)
    second = (y >= 42) & (y < 46) & (x >= 15) & (x < 81)
    rgb, rgba, labels = (
        evidence.target.copy(),
        evidence.rgba.copy(),
        evidence.labels.copy(),
    )
    rgb[~evidence.empty] = (190, 160, 130)
    rgb[first], rgb[second] = 20, 90
    rgba[~evidence.empty, :3] = rgb[~evidence.empty] / 255
    labels[first | second] = 9
    labels[second & (x >= 48)] = 10
    return replace(
        evidence,
        target=rgb,
        smooth=rgb,
        coarse=rgb,
        rgba=rgba,
        labels=labels,
        drawn=first | second,
        line=first | second,
    )


def test_complete_two_style_edit_can_fit_after_a_transient_class_increase(monkeypatch):
    evidence = two_styles()
    frontier, state, options = prepared(evidence, layers=True)
    # Deliberate width override covers the four-pixel source observations so
    # this control isolates complete-class budgeting rather than width fitting.
    options = replace(options, line_width=6)
    factory = factory_for(evidence, options, monkeypatch, 10)
    edits = list(factory(state, Work.start(20)))
    assert edits, factory.diagnostics
    assert factory.diagnostics["source_class_roots_peak"] == 10
    assert factory.diagnostics["source_final_cells_peak"] == 10
    for edit in edits:
        models = edit.details["core_material_cells"]["stroke_models"]
        assert len(models) == 2
        assert all(m["runs"] == 1 for m in models)
        assert len(edit.partition.atoms.cuts) == 1
        Operators(evidence, build(evidence), options).validate_partition(
            edit.partition, Work.start(10)
        )
        svg = export_svg(edit.document)
        full = frontier.policy.evaluate(svg)
        assert full.valid, full.rejections
        actual = render(svg, evidence.source_size)
        np.testing.assert_allclose(actual[20, 30, :3] * 255, 20, atol=2)
        np.testing.assert_allclose(actual[44, 30, :3] * 255, 90, atol=2)
        np.testing.assert_array_equal(
            actual[..., 3], render(state.svg, evidence.source_size)[..., 3]
        )
        assert edit.partition.follows(state.partition)
        local = LocalPolicy(frontier.policy).update(
            state.snapshot, svg, edit.bounds, full.structure
        )
        assert local.canvas.matches(actual)
        assert local.evaluation.terms == pytest.approx(full.terms, abs=2e-7)


def test_cancelled_combined_style_classification_never_publishes_the_first_style(
    monkeypatch,
):
    evidence = two_styles()
    _frontier, state, options = prepared(evidence, layers=True)
    options = replace(options, line_width=6)
    factory = factory_for(evidence, options, monkeypatch, 10)
    work = Work.start(10)
    before = export_svg(state.document)
    owned = core_cells.owned_model

    def stopped(*args, **kwargs):
        value = owned(*args, **kwargs)
        work.stop.set()
        return value

    monkeypatch.setattr(core_cells, "owned_model", stopped)
    assert list(factory(state, work)) == []
    assert factory.diagnostics["cells"] == factory.diagnostics["proposals"] == 0
    assert state.partition.atoms is None
    assert export_svg(state.document) == before
