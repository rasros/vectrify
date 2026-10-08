"""Both source interpretations enter native search with bounded lifetimes."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pathops
import pytest

from tests.refine.test_cel_plan_families import prepared
from tests.refine.test_cel_plan_ink_models import drawing
from vectrify.document import export_svg, load_project, save_project
from vectrify.refine.cel_plan import joint_cells
from vectrify.refine.cel_plan.core_cells import CoreCells
from vectrify.refine.cel_plan.families import Families
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.joint_cells import JointCells
from vectrify.refine.cel_plan.model import Work
from vectrify.refine.cel_plan.proposals import Operators
from vectrify.refine.cel_plan.score import render, svg_metrics
from vectrify.refine.cel_plan.search import search


def fixture(gap=False, alpha=128, block=8):
    evidence, _ink = drawing(alpha, gap=gap)
    y, x = np.indices(evidence.labels.shape)
    labels = np.where(
        evidence.empty, 0, 1 + (y - 8) // block * (80 // block) + (x - 8) // block
    )
    evidence = replace(
        evidence,
        labels=labels.astype(np.int32),
        opacity=evidence.opacity
        if evidence.opacity is not None
        else evidence.rgba[..., 3],
    )
    frontier, state, options = prepared(evidence, layers=True)
    graph = build(evidence)
    return evidence, graph, frontier, state, options


@pytest.mark.parametrize("gap", [False, True])
@pytest.mark.parametrize("alpha", [255, 128, 64])
def test_shared_scheduler_preserves_each_standalone_pool_and_native_source_contracts(
    gap, alpha
):
    evidence, graph, frontier, state, options = fixture(gap, alpha)
    factory = JointCells(Families(evidence, graph, options), options, minimum_paths=3)
    proposals = list(factory(state, Work.start(15)))
    assert proposals
    modes = [p.details["joint_cell_search"]["ink_fit"] for p in proposals]
    assert modes[:2] == ["source-intervals", "source-widths"]
    validator = Operators(evidence, graph, options)
    for mode in ("source-intervals", "source-widths"):
        baseline = CoreCells(
            Families(evidence, graph, options),
            options,
            joint=True,
            grouping="ward",
            boundary_fit="anchored",
            ink_support="connected",
            ink_roles="fitted",
            ink_coverage="fractional",
            ink_fit=mode,
            facet_fit="regional",
            atom_layout="residual",
        )
        original = list(baseline(state, Work.start(15)))
        actual = [
            p for p in proposals if p.details["joint_cell_search"]["ink_fit"] == mode
        ]
        assert len(original) == len(actual)
        for a, b in zip(original, actual, strict=True):
            a_svg, b_svg = export_svg(a.document), export_svg(b.document)
            np.testing.assert_array_equal(
                render(a_svg, evidence.source_size), render(b_svg, evidence.source_size)
            )
            assert svg_metrics(a_svg) == svg_metrics(b_svg)
            assert a.partition == b.partition
            assert b.parameters[-1] == ("ink-fit", mode)
            validator.validate_partition(b.partition, Work.start(10))
            assert frontier.policy.evaluate(b_svg).valid
            document, _ = load_project(save_project(b.document))
            assert export_svg(document) == b_svg
    assert factory.diagnostics["extractions"] == 1
    assert factory.diagnostics["reuses"] == 1
    assert state.partition.atoms is None


def test_paired_interpretations_are_checked_by_real_local_search_and_checkpoint():
    evidence, graph, frontier, state, options = fixture()
    factory = JointCells(Families(evidence, graph, options), options, minimum_paths=3)
    validator = Operators(evidence, graph, options)

    class OnlyJoint:
        def __call__(self, current, work):
            if current.edits:
                return
            yield from factory(current, work)

        def validate_partition(self, partition, work):
            validator.validate_partition(partition, work)

    before = frontier.baseline
    result = search(frontier, options, Work.start(15), OnlyJoint())
    assert result["accepted"] > 0
    assert result["score_disagreements"] == 0
    assert result["checkpointed"] > 0
    assert frontier.baseline is before
    selected = frontier.select(50)
    assert selected.metrics["nodes"] < state.snapshot.evaluation.structure["nodes"]
    assert "joint_cell_search" in selected.metrics
    assert frontier.policy.evaluate(selected.svg).valid


@pytest.mark.parametrize("fail", ["source-intervals", "source-widths"])
def test_boolean_failure_keeps_other_interpretation_and_closes_both_cursors(
    fail, monkeypatch
):
    evidence, graph, _, state, options = fixture()
    closed = []

    from vectrify.refine.cel_plan.local import Box
    from vectrify.refine.cel_plan.search import Proposal

    edit = Proposal(
        "joint-core-cells",
        (),
        (),
        state.key,
        state.document,
        Box(0, 0, *evidence.source_size),
        partition=state.partition,
    )

    def bounded(self, _state, _work):
        try:
            if self.ink_fit == fail:
                raise pathops.PathOpsError("Injected interpretation failure")
            yield edit
        finally:
            closed.append(self.ink_fit)

    monkeypatch.setattr(CoreCells, "__call__", bounded)
    factory = JointCells(SimpleNamespace(evidence=evidence, graph=graph), options)
    found = list(factory(state, Work.start(10)))
    assert len(found) == 1
    assert set(closed) == {"source-intervals", "source-widths"}
    assert factory.diagnostics["boolean_failures"] == 1


@pytest.mark.parametrize("reason", ["cancel", "limit", "close"])
def test_early_exit_releases_shared_discovery_and_original_state(reason, monkeypatch):
    evidence, graph, _, state, options = fixture()
    caches = []
    original = joint_cells.InkDiscovery

    def capture():
        cache = original()
        caches.append(cache)
        return cache

    monkeypatch.setattr(joint_cells, "InkDiscovery", capture)
    if reason == "limit":
        monkeypatch.setattr(joint_cells, "MAX_PROPOSALS", 1)
    factory = JointCells(Families(evidence, graph, options), options, minimum_paths=3)
    work = Work.start(10)
    cursor = iter(factory(state, work))
    first = next(cursor)
    assert first.details["joint_cell_search"]["ink_fit"] == "source-intervals"
    assert caches[0]._value is not None
    if reason == "cancel":
        work.stop.set()
    if reason == "close":
        cursor.close()
    else:
        assert list(cursor) == []
    assert caches[0]._value is None
    assert state.partition.atoms is None
    assert factory.diagnostics["extractions"] == 1
    assert factory.diagnostics["bounded"] == (reason == "limit")


@pytest.mark.parametrize("quality", ["fast", "balanced", "high"])
@pytest.mark.parametrize("block", [4, 8])
def test_only_high_gives_a_fragmented_seed_the_first_joint_opportunity(quality, block):
    evidence, graph, _, state, options = fixture(block=block)
    options = replace(options, quality=quality)
    factory = Families(evidence, graph, options)
    cursor = iter(factory(state, Work.start(15)))
    try:
        proposal = next(cursor)
        if quality == "high" and block == 4:
            assert proposal.operator == "joint-core-cells"
            assert (
                proposal.details["joint_cell_search"]["ink_fit"] == "source-intervals"
            )
        else:
            assert "joint_cell_search" not in (proposal.details or {})
    finally:
        cursor.close()
    if quality == "high":
        assert factory._joint_models.diagnostics["proposals"] == (block == 4)
    else:
        assert factory._joint_models is None


@pytest.mark.parametrize("quality", ["fast", "balanced", "high"])
def test_joint_parent_can_resume_with_or_without_the_high_only_opacity_slot(quality):
    evidence, graph, _, state, options = fixture()
    options = replace(options, quality=quality)
    parent = replace(state, edits=({"operator": "joint-core-cells"},))
    operators = Operators(evidence, graph, options)
    cursor = iter(operators(parent, Work.start(15)))
    try:
        proposal = next(cursor)
        assert proposal.parent == parent.key
        assert proposal.partition is not None
    finally:
        cursor.close()
