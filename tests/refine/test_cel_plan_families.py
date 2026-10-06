"""Family replacements remove subdivisions under native/full acceptance."""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from tests.refine.test_cel_plan_ownership import stripes
from vectrify.document import export_svg, import_svg
from vectrify.refine.cel_plan.export import export
from vectrify.refine.cel_plan.families import Families
from vectrify.refine.cel_plan.frontier import Frontier
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Options, Work
from vectrify.refine.cel_plan.ownership import Partition
from vectrify.refine.cel_plan.policy import Policy
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.search import State, search


def prepared(evidence, *, layers=False):
    options = Options(refine=False)
    svg, details = export(
        evidence,
        evidence.labels,
        options,
        Work.start(10),
        layers=layers,
        conservative=evidence.opacity is None,
    )
    policy = Policy(evidence.rgba)
    frontier = Frontier(policy)
    assert frontier.add(svg, "Partitioned", details)
    frontier.freeze_normalizer()
    entry = frontier.entries[0]
    state = State(
        import_svg(svg),
        svg,
        LocalPolicy(policy).start(svg, entry.evaluation),
        entry.key,
        details,
        partition=Partition.from_metadata(details["planning_surfaces"]),
    )
    return frontier, state, options


@pytest.mark.parametrize("alpha", [255, 64])
@pytest.mark.parametrize("hole", [False, True])
def test_connected_family_becomes_one_surface_preserving_alpha_and_holes(alpha, hole):
    evidence = stripes(alpha=alpha, hole=hole)
    frontier, state, options = prepared(evidence, layers=alpha != 255)
    initial_nodes = state.snapshot.evaluation.structure["nodes"]
    result = search(
        frontier, options, Work.start(10), Families(evidence, build(evidence), options)
    )
    selected = frontier.select(50)
    assert result["accepted"] > 0
    assert result["score_disagreements"] == 0
    assert selected.metrics["nodes"] < initial_nodes / 2
    assert selected.metrics["local_edits"][0]["operator"] == "family-surface"
    document = import_svg(selected.svg)
    partition = Partition.from_metadata(selected.metrics["planning_surfaces"])
    assert partition is not None
    partition.validate(document)
    assert len([s for s in partition.surfaces if s.role == "surface"]) == 1
    assert selected.metrics["regions"] == 1
    assert set(partition.owners) == set(state.partition.owners)
    actual = render(export_svg(document), evidence.source_size)
    np.testing.assert_array_equal(actual[..., 3], evidence.rgba[..., 3])
    if hole:
        assert actual[24:40, 40:56, 3].max() == 0
    assert len([s for s in state.partition.surfaces if s.role == "surface"]) == 8


def test_gradient_family_removes_paint_and_geometry_fragmentation():
    evidence = stripes(gradient=True, alpha=128)
    frontier, _state, options = prepared(evidence, layers=True)
    initial = frontier.select(100)
    factory = Families(evidence, build(evidence), options)

    def gradient_only(state, work):
        # Validate the complete gradient interpretation independently. A flat
        # alternative may win common-frontier selection at these initial weights.
        for proposal in factory(state, work):
            if proposal.parameters[0] == "gradient" and len(proposal.ids) == 8:
                yield proposal
                return

    result = search(frontier, options, Work.start(10), gradient_only)
    selected = frontier.select(100)
    assert result["accepted"] > 0
    assert result["score_disagreements"] == 0
    assert selected.metrics["gradients"] == 1
    assert selected.metrics["nodes"] < initial.metrics["nodes"] / 2
    assert selected.metrics["objective"] < initial.metrics["objective"]
    assert selected.metrics["local_edits"][0]["parameters"][0] == "gradient"
    assert frontier.policy.evaluate(selected.svg).valid


def test_ownership_updates_after_native_acceptance_not_during_proposal_creation():
    evidence = stripes()
    frontier, state, options = prepared(evidence)
    factory = Families(evidence, build(evidence), options)
    proposal = next(factory(state, Work.start(10)))
    assert len(proposal.partition.surfaces) < len(state.partition.surfaces)
    assert len(state.partition.surfaces) == 8
    assert sum(e.tag == "path" for e in import_svg(state.svg).elements()) == 8
    assert len(frontier.entries) == 1


def test_fixed_width_ink_and_different_components_do_not_enter_surface_families():
    evidence = stripes()
    _frontier, state, options = prepared(evidence)
    graph = build(evidence)
    graph = replace(graph, regions=tuple(replace(r, fixed=True) for r in graph.regions))
    assert list(Families(evidence, graph, options)(state, Work.start(10))) == []
    graph = build(evidence)
    graph = replace(
        graph, regions=tuple(replace(r, component=r.id) for r in graph.regions)
    )
    assert list(Families(evidence, graph, options)(state, Work.start(10))) == []


def test_gradient_family_uses_native_reference_frame_inside_a_scaled_offset_group():
    evidence = stripes(gradient=True, alpha=128)
    native = np.zeros((128, 160, 4), dtype=np.uint8)
    native[30:62, 20:68] = np.asarray(
        Image.fromarray(np.rint(evidence.rgba * 255).astype(np.uint8)).resize(
            (48, 32), Image.Resampling.NEAREST
        )
    )
    evidence = replace(
        evidence,
        rgba=native.astype(np.float32) / 255,
        source_size=(160, 128),
        offset=(20, 30),
        scale=(2, 2),
    )
    frontier, state, options = prepared(evidence, layers=True)
    factory = Families(evidence, build(evidence), options)
    edit = next(
        proposal
        for proposal in factory(state, Work.start(10))
        if proposal.parameters[0] == "gradient" and len(proposal.ids) == 8
    )
    svg = export_svg(edit.document)
    updated = LocalPolicy(frontier.policy).update(
        state.snapshot,
        svg,
        edit.bounds,
        frontier.policy.evaluate(svg).structure,
    )
    assert frontier.checkpoint(
        svg, "Offset gradient", {}, updated.evaluation, raster=updated.canvas
    )
    pixels = render(svg, (160, 128))
    assert (
        np.mean(np.abs(pixels[34:58, 25:63, :3] - evidence.rgba[34:58, 25:63, :3]))
        < 2 / 255
    )
    np.testing.assert_array_equal(pixels[..., 3], evidence.rgba[..., 3])


def test_structural_edit_cannot_drop_source_members_before_native_evaluation():
    evidence = stripes()
    frontier, _state, options = prepared(evidence)
    factory = Families(evidence, build(evidence), options)

    def missing(state, work):
        for edit in factory(state, work):
            surfaces = tuple(
                replace(s, members=tuple(m for m in s.members if m != 1))
                for s in edit.partition.surfaces
                if s.members != (1,)
            )
            yield replace(edit, partition=Partition(surfaces))
            return

    result = search(frontier, options, Work.start(10), missing)
    assert result["attempted"] == 0
    assert result["accepted"] == 0
    assert result["decisions"][0]["rejections"] == ["invalid-surface-ownership"]


def test_partial_family_preserves_neighbor_geometry_and_canonical_boundary_ids():
    evidence = stripes(gradient=True, alpha=128)
    _frontier, state, options = prepared(evidence, layers=True)
    graph = build(evidence)
    edit = next(
        proposal
        for proposal in Families(evidence, graph, options)(state, Work.start(10))
        if 1 < len(proposal.ids) < 8
    )
    assert set(edit.partition.owners) == set(state.partition.owners)
    for surface in state.partition.surfaces:
        if surface.id not in edit.ids:
            assert edit.document.geometry_for(
                surface.id
            ) == state.document.geometry_for(surface.id)
    before = {edge.boundary for edge in state.partition.edges(graph)}
    after = {edge.boundary for edge in edit.partition.edges(graph)}
    assert after < before
    assert {edge.id for edge in graph.boundaries}.issuperset(after)
