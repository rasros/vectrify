"""Family replacements remove subdivisions under native/full acceptance."""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from tests.refine.test_cel_plan_ownership import stripes
from vectrify.document import export_svg, import_svg
from vectrify.refine.cel_plan import families
from vectrify.refine.cel_plan.export import export
from vectrify.refine.cel_plan.families import Families
from vectrify.refine.cel_plan.frontier import Frontier
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.local import HALO, MAX_CROP_PIXELS, LocalPolicy
from vectrify.refine.cel_plan.model import Options, Work
from vectrify.refine.cel_plan.opacity import fit, fit_samples
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


@pytest.mark.parametrize("crop_pixels", [47, 113, 4096])
def test_streamed_family_samples_match_dense_row_order_and_rgba_models(
    crop_pixels,
    monkeypatch,
):
    evidence = stripes(gradient=True, alpha=128, hole=True)
    # More than 4096 matching pixels exercise the stride across chunk edges.
    fields = (
        "rgba",
        "target",
        "smooth",
        "coarse",
        "empty",
        "foreground",
        "line",
        "drawn",
        "darkness",
        "texture",
        "labels",
        "opacity",
    )
    evidence = replace(
        evidence,
        **{name: np.repeat(getattr(evidence, name), 3, axis=0) for name in fields},
        source_size=(96, 192),
    )
    graph = build(evidence)
    factory = Families(evidence, graph, Options())
    monkeypatch.setattr(families, "MAX_CROP_PIXELS", crop_pixels)
    members = (1, 2, 4, 5, 7, 8)
    own = np.isin(graph.labels, members) & ~evidence.empty
    y, x = np.nonzero(own)
    step = (len(x) + 4095) // 4096
    expected_xy = np.column_stack((x[::step] + 0.5, y[::step] + 0.5))
    expected_rgba = np.column_stack(
        (
            evidence.target[y[::step], x[::step]] / 255,
            evidence.opacity[y[::step], x[::step]],
        )
    )
    xy, rgba, count = factory.samples(members, Work.start(10))
    assert count == len(x)
    assert len(xy) <= 4096
    np.testing.assert_array_equal(xy, expected_xy)
    np.testing.assert_array_equal(rgba, expected_rgba)
    assert fit_samples(xy, rgba) == fit(evidence.target, evidence.opacity, own)


def test_interrupted_family_sampling_does_not_publish_partial_samples(monkeypatch):
    evidence = stripes(gradient=True, alpha=128)
    factory = Families(evidence, build(evidence), Options())
    monkeypatch.setattr(families, "MAX_CROP_PIXELS", 113)
    work = Work.start(10)
    original = families.np.isin
    calls = []

    def stop(*args):
        result = original(*args)
        calls.append(1)
        work.stop.set()
        return result

    monkeypatch.setattr(families.np, "isin", stop)
    assert factory.samples(tuple(range(1, 9)), work) is None
    assert len(calls) == 1


def test_long_surface_family_reaches_native_tiled_acceptance_and_full_checkpoint():
    evidence = stripes(gradient=True, alpha=128)
    fields = (
        "rgba",
        "target",
        "smooth",
        "coarse",
        "empty",
        "foreground",
        "line",
        "drawn",
        "darkness",
        "texture",
        "labels",
        "opacity",
    )
    evidence = replace(
        evidence,
        **{
            name: np.repeat(np.repeat(getattr(evidence, name), 32, axis=0), 2, axis=1)
            for name in fields
        },
        source_size=(192, 2048),
    )
    frontier, state, options = prepared(evidence, layers=True)
    factory = Families(evidence, build(evidence), options)
    selected_proposal = []

    def long_gradient(state, work):
        for proposal in factory(state, work):
            if proposal.parameters[0] == "gradient" and len(proposal.ids) == 8:
                selected_proposal.append(proposal)
                yield proposal
                return

    result = search(frontier, options, Work.start(20), long_gradient)
    assert len(selected_proposal) == 1
    assert (
        selected_proposal[0]
        .bounds.expand(2 * HALO, state.snapshot.canvas.root.shape)
        .area
        > MAX_CROP_PIXELS
    )
    assert result["accepted"] == 1
    assert result["checkpointed"] == 1
    assert result["tiles_scored"] > 1
    assert result["score_disagreements"] == 0
    selected = frontier.select(100)
    assert selected.metrics["nodes"] < state.snapshot.evaluation.structure["nodes"]
    assert selected.metrics["gradients"] == 1
    assert selected.metrics["regions"] == 1
    assert frontier.policy.evaluate(selected.svg).valid


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


def strong_boundaries(graph):
    return replace(
        graph,
        boundaries=tuple(
            replace(edge, line_support=1) if min(edge.left, edge.right) > 0 else edge
            for edge in graph.boundaries
        ),
    )


def test_coarse_ink_on_shade_steps_can_offer_a_native_valid_surface():
    evidence = stripes(alpha=128)
    target = evidence.target.copy()
    for region in range(1, 9):
        target[evidence.labels == region, 0] += region
    rgba = evidence.rgba.copy()
    rgba[..., :3] = target / 255
    evidence = replace(evidence, target=target, rgba=rgba)
    frontier, state, options = prepared(evidence, layers=True)
    graph = strong_boundaries(build(evidence))
    factory = Families(evidence, graph, options)
    groups = list(factory._groups(state, Work.start(10)))
    assert any(len(ids) == 8 for ids, _ in groups)
    assert factory.diagnostics["shade_boundaries"] == 7
    assert factory.diagnostics["supported_ridges"] == 0
    cached = dict(factory.diagnostics)
    assert list(factory._groups(state, Work.start(10))) == groups
    assert factory.diagnostics == cached
    result = search(frontier, options, Work.start(10), factory)
    assert result["accepted"] > 0
    assert result["checkpointed"] > 0
    assert result["score_disagreements"] == 0
    selected = frontier.select(50)
    assert selected.metrics["nodes"] < state.snapshot.evaluation.structure["nodes"]
    np.testing.assert_array_equal(
        render(selected.svg, evidence.source_size)[..., 3], evidence.rgba[..., 3]
    )


def test_supported_ridge_cannot_be_removed_by_a_weak_alternate_route():
    evidence = stripes(alpha=128)
    target = evidence.target.copy()
    target[8:56, 16:20] = 30
    evidence = replace(evidence, target=target)
    _frontier, state, options = prepared(evidence, layers=True)
    graph = build(evidence)
    edge = next(e for e in graph.boundaries if {e.left, e.right} == {1, 2})
    # Probe a triangular adjacency: a weak alternate route must not bypass
    # the directly supported ink between the first two source owners.
    graph = replace(
        graph,
        boundaries=(
            replace(edge, line_support=1),
            replace(edge, id=100, left=1, right=3, line_support=0),
            replace(edge, id=101, left=2, right=3, line_support=0),
        ),
    )
    factory = Families(evidence, graph, options)
    groups = list(factory._groups(state, Work.start(10)))
    first, second = state.partition.owners[1], state.partition.owners[2]
    assert groups
    assert all(not {first, second}.issubset(ids) for ids, _ in groups)
    assert factory.diagnostics["supported_ridges"] == 1
    assert factory.diagnostics["shade_boundaries"] == 0


@pytest.mark.parametrize("limit", ["points", "proofs", "short"])
def test_unresolved_strong_boundaries_remain_protected(limit, monkeypatch):
    evidence = stripes()
    _frontier, state, options = prepared(evidence)
    graph = strong_boundaries(build(evidence))
    if limit == "points":
        monkeypatch.setattr(families, "MAX_BOUNDARY_POINTS", 4)
    elif limit == "proofs":
        monkeypatch.setattr(families, "MAX_BOUNDARY_PROOFS", 0)
    else:
        graph = replace(
            graph,
            boundaries=tuple(
                replace(edge, points=edge.points[:4]) for edge in graph.boundaries
            ),
        )
    factory = Families(evidence, graph, options)
    assert list(factory._groups(state, Work.start(10))) == []
    assert factory.diagnostics["ridge_proofs"] == 0
    assert factory.diagnostics["unresolved_boundaries"] == 7


def test_interrupted_ridge_proof_is_not_cached_as_a_shade_boundary(monkeypatch):
    evidence = stripes()
    _frontier, state, options = prepared(evidence)
    factory = Families(evidence, strong_boundaries(build(evidence)), options)
    work = Work.start(10)

    def interrupted(*_args, **_kwargs):
        work.stop.set()

    monkeypatch.setattr(families, "measure", interrupted)
    assert list(factory._groups(state, work)) == []
    assert factory.ridges == {}
    assert factory.diagnostics["shade_boundaries"] == 0
