"""Joint component cells retain complete source support and native alpha."""

from dataclasses import replace

import numpy as np
import pytest

from tests.helpers import required
from tests.refine.test_cel_plan_core_cells import material
from tests.refine.test_cel_plan_families import prepared
from tests.refine.test_cel_plan_piecewise_surfaces import marked_step
from vectrify.document import export_svg, load_project, save_project
from vectrify.document.svg import parse_path
from vectrify.refine import cel
from vectrify.refine.cel_plan import core_cells
from vectrify.refine.cel_plan.core_cells import CoreCells, _filled, _supported_outlines
from vectrify.refine.cel_plan.families import Families
from vectrify.refine.cel_plan.geometry import Boundaries
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Work
from vectrify.refine.cel_plan.ownership import Partition
from vectrify.refine.cel_plan.policy import Policy
from vectrify.refine.cel_plan.proposals import Operators
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.search import search


@pytest.mark.parametrize("step", [8, 10, 16])
def test_thin_material_between_two_fills_keeps_both_shared_boundaries(step):
    y, x = np.mgrid[:48, :48]
    border = 22 - y // step
    labels = np.where(x < border, 1, 3).astype(np.int32)
    strip = (x == border) & (y >= 10) & (y < 31)
    labels[strip] = 2
    models = Boundaries()

    def boundary(points, _tolerance):
        nodes = models(points, 1.5)
        if models.decisions[-1]["model"] == "curve":
            nodes = [("L", tuple(p)) for p in cel.simplify(points, 0.75)[1:]]
        return nodes

    before = cel.region_outlines(labels, 0, fit_boundary=boundary)
    assert not _filled(parse_path(before[2]), "evenodd").subpaths
    after, retried = _supported_outlines(labels, boundary, lambda: None)
    assert retried == 1
    assert _filled(parse_path(after[2]), "evenodd").subpaths
    svg = (
        '<svg width="48" height="48">'
        + "".join(
            f'<path d="{after[i]}" fill="{paint}"/>'
            for i, paint in ((1, "red"), (2, "lime"), (3, "blue"))
        )
        + "</svg>"
    )
    actual = render(svg, (48, 48))
    np.testing.assert_array_equal(actual[10:31, :, 3], 1)
    np.testing.assert_array_equal(
        actual[strip, :3], np.tile((0, 1, 0), (strip.sum(), 1))
    )
    # The neighboring material shares the restored chain, without overlaps.
    np.testing.assert_array_equal(actual[..., 1].sum(), strip.sum())


@pytest.mark.parametrize("grouping", ["ward", "paint-fit"])
@pytest.mark.parametrize("layout", ["regions", "ink-planes"])
@pytest.mark.parametrize("boundary_fit", ["curve", "anchored"])
def test_joint_material_budget_keeps_short_hatching_as_supported_ink(
    grouping, layout, boundary_fit
):
    evidence = material(128)
    labels, target, rgba = (
        evidence.labels.copy(),
        evidence.target.copy(),
        evidence.rgba.copy(),
    )
    drawn = np.zeros_like(evidence.empty)
    for i, x in enumerate((20, 35, 50, 65), start=9):
        for y in range(14, 44):
            xx = x + (y - 14) // 3
            labels[y, xx] = i
            target[y, xx] = 10
            rgba[y, xx, :3] = 10 / 255
            drawn[y, xx] = True
    evidence = replace(
        evidence,
        labels=labels,
        target=target,
        smooth=target,
        coarse=target,
        rgba=rgba,
        drawn=drawn,
        line=drawn,
    )
    frontier, state, options = prepared(evidence, layers=True)
    factory = CoreCells(
        Families(evidence, build(evidence), options),
        options,
        joint=True,
        grouping=grouping,
        boundary_fit=boundary_fit,
        layout=layout,
        ink_support="connected" if layout == "ink-planes" else "paired",
    )
    edits = list(factory(state, Work.start(10)))
    assert edits
    for edit in edits:
        actual = render(export_svg(edit.document), evidence.source_size)
        assert np.mean(cel.lightness(actual[drawn, :3] * 255) < 100) >= 0.9
        assert frontier.policy.evaluate(export_svg(edit.document)).valid
        assert edit.partition is not None
        assert state.partition is not None
        assert edit.partition.follows(state.partition)
    assert factory.diagnostics["hierarchy_ink_cells_peak"] >= 1
    assert factory.diagnostics["hierarchy_ink_paint_links_peak"] >= 3


@pytest.mark.parametrize("grouping", ["ward", "paint-fit"])
@pytest.mark.parametrize("layout", ["regions", "ink-planes"])
@pytest.mark.parametrize("boundary_fit", ["curve", "anchored"])
def test_more_than_64_ink_islands_share_paint_without_bridges_or_lost_owners(
    grouping, layout, boundary_fit
):
    evidence = material(128)
    labels, target, rgba = (
        evidence.labels.copy(),
        evidence.target.copy(),
        evidence.rgba.copy(),
    )
    drawn = np.zeros_like(evidence.empty)
    for i in range(70):
        x, y = 10 + (i % 14) * 5, 10 + (i // 14) * 8
        labels[y : y + 4, x : x + 4] = i + 9
        target[y : y + 4, x : x + 4] = 10
        rgba[y : y + 4, x : x + 4, :3] = 10 / 255
        drawn[y : y + 4, x : x + 4] = True
    evidence = replace(
        evidence,
        labels=labels,
        target=target,
        smooth=target,
        coarse=target,
        rgba=rgba,
        drawn=drawn,
        line=drawn,
    )
    frontier, state, options = prepared(evidence, layers=True)
    factory = CoreCells(
        Families(evidence, build(evidence), options),
        options,
        joint=True,
        grouping=grouping,
        boundary_fit=boundary_fit,
        layout=layout,
        ink_support="connected" if layout == "ink-planes" else "paired",
    )
    edits = list(factory(state, Work.start(15)))
    assert edits
    for edit in edits:
        actual = render(export_svg(edit.document), evidence.source_size)
        dark = cel.lightness(actual[..., :3] * 255) < 50
        np.testing.assert_array_equal(dark & ~evidence.empty, drawn)
        assert frontier.policy.evaluate(export_svg(edit.document)).valid
        assert edit.partition is not None
        assert state.partition is not None
        assert edit.partition.follows(state.partition)
        if layout == "regions":
            assert set(edit.partition.owners) == set(state.partition.owners)
            assert edit.partition.atoms is not None
            assert len(edit.partition.atoms.cuts) == 0
        else:
            # Material planes may split a background atom. None of the 70
            # independently owned ink islands may be split or disappear.
            islands = tuple(int(i) for i in np.unique(build(evidence).labels[drawn]))
            assert edit.partition.atoms is not None
            assert edit.partition.atoms.descendants(islands) == islands
            assert set(islands).issubset(edit.partition.owners)
            Operators(evidence, build(evidence), options).validate_partition(
                edit.partition, Work.start(10)
            )
    assert 1 <= factory.diagnostics["hierarchy_ink_cells_peak"] < 64
    assert factory.diagnostics["hierarchy_ink_paint_links_peak"] >= 69


@pytest.mark.parametrize("grouping", ["ward", "paint-fit"])
@pytest.mark.parametrize("layout", ["regions", "ink-planes"])
def test_broad_flat_dark_material_is_not_ink_even_when_cel_marks_it_drawn(
    grouping, layout
):
    evidence = material(128)
    target, rgba = evidence.target.copy(), evidence.rgba.copy()
    target[~evidence.empty] = 100
    rgba[~evidence.empty, :3] = 100 / 255
    evidence = replace(
        evidence,
        target=target,
        smooth=target,
        coarse=target,
        rgba=rgba,
        drawn=~evidence.empty,
        line=~evidence.empty,
    )
    _, state, options = prepared(evidence, layers=True)
    factory = CoreCells(
        Families(evidence, build(evidence), options),
        options,
        joint=True,
        grouping=grouping,
        boundary_fit="curve",
        layout=layout,
        ink_support="connected" if layout == "ink-planes" else "paired",
    )
    assert list(factory(state, Work.start(10)))
    assert factory.diagnostics["hierarchy_ink_cells_peak"] == 0


@pytest.mark.parametrize("grouping", ["ward", "paint-fit"])
def test_co_paint_links_share_the_component_discovery_edge_bound(grouping, monkeypatch):
    evidence = material(128)
    _, state, options = prepared(evidence, layers=True)
    monkeypatch.setattr(core_cells, "MAX_REGION_EDGES", 64)
    monkeypatch.setattr(core_cells, "ink_paint_links", lambda *_: [(0, 1)] * 65)
    factory = CoreCells(
        Families(evidence, build(evidence), options),
        options,
        joint=True,
        grouping=grouping,
        boundary_fit="curve",
    )
    assert list(factory(state, Work.start(10))) == []
    assert factory.diagnostics["bounded"] == 1
    assert factory.diagnostics["proposals"] == 0
    assert state.snapshot.canvas.matches(render(state.svg, evidence.source_size))


@pytest.mark.parametrize("grouping", ["ward", "paint-fit"])
def test_offline_dynamic_component_publishes_through_full_native_checkpoint(grouping):
    evidence = material(128)
    y, x = np.indices(evidence.labels.shape)
    labels = np.where(evidence.empty, 0, 1 + (y - 8) // 3 * 16 + (x - 8) // 5)
    evidence = replace(evidence, labels=labels.astype(np.int32))
    frontier, state, options = prepared(evidence, layers=True)
    families = Families(evidence, build(evidence), options)
    factory = CoreCells(
        families,
        options,
        joint=True,
        grouping=grouping,
        boundary_fit="curve",
    )

    validator = Operators(evidence, build(evidence), options)

    class DynamicOnly:
        def __call__(self, current, work):
            yield from factory(current, work)

        def validate_partition(self, partition, work):
            validator.validate_partition(partition, work)

    result = search(frontier, options, Work.start(15), DynamicOnly())
    assert result["accepted"] > 0
    assert result["score_disagreements"] == 0
    selected = frontier.select(50)
    assert selected.metrics["nodes"] < state.snapshot.evaluation.structure["nodes"]
    assert any(
        e["operator"] == "joint-core-cells" for e in selected.metrics["local_edits"]
    )
    assert selected.metrics["core_material_cells"]["grouping"] == grouping
    assert selected.metrics["core_material_cells"]["boundary_fit"] == "curve"
    assert factory.diagnostics["proposals"] > 0
    assert frontier.policy.evaluate(selected.svg).valid
    partition = Partition.from_metadata(selected.metrics["planning_surfaces"])
    assert partition is not None
    assert state.partition is not None
    assert partition.follows(state.partition)


@pytest.mark.parametrize("alpha", [255, 128, 64])
@pytest.mark.parametrize("hole", [False, True])
@pytest.mark.parametrize(
    ("grouping", "boundary_fit"),
    [
        ("static", "polygon"),
        ("ward", "polygon"),
        ("ward", "curve"),
        ("paint-fit", "curve"),
    ],
)
def test_joint_component_includes_every_eligible_owner_with_complete_uncut_support(
    alpha, hole, grouping, boundary_fit
):
    evidence = material(alpha, hole)
    frontier, state, options = prepared(evidence, layers=True)
    graph = build(evidence)
    factory = CoreCells(
        Families(evidence, graph, options),
        options,
        joint=True,
        grouping=grouping,
        boundary_fit=boundary_fit,
    )
    edits = list(factory(state, Work.start(15)))
    assert edits
    for edit in edits:
        assert edit.operator == "joint-core-cells"
        assert edit.details is not None
        assert edit.details["core_material_cells"]["joint"]
        assert edit.partition is not None
        assert edit.partition.atoms is not None
        assert edit.partition.atoms.cuts == ()
        assert state.partition is not None
        assert edit.partition.follows(state.partition)
        Operators(evidence, graph, options).validate_partition(
            edit.partition, Work.start(10)
        )
        assert set(edit.partition.owners) == set(state.partition.owners)
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
        reloaded, _ = load_project(save_project(edit.document))
        required(Partition.from_metadata(edit.partition.metadata())).validate(reloaded)
        np.testing.assert_array_equal(
            render(export_svg(reloaded), evidence.source_size), actual
        )
    assert (
        min(frontier.policy.evaluate(export_svg(edit.document)).cost for edit in edits)
        < state.snapshot.evaluation.cost
    )


@pytest.mark.parametrize("protection", ["paint", "locked", "pinned", "fixed"])
@pytest.mark.parametrize(
    ("grouping", "layout"),
    [
        ("static", "regions"),
        ("ward", "regions"),
        ("paint-fit", "regions"),
        ("ward", "ink-planes"),
        ("paint-fit", "ink-planes"),
    ],
)
def test_independently_protected_mark_keeps_paint_geometry_and_primary_source_owner(
    protection,
    grouping,
    layout,
):
    evidence = marked_step(128)
    frontier, state, options = prepared(evidence, layers=True)
    graph = build(evidence)
    assert state.partition is not None
    oid = state.partition.owners[9]
    document = state.document
    if protection == "paint":
        state = replace(state, details={**state.details, "paint_constraints": [oid]})
    elif protection == "locked":
        document = document.replace_element(
            replace(document.element(oid), locks=frozenset({"geometry"}))
        )
    elif protection == "pinned":
        geometry = document.geometry_for(oid)
        sub = geometry.subpaths[0]
        document = document.replace_geometry(
            replace(
                geometry,
                subpaths=(
                    replace(
                        sub, nodes=(replace(sub.nodes[0], pinned=True), *sub.nodes[1:])
                    ),
                    *geometry.subpaths[1:],
                ),
            )
        )
    else:
        graph = replace(
            graph,
            regions=tuple(
                replace(r, fixed=True) if r.id == 9 else r for r in graph.regions
            ),
        )
    state = replace(state, document=document)
    factory = CoreCells(
        Families(evidence, graph, options),
        options,
        joint=True,
        grouping=grouping,
        layout=layout,
        ink_support="connected" if layout == "ink-planes" else "paired",
        boundary_fit="anchored" if layout == "ink-planes" else "polygon",
    )
    edits = list(factory(state, Work.start(15)))
    assert edits
    for edit in edits:
        assert edit.partition is not None
        assert edit.partition.owners[9] == oid
        assert edit.document.element(oid) == state.document.element(oid)
        assert edit.document.geometry_for(oid) == state.document.geometry_for(oid)
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
        assert frontier.policy.evaluate(export_svg(edit.document)).valid


def test_joint_discovery_stop_keeps_the_original_checkpoint_and_namespace():
    evidence = material(128)
    _, state, options = prepared(evidence, layers=True)
    factory = CoreCells(
        Families(evidence, build(evidence), options), options, joint=True
    )
    work = Work.start(10)
    work.stop.set()
    assert list(factory(state, work)) == []
    assert state.partition is not None
    assert state.partition.atoms is None
    assert state.snapshot.canvas.matches(render(state.svg, evidence.source_size))


def test_joint_carrier_fringe_is_scored_without_waiving_material_opacity_checks():
    evidence = material(128)
    _, state, options = prepared(evidence, layers=True)
    assert state.partition is not None
    oid = state.partition.owners[1]
    element = state.document.element(oid)
    attrs = {**dict(element.attributes), "transform": "translate(-0.4 0)"}
    document = state.document.replace_element(
        replace(element, attributes=tuple(attrs.items()))
    )
    before = export_svg(document)
    truth = render(before, evidence.source_size)
    labels = evidence.labels.copy()
    labels[8:56, 7] = 1
    empty = truth[..., 3] == 0
    target = truth[..., :3] * 255
    evidence = replace(
        evidence,
        rgba=truth,
        labels=labels,
        empty=empty,
        target=target,
        smooth=target,
        coarse=target,
        opacity=truth[..., 3],
    )
    policy = Policy(truth)
    baseline = policy.evaluate(before)
    assert baseline.valid
    policy.establish(baseline)
    state = replace(
        state,
        document=document,
        svg=before,
        snapshot=LocalPolicy(policy).start(before, baseline),
    )
    factory = CoreCells(
        Families(evidence, build(evidence), options), options, joint=True
    )
    edits = list(factory(state, Work.start(10)))
    assert edits
    for edit in edits:
        assert oid not in {s.id for s in required(edit.partition).surfaces}
        svg = export_svg(edit.document)
        actual = render(svg, evidence.source_size)
        # The fringe changes; material support and the independent policy still hold.
        assert not np.array_equal(actual[..., 3], truth[..., 3])
        evaluation = policy.evaluate(svg)
        assert evaluation.valid
        assert evaluation.terms["alpha"] > 0
        assert evaluation.terms["opacity_missing_pixels"] == 0
        updated = LocalPolicy(policy).update(
            state.snapshot, svg, edit.bounds, evaluation.structure
        )
        assert updated.canvas.matches(actual)
        assert updated.evaluation.terms == pytest.approx(evaluation.terms, abs=2e-7)


def test_tiny_independently_owned_material_survives_joint_contour_fitting():
    evidence = material(128)
    labels, target, rgba = (
        evidence.labels.copy(),
        evidence.target.copy(),
        evidence.rgba.copy(),
    )
    labels[10, 10] = 9
    target[10, 10] = (250, 230, 10)
    rgba[10, 10, :3] = target[10, 10] / 255
    evidence = replace(
        evidence, labels=labels, target=target, smooth=target, coarse=target, rgba=rgba
    )
    frontier, state, options = prepared(evidence, layers=True)
    graph = build(evidence)
    assert state.partition is not None
    oid = state.partition.owners[9]
    original = state.document.geometry_for(oid)
    geometry = replace(parse_path("M10 10H11V11H10Z"), id=original.id)
    document = state.document.replace_geometry(geometry)
    svg = export_svg(document)
    baseline = frontier.policy.evaluate(svg)
    assert baseline.valid
    state = replace(
        state,
        document=document,
        svg=svg,
        snapshot=LocalPolicy(frontier.policy).start(svg, baseline),
    )
    edits = list(
        CoreCells(Families(evidence, graph, options), options, joint=True)(
            state, Work.start(10)
        )
    )
    assert edits
    faithful = []
    for edit in edits:
        assert edit.partition is not None
        Operators(evidence, graph, options).validate_partition(
            edit.partition, Work.start(10)
        )
        actual = render(export_svg(edit.document), evidence.source_size)
        if np.max(np.abs(actual[10, 10, :3] - rgba[10, 10, :3])) <= 1 / 255:
            faithful.append(edit)
    assert faithful
    assert all(
        frontier.policy.evaluate(export_svg(edit.document)).valid for edit in faithful
    )
