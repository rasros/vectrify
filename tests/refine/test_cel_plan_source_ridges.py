"""Source ridges replace palette fragments with complete exact ownership."""

from dataclasses import replace

import numpy as np
import pathops
import pytest

from tests.refine.test_cel_plan_families import prepared
from tests.refine.test_cel_plan_ink_contours import ring
from vectrify.document import Editor, Selection, export_svg, load_project, save_project
from vectrify.document.join import curve_path, path_style, transformed_geometry
from vectrify.document.paint import GradientStop, LinearGradient
from vectrify.document.topology import inverse_matrix
from vectrify.refine.cel_plan import source_ridges
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.ink_replace import InkReplacement
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import StageInterruptedError, Work
from vectrify.refine.cel_plan.ownership import Partition
from vectrify.refine.cel_plan.proposals import Operators
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.search import search
from vectrify.refine.cel_plan.source_ridges import SourceRidges


def fragmented(*, alpha=128, mixed=True, gap=False, hole=False):
    evidence = ring(alpha=alpha, gap=gap, hole=hole)
    target, labels, drawn = (
        evidence.target.copy(),
        evidence.labels.copy(),
        evidence.drawn.copy(),
    )
    # Adjacent ink colors cannot form the legacy palette-distance family.
    for i in range(2, 14):
        target[labels == i] = (65, 8, 5) if i % 2 else (12, 65, 5)
    if mixed:
        # This independently owned mark shares an atom with one rim fragment.
        # Its complete source support and vector geometry must remain behind.
        labels[4:9, 83:88] = 2
        target[4:9, 83:88] = (12, 65, 5)
        drawn[4:9, 83:88] = True
    rgba = np.concatenate((target / 255, evidence.opacity[..., None]), axis=-1)
    return replace(
        evidence,
        labels=labels,
        target=target,
        smooth=target,
        coarse=target,
        rgba=rgba,
        drawn=drawn,
        line=drawn,
    )


@pytest.mark.parametrize("alpha", [255, 128, 64])
def test_complete_source_ridge_crosses_palette_and_keeps_mixed_owner_mark(alpha):
    evidence = fragmented(alpha=alpha)
    frontier, state, options = prepared(evidence, layers=True)
    graph = build(evidence)
    legacy = InkReplacement(evidence, graph, options)
    assert not [
        p
        for p in legacy.proposals(state, Work.start(10))
        if p.details["ink_replacement"].get("boundary_model") == "source-rim"
    ]
    factory = SourceRidges(evidence, graph, options)
    edits = list(factory.proposals(state, Work.start(20)))
    assert edits, (
        factory.diagnostics,
        factory.rim_diagnostics,
        factory.restoration_rejections,
    )
    edit = min(
        edits, key=lambda p: frontier.policy.evaluate(export_svg(p.document)).cost
    )
    assert edit.details["source_ridge"]["cuts"] >= 1
    assert edit.details["ink_replacement"]["boundary_models"] == ("ellipse", "ellipse")
    assert edit.partition.follows(state.partition)
    Operators(evidence, graph, options).validate_partition(
        edit.partition, Work.start(10)
    )
    assert set(edit.partition.owners) == set(
        edit.partition.atoms.descendants(state.partition.owners)
    )
    # Retired atoms cannot be reused as proof of majority ownership.
    labels = edit.partition.atoms.labels(graph, Work.start(10))
    mark_atoms = set(np.unique(labels[4:9, 83:88]))
    assert {edit.partition.owners[i] for i in mark_atoms} == {"cel-fill-2"}
    assert all(
        not set(s.members).intersection(mark_atoms)
        for s in edit.partition.surfaces
        if s.role == "overlay"
    )
    actual = render(export_svg(edit.document), evidence.source_size)
    before = render(state.svg, evidence.source_size)
    np.testing.assert_array_equal(actual[..., 3], before[..., 3])
    np.testing.assert_array_equal(actual[3:10, 82:89], before[3:10, 82:89])
    full = frontier.policy.evaluate(export_svg(edit.document))
    assert full.valid
    assert full.cost < state.snapshot.evaluation.cost
    assert full.structure["nodes"] < state.snapshot.evaluation.structure["nodes"]
    local = LocalPolicy(frontier.policy).update(
        state.snapshot, export_svg(edit.document), edit.bounds, full.structure
    )
    assert local.canvas.matches(actual)
    assert local.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
    assert frontier.checkpoint(
        export_svg(edit.document),
        "Source ridge",
        edit.details,
        local.evaluation,
        raster=local.canvas,
    )
    restored, _ = load_project(save_project(edit.document))
    np.testing.assert_array_equal(
        render(export_svg(restored), evidence.source_size), actual
    )
    Partition.from_metadata(edit.partition.metadata()).validate(restored)
    assert state.partition == Partition.from_metadata(
        state.details["planning_surfaces"]
    )
    np.testing.assert_array_equal(graph.labels, evidence.labels)


def test_unpublished_owner_cut_is_exact_and_complete_before_rim_fitting():
    evidence = fragmented()
    _frontier, state, options = prepared(evidence, layers=True)
    graph = build(evidence)
    factory = SourceRidges(evidence, graph, options)
    box, own, _radius, _signature = max(
        factory.bands(Work.start(10)), key=lambda b: b[1].sum()
    )
    current, _evidence, _graph, ink_ids, _members, old_ids = factory.prepare(
        state, box, own, Work.start(10)
    )
    assert "cel-fill-2" in old_ids
    remainder = curve_path(current.document.geometry_for("cel-fill-2"))
    selected = curve_path(
        current.document.geometry_for(
            next(i for i in ink_ids if i.startswith("cel-fill-2-ridge-"))
        )
    )
    original = curve_path(
        state.document.geometry_for("cel-fill-2"),
        path_style(state.document, state.document.element("cel-fill-2"))["fill-rule"],
    )
    assert abs(pathops.op(remainder, selected, pathops.PathOp.INTERSECTION).area) < 1e-8
    assert (
        abs(
            pathops.op(
                pathops.op(remainder, selected, pathops.PathOp.UNION),
                original,
                pathops.PathOp.XOR,
            ).area
        )
        < 1e-8
    )
    labels = current.partition.atoms.labels(graph, Work.start(10))
    selected_mask = np.isin(labels, _members)
    classification = np.zeros(graph.labels.shape, bool)
    classification[box.slices] = own
    np.testing.assert_array_equal(selected_mask, classification)
    assert current.partition.follows(state.partition)


@pytest.mark.parametrize(
    "kind", ["gap", "alpha-hole", "no-core", "explicit-width", "source-width"]
)
def test_unproved_ridge_never_replaces_seed(kind):
    evidence = fragmented(gap=kind == "gap", hole=kind == "alpha-hole")
    if kind == "source-width":
        evidence = replace(evidence, filled_line_width=2)
    _frontier, state, options = prepared(evidence, layers=kind != "no-core")
    if kind == "explicit-width":
        options = replace(options, line_width=2)
    factory = SourceRidges(evidence, build(evidence), options)
    assert not list(factory.proposals(state, Work.start(20)))
    assert state.partition == Partition.from_metadata(
        state.details["planning_surfaces"]
    )


@pytest.mark.parametrize(
    "limit", ["MAX_PATHS", "MAX_NODES", "MAX_CAVITY_AREA", "MAX_BANDS"]
)
def test_bounded_discovery_keeps_original_ownership(limit, monkeypatch):
    evidence = fragmented()
    _frontier, state, options = prepared(evidence, layers=True)
    monkeypatch.setattr(source_ridges, limit, 0)
    factory = SourceRidges(evidence, build(evidence), options)
    assert not list(factory.proposals(state, Work.start(10)))
    assert state.partition == Partition.from_metadata(
        state.details["planning_surfaces"]
    )


def test_stop_during_atom_cut_never_emits_partial_drawing(monkeypatch):
    evidence = fragmented()
    _frontier, state, options = prepared(evidence, layers=True)

    def interrupted(*_args, **_kwargs):
        raise StageInterruptedError("Stopped during ridge cut")

    monkeypatch.setattr(source_ridges.Atoms, "split", interrupted)
    factory = SourceRidges(evidence, build(evidence), options)
    assert not list(factory.proposals(state, Work.start(10)))
    assert state.partition == Partition.from_metadata(
        state.details["planning_surfaces"]
    )


def test_reflected_sheared_frame_preserves_native_source_cut():
    evidence = fragmented()
    frontier, state, options = prepared(evidence, layers=True)
    editor = Editor(state.document, selection=Selection(whole_document=True))
    matrix = (-1, 0.15, 0.2, 1, 100, -10)
    paths = [e.id for e in state.document.elements() if e.tag == "path"]
    with editor.transaction("Reexpress local frame") as tx:
        for oid in paths:
            tx.replace_geometry(
                oid,
                transformed_geometry(
                    state.document.geometry_for(oid), inverse_matrix(matrix)
                ),
            )
            tx.set_attributes(oid, {"transform": "matrix(-1 .15 .2 1 100 -10)"})
    state = replace(state, document=editor.snapshot.document)
    factory = SourceRidges(evidence, build(evidence), options)
    edits = list(factory.proposals(state, Work.start(20)))
    assert edits, (factory.diagnostics, factory.restoration_rejections)
    valid = []
    for edit in edits:
        actual = render(export_svg(edit.document), evidence.source_size)
        np.testing.assert_array_equal(
            actual[..., 3], render(state.svg, evidence.source_size)[..., 3]
        )
        evaluation = frontier.policy.evaluate(export_svg(edit.document))
        if evaluation.valid:
            valid.append(evaluation)
    assert valid
    assert min(e.cost for e in valid) < state.snapshot.evaluation.cost


def test_operator_search_publishes_source_split_after_independent_checkpoint():
    evidence = fragmented()
    frontier, state, options = prepared(evidence, layers=True)
    ops = Operators(evidence, build(evidence), options)

    class OnlyRidges:
        def __call__(self, current, work):
            yield from SourceRidges(
                evidence, ops.branch(current.partition, work).graph, options
            )(current, work)

        validate_partition = ops.validate_partition

    report = search(frontier, options, Work.start(15), OnlyRidges())
    assert report["accepted"] > 0
    assert report["score_disagreements"] == 0
    selected = frontier.select(50)
    assert selected.metrics["nodes"] < state.snapshot.evaluation.structure["nodes"]
    assert any(e["operator"] == "source-ridge" for e in selected.metrics["local_edits"])
    partition = Partition.from_metadata(selected.metrics["planning_surfaces"])
    ops.validate_partition(partition, Work.start(10))


@pytest.mark.parametrize("protection", ["locked", "pinned"])
@pytest.mark.parametrize("oid", ["cel-fill-2", "cel-fill-0", "cel-fill-1"])
def test_protected_existing_ink_or_neighbor_owner_is_not_cut(protection, oid):
    evidence = fragmented()
    _frontier, state, options = prepared(evidence, layers=True)
    document = state.document
    if protection == "locked":
        document = document.replace_element(
            replace(document.element(oid), locks=frozenset({"geometry"}))
        )
    else:
        geometry = document.geometry_for(oid)
        contour = geometry.subpaths[0]
        geometry = replace(
            geometry,
            subpaths=(
                replace(
                    contour,
                    nodes=(replace(contour.nodes[0], pinned=True), *contour.nodes[1:]),
                ),
                *geometry.subpaths[1:],
            ),
        )
        document = document.replace_geometry(geometry)
    state = replace(state, document=document)
    factory = SourceRidges(evidence, build(evidence), options)
    edits = list(factory.proposals(state, Work.start(10)))
    for edit in edits:
        assert edit.document.element(oid) == document.element(oid)
        assert edit.document.geometry_for(oid) == document.geometry_for(oid)
    if oid == "cel-fill-2":
        assert not edits
        assert factory.diagnostics["owner_exclusions"] > 0
    else:
        assert factory.restoration_rejections["protected-neighbor"] > 0


def test_source_ridge_preserves_an_existing_atom_namespace():
    evidence = fragmented()
    _frontier, state, options = prepared(evidence, layers=True)
    graph = build(evidence)
    factory = SourceRidges(evidence, graph, options)
    box, own, _radius, _sig = max(
        factory.bands(Work.start(10)), key=lambda b: b[1].sum()
    )
    current, branch_evidence, branch_graph, _ids, _members, _old = factory.prepare(
        state, box, own, Work.start(10)
    )
    # The second, narrower interpretation cuts existing active child atoms.
    smaller = min(factory.bands(Work.start(10)), key=lambda b: b[1].sum())
    repeated = SourceRidges(branch_evidence, branch_graph, options).prepare(
        current, smaller[0], smaller[1], Work.start(10)
    )
    assert repeated is not None
    next_state, _evidence, next_graph, _ids, _members, _old = repeated
    assert next_state.partition.atoms.extends(current.partition.atoms)
    assert next_state.partition.follows(current.partition)
    ops = Operators(evidence, graph, options)
    ops.validate_partition(next_state.partition, Work.start(10))
    np.testing.assert_array_equal(
        ops.branch(next_state.partition, Work.start(10)).graph.labels, next_graph.labels
    )


def test_cavity_count_limit_precedes_object_inventory(monkeypatch):
    evidence = fragmented()
    factory = SourceRidges(
        evidence, build(evidence), prepared(evidence, layers=True)[2]
    )
    monkeypatch.setattr(source_ridges, "MAX_CAVITY_COMPONENTS", 0)

    def inventory(*_args, **_kwargs):
        pytest.fail("Unbounded cavity inventory must not be allocated")

    monkeypatch.setattr(source_ridges, "find_objects", inventory)
    assert not list(factory.bands(Work.start(10)))
    assert factory.diagnostics["cavity_component_limits"] == 1


@pytest.mark.parametrize("invisible", ["selected", "retained"])
def test_exact_source_cut_can_retain_a_zero_area_ledger_side(invisible):
    evidence = fragmented()
    _frontier, state, options = prepared(evidence, layers=True)
    original = state.document.geometry_for("cel-fill-2")
    dot = [
        s
        for s in original.subpaths
        if curve_path(replace(original, subpaths=(s,))).bounds[3] < 12
    ]
    rim = [s for s in original.subpaths if s not in dot]
    assert dot
    assert rim
    document = state.document.replace_geometry(
        replace(original, subpaths=tuple(dot if invisible == "selected" else rim))
    )
    state = replace(state, document=document, svg=export_svg(document))
    graph = build(evidence)
    bounded = SourceRidges(evidence, graph, options)
    box, own, _radius, _sig = max(
        bounded.bands(Work.start(10)), key=lambda b: b[1].sum()
    )
    assert bounded.prepare(state, box, own, Work.start(10)) is None
    ops = Operators(evidence, graph, options)
    exact = SourceRidges(
        evidence, graph, options, resolver=ops.branch, allow_invisible=True
    )
    result = exact.prepare(state, box, own, Work.start(10))
    assert result is not None
    assert result.state.partition is not None
    assert exact.diagnostics["invisible_fragments"] == 1
    assert result.state.partition.follows(state.partition)
    ops.validate_partition(result.state.partition, Work.start(10))
    before = render(export_svg(document), evidence.source_size)
    np.testing.assert_array_equal(
        render(export_svg(result.state.document), evidence.source_size), before
    )
    restored, _ = load_project(save_project(result.state.document))
    np.testing.assert_array_equal(
        render(export_svg(restored), evidence.source_size), before
    )


def test_source_discovery_gets_live_slice_without_removing_existing_operators(
    monkeypatch,
):
    evidence = fragmented()
    _frontier, state, options = prepared(evidence, layers=True)
    ops = Operators(evidence, build(evidence), options)
    names = (
        "families",
        "overlays",
        "paint",
        "geometry",
        "ink",
        "replacements",
        "ridges",
    )
    for name in names:

        def cursor(_current, work, name=name):
            assert work.remaining > 0
            yield replace(state, key=name)

        monkeypatch.setattr(ops, name, cursor)
    iterator = ops(state, Work.start(10))
    first_cycle = [next(iterator).key for _ in names]
    iterator.close()
    assert first_cycle[0] == "families"
    assert sorted(first_cycle) == sorted(names)


def test_partial_owner_keeps_private_gradient_and_native_mark_pixels():
    evidence = fragmented()
    frontier, state, options = prepared(evidence, layers=True)
    editor = Editor(state.document, selection=Selection(whole_document=True))
    gradient = LinearGradient(
        (0, 0), (96, 64), (GradientStop(0, "#0c4105"), GradientStop(1, "#195214"))
    )
    with editor.transaction("Set source owner paint") as tx:
        tx.set_fill("cel-fill-2", gradient)
    document = editor.snapshot.document
    svg = export_svg(document)
    before = render(svg, evidence.source_size)
    state = replace(
        state,
        document=document,
        svg=svg,
        snapshot=LocalPolicy(frontier.policy).start(svg, frontier.policy.evaluate(svg)),
    )
    edits = list(
        SourceRidges(evidence, build(evidence), options).proposals(
            state, Work.start(20)
        )
    )
    assert edits
    server = document.element("cel-fill-2").get("fill")
    for edit in edits:
        assert edit.document.element("cel-fill-2").get("fill") == server
        assert [
            e for e in edit.document.elements() if e.paint_owner == "cel-fill-2"
        ] == [e for e in document.elements() if e.paint_owner == "cel-fill-2"]
        np.testing.assert_array_equal(
            render(export_svg(edit.document), evidence.source_size)[3:10, 82:89],
            before[3:10, 82:89],
        )


def test_native_offset_and_scale_preserve_complete_source_atom_cut():
    evidence = fragmented()
    native = np.zeros((128, 160, 4), np.float32)
    native[30:62, 20:68] = evidence.rgba[::2, ::2]
    evidence = replace(
        evidence, rgba=native, source_size=(160, 128), offset=(20, 30), scale=(2, 2)
    )
    frontier, state, options = prepared(evidence, layers=True)
    graph = build(evidence)
    edits = list(
        SourceRidges(evidence, graph, options).proposals(state, Work.start(20))
    )
    assert edits
    valid = [
        edit
        for edit in edits
        if frontier.policy.evaluate(export_svg(edit.document)).valid
    ]
    assert valid
    for edit in valid:
        Operators(evidence, graph, options).validate_partition(
            edit.partition, Work.start(10)
        )
        svg = export_svg(edit.document)
        actual = render(svg, evidence.source_size)
        np.testing.assert_array_equal(
            actual[..., 3], render(state.svg, evidence.source_size)[..., 3]
        )
        full = frontier.policy.evaluate(svg)
        local = LocalPolicy(frontier.policy).update(
            state.snapshot, svg, edit.bounds, full.structure
        )
        assert local.canvas.matches(actual)
        assert local.evaluation.terms == pytest.approx(full.terms, abs=2e-7)


def test_registered_source_cut_reuses_accounted_graph_in_native_validation():
    evidence = fragmented()
    _frontier, state, options = prepared(evidence, layers=True)
    ops = Operators(evidence, build(evidence), options)
    box, own, _radius, _sig = max(
        ops.ridges.bands(Work.start(10)), key=lambda b: b[1].sum()
    )
    cut = ops.ridges.prepare(state, box, own, Work.start(10))
    assert cut is not None
    assert cut.branch is ops.branch(cut.state.partition, Work.start(10))
    assert cut.graph is cut.branch.graph
    ops.validate_partition(cut.state.partition, Work.start(10))
    assert ops.schedule_diagnostics["source_graph_rebuilds"] == 1
    smaller = min(ops.ridges.bands(Work.start(10)), key=lambda b: b[1].sum())
    refined = cut.branch.ridges.prepare(
        cut.state, smaller[0], smaller[1], Work.start(10)
    )
    assert refined is not None
    assert refined.branch is ops.branch(refined.state.partition, Work.start(10))
    ops.validate_partition(refined.state.partition, Work.start(10))
    assert ops.schedule_diagnostics["source_graph_rebuilds"] == 2


def test_complete_atom_reassignment_does_not_allocate_namespace_or_graph_copy():
    evidence = fragmented(mixed=False)
    _frontier, state, options = prepared(evidence, layers=True)
    ops = Operators(evidence, build(evidence), options)
    box, own, _radius, _sig = max(
        ops.ridges.bands(Work.start(10)), key=lambda b: b[1].sum()
    )
    cut = ops.ridges.prepare(state, box, own, Work.start(10))
    assert cut is not None
    assert cut.state.partition.atoms is None
    assert cut.graph is ops.graph
    ops.validate_partition(cut.state.partition, Work.start(10))
    assert ops.schedule_diagnostics["source_graph_rebuilds"] == 0


def test_cursor_close_preserves_emitted_ridge_diagnostics():
    evidence = fragmented()
    _frontier, state, options = prepared(evidence, layers=True)
    factory = SourceRidges(evidence, build(evidence), options)
    cursor = factory.proposals(state, Work.start(10))
    next(cursor)
    cursor.close()
    assert factory.diagnostics["proposals"] == 1
    assert factory.rim_diagnostics["rim_proposals"] == 1
    assert factory.rim_diagnostics["rim_compact_underpaints"] == 1
