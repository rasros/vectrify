"""Coherent piecewise replacements need no accepted single-paint intermediate."""

from dataclasses import replace

import numpy as np
import pathops
import pytest

from tests.refine.test_cel_plan_families import prepared
from tests.refine.test_cel_plan_ownership import stripes
from vectrify.document import (
    Editor,
    Element,
    Geometry,
    PathNode,
    Selection,
    Subpath,
    export_svg,
    load_project,
    save_project,
)
from vectrify.document.hit_test import multiply
from vectrify.document.join import (
    curve_path,
    path_geometry,
    transformed_geometry,
    union_geometry,
)
from vectrify.document.redraw import root_matrix
from vectrify.document.topology import inverse_matrix
from vectrify.refine.cel_plan import piecewise_surfaces
from vectrify.refine.cel_plan.families import Families
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Work
from vectrify.refine.cel_plan.nested import enclosed
from vectrify.refine.cel_plan.ownership import Partition, Surface, exported
from vectrify.refine.cel_plan.piecewise_surfaces import PiecewiseSurfaces
from vectrify.refine.cel_plan.proposals import Operators
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.search import search


def step(alpha=255, hole=False, diagonal=False, gradient=False):
    evidence = stripes(alpha=alpha, gradient=True, hole=hole)
    y, x = np.indices(evidence.labels.shape)
    side = (x + 0.5 + y + 0.5) / np.sqrt(2) < 56 if diagonal else x < 43
    target = evidence.target.copy()
    target[(~evidence.empty) & side] = (32, 100, 60)
    target[(~evidence.empty) & ~side] = (180, 100, 60)
    if gradient:
        target[(~evidence.empty) & side, 0] += 0.8 * x[(~evidence.empty) & side]
        target[(~evidence.empty) & ~side, 0] += 0.6 * y[(~evidence.empty) & ~side]
    rgba = evidence.rgba.copy()
    rgba[~evidence.empty, :3] = target[~evidence.empty] / 255
    return replace(evidence, target=target, smooth=target, coarse=target, rgba=rgba)


def marked_step(alpha=255):
    evidence = step(alpha)
    labels, target, rgba = (
        evidence.labels.copy(),
        evidence.target.copy(),
        evidence.rgba.copy(),
    )
    labels[24:40, 32:56] = 9
    target[24:40, 32:56] = (250, 220, 30)
    rgba[24:40, 32:56, :3] = np.array((250, 220, 30)) / 255
    return replace(
        evidence, labels=labels, target=target, smooth=target, coarse=target, rgba=rgba
    )


@pytest.mark.parametrize("alpha", [255, 128])
def test_two_shade_surfaces_continue_beneath_one_unchanged_crossing_mark(alpha):
    evidence = marked_step(alpha)
    frontier, state, options = prepared(evidence, layers=alpha != 255)
    marker = state.partition.owners[9]
    factory = PiecewiseSurfaces(Families(evidence, build(evidence), options), options)
    edits = list(factory(state, Work.start(10)))
    continued = [p for p in edits if p.parameters[0].startswith("continued-")]
    assert continued
    edit = continued[0]
    assert edit.partition.follows(state.partition)
    assert edit.partition.owners[9] == marker
    assert edit.document.element(marker) == state.document.element(marker)
    assert edit.document.geometry_for(marker) == state.document.geometry_for(marker)
    shades = [s for s in edit.partition.surfaces if s.covered]
    assert len(shades) == 2
    assert all(s.covered == (9,) for s in shades)
    parent = edit.document.ancestry(marker)[-2]
    positions = {child.id: i for i, child in enumerate(parent.children)}
    assert all(positions[s.id] < positions[marker] for s in shades)
    svg = export_svg(edit.document)
    full = frontier.policy.evaluate(svg)
    assert full.valid
    assert full.cost < state.snapshot.evaluation.cost
    actual = render(svg, evidence.source_size)
    np.testing.assert_array_equal(
        actual[..., 3], render(state.svg, evidence.source_size)[..., 3]
    )
    np.testing.assert_array_equal(
        actual[28:36, 36:52, :3],
        render(state.svg, evidence.source_size)[28:36, 36:52, :3],
    )
    local = LocalPolicy(frontier.policy).update(
        state.snapshot, svg, edit.bounds, full.structure
    )
    assert local.canvas.matches(actual)
    assert local.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
    Operators(evidence, build(evidence), options).validate_partition(
        edit.partition, Work.start(10)
    )
    saved, _ = load_project(save_project(edit.document))
    Partition.from_metadata(edit.partition.metadata()).validate(saved)


def test_fragmented_source_atoms_do_not_limit_compact_whole_owner_family():
    evidence = step()
    frontier, state, options = prepared(evidence)
    y, x = np.indices(evidence.labels.shape)
    # 960 small source atoms, already owned by eight editable strips. The
    # drawing and source colors remain the same; only atomic lineage is finer.
    labels = np.where(evidence.empty, 0, 1 + ((y - 8) // 2) * 40 + (x - 8) // 2)
    fine = replace(evidence, labels=labels.astype(np.int32))
    metadata = exported(fine, evidence.labels, frozenset(range(1, 9)), Work.start(10))
    state = replace(state, partition=Partition.from_metadata(metadata))
    factory = PiecewiseSurfaces(Families(fine, build(fine), options), options)
    edits = list(factory(state, Work.start(10)))
    whole = [p for p in edits if p.details["piecewise_surface"]["removed_paths"] == 6]
    assert whole
    edit = whole[0]
    assert len(edit.details["piecewise_surface"]["source_members"]) == 960
    assert edit.partition.follows(state.partition)
    assert len(edit.partition.atoms.cuts) == 24
    assert frontier.policy.evaluate(export_svg(edit.document)).valid
    Operators(fine, build(fine), options).validate_partition(
        edit.partition, Work.start(10)
    )


def test_unsupported_source_namespace_and_roi_remain_bounded(monkeypatch):
    evidence = step()
    _, state, options = prepared(evidence)
    monkeypatch.setattr(piecewise_surfaces, "MAX_MEMBERS", 8)
    factory = PiecewiseSurfaces(Families(evidence, build(evidence), options), options)
    assert list(factory(state, Work.start(10))) == []
    assert factory.diagnostics["group_limits"] == 1


@pytest.mark.parametrize("kind", ["partial-owner", "transparent", "alpha-hole"])
def test_continuation_cannot_hide_partial_ownership_transparency_or_a_true_hole(kind):
    evidence = marked_step(128)
    if kind == "alpha-hole":
        labels, rgba, empty = (
            evidence.labels.copy(),
            evidence.rgba.copy(),
            evidence.empty.copy(),
        )
        labels[24:40, 32:56] = 0
        rgba[24:40, 32:56] = 0
        empty[24:40, 32:56] = True
        opacity = evidence.opacity.copy()
        opacity[24:40, 32:56] = 0
        evidence = replace(
            evidence, labels=labels, rgba=rgba, empty=empty, opacity=opacity
        )
    _, state, options = prepared(evidence, layers=True)
    if kind != "alpha-hole":
        marker = state.partition.owners[9]
        editor = Editor(state.document, selection=Selection(whole_document=True))
        if kind == "transparent":
            with editor.transaction("Partial current mark paint") as tx:
                tx.set_attributes(marker, {"fill-opacity": "0.5"})
        else:
            distant = state.partition.owners[1]
            shape = union_geometry(
                [state.document.geometry_for(i) for i in (marker, distant)],
                [{"fill-rule": "nonzero"}] * 2,
            )
            with editor.transaction("One owner also covers a distant fragment") as tx:
                tx.replace_geometry(marker, shape)
                tx.delete_objects(frozenset((distant,)))
            state = replace(
                state,
                partition=state.partition.replace(
                    (marker, distant), (Surface(marker, (1, 9)),)
                ),
            )
        state = replace(state, document=editor.snapshot.document)
    factory = PiecewiseSurfaces(Families(evidence, build(evidence), options), options)
    edits = list(factory(state, Work.start(10)))
    assert not any(p.parameters[0].startswith("continued-") for p in edits)
    reason = {
        "partial-owner": "partially-enclosed-owner",
        "transparent": "mark-style-or-frame",
        "alpha-hole": "alpha-hole",
    }[kind]
    if kind == "partial-owner":
        # The bounded seed prefix can exclude this mismatched owner earlier.
        # Exercise the enclosure proof independently on the surviving family.
        members = tuple(range(2, 9))
        box, own, _, _ = factory._samples(members, Work.start(10))
        ids = tuple(sorted({state.partition.owners[i] for i in members}))
        rejected = {}
        assert (
            enclosed(
                evidence,
                factory.families.graph,
                state,
                ids,
                box,
                own,
                ids[-1],
                Work.start(10),
                rejections=rejected,
            )
            is None
        )
        assert rejected[reason] == 1
        return
    assert factory.diagnostics["enclosure_exclusions"].get(reason, 0) > 0


def test_continuation_requires_actual_core_under_the_mark_as_well_as_material():
    evidence = marked_step(128)
    _, state, options = prepared(evidence, layers=True)
    marker = state.partition.owners[9]
    base = next(s for s in state.partition.surfaces if s.role == "underlay")
    # Its metadata still lists every member. Its actual geometry lacks the mark.
    inner = transformed_geometry(
        state.document.geometry_for(marker), (0.25, 0, 0, 0.25, 33, 24)
    )
    inner = transformed_geometry(
        inner,
        multiply(
            inverse_matrix(root_matrix(state.document, base.id)),
            root_matrix(state.document, marker),
        ),
    )
    cut = path_geometry(
        pathops.op(
            curve_path(state.document.geometry_for(base.id)),
            curve_path(inner),
            pathops.PathOp.DIFFERENCE,
        )
    )
    editor = Editor(state.document, selection=Selection(whole_document=True))
    with editor.transaction("Unproved underpaint below retained mark") as tx:
        tx.replace_geometry(base.id, cut)
    state = replace(state, document=editor.snapshot.document)
    factory = PiecewiseSurfaces(Families(evidence, build(evidence), options), options)
    edits = list(factory(state, Work.start(10)))
    assert edits
    assert not any(p.parameters[0].startswith("continued-") for p in edits)
    assert factory.diagnostics["core_exclusions"] > 0


@pytest.mark.parametrize("overlap", [False, True])
def test_continuing_both_sides_cannot_reorder_a_mark_across_unrelated_overlap(overlap):
    evidence = marked_step(128)
    _, state, options = prepared(evidence, layers=True)
    marker = state.partition.owners[9]
    parent = state.document.ancestry(marker)[-2]
    editor = Editor(state.document, selection=Selection(whole_document=True))
    # Move the existing mark below the family. Continuing both surfaces must
    # move it back above them, crossing the independent paint at index two.
    x = 42 if overlap else 100
    shape = Geometry(
        "crossed-paint-geometry",
        (
            Subpath(
                "crossed-paint-contour",
                tuple(
                    PathNode(str(i), "M" if i == 0 else "L", p)
                    for i, p in enumerate(((x, 30), (x + 2, 30), (x + 2, 34), (x, 34)))
                ),
                True,
            ),
        ),
    )
    with editor.transaction("Mark below intervening paint") as tx:
        tx.reorder_object(marker, 1)
        tx.insert_object(
            parent.id,
            Element(
                "crossed-paint", "path", (("fill", "#0040ff"),), geometry_id=shape.id
            ),
            index=2,
            geometries=(shape,),
        )
    state = replace(state, document=editor.snapshot.document)
    factory = PiecewiseSurfaces(Families(evidence, build(evidence), options), options)
    edits = list(factory(state, Work.start(10)))
    assert edits
    continued = [p for p in edits if p.parameters[0].startswith("continued-")]
    if overlap:
        assert not continued
        assert factory.diagnostics["order_exclusions"] > 0
    else:
        assert continued
    assert all(p.document.geometry_for("crossed-paint") == shape for p in edits)


@pytest.mark.parametrize("alpha", [255, 128])
@pytest.mark.parametrize("hole", [False, True])
@pytest.mark.parametrize("gradient", [False, True])
def test_joint_full_family_two_paints_exact_source_split_native_local_reload(
    alpha, hole, gradient
):
    evidence = step(alpha, hole, gradient=gradient)
    frontier, state, options = prepared(evidence, layers=alpha != 255)
    graph = build(evidence)
    families = Families(evidence, graph, options)
    # A single material fit cannot offer all eight distant shade fragments.
    assert not any(
        len(p.ids) == 8 for p in families.surface_models(state, Work.start(10))
    )
    edits = list(PiecewiseSurfaces(families, options)(state, Work.start(10)))
    whole = [p for p in edits if len(p.ids) == 9]
    assert whole
    edit = whole[0]
    assert edit.partition.follows(state.partition)
    assert len(edit.partition.atoms.cuts) == 1
    assert edit.details["piecewise_surface"]["removed_paths"] == 6
    Operators(evidence, graph, options).validate_partition(
        edit.partition, Work.start(10)
    )
    svg = export_svg(edit.document)
    full = frontier.policy.evaluate(svg)
    assert full.valid
    assert full.cost < state.snapshot.evaluation.cost
    assert len([s for s in edit.partition.surfaces if s.role == "surface"]) == 2
    actual = render(svg, evidence.source_size)
    np.testing.assert_array_equal(
        actual[..., 3], render(state.svg, evidence.source_size)[..., 3]
    )
    local = LocalPolicy(frontier.policy).update(
        state.snapshot, svg, edit.bounds, full.structure
    )
    assert local.canvas.matches(actual)
    assert local.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
    saved, _ = load_project(save_project(edit.document))
    Partition.from_metadata(edit.partition.metadata()).validate(saved)
    np.testing.assert_array_equal(
        render(export_svg(saved), evidence.source_size), actual
    )
    assert state.partition.atoms is None
    assert len([s for s in state.partition.surfaces if s.role == "surface"]) == 8


def test_native_search_publishes_compound_edit_and_child_graph_composes():
    evidence = step(128, hole=True)
    frontier, _, options = prepared(evidence, layers=True)
    factory = Operators(evidence, build(evidence), options)
    report = search(frontier, options, Work.start(10), factory)
    assert report["accepted"] > 0
    assert report["score_disagreements"] == 0
    selected = frontier.select(100)
    assert any(
        e["operator"] == "piecewise-surface" for e in selected.metrics["local_edits"]
    )
    partition = Partition.from_metadata(selected.metrics["planning_surfaces"])
    branch = factory.branch(partition, Work.start(10))
    assert branch.families.graph.source_atoms == partition.atoms.key
    assert branch.replacements.graph is branch.overlays.graph is branch.families.graph


@pytest.mark.parametrize("alpha", [255, 128])
def test_diagonal_line_and_independent_current_frames_are_preserved(alpha):
    evidence = step(alpha, diagonal=True)
    frontier, state, options = prepared(evidence, layers=alpha != 255)
    editor = Editor(state.document, selection=Selection(whole_document=True))
    for surface in state.partition.surfaces:
        if surface.role != "surface":
            continue
        with editor.transaction("Equivalent child frame") as tx:
            tx.set_attributes(surface.id, {"transform": "translate(5 -3)"})
            tx.replace_geometry(
                surface.id,
                transformed_geometry(
                    state.document.geometry_for(surface.id), (1, 0, 0, 1, -5, 3)
                ),
            )
    state = replace(state, document=editor.snapshot.document)
    edits = list(
        PiecewiseSurfaces(Families(evidence, build(evidence), options), options)(
            state, Work.start(10)
        )
    )
    assert edits
    for edit in edits:
        assert all(abs(v) > 0.5 for v in edit.parameters[1])
        assert frontier.policy.evaluate(export_svg(edit.document)).valid


def test_complete_owner_outlier_remains_editable_and_unchanged():
    evidence = step()
    target = evidence.target.copy()
    rgba = evidence.rgba.copy()
    target[32, 23] = (0, 0, 255)
    rgba[32, 23, :3] = (0, 0, 1)
    evidence = replace(evidence, target=target, smooth=target, coarse=target, rgba=rgba)
    _, state, options = prepared(evidence)
    factory = PiecewiseSurfaces(Families(evidence, build(evidence), options), options)
    edits = list(factory(state, Work.start(10)))
    assert edits
    oid = state.partition.owners[2]
    for edit in edits:
        assert oid not in edit.ids
        assert edit.partition.owners[2] == oid
        assert edit.document.geometry_for(oid) == state.document.geometry_for(oid)


@pytest.mark.parametrize("shift", [-30, 60])
def test_intervening_geometry_requires_order_proof(shift):
    evidence = step()
    _, state, options = prepared(evidence)
    editor = Editor(state.document, selection=Selection(whole_document=True))
    parent = state.document.ancestry(state.partition.owners[4])[-2]
    index = next(
        i for i, e in enumerate(parent.children) if e.id == state.partition.owners[4]
    )
    x = 40 + shift
    shape = Geometry(
        "stationary-geometry",
        (
            Subpath(
                "stationary-contour",
                (
                    PathNode("a", "M", (x, 16)),
                    PathNode("b", "L", (x + 8, 16)),
                    PathNode("c", "L", (x + 8, 20)),
                    PathNode("d", "L", (x, 20)),
                ),
                True,
            ),
        ),
    )
    with editor.transaction("Stationary intervening paint") as tx:
        tx.insert_object(
            parent.id,
            Element("stationary", "path", (("fill", "#0040ff"),), geometry_id=shape.id),
            index=index + 1,
            geometries=(shape,),
        )
    state = replace(state, document=editor.snapshot.document)
    factory = PiecewiseSurfaces(Families(evidence, build(evidence), options), options)
    edits = list(factory(state, Work.start(10)))
    if shift < 0:
        assert not edits
        assert factory.diagnostics["order_exclusions"] > 0
    else:
        assert edits
        for edit in edits:
            assert "stationary" not in edit.ids


def test_actual_rgba_core_and_limits_are_required(monkeypatch):
    evidence = step(128)
    _, state, options = prepared(evidence, layers=True)
    base = next(s for s in state.partition.surfaces if s.role == "underlay")
    editor = Editor(state.document, selection=Selection(whole_document=True))
    with editor.transaction("Remove core") as tx:
        tx.delete_objects(frozenset((base.id,)))
    changed = replace(
        state,
        document=editor.snapshot.document,
        partition=Partition(
            tuple(s for s in state.partition.surfaces if s.id != base.id)
        ),
    )
    factory = PiecewiseSurfaces(Families(evidence, build(evidence), options), options)
    assert list(factory(changed, Work.start(10))) == []
    assert factory.diagnostics["core_exclusions"] > 0
    monkeypatch.setattr(piecewise_surfaces, "MAX_PIXELS", 1)
    assert list(factory(state, Work.start(10))) == []
    assert factory.diagnostics["group_limits"] > 0


def test_compact_whole_contour_removes_jagged_fragments_with_actual_coverage():
    evidence = step(128)
    frontier, state, options = prepared(evidence, layers=True)
    # A subpixel ripple is evidence for a long straight run, within an existing
    # opaque underlayer. Give it many serialized points to expose compaction.
    points = [(8.0, 56.0), (8.0, 11.0)]
    points.extend((float(x), 11.0 if x % 2 == 0 else 10.0) for x in range(9, 89))
    points.extend(((88.0, 56.0),))
    wave = Geometry(
        "wave",
        (
            Subpath(
                "wave",
                tuple(
                    PathNode(f"n{i}", "M" if i == 0 else "L", p)
                    for i, p in enumerate(points)
                ),
                True,
            ),
        ),
    )
    editor = Editor(state.document, selection=Selection(whole_document=True))
    ids = tuple(s.id for s in state.partition.surfaces if s.role == "surface")
    for oid in ids:
        path = pathops.op(
            curve_path(state.document.geometry_for(oid)),
            curve_path(wave),
            pathops.PathOp.INTERSECTION,
        )
        with editor.transaction("Fragmented rippled surface") as tx:
            tx.replace_geometry(oid, path_geometry(path))
    state = replace(state, document=editor.snapshot.document)
    state = replace(state, svg=export_svg(state.document))
    initial = frontier.policy.evaluate(state.svg)
    assert initial.valid
    state = replace(
        state, snapshot=LocalPolicy(frontier.policy).start(state.svg, initial)
    )
    factory = PiecewiseSurfaces(Families(evidence, build(evidence), options), options)
    edits = list(factory(state, Work.start(10)))
    compact = [p for p in edits if p.parameters[0] == "contained-lines"]
    assert compact
    edit = compact[0]
    full = frontier.policy.evaluate(export_svg(edit.document))
    assert full.valid
    assert full.structure["nodes"] < initial.structure["nodes"] / 2
    endpoints = {
        n.endpoint
        for s in edit.partition.surfaces
        if s.role == "surface"
        for path in edit.document.geometry_for(s.id).subpaths
        for n in path.nodes
    }
    assert {(8.0, 11.0), (88.0, 11.0), (8.0, 56.0), (88.0, 56.0)}.issubset(endpoints)
    np.testing.assert_array_equal(
        render(export_svg(edit.document), evidence.source_size)[..., 3],
        render(state.svg, evidence.source_size)[..., 3],
    )
    # Complementary children are contained in the exact old family, preserving
    # paint order and original holes even when the contour model is smaller.
    old_union = union_geometry(
        [state.document.geometry_for(oid) for oid in ids],
        [{"fill-rule": "nonzero"} for _ in ids],
    )
    survivors = [s.id for s in edit.partition.surfaces if s.role == "surface"]
    for oid in survivors:
        outside = pathops.op(
            curve_path(edit.document.geometry_for(oid)),
            curve_path(old_union),
            pathops.PathOp.DIFFERENCE,
        )
        assert abs(outside.area) <= 1e-8


def test_stop_during_complete_screen_does_not_publish_partial_family(monkeypatch):
    evidence = step()
    _, state, options = prepared(evidence)
    work = Work.start(10)
    original = piecewise_surfaces.prediction_pair

    def stop(*args, **kwargs):
        result = original(*args, **kwargs)
        work.stop.set()
        return result

    monkeypatch.setattr(piecewise_surfaces, "prediction_pair", stop)
    factory = PiecewiseSurfaces(Families(evidence, build(evidence), options), options)
    assert list(factory(state, work)) == []
    assert state.partition.atoms is None
