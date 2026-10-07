"""Source material cells replace fragments without changing coverage or marks."""

from dataclasses import replace

import numpy as np
import pytest

from tests.refine.test_cel_plan_families import prepared
from tests.refine.test_cel_plan_piecewise_surfaces import marked_step, step
from vectrify.document import (
    Editor,
    Element,
    Selection,
    export_svg,
    load_project,
    save_project,
)
from vectrify.document.svg import parse_path
from vectrify.refine.cel_plan import atoms as atom_module
from vectrify.refine.cel_plan.atoms import Atoms
from vectrify.refine.cel_plan.core_cells import CoreCells
from vectrify.refine.cel_plan.families import Families
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Work
from vectrify.refine.cel_plan.nested import in_core
from vectrify.refine.cel_plan.ownership import Partition
from vectrify.refine.cel_plan.proposals import Operators
from vectrify.refine.cel_plan.score import render


def material(alpha=128, hole=False):
    evidence = step(alpha, hole=hole)
    target, rgba = evidence.target.copy(), evidence.rgba.copy()
    _y, x = np.indices(evidence.labels.shape)
    target[(x >= 70) & ~evidence.empty] = (60, 100, 200)
    target[(x >= 35) & (x < 70) & ~evidence.empty] = (180, 100, 60)
    rgba[~evidence.empty, :3] = target[~evidence.empty] / 255
    return replace(
        evidence,
        target=target,
        smooth=target,
        coarse=target,
        rgba=rgba,
        opacity=rgba[..., 3],
    )


def test_multiway_atoms_replay_exact_complete_support_in_one_namespace():
    evidence = material()
    graph = build(evidence)
    original = Atoms.original(graph)
    y, x = np.indices(graph.labels.shape)
    classes = ((x // 2 + y // 6) % 4).astype(np.uint8)
    refined, groups = original.partition(graph, range(1, 9), classes, 4, Work.start(10))
    assert len(refined.cuts) == 24
    replay = refined.labels(graph, Work.start(10))
    for index, group in enumerate(groups):
        np.testing.assert_array_equal(
            np.isin(replay, group), (~evidence.empty) & (classes == index)
        )
    assert set(refined.descendants(range(1, 9))) == {
        i for group in groups for i in group
    }
    assert original.cuts == ()
    assert not replay.flags.writeable


def test_multiway_split_bounds_fail_atomically(monkeypatch):
    graph = build(material())
    original = Atoms.original(graph)
    y, x = np.indices(graph.labels.shape)
    classes = ((x // 2 + y // 6) % 4).astype(np.uint8)
    monkeypatch.setattr(atom_module, "MAX_CUTS", 2)
    with pytest.raises(ValueError, match="bounds"):
        original.partition(graph, range(1, 9), classes, 4, Work.start(10))
    assert original.cuts == ()


def test_multiway_split_cannot_cut_protected_source_atom():
    graph = build(material())
    graph = replace(
        graph,
        regions=tuple(
            replace(r, fixed=True) if r.id == 1 else r for r in graph.regions
        ),
    )
    classes = np.broadcast_to(
        np.arange(graph.labels.shape[1]) % 3, graph.labels.shape
    ).astype(np.uint8)
    with pytest.raises(ValueError, match="Protected"):
        Atoms.original(graph).partition(graph, range(1, 9), classes, 3, Work.start(10))


@pytest.mark.parametrize("alpha", [255, 128])
@pytest.mark.parametrize("hole", [False, True])
def test_whole_core_materials_remove_fragments_with_exact_alpha_ownership_reload(
    alpha, hole
):
    evidence = material(alpha, hole)
    frontier, state, options = prepared(evidence, layers=True)
    graph = build(evidence)
    factory = CoreCells(Families(evidence, graph, options), options)
    edits = list(factory(state, Work.start(10)))
    assert edits
    edit = max(
        (
            p
            for p in edits
            if p.details["core_material_cells"]["region_threshold"] is None
        ),
        key=lambda p: p.parameters[0],
    )
    assert edit.parameters[0] >= 2
    assert edit.details["core_material_cells"]["removed_paths"] >= 5
    assert edit.partition.follows(state.partition)
    Operators(evidence, graph, options).validate_partition(
        edit.partition, Work.start(10)
    )
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
    assert full.cost < state.snapshot.evaluation.cost
    actual = render(svg, evidence.source_size)
    np.testing.assert_array_equal(
        actual[..., 3], render(state.svg, evidence.source_size)[..., 3]
    )
    if hole:
        assert actual[24:40, 40:56, 3].max() == 0
    local = LocalPolicy(frontier.policy).update(
        state.snapshot, svg, edit.bounds, full.structure
    )
    assert local.canvas.matches(actual)
    assert local.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
    reloaded, _ = load_project(save_project(edit.document))
    Partition.from_metadata(edit.partition.metadata()).validate(reloaded)
    np.testing.assert_array_equal(
        render(export_svg(reloaded), evidence.source_size), actual
    )
    assert state.partition.atoms is None


@pytest.mark.parametrize("alpha", [255, 128])
def test_owned_mark_keeps_its_geometry_paint_and_primary_owner(alpha):
    evidence = marked_step(alpha)
    evidence = replace(evidence, opacity=evidence.rgba[..., 3])
    frontier, state, options = prepared(evidence, layers=True)
    marker = state.partition.owners[9]
    state = replace(
        state,
        details={
            **state.details,
            "paint_constraints": [*state.details.get("paint_constraints", ()), marker],
        },
    )
    factory = CoreCells(Families(evidence, build(evidence), options), options)
    edits = list(factory(state, Work.start(10)))
    assert edits
    for edit in edits:
        assert edit.partition.owners[9] == marker
        assert edit.document.element(marker) == state.document.element(marker)
        assert edit.document.geometry_for(marker) == state.document.geometry_for(marker)
        actual = render(export_svg(edit.document), evidence.source_size)
        np.testing.assert_array_equal(
            actual[28:36, 36:52], render(state.svg, evidence.source_size)[28:36, 36:52]
        )
        assert frontier.policy.evaluate(export_svg(edit.document)).valid


@pytest.mark.parametrize("kind", ["partial-core", "unknown-object"])
def test_unsupported_current_compositing_does_not_emit_a_global_replacement(kind):
    evidence = marked_step(128)
    _, state, options = prepared(evidence, layers=True)
    base = next(s for s in state.partition.surfaces if s.role == "underlay")
    marker = state.partition.owners[9]
    state = replace(state, details={**state.details, "paint_constraints": [marker]})
    editor = Editor(state.document, selection=Selection(whole_document=True))
    with editor.transaction("Unsupported compositing") as tx:
        if kind == "unknown-object":
            shape = parse_path("M30 12H40V20H30Z")
            tx.insert_object(
                state.document.ancestry(base.id)[-2].id,
                Element(
                    "unknown", "path", (("fill", "#ff00ff"),), geometry_id=shape.id
                ),
                geometries=(shape,),
            )
        else:
            tx.set_attributes(
                base.id if kind == "partial-core" else marker, {"fill-opacity": "0.5"}
            )
    changed = replace(state, document=editor.snapshot.document)
    factory = CoreCells(Families(evidence, build(evidence), options), options)
    assert list(factory(changed, Work.start(10))) == []


def test_round_coverage_contour_has_a_proved_native_opaque_interior():
    evidence = material()
    _, state, options = prepared(evidence, layers=True)
    base = next(s for s in state.partition.surfaces if s.role == "underlay")
    shape = parse_path("M80 32C80 50 60 58 42 50C10 55 8 20 30 10C50 0 80 8 80 32Z")
    editor = Editor(state.document, selection=Selection(whole_document=True))
    with editor.transaction("Curved actual core") as tx:
        tx.replace_geometry(base.id, shape)
    changed = replace(state, document=editor.snapshot.document)
    factory = CoreCells(Families(evidence, build(evidence), options), options)
    interior, box, opaque = factory._interior(changed, base, shape, Work.start(10))
    from vectrify.document.redraw import root_matrix

    coverage = factory._mask(interior, root_matrix(changed.document, base.id), box)
    assert coverage.max() == 1
    assert not ((coverage > 0) & (opaque != 1)).any()


def test_cancelled_component_model_keeps_original_state():
    evidence = material()
    _, state, options = prepared(evidence, layers=True)
    work = Work.start(10)
    work.stop.set()
    assert (
        list(
            CoreCells(Families(evidence, build(evidence), options), options)(
                state, work
            )
        )
        == []
    )
    assert state.partition.atoms is None


def test_coarse_cel_darkness_does_not_freeze_a_monotone_material_step():
    evidence = step(128)
    evidence = replace(evidence, drawn=~evidence.empty)
    _, state, options = prepared(evidence, layers=True)
    factory = CoreCells(Families(evidence, build(evidence), options), options)
    edits = list(factory(state, Work.start(10)))
    assert edits
    assert factory.diagnostics["ridge_pixels"] == 0
    assert factory.diagnostics["selected_paths"] == 8


def test_paired_source_trough_retains_its_owned_ink_without_a_manual_hold():
    evidence = step(128)
    labels, rgb, rgba = (
        evidence.labels.copy(),
        evidence.target.copy(),
        evidence.rgba.copy(),
    )
    labels[12:52, 42:44] = 9
    rgb[12:52, 42:44] = (5, 5, 5)
    rgba[12:52, 42:44, :3] = 5 / 255
    drawn = np.zeros(evidence.empty.shape, bool)
    drawn[12:52, 42:44] = True
    evidence = replace(
        evidence,
        labels=labels,
        target=rgb,
        smooth=rgb,
        coarse=rgb,
        rgba=rgba,
        drawn=drawn,
    )
    _, state, options = prepared(evidence, layers=True)
    marker = state.partition.owners[9]
    factory = CoreCells(Families(evidence, build(evidence), options), options)
    edits = list(factory(state, Work.start(10)))
    assert edits
    assert factory.diagnostics["ridge_owners"] >= 1
    for edit in edits:
        assert edit.partition.owners[9] == marker
        assert edit.document.element(marker) == state.document.element(marker)
        assert edit.document.geometry_for(marker) == state.document.geometry_for(marker)


@pytest.mark.parametrize("alpha", [255, 128])
def test_connected_materials_use_compact_source_contours_and_complete_uncut_atoms(
    alpha,
):
    evidence = material(alpha, hole=True)
    target = evidence.target.copy()
    _y, x = np.indices(evidence.empty.shape)
    target[(x < 38) & ~evidence.empty] = (32, 100, 60)
    target[(x >= 38) & (x < 68) & ~evidence.empty] = (180, 100, 60)
    target[(x >= 68) & ~evidence.empty] = (60, 100, 200)
    rgba = evidence.rgba.copy()
    rgba[~evidence.empty, :3] = target[~evidence.empty] / 255
    evidence = replace(evidence, target=target, smooth=target, coarse=target, rgba=rgba)
    frontier, state, options = prepared(evidence, layers=True)
    graph = build(evidence)
    factory = CoreCells(Families(evidence, graph, options), options)
    region_edits = [
        p
        for p in factory(state, Work.start(10))
        if p.details["core_material_cells"]["region_threshold"] is not None
    ]
    assert region_edits
    for edit in region_edits:
        assert edit.parameters[0] == 3
        assert edit.partition.atoms.cuts == ()
        assert edit.details["core_material_cells"]["removed_paths"] == 6
        Operators(evidence, graph, options).validate_partition(
            edit.partition, Work.start(10)
        )
        full = frontier.policy.evaluate(export_svg(edit.document))
        assert full.valid
        assert full.cost < state.snapshot.evaluation.cost
        actual = render(export_svg(edit.document), evidence.source_size)
        np.testing.assert_array_equal(
            actual[..., 3], render(state.svg, evidence.source_size)[..., 3]
        )
        assert actual[24:40, 40:56, 3].max() == 0


def test_source_alpha_contrast_does_not_prove_an_intrinsic_ink_trough():
    evidence = step(128)
    _, state, options = prepared(evidence, layers=True)
    rgb, opacity = evidence.target.copy(), evidence.opacity.copy()
    rgb[12:52, 42:44] = (5, 5, 5)
    opacity[12:52, :42] = 0.1
    opacity[12:52, 44:] = 0.1
    drawn = np.zeros(evidence.empty.shape, bool)
    drawn[12:52, 42:44] = True
    evidence = replace(evidence, target=rgb, opacity=opacity, drawn=drawn)
    factory = CoreCells(Families(evidence, build(evidence), options), options)
    assert factory._ridge_owners(state, Work.start(10)) == set()


@pytest.mark.parametrize("stop_opacity", [1, 0.5])
def test_actual_gradient_core_requires_opaque_stops_inside_the_partial_group(
    stop_opacity,
):
    from vectrify.document.paint import GradientStop, LinearGradient

    evidence = material(128)
    _, state, options = prepared(evidence, layers=True)
    base = next(s for s in state.partition.surfaces if s.role == "underlay")
    surface = next(s for s in state.partition.surfaces if s.role == "surface")
    editor = Editor(state.document, selection=Selection(whole_document=True))
    with editor.transaction("Gradient coverage core") as tx:
        tx.set_fill(
            base.id,
            LinearGradient(
                (0, 0),
                (80, 48),
                (
                    GradientStop(0, "#20643c", 1),
                    GradientStop(1, "#b4643c", stop_opacity),
                ),
            ),
        )
    state = replace(state, document=editor.snapshot.document)
    assert in_core(
        state,
        surface.members,
        surface.id,
        state.document.geometry_for(surface.id),
        Work.start(10),
    ) == (stop_opacity == 1)
    if stop_opacity != 1:
        assert (
            list(
                CoreCells(Families(evidence, build(evidence), options), options)(
                    state, Work.start(10)
                )
            )
            == []
        )


def test_exact_three_plane_material_prefix_is_not_skipped():
    evidence = material(128)
    _, state, options = prepared(evidence, layers=True)
    factory = CoreCells(Families(evidence, build(evidence), options), options)
    planes = [p for p in factory(state, Work.start(10)) if p.parameters[2] is None]
    assert [p.parameters[0] for p in planes] == [3]
    assert planes[0].details["core_material_cells"]["source_squared_error"] == 0
    assert len(planes[0].partition.atoms.cuts) > 0


@pytest.mark.parametrize("retained", [True, False])
def test_partial_paint_inside_actual_opaque_core_preserves_complete_native_alpha(
    retained,
):
    evidence = marked_step(128) if retained else material(128)
    frontier, state, options = prepared(evidence, layers=True)
    marker = state.partition.owners[9 if retained else 1]
    if retained:
        state = replace(state, details={**state.details, "paint_constraints": [marker]})
    editor = Editor(state.document, selection=Selection(whole_document=True))
    with editor.transaction("Partial child in opaque core") as tx:
        tx.set_attributes(marker, {"fill-opacity": "0.5"})
    state = replace(state, document=editor.snapshot.document)
    factory = CoreCells(Families(evidence, build(evidence), options), options)
    edits = list(factory(state, Work.start(10)))
    assert edits
    original = render(export_svg(state.document), evidence.source_size)
    for edit in edits:
        if retained:
            assert edit.document.element(marker) == state.document.element(marker)
            assert edit.document.geometry_for(marker) == state.document.geometry_for(
                marker
            )
        actual = render(export_svg(edit.document), evidence.source_size)
        np.testing.assert_array_equal(actual[..., 3], original[..., 3])
        assert frontier.policy.evaluate(export_svg(edit.document)).valid


def test_native_coverage_proof_keeps_old_paint_on_the_antialiased_core_edge():
    evidence = material(128)
    _, state, options = prepared(evidence, layers=True)
    base = next(s for s in state.partition.surfaces if s.role == "underlay")
    parent = state.document.ancestry(base.id)[-2]
    editor = Editor(state.document, selection=Selection(whole_document=True))
    with editor.transaction("Fractional coverage frame") as tx:
        tx.set_attributes(
            parent.id, {"transform": parent.get("transform", "") + " translate(0.25 0)"}
        )
    state = replace(state, document=editor.snapshot.document)
    factory = CoreCells(Families(evidence, build(evidence), options), options)
    edits = list(factory(state, Work.start(10)))
    assert edits
    assert factory.diagnostics["alpha_exclusions"] >= 2
    original = render(export_svg(state.document), evidence.source_size)
    for edit in edits:
        assert edit.partition.owners[1] == state.partition.owners[1]
        assert edit.partition.owners[8] == state.partition.owners[8]
        actual = render(export_svg(edit.document), evidence.source_size)
        np.testing.assert_array_equal(actual[..., 3], original[..., 3])


def test_carrier_keeps_evenodd_holes_whose_contours_have_the_same_winding():
    evidence = material(128, hole=True)
    _, state, options = prepared(evidence, layers=True)
    base = next(s for s in state.partition.surfaces if s.role == "underlay")
    shape = parse_path("M0 0H80V48H0Z M32 16H48V32H32Z")
    editor = Editor(state.document, selection=Selection(whole_document=True))
    with editor.transaction("Same winding evenodd carrier") as tx:
        tx.replace_geometry(base.id, shape)
        tx.set_attributes(base.id, {"fill-rule": "evenodd"})
    state = replace(state, document=editor.snapshot.document)
    edits = list(
        CoreCells(Families(evidence, build(evidence), options), options)(
            state, Work.start(10)
        )
    )
    assert edits
    original = render(export_svg(state.document), evidence.source_size)
    for edit in edits:
        assert edit.document.geometry_for(base.id) == state.document.geometry_for(
            base.id
        )
        assert edit.document.element(base.id).get("fill-rule") == "evenodd"
        actual = render(export_svg(edit.document), evidence.source_size)
        np.testing.assert_array_equal(actual[..., 3], original[..., 3])
        assert actual[24:40, 40:56, 3].max() == 0
