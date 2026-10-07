"""Supported facet cuts compose as true source atoms in production search."""

from dataclasses import replace

import numpy as np
import pytest

from tests.refine.test_cel_plan_families import prepared
from tests.refine.test_cel_plan_surface_models import ramp
from vectrify.document import Editor, Selection, export_svg, load_project, save_project
from vectrify.document.join import transformed_geometry
from vectrify.refine.cel_plan import proposals, surface_splits
from vectrify.refine.cel_plan.families import Families
from vectrify.refine.cel_plan.frontier import Frontier
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Work
from vectrify.refine.cel_plan.ownership import Partition
from vectrify.refine.cel_plan.policy import Policy
from vectrify.refine.cel_plan.proposals import Operators
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.search import Rejections, identity, search
from vectrify.refine.cel_plan.surface_splits import SurfaceSplits


def setup(alpha=255, hole=False, *, diagonal=False, single=False):
    e = ramp(alpha, hole)
    _, state, options = prepared(e, layers=alpha != 255)
    family = Families(e, build(e), options)
    merged = next(
        p for p in family.surface_models(state, Work.start(10)) if len(p.ids) == 8
    )
    y, x = np.indices(e.labels.shape)
    left = (x + 0.5 + y + 0.5) / np.sqrt(2) < 56 if diagonal else x < 43
    target = e.target.copy()
    target[~e.empty & left] = (32, 100, 60)
    target[~e.empty & ~left] = (180, 100, 60)
    rgba = e.rgba.copy()
    rgba[~e.empty, :3] = target[~e.empty] / 255
    e = replace(e, target=target, smooth=target, coarse=target, rgba=rgba)
    partition = merged.partition
    if single:
        e = replace(e, labels=np.where(e.empty, 0, 1).astype(np.int32))
        partition = Partition(
            tuple(replace(s, members=(1,)) for s in partition.surfaces)
        )
    policy = Policy(e.rgba)
    frontier = Frontier(policy)
    details = {
        **state.details,
        **merged.details,
        "planning_surfaces": partition.metadata(),
    }
    svg = export_svg(merged.document)
    assert frontier.add(svg, "Coherent surface", details)
    frontier.freeze_normalizer()
    state = replace(
        state,
        document=merged.document,
        svg=svg,
        details=details,
        snapshot=LocalPolicy(policy).start(svg, frontier.entries[0].evaluation),
        partition=partition,
    )
    return e, frontier, state, options


@pytest.mark.parametrize("alpha", [255, 128])
@pytest.mark.parametrize("hole", [False, True])
@pytest.mark.parametrize("single", [False, True])
def test_two_editable_paints_split_real_atoms_with_native_local_and_reload_agreement(
    alpha, hole, single
):
    evidence, frontier, state, options = setup(alpha, hole, single=single)
    factory = SurfaceSplits(Families(evidence, build(evidence), options), options)
    edits = list(factory(state, Work.start(10)))
    assert edits
    edit = edits[0]
    assert len(edit.partition.atoms.cuts) == 1
    assert edit.partition.follows(state.partition)
    branch = edit.partition.atoms.graph(evidence, build(evidence), Work.start(10))
    assert any({e.left, e.right} == set(edit.ids) for e in edit.partition.edges(branch))
    svg = export_svg(edit.document)
    full = frontier.policy.evaluate(svg)
    assert full.valid
    assert full.visual < state.snapshot.evaluation.visual
    local = LocalPolicy(frontier.policy).update(
        state.snapshot, svg, edit.bounds, full.structure
    )
    assert local.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
    assert local.canvas.matches(render(svg, evidence.source_size))
    np.testing.assert_array_equal(
        render(svg, evidence.source_size)[..., 3],
        render(state.svg, evidence.source_size)[..., 3],
    )
    saved, _ = load_project(save_project(edit.document))
    restored = Partition.from_metadata(edit.partition.metadata())
    restored.validate(saved)
    np.testing.assert_array_equal(
        render(export_svg(saved), evidence.source_size),
        render(svg, evidence.source_size),
    )
    assert state.partition.atoms is None
    if alpha != 255:
        old = next(s for s in state.partition.surfaces if s.role == "underlay")
        new = next(s for s in edit.partition.surfaces if s.role == "underlay")
        assert old.members != new.members
        assert edit.document.geometry_for(old.id) == state.document.geometry_for(old.id)


@pytest.mark.parametrize("alpha", [255, 128])
def test_diagonal_cut_and_current_child_transform_keep_complete_support(alpha):
    evidence, frontier, state, options = setup(alpha, diagonal=True, single=True)
    oid = state.partition.owners[1]
    editor = Editor(state.document, selection=Selection(whole_document=True))
    with editor.transaction("Equivalent rotated child frame") as tx:
        tx.set_attributes(oid, {"transform": "matrix(0 1 -1 0 96 0)"})
        tx.replace_geometry(
            oid,
            transformed_geometry(
                state.document.geometry_for(oid), (0, -1, 1, 0, 0, 96)
            ),
        )
    state = replace(state, document=editor.snapshot.document)
    factory = SurfaceSplits(Families(evidence, build(evidence), options), options)
    edits = list(factory(state, Work.start(10)))
    assert edits
    edit = edits[0]
    assert all(abs(v) > 0.5 for v in edit.parameters[0])
    assert frontier.policy.evaluate(export_svg(edit.document)).valid
    atoms = edit.partition.atoms
    labels = atoms.labels(build(evidence), Work.start(10))
    assert sum(int((labels == i).sum()) for i in edit.partition.owners) == int(
        (~evidence.empty).sum()
    )


def test_search_publishes_split_and_rebuilds_all_child_operators(monkeypatch):
    evidence, frontier, state, options = setup(128, single=True)
    graph = build(evidence)
    factory = Operators(evidence, graph, options)
    report = search(frontier, options, Work.start(10), factory)
    assert report["accepted"] > 0
    assert report["score_disagreements"] == 0
    selected = frontier.select(100)
    assert selected.metrics["planning_surfaces"].get("source_atoms")
    partition = Partition.from_metadata(selected.metrics["planning_surfaces"])
    branch = factory.branch(partition, Work.start(10))
    assert branch.graph.labels is branch.evidence.labels
    assert branch.families.graph is branch.replacements.graph is branch.overlays.graph
    assert factory.graph is graph
    assert factory.evidence is evidence
    assert branch.graph.source_atoms == partition.atoms.key
    # Same pixels cannot alias a state or rejection proof in another graph.
    assert identity(state.svg, partition) != identity(state.svg, state.partition)
    monkeypatch.setattr(proposals, "MAX_BRANCH_BYTES", 1)
    new_factory = Operators(evidence, graph, options)
    with pytest.raises(ValueError, match="bounds"):
        new_factory.validate_partition(partition, Work.start(10))


def test_complete_paint_outlier_and_actual_rgba_core_protect_source(monkeypatch):
    evidence, _, state, options = setup(128, single=True)
    target = evidence.target.copy()
    target[30, 30] = (255, 0, 255)
    bad = replace(evidence, target=target)
    factory = SurfaceSplits(Families(bad, build(bad), options), options)
    assert list(factory(state, Work.start(10))) == []
    assert factory.diagnostics["paint_exclusions"] > 0
    base = next(s for s in state.partition.surfaces if s.role == "underlay")
    editor = Editor(state.document, selection=Selection(whole_document=True))
    with editor.transaction("Remove real core") as tx:
        tx.delete_objects(frozenset((base.id,)))
    changed = replace(
        state,
        document=editor.snapshot.document,
        partition=Partition(
            tuple(s for s in state.partition.surfaces if s.id != base.id)
        ),
    )
    factory = SurfaceSplits(Families(evidence, build(evidence), options), options)
    assert list(factory(changed, Work.start(10))) == []
    assert factory.diagnostics["core_exclusions"] > 0
    monkeypatch.setattr(surface_splits, "MAX_PIXELS", 1)
    assert list(factory(state, Work.start(10))) == []
    assert state.partition.atoms is None


def test_stop_and_fixed_source_do_not_return_partial_splits(monkeypatch):
    evidence, _, state, options = setup(single=True)
    graph = build(evidence)
    fixed = replace(
        graph, regions=(graph.regions[0], replace(graph.regions[1], fixed=True))
    )
    assert (
        list(
            SurfaceSplits(Families(evidence, fixed, options), options)(
                state, Work.start(10)
            )
        )
        == []
    )
    work = Work.start(10)
    original = surface_splits.prediction

    def stop(*args, **kwargs):
        result = original(*args, **kwargs)
        work.stop.set()
        return result

    monkeypatch.setattr(surface_splits, "prediction", stop)
    assert (
        list(SurfaceSplits(Families(evidence, graph, options), options)(state, work))
        == []
    )
    assert state.partition.atoms is None


def test_rejection_proofs_and_active_branch_memory_do_not_alias_siblings(monkeypatch):
    evidence, _, state, options = setup(single=True)
    graph = build(evidence)
    factory = Operators(evidence, graph, options)
    edit = next(iter(SurfaceSplits(factory.families, options)(state, Work.start(10))))
    original = edit.partition.atoms.original(graph)
    y, x = np.indices(graph.labels.shape)
    sibling_atoms, a, b = original.split(graph, (1,), y < 32, Work.start(10))
    sibling = state.partition.split(
        (state.partition.owners[1],),
        (
            replace(edit.partition.surfaces[0], members=a),
            replace(edit.partition.surfaces[1], members=b),
        ),
        sibling_atoms,
    )
    cache = Rejections()
    assert cache.key(state, edit, 1) != cache.key(
        state, replace(edit, partition=sibling), 1
    )
    first = factory.branch(edit.partition, Work.start(10))
    size = factory._graph_bytes(first.graph)
    monkeypatch.setattr(proposals, "MAX_BRANCH_BYTES", size * 2 + 4096)
    second = factory.branch(sibling, Work.start(10))
    third_atoms, a, b = original.split(graph, (1,), x < 28, Work.start(10))
    third = state.partition.split(
        (state.partition.owners[1],),
        (
            replace(edit.partition.surfaces[0], members=a),
            replace(edit.partition.surfaces[1], members=b),
        ),
        third_atoms,
    )
    with pytest.raises(ValueError, match="Active source graphs"):
        factory.branch(third, Work.start(10))
    assert first.graph.source_atoms == edit.partition.atoms.key
    assert second.graph.source_atoms == sibling.atoms.key
