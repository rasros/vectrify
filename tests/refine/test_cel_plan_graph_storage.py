"""Sharing saves retained storage without weakening source namespace checks."""

from dataclasses import replace

import numpy as np
import pytest

from tests.refine.test_cel_plan_ownership import stripes
from vectrify.refine.cel_plan import atoms as atom_module
from vectrify.refine.cel_plan import proposals
from vectrify.refine.cel_plan.atoms import Atoms
from vectrify.refine.cel_plan.graph import build, shared
from vectrify.refine.cel_plan.model import Options, StageInterruptedError, Work
from vectrify.refine.cel_plan.ownership import Partition, Surface
from vectrify.refine.cel_plan.proposals import Operators


def partition(graph, atoms=None):
    return Partition(
        tuple(
            Surface(f"owner-{r.id}", (r.id,))
            for r in graph.regions
            if r.area and r.id not in graph.hidden
        ),
        atoms,
    )


def test_original_namespace_shares_immutable_graph_without_rebuilding(monkeypatch):
    evidence = stripes(alpha=128)
    graph = build(evidence)
    atoms = Atoms.original(graph)
    original_pixels = graph.labels.copy()

    def unexpected(*_args, **_kwargs):
        raise AssertionError("Uncut source graph must not be rebuilt")

    monkeypatch.setattr(atom_module, "build", unexpected)
    ops = Operators(evidence, graph, Options())
    current = partition(graph, atoms)
    branch = ops.branch(current, Work.start(10))
    assert branch.graph.labels is graph.labels
    assert branch.graph.regions is graph.regions
    assert branch.graph.boundaries is graph.boundaries
    assert branch.graph.source_atoms == atoms.key
    assert ops.branch(current, Work.start(10)) is branch
    ops.validate_partition(current, Work.start(10))
    assert ops.schedule_diagnostics["source_graph_views"] == 1
    assert ops.schedule_diagnostics["source_graph_rebuilds"] == 0
    assert ops.schedule_diagnostics["source_graph_cache_bytes"] <= 1024
    with pytest.raises(ValueError, match="read-only"):
        branch.graph.labels[8, 8] = 0
    with pytest.raises(ValueError, match="read-only"):
        branch.graph.boundaries[0].points[0] = 0
    np.testing.assert_array_equal(graph.labels, original_pixels)


def test_cut_reuses_only_complete_unchanged_region_and_boundary_values():
    evidence = stripes(alpha=128)
    original = build(evidence)
    atoms, _, _ = Atoms.original(original).split(
        original, (2,), np.indices(original.labels.shape)[0] < 32, Work.start(10)
    )
    cut = atoms.graph(evidence, original, Work.start(10))
    rebuilt = build(evidence, atoms.labels(original, Work.start(10)))
    assert cut.regions == rebuilt.regions
    assert cut.regions[1] is original.regions[1]
    assert cut.regions[2] is not original.regions[2]
    assert len({b.id for b in cut.boundaries}) == len(cut.boundaries)
    old_ids = {id(b) for b in original.boundaries}
    reused = [b for b in cut.boundaries if id(b) in old_ids]
    changed = [b for b in cut.boundaries if id(b) not in old_ids]
    assert reused
    assert changed
    assert all(b.id > max(b.id for b in original.boundaries) for b in changed)
    assert all(not b.points.flags.writeable for b in cut.boundaries)
    assert sorted(
        (b.left, b.right, b.line_support, b.points.tobytes()) for b in cut.boundaries
    ) == sorted(
        (b.left, b.right, b.line_support, b.points.tobytes())
        for b in rebuilt.boundaries
    )
    assert cut.source_atoms == atoms.key


@pytest.mark.parametrize("change", ["points", "support"])
def test_same_chain_endpoints_do_not_prove_unchanged_geometry_or_evidence(change):
    graph = build(stripes())
    edge = next(b for b in graph.boundaries if len(b.points) > 3)
    points = edge.points.copy()
    if change == "points":
        points[1] += 0.125
    points.flags.writeable = False
    changed = replace(
        edge,
        points=points,
        line_support=0.5 if change == "support" else edge.line_support,
    )
    result = shared(replace(graph, boundaries=(changed,)), graph, Work.start(10))
    assert result.boundaries[0] is not edge
    assert result.boundaries[0].id > max(b.id for b in graph.boundaries)


def test_original_namespace_does_not_block_a_cut_that_fits_the_cache(monkeypatch):
    evidence = stripes(alpha=128)
    graph = build(evidence)
    original = Atoms.original(graph)
    ops = Operators(evidence, graph, Options())
    parent = ops.branch(partition(graph, original), Work.start(10))
    atoms, _, _ = original.split(
        graph, (2,), np.indices(graph.labels.shape)[0] < 32, Work.start(10)
    )
    expected = atoms.graph(evidence, graph, Work.start(10))
    # A single cut fits, while two independently copied full graphs would not.
    limit = ops._graph_bytes(expected) + 1024
    assert ops._graph_bytes(graph) + ops._graph_bytes(expected) > limit
    monkeypatch.setattr(proposals, "MAX_BRANCH_BYTES", limit)
    current = partition(expected, atoms)
    child = ops.branch(current, Work.start(10))
    ops.validate_partition(current, Work.start(10))
    assert parent.graph.labels is graph.labels
    assert child.graph.labels is not graph.labels
    assert ops._live_graph_bytes() <= limit
    assert ops.schedule_diagnostics["source_graph_cache_peak_bytes"] <= limit
    assert ops.schedule_diagnostics["source_graph_rebuilds"] == 1


def test_array_views_charge_their_entire_retained_allocation_once():
    graph = build(stripes())
    buffer = np.arange(4000, dtype=np.float64).reshape(-1, 2)
    buffer.flags.writeable = False
    a = replace(graph.boundaries[0], points=buffer[:10])
    b = replace(graph.boundaries[1], points=buffer[10:20])
    view_graph = replace(graph, boundaries=(a, b))
    amount = Operators._graph_bytes(view_graph)
    one = Operators._graph_bytes(replace(view_graph, boundaries=(a,)))
    assert amount >= buffer.nbytes + graph.labels.nbytes
    assert amount - one == 512 + 8  # Another record/reference, same point storage.


def test_build_does_not_freeze_or_alias_mutable_caller_labels():
    evidence = stripes()
    labels = evidence.labels.copy()
    graph = build(evidence, labels)
    labels[8, 8] = 0
    assert labels.flags.writeable
    assert graph.labels[8, 8] == 1
    assert not graph.labels.flags.writeable


def test_uncut_view_still_rejects_foreign_source_and_interrupted_work():
    evidence = stripes()
    graph = build(evidence)
    atoms = Atoms.original(graph)
    with pytest.raises(ValueError, match="different original"):
        replace(atoms, source="0" * 64).graph(evidence, graph, Work.start(10))
    work = Work.start(10)
    work.stop.set()
    with pytest.raises(StageInterruptedError):
        atoms.graph(evidence, graph, work)
    with pytest.raises(StageInterruptedError):
        shared(graph, graph, work)
