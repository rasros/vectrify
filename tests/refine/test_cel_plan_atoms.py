"""Real pixel atom splits, immutable branch graphs and strict replay checks."""

from dataclasses import replace

import numpy as np
import pytest

from tests.refine.test_cel_plan_ownership import stripes
from vectrify.refine.cel_plan import atoms as module
from vectrify.refine.cel_plan.atoms import Atoms, Cut
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.model import StageInterruptedError, Work
from vectrify.refine.cel_plan.ownership import Partition, Surface


def one_atom():
    evidence = stripes(hole=True)
    return replace(evidence, labels=np.where(evidence.empty, 0, 1).astype(np.int32))


def test_exact_support_new_canonical_edge_and_immutable_parent():
    evidence = one_atom()
    graph = build(evidence)
    original = graph.labels.copy()
    root = Atoms.original(graph)
    left = np.indices(graph.labels.shape)[1] < 48
    atoms, a, b = root.split(graph, (1,), left, Work.start(10))
    branch = atoms.graph(evidence, graph, Work.start(10))
    assert a == (2,)
    assert b == (3,)
    np.testing.assert_array_equal(branch.labels == 2, (original == 1) & left)
    np.testing.assert_array_equal(branch.labels == 3, (original == 1) & ~left)
    np.testing.assert_array_equal(graph.labels, original)
    assert branch.regions[1].area == 0
    assert sum(r.area for r in branch.regions) == graph.regions[1].area
    assert not branch.labels.flags.writeable
    assert all(not edge.points.flags.writeable for edge in branch.boundaries)
    parent = Partition((Surface("body", (1,)), Surface("base", (1,), "underlay")))
    split = parent.split(("body",), (Surface("a", a), Surface("b", b)), atoms)
    assert split.follows(parent)
    assert next(s for s in split.surfaces if s.role == "underlay").members == (2, 3)
    edges = split.edges(branch)
    assert any({e.left, e.right} == {"a", "b"} for e in edges)
    with pytest.raises(ValueError, match="current source atom graph"):
        split.edges(graph)
    assert Partition.from_metadata(split.metadata()) == split


def test_successive_split_merge_secondary_coverage_and_sibling_isolation():
    evidence = one_atom()
    graph = build(evidence)
    root = Atoms.original(graph)
    y, x = np.indices(graph.labels.shape)
    first, a, b = root.split(graph, (1,), x < 48, Work.start(10))
    other, _, _ = root.split(graph, (1,), y < 32, Work.start(10))
    branch = first.graph(evidence, graph, Work.start(10))
    second, c, d = first.split(branch, a, y < 32, Work.start(10))
    assert second.extends(first)
    assert not second.extends(other)
    assert second.descendants((1,)) == (3, 4, 5)
    parent = Partition((Surface("body", (1,)),))
    p = parent.split(("body",), (Surface("a", a), Surface("b", b)), first)
    p = Partition((replace(p.surfaces[0], covered=b), p.surfaces[1]), first)
    child = p.split(("a",), (Surface("c", c, covered=b), Surface("d", d)), second)
    assert child.follows(p)
    assert not child.follows(
        parent.split(("body",), (Surface("x", (2,)), Surface("z", (3,))), other)
    )
    merged = child.replace(("c", "d"), (Surface("c", (4, 5), covered=b),))
    assert merged.atoms == second
    assert first.cuts == (first.cuts[0],)
    assert branch.source_atoms == first.key
    with pytest.raises(ValueError, match="namespace"):
        other.split(branch, (2,), x < 32, Work.start(10))


def test_foreign_incomplete_and_other_owner_runs_fail_replay():
    evidence = one_atom()
    graph = build(evidence)
    root = Atoms.original(graph)
    valid, _, _ = root.split(
        graph, (1,), np.indices(graph.labels.shape)[1] < 48, Work.start(10)
    )
    with pytest.raises(ValueError, match="different original"):
        replace(valid, source="0" * 64).labels(graph, Work.start(10))
    with pytest.raises(ValueError, match="complete support"):
        replace(root, cuts=(Cut(1, ((8, 8, 9),), (1, 1)),)).labels(
            graph, Work.start(10)
        )
    area = graph.regions[1].area
    with pytest.raises(ValueError, match="another owner's"):
        replace(root, cuts=(Cut(1, ((0, 0, 1),), (1, area - 1)),)).labels(
            graph, Work.start(10)
        )


def test_fixed_hidden_bounds_and_cancellation_cannot_publish_partial_atoms(monkeypatch):
    evidence = one_atom()
    graph = build(evidence)
    left = np.indices(graph.labels.shape)[1] < 48
    fixed = replace(
        graph, regions=(graph.regions[0], replace(graph.regions[1], fixed=True))
    )
    root = Atoms.original(graph)
    with pytest.raises(ValueError, match="Protected"):
        root.split(fixed, (1,), left, Work.start(10))
    valid, _, _ = root.split(graph, (1,), left, Work.start(10))
    with pytest.raises(ValueError, match="Protected"):
        valid.labels(fixed, Work.start(10))
    with pytest.raises(ValueError, match="Protected"):
        valid.labels(replace(graph, hidden=frozenset((0, 1))), Work.start(10))
    monkeypatch.setattr(module, "MAX_RUNS", 1)
    with pytest.raises(ValueError, match="bounds"):
        root.split(graph, (1,), left, Work.start(10))
    work = Work.start(10)
    work.stop.set()
    with pytest.raises(StageInterruptedError):
        valid.labels(graph, work)
    assert root.cuts == ()
    assert graph.labels[8, 8] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"shape": (64, "96")},
        {"count": True},
        {"cuts": None},
        {"cuts": [{"parent": 1, "left": [(8, 8, 9)], "areas": (1,)}]},
        {"cuts": [{"parent": 1, "left": [(8, 8, 10), (8, 9, 11)], "areas": (4, 1)}]},
    ],
)
def test_malformed_metadata_is_rejected(change):
    root = Atoms.original(build(one_atom()))
    with pytest.raises(ValueError, match=r"schema|runs"):
        Atoms.from_metadata({**root.metadata(), **change})
