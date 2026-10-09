"""Real pixel atom splits, immutable branch graphs and strict replay checks."""

from dataclasses import replace

import numpy as np
import pytest

from tests.helpers import required
from tests.refine.test_cel_plan_ownership import stripes
from vectrify.refine.cel_plan import atoms as module
from vectrify.refine.cel_plan.atoms import AtomLimitError, Atoms, Cut
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


def capacity_source():
    """54 mixed parents need 66 binary cuts, but only 120 direct children."""
    shape = (24, 162)
    target = np.full((*shape, 3), 180, dtype=np.float32)
    rgba = np.concatenate((target / 255, np.full((*shape, 1), 0.5)), axis=-1)
    zero = np.zeros(shape, np.float32)
    evidence = replace(
        stripes(),
        rgba=rgba,
        target=target,
        smooth=target,
        coarse=target,
        empty=zero.astype(bool),
        drawn=zero.astype(bool),
        line=zero.astype(bool),
        foreground=np.ones(shape, bool),
        darkness=zero,
        texture=zero,
        labels=np.broadcast_to(np.arange(162)[None, :] // 3, shape)
        .copy()
        .astype(np.int32),
        source_size=(162, 24),
        opacity=rgba[..., 3],
    )
    y, x = np.indices(shape)
    classes = np.where(x < 36, y // 8, np.where(y < 12, 0, 2)).astype(np.int32)
    return evidence, classes


def test_direct_partition_retains_source_pixels_without_raising_capacity():
    from vectrify.refine.cel_plan.atoms import MultiCut

    evidence, classes = capacity_source()
    graph = build(evidence)
    original = Atoms.original(graph)
    members = tuple(range(54))
    with pytest.raises(ValueError, match="bounds"):
        original.partition(graph, members, classes, 3, Work.start(10))
    atoms, cells = original.partition(
        graph, members, classes, 3, Work.start(10), compact=True
    )
    assert len(atoms.cuts) == 54 <= module.MAX_CUTS == 64
    assert atoms.namespace_count - atoms.count == 120 <= module.MAX_CHILDREN == 128
    assert sum(isinstance(c, MultiCut) for c in atoms.cuts) == 12
    assert module._run_count(atoms.cuts) <= module.MAX_RUNS == 16384
    labels = atoms.labels(graph, Work.start(10))
    lookup = np.full(atoms.namespace_count, -1)
    for index, children in enumerate(cells):
        lookup[list(children)] = index
    np.testing.assert_array_equal(lookup[labels], classes)
    np.testing.assert_array_equal(graph.labels, evidence.labels)
    assert not labels.flags.writeable
    assert Atoms.from_metadata(atoms.metadata()) == atoms
    assert atoms.metadata()["version"] == 2
    assert original.cuts == ()


def test_multiway_lineage_secondary_coverage_and_successive_binary_split():
    evidence = one_atom()
    graph = build(evidence)
    original = Atoms.original(graph)
    y, x = np.indices(graph.labels.shape)
    classes = (x // 32).astype(np.int32)
    atoms, groups = original.partition(
        graph, (1,), classes, 3, Work.start(10), compact=True
    )
    assert atoms.descendants((1,)) == (2, 3, 4)
    branch = atoms.graph(evidence, graph, Work.start(10))
    parent = Partition((Surface("body", (1,)), Surface("base", (1,), "underlay")))
    first = parent.split(
        ("body",), tuple(Surface(f"c{i}", g) for i, g in enumerate(groups)), atoms
    )
    assert first.follows(parent)
    assert next(s for s in first.surfaces if s.role == "underlay").members == (2, 3, 4)
    second, left, right = atoms.split(branch, groups[0], y < 32, Work.start(10))
    assert second.extends(atoms)
    assert second.descendants(groups[0], start=1) == (5, 6)
    child = first.split(("c0",), (Surface("a", left), Surface("b", right)), second)
    assert child.follows(first)
    assert set(next(s for s in child.surfaces if s.role == "underlay").members) == {
        3,
        4,
        5,
        6,
    }
    assert Partition.from_metadata(child.metadata()) == child
    other, _ = original.partition(
        graph, (1,), (y // 24).astype(np.int32), 3, Work.start(10), compact=True
    )
    assert not second.extends(other)
    with pytest.raises(ValueError, match="namespace"):
        other.split(branch, groups[0], y < 32, Work.start(10))
    legacy, _, _ = original.split(graph, (1,), x < 48, Work.start(10))
    assert legacy.metadata()["version"] == 1
    assert "left" in legacy.metadata()["cuts"][0]
    assert required(Atoms.from_metadata(legacy.metadata())).key == legacy.key


@pytest.mark.parametrize(
    "defect", ["overlap", "steal", "incomplete", "version", "binary-arity"]
)
def test_multiway_metadata_and_replay_reject_invalid_source_support(defect):
    from vectrify.refine.cel_plan.atoms import MultiCut

    graph = build(one_atom())
    root = Atoms.original(graph)
    _y, x = np.indices(graph.labels.shape)
    atoms, _ = root.partition(
        graph, (1,), (x // 32).astype(np.int32), 3, Work.start(10), compact=True
    )
    data = atoms.metadata()
    if defect == "overlap":
        data["cuts"][0]["groups"] = (data["cuts"][0]["groups"][0],) * 2
        data["cuts"][0]["areas"] = (1152, 1152, 1280)
    elif defect == "version":
        data["version"] = 1
    elif defect == "binary-arity":
        data["cuts"][0]["groups"] = (data["cuts"][0]["groups"][0],)
        data["cuts"][0]["areas"] = (1152, 2432)
    else:
        area = graph.regions[1].area
        groups = (
            (((0, 0, 1),), ((8, 8, 9),))
            if defect == "steal"
            else (((8, 8, 9),), ((8, 9, 10),))
        )
        broken = replace(
            root,
            cuts=(MultiCut(1, groups, (1, 1, area - (2 if defect == "steal" else 3))),),
        )
        with pytest.raises(ValueError, match=r"owner|complete support"):
            broken.labels(graph, Work.start(10))
        return
    with pytest.raises(ValueError, match=r"schema|runs|area"):
        Atoms.from_metadata(data)


@pytest.mark.parametrize("limit", ["MAX_CUTS", "MAX_CHILDREN", "MAX_RUNS"])
def test_direct_partition_capacity_and_cancellation_discard_entire_change(
    monkeypatch, limit
):
    evidence, classes = capacity_source()
    graph = build(evidence)
    root = Atoms.original(graph)
    monkeypatch.setattr(module, limit, 1)
    with pytest.raises(AtomLimitError, match="bounds") as rejected:
        root.partition(graph, range(54), classes, 3, Work.start(10), compact=True)
    assert (
        rejected.value.resource
        == {
            "MAX_CUTS": "entries",
            "MAX_CHILDREN": "children",
            "MAX_RUNS": "runs",
        }[limit]
    )
    assert rejected.value.limit == 1
    assert rejected.value.attempted > rejected.value.limit
    work = Work.start(10)
    work.stop.set()
    with pytest.raises(StageInterruptedError):
        root.partition(graph, range(54), classes, 3, work, compact=True)
    assert root.cuts == ()
    np.testing.assert_array_equal(graph.labels, evidence.labels)
