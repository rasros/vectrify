"""Residual labels compact allocation while retaining exact, revisioned lineage."""

from dataclasses import replace

import numpy as np
import pytest

from tests.refine.test_cel_plan_atoms import capacity_source, one_atom
from vectrify.refine.cel_plan import atoms as module
from vectrify.refine.cel_plan.atoms import AtomLimitError, Atoms, ResidualCut
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.model import StageInterruptedError, Work
from vectrify.refine.cel_plan.ownership import Partition, Surface


def partition(root, graph, members, classes, count, work=None):
    return root.partition(
        graph,
        members,
        classes,
        count,
        work or Work.start(10),
        compact=True,
        retain_parent=True,
    )


def test_same_source_classification_uses_only_additional_labels():
    evidence, classes = capacity_source()
    graph = build(evidence)
    root = Atoms.original(graph)
    legacy, old_cells = root.partition(
        graph, range(54), classes, 3, Work.start(10), compact=True
    )
    atoms, cells = partition(root, graph, range(54), classes, 3)
    assert len(atoms.cuts) == len(legacy.cuts) == 54
    assert all(isinstance(c, ResidualCut) for c in atoms.cuts)
    assert atoms.namespace_count - atoms.count == 66
    assert legacy.namespace_count - legacy.count == 120
    assert module.MAX_CUTS == 64
    assert module.MAX_CHILDREN == 128
    assert atoms.retired == frozenset()
    assert module._run_count(atoms.cuts) == module._run_count(legacy.cuts)
    for ledger, groups in ((atoms, cells), (legacy, old_cells)):
        lookup = np.full(ledger.namespace_count, -1)
        for i, members in enumerate(groups):
            lookup[list(members)] = i
        labels = ledger.labels(graph, Work.start(10))
        np.testing.assert_array_equal(lookup[labels], classes)
        assert not labels.flags.writeable
    np.testing.assert_array_equal(graph.labels, evidence.labels)
    assert Atoms.from_metadata(atoms.metadata()) == atoms
    assert atoms.metadata()["version"] == 3
    assert root.cuts == ()


def test_residual_can_be_split_again_then_retired_with_complete_secondary_ownership():
    evidence = one_atom()
    graph = build(evidence)
    root = Atoms.original(graph)
    y, x = np.indices(graph.labels.shape)
    first, cells = partition(root, graph, (1,), (x >= 32).astype(np.int32), 2)
    assert cells == ((2,), (1,))  # The largest child keeps the original label.
    first_graph = first.graph(evidence, graph, Work.start(10))
    assert first_graph.regions[1].area < graph.regions[1].area
    parent = Partition((Surface("body", (1,)), Surface("base", (1,), "underlay")))
    owned = parent.split(
        ("body",), (Surface("left", cells[0]), Surface("right", cells[1])), first
    )
    second, parts = partition(first, first_graph, (1,), (y >= 32).astype(np.int32), 2)
    assert second.extends(first)
    assert second.descendants((1,), start=1) == (1, 3)
    assert second.descendants((1,)) == (1, 2, 3)
    middle = owned.split(
        ("right",), (Surface("top", parts[0]), Surface("bottom", parts[1])), second
    )
    assert middle.follows(owned)
    second_graph = second.graph(evidence, graph, Work.start(10))
    third, a, b = second.split(second_graph, (1,), x < 64, Work.start(10))
    assert third.retired == frozenset((1,))
    final = middle.split(
        (
            next(
                s.id
                for s in middle.surfaces
                if s.role != "underlay" and s.members == (1,)
            ),
        ),
        (Surface("a", a), Surface("b", b)),
        third,
    )
    assert final.follows(middle)
    assert third.descendants((1,)) == (2, 3, 4, 5)
    assert next(s.members for s in final.surfaces if s.role == "underlay") == (
        2,
        3,
        4,
        5,
    )
    assert Partition.from_metadata(final.metadata()) == final
    third.graph(evidence, graph, Work.start(10))
    np.testing.assert_array_equal(
        first.labels(graph, Work.start(10)), first_graph.labels
    )
    with pytest.raises(ValueError, match="namespace"):
        partition(first, second_graph, (1,), (x < 64).astype(np.int32), 2)


@pytest.mark.parametrize(
    "defect", ["version", "false", "integer", "left", "arity", "overlap"]
)
def test_malformed_residual_metadata_is_rejected(defect):
    graph = build(one_atom())
    root = Atoms.original(graph)
    classes = np.indices(graph.labels.shape)[1] // 32
    atoms, _ = partition(root, graph, (1,), classes, 3)
    data = atoms.metadata()
    cut = data["cuts"][0]
    if defect == "version":
        data["version"] = 2
    elif defect in {"false", "integer"}:
        cut["retain_parent"] = False if defect == "false" else 1
    elif defect == "left":
        cut["left"] = cut.pop("groups")[0]
    elif defect == "arity":
        cut["groups"] = ()
    else:
        cut["groups"] = (cut["groups"][0],) * 2
    with pytest.raises(ValueError, match=r"schema|bounds|runs|area"):
        Atoms.from_metadata(data)


@pytest.mark.parametrize("defect", ["steal", "area", "reuse"])
def test_replay_checks_residual_support_and_parent_area_on_every_revision(defect):
    graph = build(one_atom())
    root = Atoms.original(graph)
    area = graph.regions[1].area
    runs = ((0, 0, 1),) if defect == "steal" else ((8, 8, 9),)
    cut = ResidualCut(1, (runs,), (1, area - (2 if defect == "area" else 1)))
    cuts = (cut,)
    if defect == "reuse":
        cuts += (ResidualCut(1, (runs,), (1, area - 2)),)
    broken = replace(root, cuts=cuts)
    with pytest.raises(ValueError, match=r"owner|complete support"):
        broken.labels(graph, Work.start(10))


@pytest.mark.parametrize("resource", ["MAX_CUTS", "MAX_CHILDREN", "MAX_RUNS"])
def test_limits_and_mid_generation_stop_publish_no_partial_ledger(
    resource, monkeypatch
):
    evidence, classes = capacity_source()
    graph = build(evidence)
    root = Atoms.original(graph)
    monkeypatch.setattr(module, resource, 1)
    with pytest.raises(AtomLimitError):
        partition(root, graph, range(54), classes, 3)
    assert root.cuts == ()
    np.testing.assert_array_equal(graph.labels, evidence.labels)


def test_cancellation_during_replay_keeps_parent_immutable():
    graph = build(one_atom())
    root = Atoms.original(graph)
    x = np.indices(graph.labels.shape)[1]
    atoms, _ = partition(root, graph, (1,), (x >= 48).astype(np.int32), 2)
    work = Work.start(10)
    work.stop.set()
    with pytest.raises(StageInterruptedError):
        atoms.labels(graph, work)
    with pytest.raises(StageInterruptedError):
        partition(root, graph, (1,), (x >= 48).astype(np.int32), 2, work)
    assert root.cuts == ()


@pytest.mark.parametrize("hidden", [False, True])
def test_protected_original_roots_remain_protected_after_retention(hidden):
    graph = build(one_atom())
    root = Atoms.original(graph)
    x = np.indices(graph.labels.shape)[1]
    atoms, _ = partition(root, graph, (1,), (x >= 48).astype(np.int32), 2)
    protected = (
        replace(graph, hidden=frozenset((1,)))
        if hidden
        else replace(
            graph, regions=(graph.regions[0], replace(graph.regions[1], fixed=True))
        )
    )
    with pytest.raises(ValueError, match="Protected"):
        atoms.labels(protected, Work.start(10))
    with pytest.raises(ValueError, match="Protected"):
        partition(root, protected, (1,), (x >= 48).astype(np.int32), 2)


def test_residual_layout_requires_compact_partition():
    graph = build(one_atom())
    root = Atoms.original(graph)
    x = np.indices(graph.labels.shape)[1]
    with pytest.raises(ValueError, match="compact"):
        root.partition(
            graph,
            (1,),
            (x >= 48).astype(np.int32),
            2,
            Work.start(10),
            retain_parent=True,
        )
