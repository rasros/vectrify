"""Stable source identities survive renumbering, layers, merges and splits."""

from dataclasses import replace

import numpy as np
import pytest

from vectrify.document import import_svg
from vectrify.refine.cel_plan.export import export
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.model import Evidence, Options, Work
from vectrify.refine.cel_plan.ownership import Partition, Surface, exported


def stripes(*, alpha=255, gradient=False, hole=False):
    height, width = 64, 96
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    pixels[8:56, 8:88] = (176, 100, 60, alpha)
    if gradient:
        pixels[8:56, 8:88, 0] = np.arange(80, dtype=np.uint8)[None, :] + 136
    if hole:
        pixels[24:40, 40:56] = 0
    shown = pixels[..., 3] > 0
    labels = np.where(shown, 1 + (np.arange(width)[None, :] - 8) // 10, 0).astype(
        np.int32
    )
    rgba = pixels.astype(np.float32) / 255
    target = np.where(shown[..., None], pixels[..., :3], 255).astype(np.float32)
    zero = np.zeros((height, width), dtype=np.float32)
    return Evidence(
        rgba,
        target,
        target.copy(),
        target.copy(),
        ~shown,
        shown,
        zero.astype(bool),
        zero.astype(bool),
        zero.copy(),
        zero.copy(),
        labels,
        (width, height),
        (0, 0),
        (1, 1),
        None,
        False,
        rgba[..., 3].copy() if alpha != 255 else None,
    )


def test_export_renumbering_keeps_original_region_ids_and_canonical_edge_owners():
    evidence = stripes()
    graph = build(evidence)
    mapping = np.array([0, 9, 9, 2, 2, 7, 7, 1, 1], dtype=np.int32)
    labels = mapping[evidence.labels]
    svg, details = export(evidence, labels, Options(gradients=False), Work.start(10))
    partition = Partition.from_metadata(details["planning_surfaces"])
    assert partition is not None
    partition.validate(import_svg(svg))
    expected = {
        "cel-fill-9": (1, 2),
        "cel-fill-2": (3, 4),
        "cel-fill-7": (5, 6),
        "cel-fill-1": (7, 8),
    }
    assert {s.id: s.members for s in partition.surfaces} == expected
    assert set(partition.owners) == set(range(1, 9))
    original = {edge.id: edge for edge in graph.boundaries}
    for edge in partition.edges(graph):
        assert edge.left == partition.owners.get(original[edge.boundary].left)
        assert edge.right == partition.owners.get(original[edge.boundary].right)
        assert edge.left != edge.right
    assert len(partition.edges(graph)) < len(graph.boundaries)


def test_successive_merges_and_split_keep_members_and_sibling_ownership():
    initial = Partition(tuple(Surface(f"p{i}", (i,)) for i in range(1, 5)))
    first = initial.replace(("p1", "p2"), (Surface("p1", (1, 2)),))
    second = first.replace(("p1", "p3"), (Surface("p1", (1, 2, 3)),))
    split = second.replace(("p1",), (Surface("p1", (1,)), Surface("p2", (2, 3))))
    sibling = first.replace(("p3", "p4"), (Surface("p3", (3, 4)),))
    assert first.owners == {1: "p1", 2: "p1", 3: "p3", 4: "p4"}
    assert sibling.owners == {1: "p1", 2: "p1", 3: "p3", 4: "p3"}
    assert split.owners == {1: "p1", 2: "p2", 3: "p2", 4: "p4"}
    assert initial.surfaces[0].members == (1,)
    assert Partition.from_metadata(second.metadata()) == second
    with pytest.raises(ValueError, match="retain all"):
        first.replace(("p1",), (Surface("p1", (1,)),))
    with pytest.raises(ValueError, match="multiple primary"):
        Partition((Surface("a", (1,)), Surface("b", (1,))))


def test_opacity_base_is_secondary_coverage_not_duplicate_primary_ownership():
    evidence = stripes(alpha=64, hole=True)
    svg, details = export(
        evidence,
        evidence.labels,
        Options(gradients=False),
        Work.start(10),
        layers=True,
    )
    partition = Partition.from_metadata(details["planning_surfaces"])
    assert partition is not None
    partition.validate(import_svg(svg))
    underlay = next(s for s in partition.surfaces if s.role == "underlay")
    assert underlay.members == tuple(range(1, 9))
    assert set(partition.owners) == set(range(1, 9))
    assert underlay.id not in partition.owners.values()
    with pytest.raises(ValueError, match="primary"):
        partition.replace((underlay.id,), (Surface("x", underlay.members),))


def test_independent_subdivision_requires_new_graph_atoms_before_owned_edits():
    evidence = stripes()
    source = np.where(~evidence.empty, 1, 0).astype(np.int32)
    evidence = replace(evidence, labels=source)
    # Eight output strips split a single source region. Their geometry remains
    # exportable, but the original atomic graph cannot describe that ownership.
    labels = stripes().labels
    metadata = exported(evidence, labels, frozenset(range(1, 9)), Work.start(10))
    assert metadata["complete"] is False
    assert metadata["reason"] == "source-region-split"
    assert Partition.from_metadata(metadata) is None
