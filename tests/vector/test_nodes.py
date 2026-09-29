"""The node moves change only what they may, and keep every node's identity."""

import random

import pytest

from tests.document.test_topology import linked_editor
from vectrify.document import import_svg
from vectrify.vector import nodes
from vectrify.vector.nodes import Frozen, Paths, frozen

SVG = (
    '<svg width="100" height="100">'
    '<path id="p" d="M10 10 C30 0 70 0 90 10 L90 90 L10 90 Z" fill="red"/>'
    '<path id="s" d="M10 50 L90 50" stroke="black" stroke-width="2" fill="none"/>'
    "</svg>"
)


def state(svg=SVG, ids=("p", "s")):
    document = import_svg(svg)
    paths = Paths(
        {oid: document.geometry_for(oid) for oid in ids},
        {"s": 2.0} if "s" in ids else {},
    )
    return document, paths


def ids(geometry):
    return [n.id for s in geometry.subpaths for n in s.nodes]


def test_nudge_moves_one_thing_and_keeps_node_ids():
    document, paths = state()
    fixed = frozen(document, paths)
    rng = random.Random(1)
    for _ in range(50):
        moved = nodes.nudge(paths, rng, fixed)
        assert moved is not None
        assert moved.key() != paths.key()
        for oid in paths.geometries:
            assert ids(moved.geometries[oid]) == ids(paths.geometries[oid])


def test_split_adds_one_node_and_an_unpushed_cubic_split_keeps_the_curve():
    document, paths = state(ids=("p",))
    fixed = frozen(document, paths)
    rng = random.Random(2)
    split = nodes.split(paths, rng, fixed)
    assert split is not None
    assert split.nodes() == paths.nodes() + 1
    old = set(ids(paths.geometries["p"]))
    assert old <= set(ids(split.geometries["p"]))


def test_an_unpushed_cubic_split_lands_on_the_curve():
    curve = import_svg(
        '<svg width="100" height="100"><path id="p" d="M0 0 C0 60 100 60 100 0"/></svg>'
    ).geometry_for("p")
    paths = Paths({"p": curve}, {})
    still = Frozen(frozenset(), frozenset(), {"p": 0.0})
    halves = nodes.split(paths, random.Random(3), still)
    assert halves is not None
    first, second = halves.geometries["p"].subpaths[0].nodes[1:]

    def at(t):
        u = 1 - t
        return (
            3 * u * t * t * 100 + t**3 * 100,
            3 * u * u * t * 60 + 3 * u * t * t * 60,
        )

    x, y = first.endpoint
    assert (
        min(
            abs(x - px) + abs(y - py)
            for px, py in map(at, [i / 2000 for i in range(2001)])
        )
        < 0.2
    )
    assert second.endpoint == (100.0, 0.0)


def test_remove_keeps_a_closed_contour_at_three_nodes():
    document, paths = state(ids=("p",))
    fixed = frozen(document, paths)
    rng = random.Random(4)
    fewer = nodes.remove(paths, rng, fixed)
    assert fewer is not None
    assert fewer.nodes() == paths.nodes() - 1
    assert set(ids(fewer.geometries["p"])) < set(ids(paths.geometries["p"]))
    assert nodes.remove(fewer, rng, frozen(document, fewer)) is None


def test_pinned_and_linked_nodes_do_not_move():
    editor = linked_editor()
    document = editor.snapshot.document
    paths = Paths({"fill": document.geometry_for("fill")}, {})
    fixed = frozen(document, paths)
    before = {n.id: n for s in paths.geometries["fill"].subpaths for n in s.nodes}
    rng = random.Random(5)
    for move in (nodes.nudge, nodes.split, nodes.remove):
        for _ in range(40):
            changed = move(paths, rng, fixed)
            if changed is None:
                continue
            after = {
                n.id: n for s in changed.geometries["fill"].subpaths for n in s.nodes
            }
            for node_id in fixed.nodes:
                assert after[node_id] == before[node_id]
            for node_id in fixed.endpoints:
                assert after[node_id].endpoint == before[node_id].endpoint
    assert nodes.shift(paths, rng, fixed) is None


def test_stroke_scales_only_stroked_paths():
    document, paths = state()
    thicker = nodes.stroke(paths, random.Random(6), frozen(document, paths))
    assert thicker is not None
    assert set(thicker.strokes) == {"s"}
    assert thicker.strokes["s"] != 2.0
    _, unstroked = state(ids=("p",))
    assert (
        nodes.stroke(unstroked, random.Random(6), Frozen(frozenset(), frozenset(), {}))
        is None
    )


@pytest.mark.parametrize("name", sorted(nodes.MOVES))
def test_every_move_is_registered_under_its_checkbox(name):
    assert callable(nodes.MOVES[name])
