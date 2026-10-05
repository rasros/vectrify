"""Shared edges: a neighbour's copy of an edge follows the selected path."""

from dataclasses import replace

import pytest

from vectrify.document import import_svg
from vectrify.refine.shared import (
    coordinate_indices,
    follow,
    frozen_points,
    intact,
    links,
)

# Two squares side by side sharing the edge x = 10 through a point at y = 5;
# B runs it the other way, and is drawn back to its start as a cel trace is.
SIDE = (
    '<svg width="30" height="20">'
    '<path id="a" fill="red" d="M0 0 L10 0 L10 5 L10 10 L0 10 Z"/>'
    '<path id="b" fill="blue" d="M10 0 L20 0 L20 10 L10 10 L10 5 L10 0 Z"/>'
    "</svg>"
)


def points(document, oid):
    return [n.values[-2:] for s in document.geometry_for(oid).subpaths for n in s.nodes]


def reshaped(document, oid, where):
    """*document* with *oid*'s nodes passed through *where* (a node list
    to a node list) contour by contour."""
    geometry = document.geometry_for(oid)
    subpaths = tuple(
        replace(s, nodes=tuple(where(list(s.nodes)))) for s in geometry.subpaths
    )
    return document.replace_geometry(replace(geometry, subpaths=subpaths))


def test_a_shared_edge_is_found_and_its_ends_frozen():
    document = import_svg(SIDE)
    (link,) = links(document, ["a"], ["b"])
    assert link.neighbour == "b"
    assert link.reversed
    nodes = {
        n.id: n.values for s in document.geometry_for("a").subpaths for n in s.nodes
    }
    assert {nodes[link.start], nodes[link.end]} == {(10.0, 0.0), (10.0, 10.0)}
    held = frozen_points(document, [link])
    assert {nodes[i] for i in held if i in nodes} == {(10.0, 0.0), (10.0, 10.0)}


def test_an_overlap_stops_being_an_intact_shared_run():
    document = import_svg(SIDE)
    forward = links(document, ["a"], ["b"])[0]
    reverse = links(document, ["b"], ["a"])[0]
    assert intact(document, forward)
    assert intact(document, reverse)
    geometry = document.geometry_for("a")
    middle = next(
        n for s in geometry.subpaths for n in s.nodes if n.endpoint == (10, 5)
    )
    overlap = document.replace_geometry(
        geometry.replace_node(replace(middle, values=(10.5, 5)))
    )
    assert not intact(overlap, forward)
    assert not intact(overlap, reverse)
    assert intact(follow(overlap, [forward])[0], forward)


def test_shared_coordinate_rows_include_a_drawn_closure():
    document = import_svg(SIDE)
    (link,) = links(document, ["a"], ["b"])
    assert coordinate_indices(
        document.geometry_for("a"), link.subpath, link.start, link.end
    ) == [(1, 2), (2, 3)]
    assert coordinate_indices(
        document.geometry_for("b"),
        link.neighbour_subpath,
        link.neighbour_end,
        link.neighbour_start,
    ) == [(3, 4), (4, 5)]


def test_shared_coordinate_rows_handle_holes_wrapped_cubics_and_implicit_closures():
    geometry = import_svg(
        '<svg><path id="a" d="M0 0 L30 0 L30 30 Z '
        'M10 10 C12 12 18 12 20 10 L20 20 L10 20 Z"/></svg>'
    ).geometry_for("a")
    hole = geometry.subpaths[1]
    head, cubic, _, end = hole.nodes
    assert coordinate_indices(geometry, hole.id, end.id, cubic.id) == [
        (8, 3),  # Implicit closing line back to the hole's head.
        (3, 4, 5, 6),
    ]
    assert coordinate_indices(geometry, hole.id, head.id, head.id) == [
        (3, 4, 5, 6),
        (6, 7),
        (7, 8),
        (8, 3),
    ]


def test_shared_coordinate_rows_handle_open_curves_and_missing_ids():
    geometry = import_svg(
        '<svg><path id="a" d="M1 2 C3 4 5 6 7 8 L9 10"/></svg>'
    ).geometry_for("a")
    subpath = geometry.subpaths[0]
    start, _, end = subpath.nodes
    assert coordinate_indices(geometry, subpath.id, start.id, end.id) == [
        (0, 1, 2, 3),
        (3, 4),
    ]
    assert coordinate_indices(geometry, subpath.id, end.id, start.id) is None
    assert coordinate_indices(geometry, subpath.id, start.id, "missing") is None
    assert coordinate_indices(geometry, "missing", start.id, end.id) is None


@pytest.mark.parametrize(
    ("where", "expected"),
    [
        # The point between moves out: the neighbour's moves with it.
        (
            lambda nodes: [
                replace(n, values=(12.0, 5.0)) if n.values == (10.0, 5.0) else n
                for n in nodes
            ],
            [
                (10.0, 0.0),
                (20.0, 0.0),
                (20.0, 10.0),
                (10.0, 10.0),
                (12.0, 5.0),
                (10.0, 0.0),
            ],
        ),
        # The point between goes: so does the neighbour's.
        (
            lambda nodes: [n for n in nodes if n.values != (10.0, 5.0)],
            [(10.0, 0.0), (20.0, 0.0), (20.0, 10.0), (10.0, 10.0), (10.0, 0.0)],
        ),
        # The edge bends: the neighbour bends the same way, run backwards.
        (
            lambda nodes: [
                replace(n, command="C", values=(11.0, 2.0, 11.0, 4.0, 10.0, 5.0))
                if n.values == (10.0, 5.0)
                else n
                for n in nodes
            ],
            [
                (10.0, 0.0),
                (20.0, 0.0),
                (20.0, 10.0),
                (10.0, 10.0),
                (10.0, 5.0),
                (10.0, 0.0),
            ],
        ),
    ],
)
def test_the_neighbour_follows_the_edge(where, expected):
    document = import_svg(SIDE)
    found = links(document, ["a"], ["b"])
    after, changed = follow(reshaped(document, "a", where), found)
    assert points(after, "b") == expected
    if expected == points(document, "b"):
        bent = after.geometry_for("b").subpaths[0].nodes[-1]
        assert bent.command == "C"
        assert bent.values == (11.0, 4.0, 11.0, 2.0, 10.0, 0.0)
    assert changed == {"b"}


def test_a_hole_shared_whole_follows_the_island():
    document = import_svg(
        '<svg width="30" height="30">'
        '<path id="ring" fill="blue" fill-rule="evenodd" '
        'd="M0 0 L30 0 L30 30 L0 30 Z M10 10 L10 20 L20 20 L20 10 L10 10 Z"/>'
        '<path id="island" fill="red" d="M10 10 L20 10 L20 20 L10 20 L10 10 Z"/>'
        "</svg>"
    )
    found = links(document, ["island"], ["ring"])
    assert len(found) == 1
    moved = reshaped(
        document,
        "island",
        lambda nodes: [
            replace(n, values=(21.0, 21.0)) if n.values == (20.0, 20.0) else n
            for n in nodes
        ],
    )
    after, _ = follow(moved, found)
    hole = [n.values[-2:] for n in after.geometry_for("ring").subpaths[1].nodes]
    assert (21.0, 21.0) in hole
    assert (20.0, 20.0) not in hole
    assert hole[0] == hole[-1]


def test_a_run_whose_end_is_gone_is_left_alone():
    document = import_svg(SIDE)
    found = links(document, ["a"], ["b"])
    gone = reshaped(
        document, "a", lambda nodes: [n for n in nodes if n.values != (10.0, 10.0)]
    )
    after, changed = follow(gone, found)
    assert not changed
    assert points(after, "b") == points(document, "b")
