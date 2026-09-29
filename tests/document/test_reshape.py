"""Reshaping a path keeps the identity of every node it does not remove."""

from dataclasses import replace

import pytest

from tests.document.test_document import select
from tests.document.test_topology import linked_editor
from vectrify.document import Editor, EditRejectedError, PathNode, import_svg

SQUARE = (
    '<svg width="40" height="40">'
    '<path id="p" d="M0 0 L20 0 L20 20 L0 20 Z" fill="red"/></svg>'
)


def square():
    editor = Editor(import_svg(SQUARE), selection=select("p"))
    return editor, editor.snapshot.document.geometry_for("p")


def with_nodes(geometry, nodes):
    subpath = replace(geometry.subpaths[0], nodes=tuple(nodes))
    return replace(geometry, subpaths=(subpath,))


def test_moving_and_inserting_nodes_keeps_the_others_ids():
    editor, geometry = square()
    m, a, b, c = geometry.subpaths[0].nodes
    extra = PathNode("new-node", "L", (20, 10))
    moved = replace(b, values=(22.0, 22.0))
    with editor.transaction("Reshape") as tx:
        tx.reshape_path("p", with_nodes(geometry, [m, a, extra, moved, c]))
    after = editor.snapshot.document.geometry_for("p").subpaths[0].nodes
    assert [n.id for n in after] == [m.id, a.id, "new-node", b.id, c.id]
    assert after[3].values == (22.0, 22.0)
    assert editor.undo_labels == ("Reshape",)


def test_moving_needs_geometry_and_changing_the_node_count_needs_structure():
    editor, geometry = square()
    m, a, b, c = geometry.subpaths[0].nodes
    moved = with_nodes(geometry, [m, a, replace(b, values=(21.0, 21.0)), c])
    with editor.transaction("Move", allowed=frozenset({"geometry"})) as tx:
        tx.reshape_path("p", moved)
    fewer = with_nodes(editor.snapshot.document.geometry_for("p"), [m, a, c])
    tx = editor.transaction("Remove", allowed=frozenset({"geometry"}))
    with pytest.raises(EditRejectedError, match="structure"):
        tx.reshape_path("p", fewer)


def test_pinned_endpoints_can_neither_move_nor_go():
    editor, geometry = square()
    b = geometry.subpaths[0].nodes[2]
    editor.pin_node("p", b.id)
    geometry = editor.snapshot.document.geometry_for("p")
    m, a, b, c = geometry.subpaths[0].nodes
    for nodes in ([m, a, c], [m, a, replace(b, values=(25.0, 25.0)), c]):
        tx = editor.transaction("Reshape")
        with pytest.raises(EditRejectedError, match="pinned"):
            tx.reshape_path("p", with_nodes(geometry, nodes))


def test_linked_boundary_edges_must_come_through_unchanged():
    editor = linked_editor()
    editor.select(select("fill"))
    geometry = editor.snapshot.document.geometry_for("fill")
    m, curve, down, left = geometry.subpaths[0].nodes
    # The linked edge ends at the curve node: moving its end breaks the link,
    # moving a node elsewhere does not.
    tx = editor.transaction("Reshape")
    with pytest.raises(EditRejectedError, match="Linked boundary"):
        tx.reshape_path(
            "fill",
            with_nodes(
                geometry, [m, replace(curve, values=(16, 4, 40, 4, 50, 8)), down, left]
            ),
        )
    with editor.transaction("Reshape") as tx:
        tx.reshape_path(
            "fill",
            with_nodes(geometry, [m, curve, down, replace(left, values=(6.0, 42.0))]),
        )


LINE = (
    '<svg width="40" height="40">'
    '<path id="p" d="M0 0 L10 0 L20 10" fill="none" stroke="black"/></svg>'
)


def middle(editor):
    return editor.snapshot.document.geometry_for("p").subpaths[0].nodes


def handles(editor, count):
    nodes = middle(editor)
    with editor.transaction("Handles") as tx:
        tx.set_node_handles("p", nodes[1].id, count)
    return middle(editor)


def test_two_handles_make_a_smooth_point_and_none_make_a_corner_again():
    editor = Editor(import_svg(LINE), selection=select("p"))
    ids = [n.id for n in middle(editor)]
    start, point, end = handles(editor, 2)
    assert [n.id for n in (start, point, end)] == ids
    assert (point.command, end.command) == ("C", "C")
    incoming, outgoing = point.values[2:4], end.values[0:2]
    # In line through the point, on opposite sides of it.
    cross = (incoming[0] - 10) * (outgoing[1] - 0) - (incoming[1] - 0) * (
        outgoing[0] - 10
    )
    assert abs(cross) < 1e-9
    assert (incoming[0] - 10) * (outgoing[0] - 10) < 0
    assert point.endpoint == (10.0, 0.0)
    _, point, end = handles(editor, 0)
    assert (point.command, end.command) == ("L", "L")
    assert [n.id for n in middle(editor)] == ids


def test_one_handle_curves_the_way_in_and_switches_sides_when_asked_again():
    editor = Editor(import_svg(LINE), selection=select("p"))
    _, point, end = handles(editor, 1)
    assert (point.command, end.command) == ("C", "L")
    _, point, end = handles(editor, 1)
    assert (point.command, end.command) == ("L", "C")


def test_handles_respect_permissions_and_ask_for_a_segment():
    editor = Editor(import_svg(LINE), selection=select("p"))
    tx = editor.transaction("Handles", allowed=frozenset({"paint"}))
    with pytest.raises(EditRejectedError, match="not permitted"):
        tx.set_node_handles("p", middle(editor)[1].id, 2)
    lonely = Editor(
        import_svg('<svg width="9" height="9"><path id="p" d="M1 1"/></svg>'),
        selection=select("p"),
    )
    tx = lonely.transaction("Handles")
    with pytest.raises(EditRejectedError, match="no segment"):
        tx.set_node_handles("p", middle(lonely)[0].id, 2)
