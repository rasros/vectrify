"""Reshaping a path keeps the identity of every node it does not remove."""

from dataclasses import replace

import pytest

from tests.document.test_document import select
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


@pytest.mark.parametrize("position", [0, -1])
@pytest.mark.parametrize("explicit_close", [False, True])
def test_closed_path_endpoints_change_both_sides_of_the_join(position, explicit_close):
    path = "M0 0 L20 0 L20 20 L0 20" + (" L0 0" if explicit_close else "")
    editor = Editor(
        import_svg(f'<svg><path id="p" d="{path} Z"/></svg>'),
        selection=select("p"),
    )
    original = middle(editor)
    target = original[position]
    seam = position == 0 or explicit_close

    def change(count):
        with editor.transaction("Handles") as tx:
            tx.set_node_handles("p", target.id, count)
        nodes = middle(editor)
        incoming = nodes[-1] if seam else nodes[-2]
        outgoing = nodes[1] if seam else nodes[-1]
        return nodes, incoming, outgoing

    nodes, incoming, outgoing = change(2)
    assert nodes[-1].endpoint == nodes[0].endpoint
    assert [n.id for n in nodes[: len(original)]] == [n.id for n in original]
    assert incoming.command == outgoing.command == "C"
    assert incoming.values[2:4] != target.endpoint
    assert outgoing.values[:2] != target.endpoint
    assert incoming.endpoint == target.endpoint
    # Removing handles at either representation removes both visible handles.
    _, incoming, outgoing = change(0)
    assert incoming.command == outgoing.command == "L"
    _, incoming, outgoing = change(1)
    assert (incoming.command, outgoing.command) == ("C", "L")
    _, incoming, outgoing = change(1)
    assert (incoming.command, outgoing.command) == ("L", "C")


@pytest.mark.parametrize("position", [0, -1])
def test_open_endpoints_keep_their_only_handle_when_one_is_requested_again(position):
    editor = Editor(import_svg(LINE), selection=select("p"))
    target = middle(editor)[position]
    for count in (1, 1, 2):
        with editor.transaction("Handles") as tx:
            tx.set_node_handles("p", target.id, count)
        nodes = middle(editor)
        segment = nodes[1] if position == 0 else nodes[-1]
        offset = 0 if position == 0 else 2
        assert segment.command == "C"
        assert segment.values[offset : offset + 2] != target.endpoint
        assert len(nodes) == 3
    with editor.transaction("Handles") as tx:
        tx.set_node_handles("p", target.id, 0)
    assert all(n.command != "C" for n in middle(editor))


@pytest.mark.parametrize("position", [0, -1])
def test_removing_handles_at_a_curved_join_preserves_the_neighbours_handles(position):
    editor = Editor(
        import_svg(
            '<svg><path id="p" d="M0 0 C5 -5 25 -5 20 0 C25 5 5 5 0 0 Z"/></svg>'
        ),
        selection=select("p"),
    )
    original = middle(editor)
    with editor.transaction("Handles") as tx:
        tx.set_node_handles("p", original[position].id, 0)
    start, outgoing, incoming = middle(editor)
    assert outgoing.values[:2] == incoming.values[2:4] == start.endpoint
    assert outgoing.values[2:4] == original[1].values[2:4]
    assert incoming.values[:2] == original[-1].values[:2]
    editor.undo()
    assert middle(editor) == original


def test_a_two_point_closed_path_uses_its_neighbour_for_the_handle_direction():
    editor = Editor(
        import_svg('<svg><path id="p" d="M0 0 L20 0 Z"/></svg>'),
        selection=select("p"),
    )
    with editor.transaction("Handles") as tx:
        tx.set_node_handles("p", middle(editor)[0].id, 2)
    start, outgoing, incoming = middle(editor)
    assert incoming.values[2:4] != start.endpoint
    assert outgoing.values[:2] != start.endpoint


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


WAVE = (
    '<svg width="40" height="40">'
    '<path id="p" d="M0 0 C0 10 10 10 10 0 C10 -10 20 -10 20 0" fill="none"/></svg>'
)


@pytest.mark.parametrize("count", [0, 1])
def test_moving_a_point_keeps_its_retracted_handles_on_it(count):
    editor = Editor(import_svg(WAVE), selection=select("p"))
    _, point, end = handles(editor, count)
    before = (point.values[2:4] != point.endpoint) + (end.values[0:2] != point.endpoint)
    assert before == count
    # A drag sends the point's own values with only the endpoint moved.
    with editor.transaction("Move") as tx:
        tx.update_node("p", point.id, (*point.values[:4], 12.0, 2.0))
    _, point, end = middle(editor)
    assert point.endpoint == (12.0, 2.0)
    after = (point.values[2:4] != point.endpoint) + (end.values[0:2] != point.endpoint)
    assert after == count


def test_moving_a_point_takes_both_of_its_handles_along():
    editor = Editor(import_svg(WAVE), selection=select("p"))
    start, point, _ = middle(editor)
    with editor.transaction("Move") as tx:
        tx.update_node("p", point.id, (*point.values[:4], 13.0, 4.0))
    after = middle(editor)
    assert after[1].values == (0, 10, 13, 14, 13, 4)
    assert after[2].values == (13, -6, 20, -10, 20, 0)
    assert after[0] == start


def test_moving_a_handle_moves_only_that_handle():
    editor = Editor(import_svg(WAVE), selection=select("p"))
    _, point, end = middle(editor)
    with editor.transaction("Handle") as tx:
        tx.update_node("p", end.id, (15.0, -12.0, *end.values[2:]))
    with editor.transaction("Handle") as tx:
        tx.update_node("p", point.id, (0, 10, 7.0, 11.0, *point.endpoint))
    after = middle(editor)
    assert after[1].values == (0, 10, 7, 11, 10, 0)
    assert after[2].values == (15, -12, 20, -10, 20, 0)
