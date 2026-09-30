"""Deleting points, start points included, and whole contours."""

import pytest

from vectrify.document import Editor, EditRejectedError, Selection, import_svg


def editor(d, *extra):
    paths = "".join(f'<path id="{i}" d="{p}"/>' for i, p in (("p", d), *extra))
    result = Editor(import_svg(f'<svg width="40" height="40">{paths}</svg>'))
    result.select(Selection(object_ids=frozenset({"p"})))
    return result


def nodes(e, contour=0):
    return e.snapshot.document.geometry_for("p").subpaths[contour].nodes


def delete(e, index, contour=0):
    node = nodes(e, contour)[index]
    with e.transaction("Delete node") as tx:
        tx.delete_node("p", node.id)
    return node


def shape(e, contour=0):
    return [(n.command, n.values) for n in nodes(e, contour)]


def test_deleting_an_open_contours_start_makes_the_next_point_the_start():
    e = editor("M0 0 L10 0 C13 3 17 3 20 0 L30 0")
    before = e.snapshot.document
    second = nodes(e)[1].id
    delete(e, 0)
    assert shape(e) == [
        ("M", (10.0, 0.0)),
        ("C", (13.0, 3.0, 17.0, 3.0, 20.0, 0.0)),
        ("L", (30.0, 0.0)),
    ]
    assert nodes(e)[0].id == second
    assert e.undo().document == before


def test_deleting_a_closed_contours_start_closes_through_its_neighbours():
    e = editor("M0 0 L10 0 L10 10 L0 10 Z")
    delete(e, 0)
    assert shape(e) == [("M", (10.0, 0.0)), ("L", (10.0, 10.0)), ("L", (0.0, 10.0))]
    assert e.snapshot.document.geometry_for("p").subpaths[0].closed


def test_a_curve_into_the_new_start_keeps_closing_the_contour():
    e = editor("M0 0 C3 -3 7 -3 10 0 L10 10 L0 10 Z")
    delete(e, 0)
    assert shape(e) == [
        ("M", (10.0, 0.0)),
        ("L", (10.0, 10.0)),
        ("L", (0.0, 10.0)),
        ("C", (3.0, -3.0, 7.0, -3.0, 10.0, 0.0)),
    ]


@pytest.mark.parametrize("index", [0, -1])
def test_a_contour_ending_on_its_start_deletes_both_nodes_of_that_point(index):
    e = editor("M0 0 L10 0 L10 10 L0 10 L0 0 Z")
    delete(e, index)
    assert shape(e) == [("M", (10.0, 0.0)), ("L", (10.0, 10.0)), ("L", (0.0, 10.0))]


def test_a_pinned_start_is_kept():
    e = editor("M0 0 L10 0 L20 0")
    e.pin_node("p", nodes(e)[0].id)
    before = e.snapshot
    with (
        pytest.raises(EditRejectedError, match="pinned"),
        e.transaction("Delete node") as tx,
    ):
        tx.delete_node("p", nodes(e)[0].id)
    assert e.snapshot == before


def test_an_open_contour_down_to_one_point_is_deleted_with_its_path():
    e = editor("M0 0 L10 0", ("q", "M20 20 L30 20 L30 30 Z"))
    before = e.snapshot.document
    delete(e, 0)
    doc = e.snapshot.document
    assert "p" not in {el.id for el in doc.elements()}
    assert e.snapshot.selection == Selection()
    assert e.undo().document == before


def test_a_closed_contour_down_to_two_points_is_deleted_but_others_stay():
    e = editor("M0 0 L30 0 L30 30 L0 30 Z M10 10 L12 10 L11 12 Z")
    delete(e, 1, contour=1)
    subpaths = e.snapshot.document.geometry_for("p").subpaths
    assert len(subpaths) == 1
    assert [n.endpoint for n in subpaths[0].nodes] == [
        (0, 0),
        (30, 0),
        (30, 30),
        (0, 30),
    ]


def test_deleting_the_last_point_of_a_speck_deletes_the_path():
    e = editor("M5 5 L5.1 5 L5.1 5.1 Z")
    delete(e, 2)
    assert "p" not in {el.id for el in e.snapshot.document.elements()}


def delete_contour(e, contour):
    with e.transaction("Delete contour") as tx:
        tx.delete_contour("p", nodes(e, contour)[1].id)


def test_delete_contour_removes_the_points_contour_then_the_path():
    e = editor("M0 0 L30 0 L30 30 L0 30 Z M10 10 L20 10 L20 20 L10 20 Z")
    before = e.snapshot.document
    delete_contour(e, 1)
    assert len(e.snapshot.document.geometry_for("p").subpaths) == 1
    delete_contour(e, 0)
    assert "p" not in {el.id for el in e.snapshot.document.elements()}
    e.undo()
    assert e.undo().document == before


def test_delete_contour_refuses_pinned_points_and_shared_geometry():
    e = editor("M0 0 L30 0 L30 30 Z")
    e.pin_node("p", nodes(e)[2].id)
    before = e.snapshot
    with (
        pytest.raises(EditRejectedError, match="Unpin"),
        e.transaction("Delete contour") as tx,
    ):
        tx.delete_contour("p", nodes(e)[1].id)
    assert e.snapshot == before
    shared = Editor(
        import_svg(
            '<svg><defs><path id="p" d="M0 0 L30 0 L30 30 Z"/></defs>'
            '<use id="u" href="#p"/></svg>'
        )
    )
    shared.select(Selection(object_ids=frozenset({"p"})))
    node = shared.snapshot.document.geometry_for("p").subpaths[0].nodes[1]
    with pytest.raises(EditRejectedError), shared.transaction("Delete") as tx:
        tx.delete_contour("p", node.id)
