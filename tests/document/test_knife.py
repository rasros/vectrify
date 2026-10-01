"""Cut filled paths along a straight line into pieces that meet exactly."""

import pathops
import pytest

from tests.document.test_document import select
from vectrify.document import Editor, EditRejectedError, import_svg
from vectrify.document.contact import edges
from vectrify.document.join import curve_path

CIRCLE = (
    "M10 0C15.5 0 20 4.5 20 10C20 15.5 15.5 20 10 20"
    "C4.5 20 0 15.5 0 10C0 4.5 4.5 0 10 0Z"
)


def drawing(data="M0 0H10V10H0Z", attrs='fill="red"', wrap=("", "")):
    return import_svg(
        f'<svg width="100" height="100">{wrap[0]}'
        f'<path id="p" d="{data}" {attrs}/>{wrap[1]}</svg>'
    )


def cut(document, start, end, *ids):
    editor = Editor(document, selection=select(*(ids or ("p",))))
    with editor.transaction("Cut with knife") as tx:
        pieces = tx.cut_paths(start, end)
    return editor, pieces


def area(document, oid):
    return abs(curve_path(document.geometry_for(oid)).area)


def test_square_splits_into_two_rectangles_with_its_paint_and_place():
    document = import_svg(
        '<svg width="100" height="100"><path id="back" d="M0 0H1V1Z"/>'
        '<path id="p" d="M0 0H10V10H0Z" fill="red" opacity=".5"/>'
        '<path id="front" d="M0 0H1V1Z"/></svg>'
    )
    editor, pieces = cut(document, (3, -5), (3, 15))
    doc = editor.snapshot.document
    assert pieces[0] == "p"
    assert len(pieces) == 2
    assert [c.id for c in doc.root.children] == ["back", *pieces, "front"]
    assert sorted(area(doc, oid) for oid in pieces) == pytest.approx([30, 70])
    for oid in pieces:
        assert doc.element(oid).get("fill") == "red"
        assert doc.element(oid).get("opacity") == ".5"
        xs = {n.endpoint[0] for s in doc.geometry_for(oid).subpaths for n in s.nodes}
        assert xs in ({0.0, 3.0}, {3.0, 10.0})
    assert doc.geometry_for("p").id == document.geometry_for("p").id
    assert editor.snapshot.selection.object_ids == frozenset(pieces)


def test_circle_keeps_its_curves():
    editor, pieces = cut(drawing(CIRCLE), (-5, -5), (25, 25))
    doc = editor.snapshot.document
    for oid in pieces:
        commands = [n.command for s in doc.geometry_for(oid).subpaths for n in s.nodes]
        assert commands.count("C") >= 2
        assert commands.count("L") == 1
    assert sum(area(doc, oid) for oid in pieces) == pytest.approx(
        area(drawing(CIRCLE), "p")
    )


@pytest.mark.parametrize(
    ("start", "end"),
    [((5, -5), (5, 5)), ((5, -5), (5, -1)), ((-5, 20), (20, 20))],
)
def test_line_that_does_not_cross_leaves_the_path_alone(start, end):
    document = drawing()
    editor = Editor(document, selection=select("p"))
    with (
        pytest.raises(EditRejectedError, match="Drag the knife across"),
        editor.transaction("Cut with knife") as tx,
    ):
        tx.cut_paths(start, end)
    assert editor.snapshot.document == document


def test_only_crossed_paths_are_cut_and_unpainted_paths_are_skipped():
    document = import_svg(
        '<svg width="100" height="100"><path id="p" d="M0 0H10V10H0Z"/>'
        '<path id="q" d="M50 0H60V10H50Z"/>'
        '<path id="s" d="M0 5H10" fill="none" stroke="black"/>'
        '<path id="t" d="M0 6H10" fill="none"/></svg>'
    )
    editor, pieces = cut(document, (5, -5), (5, 15), "p", "q", "s", "t")
    doc = editor.snapshot.document
    assert len(pieces) == 4
    assert "q" not in pieces
    assert "t" not in pieces
    assert doc.geometry_for("q") == document.geometry_for("q")
    assert doc.geometry_for("t") == document.geometry_for("t")
    # The stroke comes apart where the knife crosses it.
    assert sorted(
        [n.endpoint for n in doc.geometry_for(oid).subpaths[0].nodes]
        for oid in pieces[2:]
    ) == [[(0, 5), (5, 5)], [(5, 5), (10, 5)]]


def test_transformed_path_is_cut_where_the_line_is_on_screen():
    document = drawing(
        wrap=('<g transform="translate(20 0)"><g transform="scale(2)">', "</g></g>")
    )
    # On screen the square spans x 20..40; x=25 is a quarter of the way in.
    editor, pieces = cut(document, (25, -5), (25, 30))
    doc = editor.snapshot.document
    assert sorted(area(doc, oid) for oid in pieces) == pytest.approx([25, 75])
    assert all(doc.ancestry(oid)[-2].get("transform") == "scale(2)" for oid in pieces)


@pytest.mark.parametrize("rule", ["nonzero", "evenodd"])
def test_path_with_a_hole_keeps_the_hole_open(rule):
    data = (
        "M0 0H10V10H0Z M3 3V7H7V3Z"
        if rule == "nonzero"
        else "M0 0H10V10H0Z M3 3H7V7H3Z"
    )
    editor, pieces = cut(drawing(data, f'fill-rule="{rule}"'), (5, -5), (5, 15))
    doc = editor.snapshot.document
    assert [area(doc, oid) for oid in pieces] == pytest.approx([42, 42])
    for oid in pieces:
        path = curve_path(doc.geometry_for(oid))
        assert not path.contains((5.5, 5))
        assert not path.contains((4.5, 5))


def seam(doc, oid, x):
    """The piece's straight edges along the vertical line at *x*."""
    return {
        tuple(sorted(e.points))
        for e in edges(doc, doc.geometry_for(oid).id)
        if len(e.points) == 2
        and e.points[0] != e.points[1]
        and all(p[0] == x for p in e.points)
    }


def test_both_pieces_meet_on_the_same_seam_nodes():
    editor, pieces = cut(drawing(CIRCLE), (10, -5), (10, 25))
    doc = editor.snapshot.document
    assert seam(doc, pieces[0], 10) == seam(doc, pieces[1], 10) == {((10, 0), (10, 20))}


def test_editing_a_seam_point_moves_only_its_piece():
    editor, pieces = cut(drawing(), (4, -5), (4, 15))
    doc = editor.snapshot.document
    other = doc.geometry_for(pieces[1])
    node = next(
        n
        for s in doc.geometry_for(pieces[0]).subpaths
        for n in s.nodes
        if n.endpoint[0] == 4
    )
    with editor.transaction("Edit seam") as tx:
        tx.update_node(pieces[0], node.id, (5, node.endpoint[1]))
    assert editor.snapshot.document.geometry_for(pieces[1]) == other


@pytest.mark.parametrize("data", ["M0 0H10V10H0Z", CIRCLE])
def test_join_merges_the_pieces_back_into_one_path(data):
    document = drawing(data)
    whole = abs(curve_path(document.geometry_for("p")).area)
    editor, pieces = cut(document, (4, -5), (4, 25))
    with editor.transaction("Join outlines") as tx:
        joined = tx.join_paths(frozenset(pieces))
    doc = editor.snapshot.document
    assert [e.id for e in doc.elements() if e.tag == "path"] == [joined]
    geometry = doc.geometry_for(joined)
    assert len(geometry.subpaths) == 1
    assert abs(curve_path(geometry).area) == pytest.approx(whole, rel=1e-6)
    assert dict(doc.element(joined).attributes)["fill"] == "red"


def test_undo_restores_the_original_path():
    document = drawing()
    editor, _ = cut(document, (5, -5), (5, 15))
    editor.undo()
    assert editor.snapshot.document == document
    assert editor.snapshot.selection == select("p")
    assert editor.undo_labels == ()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda e, _: e.set_locks("p", frozenset({"geometry"})), "locked"),
        (
            lambda e, d: e.pin_node("p", d.geometry_for("p").subpaths[0].nodes[0].id),
            "Unpin",
        ),
    ],
)
def test_locked_or_pinned_paths_are_refused(change, message):
    document = drawing()
    editor = Editor(document, selection=select("p"))
    change(editor, document)
    before = editor.snapshot.document
    with (
        pytest.raises(EditRejectedError, match=message),
        editor.transaction("Cut with knife") as tx,
    ):
        tx.cut_paths((5, -5), (5, 15))
    assert editor.snapshot.document == before


def test_shared_geometry_is_refused_but_pieces_cut_again():
    shared = import_svg(
        '<svg width="100" height="100"><path id="p" d="M0 0H10V10H0Z"/>'
        '<path id="q" d="M50 0H60V10H50Z"/></svg>'
    )
    editor = Editor(shared, selection=select("p", "q"))
    with editor.transaction("Share") as tx:
        tx.share_geometry("q", "p")
    with (
        pytest.raises(EditRejectedError, match="shared geometry"),
        editor.transaction("Cut with knife") as tx,
    ):
        tx.cut_paths((5, -5), (5, 15))
    editor, _ = cut(drawing(), (5, -5), (5, 15))
    with editor.transaction("Cut with knife") as tx:
        assert len(tx.cut_paths((2, -5), (2, 15))) == 2


def test_pieces_render_like_the_original():
    document = drawing(CIRCLE)
    editor, pieces = cut(document, (-5, 3), (25, 17))
    doc = editor.snapshot.document
    union = None
    for oid in pieces:
        path = curve_path(doc.geometry_for(oid))
        union = path if union is None else pathops.op(union, path, pathops.PathOp.UNION)
    assert union is not None
    difference = pathops.op(
        curve_path(document.geometry_for("p")), union, pathops.PathOp.XOR
    )
    assert abs(difference.area) < 1e-3
