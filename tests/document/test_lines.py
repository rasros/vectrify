"""Break, cut, join and outline drawn lines."""

import pytest

from tests.document.test_document import select
from vectrify.document import Editor, EditRejectedError, import_svg
from vectrify.document.join import curve_path
from vectrify.document.lines import stroke_outline

STROKE = 'fill="none" stroke="black" stroke-width="2"'


def drawing(*paths):
    return import_svg(
        '<svg width="100" height="100">'
        + "".join(f'<path id="{oid}" d="{d}" {STROKE}/>' for oid, d in paths)
        + "</svg>"
    )


def contours(document, oid):
    return [
        ([n.endpoint for n in s.nodes], s.closed)
        for s in document.geometry_for(oid).subpaths
    ]


def node_ids(document, oid):
    return [[n.id for n in s.nodes] for s in document.geometry_for(oid).subpaths]


def edit(document, label, *ids):
    editor = Editor(document, selection=select(*ids))
    return editor, editor.transaction(label)


def test_breaking_an_open_line_gives_each_piece_its_own_copy_of_the_point():
    document = drawing(("p", "M0 0L10 0L20 0L30 0"))
    middle = node_ids(document, "p")[0][2]
    editor, tx = edit(document, "Break", "p")
    with tx:
        tx.break_points("p", [middle])
    doc = editor.snapshot.document
    assert contours(doc, "p") == [
        ([(0, 0), (10, 0), (20, 0)], False),
        ([(20, 0), (30, 0)], False),
    ]
    first, second = node_ids(doc, "p")
    assert first[-1] == middle
    assert second[0] != middle


def test_breaking_a_closed_contour_opens_it_at_the_point():
    document = drawing(("p", "M0 0L10 0L10 10L0 10Z"))
    corner = node_ids(document, "p")[0][2]
    editor, tx = edit(document, "Break", "p")
    with tx:
        tx.break_points("p", [corner])
    doc = editor.snapshot.document
    assert contours(doc, "p") == [
        ([(10, 10), (0, 10), (0, 0), (10, 0), (10, 10)], False)
    ]
    assert node_ids(doc, "p")[0][-1] == corner


def test_breaking_a_closed_contour_twice_gives_two_lines():
    document = drawing(("p", "M0 0L10 0L10 10L0 10Z"))
    ids = node_ids(document, "p")[0]
    editor, tx = edit(document, "Break", "p")
    with tx:
        tx.break_points("p", [ids[1], ids[3]])
    assert contours(editor.snapshot.document, "p") == [
        ([(10, 0), (10, 10), (0, 10)], False),
        ([(0, 10), (0, 0), (10, 0)], False),
    ]


def test_an_open_lines_end_is_already_free():
    document = drawing(("p", "M0 0L10 0"))
    _, tx = edit(document, "Break", "p")
    with pytest.raises(EditRejectedError, match="free already"), tx:
        tx.break_points("p", [node_ids(document, "p")[0][0]])


def test_deleting_a_segment_splits_a_line_and_opens_a_closed_contour():
    document = drawing(("p", "M0 0L10 0L20 0L30 0"), ("q", "M0 0L10 0L10 10Z"))
    line, loop = node_ids(document, "p")[0], node_ids(document, "q")[0]
    editor, tx = edit(document, "Delete segment", "p", "q")
    with tx:
        tx.delete_segments("p", frozenset(line[1:3]))
        # The closing line, from the last point back to the start.
        tx.delete_segments("q", frozenset({loop[0], loop[2]}))
    doc = editor.snapshot.document
    assert contours(doc, "p") == [
        ([(0, 0), (10, 0)], False),
        ([(20, 0), (30, 0)], False),
    ]
    assert contours(doc, "q") == [([(0, 0), (10, 0), (10, 10)], False)]


def test_deleting_the_segments_of_a_loop_drops_its_lone_points():
    document = drawing(("p", "M0 0L10 0L20 0"))
    ids = node_ids(document, "p")[0]
    editor, tx = edit(document, "Delete segment", "p")
    with tx:
        tx.delete_segments("p", frozenset(ids[:2]))
    assert contours(editor.snapshot.document, "p") == [([(10, 0), (20, 0)], False)]


def test_points_that_are_not_neighbours_have_no_segment_between():
    document = drawing(("p", "M0 0L10 0L20 0"))
    ids = node_ids(document, "p")[0]
    _, tx = edit(document, "Delete segment", "p")
    with pytest.raises(EditRejectedError, match="two points"), tx:
        tx.delete_segments("p", frozenset({ids[0], ids[2]}))


def dashes(gap):
    return drawing(
        ("a", "M0 0L10 0"), ("b", f"M{10 + gap} 0L{20 + gap} 0"), ("c", "M0 50L10 50")
    )


def test_dashes_within_reach_join_into_one_line_in_the_front_path():
    editor, tx = edit(dashes(2), "Join ends", "a", "b", "c")
    with tx:
        joined = tx.join_ends(frozenset({"a", "b", "c"}), 3, curve=False)
    doc = editor.snapshot.document
    assert joined == ("b",)
    assert [e.id for e in doc.root.children] == ["b", "c"]
    assert contours(doc, "b") == [([(0, 0), (10, 0), (12, 0), (22, 0)], False)]
    assert contours(doc, "c") == [([(0, 50), (10, 50)], False)]


def test_dashes_beyond_reach_stay_apart():
    _, tx = edit(dashes(5), "Join ends", "a", "b")
    with pytest.raises(EditRejectedError, match="close enough"), tx:
        tx.join_ends(frozenset({"a", "b"}), 3)


def test_ends_side_by_side_do_not_join():
    # Two parallel lines whose ends are near each other are not one line.
    document = drawing(("a", "M0 0L10 0"), ("b", "M0 2L10 2"))
    _, tx = edit(document, "Join ends", "a", "b")
    with pytest.raises(EditRejectedError, match="close enough"), tx:
        tx.join_ends(frozenset({"a", "b"}), 3)


def test_a_curved_bridge_leaves_each_end_along_its_line():
    editor, tx = edit(dashes(3), "Join ends", "a", "b")
    with tx:
        tx.join_ends(frozenset({"a", "b"}), 5)
    (subpath,) = editor.snapshot.document.geometry_for("b").subpaths
    bridge = subpath.nodes[2]
    assert bridge.command == "C"
    assert bridge.values == pytest.approx((11, 0, 12, 0, 13, 0))


def test_ends_meeting_at_a_junction_join_the_straightest_way():
    document = drawing(("p", "M0 0L10 0 M10 0L20 0 M10 0L10 10"))
    editor, tx = edit(document, "Join ends", "p")
    with tx:
        tx.join_ends(frozenset({"p"}), 1)
    assert sorted(contours(editor.snapshot.document, "p")) == [
        ([(0, 0), (10, 0), (20, 0)], False),
        ([(10, 0), (10, 10)], False),
    ]


def test_a_dashed_ring_closes():
    document = drawing(("p", "M2 0L10 0L10 9 M10 11L10 20L0 20L0 0L1 0"))
    editor, tx = edit(document, "Join ends", "p")
    with tx:
        tx.join_ends(frozenset({"p"}), 3, curve=False)
    ((points, closed),) = contours(editor.snapshot.document, "p")
    assert closed
    assert points[0] == points[-1]
    assert len(points) == 9


def test_two_chosen_ends_join_however_far_apart():
    document = drawing(("a", "M0 0L10 0"), ("b", "M40 30L50 30"))
    a, b = node_ids(document, "a")[0], node_ids(document, "b")[0]
    editor, tx = edit(document, "Join ends", "a", "b")
    with tx:
        tx.join_ends(frozenset({"a", "b"}), ends=[("a", a[-1]), ("b", b[0])])
    doc = editor.snapshot.document
    assert [e.id for e in doc.root.children] == ["b"]
    assert contours(doc, "b")[0][0] == [(0, 0), (10, 0), (40, 30), (50, 30)]


def test_two_chosen_ends_must_be_free_ends():
    document = drawing(("a", "M0 0L10 0L20 0"))
    ids = node_ids(document, "a")[0]
    _, tx = edit(document, "Join ends", "a")
    with pytest.raises(EditRejectedError, match="two points"), tx:
        tx.join_ends(frozenset({"a"}), ends=[("a", ids[0]), ("a", ids[1])])


def test_the_knife_cuts_an_open_line_where_it_crosses():
    document = drawing(("p", "M0 0L30 0"))
    editor = Editor(document, selection=select("p"))
    with editor.transaction("Cut with knife") as tx:
        pieces = tx.cut_paths((10, -5), (10, 5))
    doc = editor.snapshot.document
    assert len(pieces) == 2
    assert sorted(contours(doc, oid)[0][0] for oid in pieces) == [
        [(0, 0), (10, 0)],
        [(10, 0), (30, 0)],
    ]
    for oid in pieces:
        assert doc.element(oid).get("stroke") == "black"
    # The longer piece keeps the path; the shorter one comes away.
    assert contours(doc, "p") == [([(10, 0), (30, 0)], False)]


def test_the_knife_cuts_a_loop_away_from_its_line():
    # A line that loops round on itself: the knife across the loop takes it off.
    document = drawing(("p", "M0 0L20 0L20 -10L10 -10L10 10"))
    editor = Editor(document, selection=select("p"))
    with editor.transaction("Cut with knife") as tx:
        pieces = tx.cut_paths((5, -5), (25, -5))
    doc = editor.snapshot.document
    kept, cut = pieces
    assert kept == "p"
    assert contours(doc, cut) == [([(20, -5), (20, -10), (10, -10), (10, -5)], False)]
    assert sorted(contours(doc, "p")) == [
        ([(0, 0), (20, 0), (20, -5)], False),
        ([(10, -5), (10, 10)], False),
    ]


def test_the_knife_opens_a_closed_line_it_crosses_once():
    document = drawing(("p", "M0 0L10 0L10 10L0 10Z"))
    editor = Editor(document, selection=select("p"))
    with editor.transaction("Cut with knife") as tx:
        pieces = tx.cut_paths((5, -5), (5, 5))
    ((points, closed),) = contours(editor.snapshot.document, "p")
    assert pieces == ("p",)
    assert not closed
    assert points[0] == points[-1] == (5, 0)


def test_a_stroke_outline_covers_the_width_along_the_line():
    document = drawing(("p", "M0 0L10 0"))
    geometry = document.geometry_for("p")
    style = {"stroke-width": "2", "stroke-linecap": "butt"}
    assert abs(curve_path(stroke_outline(geometry, style)).area) == pytest.approx(20)
    style["stroke-linecap"] = "round"
    assert abs(curve_path(stroke_outline(geometry, style)).area) == pytest.approx(
        20 + 3.14159, rel=1e-2
    )
