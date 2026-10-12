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
U_SHAPE = "M0 0H10V10H7V3H3V10H0Z"


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


@pytest.mark.parametrize("reverse", [False, True])
def test_short_cut_across_one_arm_does_not_cut_the_other_arm(reverse):
    document = drawing(U_SHAPE)
    start, end = (-1, 6), (4, 6)
    if reverse:
        start, end = end, start
    editor, pieces = cut(document, start, end)
    doc = editor.snapshot.document
    assert len(pieces) == 2
    assert sorted(area(doc, oid) for oid in pieces) == pytest.approx([12, 60])
    main = curve_path(doc.geometry_for("p"))
    assert main.contains((8, 4))
    assert main.contains((8, 8))
    cut_edges = {
        tuple(sorted(e.points))
        for oid in pieces
        for e in edges(doc, doc.geometry_for(oid).id)
        if len(e.points) == 2 and all(p[1] == 6 for p in e.points)
    }
    assert cut_edges == {((0, 6), (3, 6))}
    assert editor.undo().document == document
    assert editor.redo().document == doc


def test_short_cut_leaves_contours_beyond_both_endpoints_attached_to_original():
    document = drawing("M0 0H10V10H0Z M0 20H10V30H0Z M0 40H10V50H0Z")
    editor, pieces = cut(document, (5, 19), (5, 31))
    doc = editor.snapshot.document
    assert len(pieces) == 2
    assert sorted(area(doc, oid) for oid in pieces) == pytest.approx([50, 250])
    main = curve_path(doc.geometry_for("p"))
    for point in ((2, 5), (8, 5), (2, 45), (8, 45)):
        assert main.contains(point)
    assert seam(doc, "p", 5) == {((5, 20), (5, 30))}


def test_diagonal_short_cut_does_not_slice_a_second_arm():
    document = drawing("M0 0L10 -10L20 0L17 3L10 -4L6 0L13 7L10 10Z")
    editor, pieces = cut(document, (5, 7), (10, 2))
    doc = editor.snapshot.document
    assert sorted(area(doc, oid) for oid in pieces) == pytest.approx([24, 120])
    assert curve_path(doc.geometry_for("p")).contains((16, 0))
    cut_edges = {
        tuple(sorted(e.points))
        for oid in pieces
        for e in edges(doc, doc.geometry_for(oid).id)
        if len(e.points) == 2 and all(abs(p[0] + p[1] - 12) < 1e-9 for p in e.points)
    }
    assert len(cut_edges) == 1
    first, second = next(iter(cut_edges))
    assert first == pytest.approx((6, 6))
    assert second == pytest.approx((9, 3))


def test_partial_cut_can_detach_several_prongs_and_preserve_their_pins():
    document = drawing("M0 0H14V10H12V3H8V10H6V3H2V10H0Z")
    editor = Editor(document, selection=select("p"))
    nodes = document.geometry_for("p").subpaths[0].nodes
    editor.pin_nodes(("p", n.id) for n in nodes)
    with editor.transaction("Cut with knife") as tx:
        pieces = tx.cut_paths((-1, 6), (9, 6))
    doc = editor.snapshot.document
    assert len(pieces) == 3
    assert sorted(area(doc, oid) for oid in pieces) == pytest.approx([8, 8, 68])
    assert curve_path(doc.geometry_for("p")).contains((13, 8))
    surviving = {
        n.id: n
        for oid in pieces
        for s in doc.geometry_for(oid).subpaths
        for n in s.nodes
    }
    for node in nodes:
        assert surviving[node.id].pinned
        assert surviving[node.id].endpoint == node.endpoint
    assert editor.snapshot.selection.object_ids == frozenset(pieces)


def test_short_cut_reconnects_curved_contours_beyond_its_end():
    document = drawing(f"M0 30H20V40H0Z {CIRCLE}")
    editor, pieces = cut(document, (5, 29), (5, 41))
    doc = editor.snapshot.document
    original_circle = curve_path(drawing(CIRCLE).geometry_for("p"))
    main = curve_path(doc.geometry_for("p"))
    circle = pathops.op(main, original_circle, pathops.PathOp.INTERSECTION)
    assert circle.area == pytest.approx(original_circle.area)
    assert seam(doc, "p", 5) == {((5, 30), (5, 40))}
    union = pathops.op(
        *(curve_path(doc.geometry_for(oid)) for oid in pieces), pathops.PathOp.UNION
    )
    difference = pathops.op(
        curve_path(document.geometry_for("p")), union, pathops.PathOp.XOR
    )
    assert abs(difference.area) < 1e-3


def test_cut_from_hole_to_outside_that_does_not_separate_material_is_refused():
    document = drawing("M0 0H10V10H0Z M3 3V7H7V3Z")
    with pytest.raises(EditRejectedError, match="Drag the knife across"):
        cut(document, (5, 5), (11, 5))


def test_short_cut_can_split_an_island_inside_a_hole_without_cutting_its_surround():
    document = drawing("M0 0H20V20H0Z M3 3V17H17V3Z M7 7H13V13H7Z")
    editor, pieces = cut(document, (10, 6), (10, 14))
    doc = editor.snapshot.document
    assert len(pieces) == 2
    assert sorted(area(doc, oid) for oid in pieces) == pytest.approx([18, 222])
    main = curve_path(doc.geometry_for("p"))
    assert main.contains((1, 1))
    assert main.contains((19, 19))
    assert not main.contains((5, 5))
    assert seam(doc, "p", 10) == {((10, 7), (10, 13))}


def test_short_cut_separates_material_that_only_touches_at_a_hole_vertex():
    document = drawing("M0 0H10V10H0Z M3 3V7L10 5Z")
    editor, pieces = cut(document, (5, 4), (11, 4))
    doc = editor.snapshot.document
    assert len(pieces) == 2
    assert sorted(area(doc, oid) for oid in pieces) == pytest.approx([1.75, 84.25])


def test_a_nonseparating_cut_through_a_hole_is_reconnected_while_another_region_cuts():
    document = drawing("M0 0H10V10H0Z M3 3V7H7V3Z M11 4H12V6H11Z")
    editor, pieces = cut(document, (5, 5), (13, 5))
    doc = editor.snapshot.document
    assert len(pieces) == 2
    assert sorted(area(doc, oid) for oid in pieces) == pytest.approx([1, 85])
    main = curve_path(doc.geometry_for("p"))
    assert not main.contains((5, 5))
    assert main.contains((8, 4))
    assert main.contains((8, 6))
    cut_edges = {
        tuple(sorted(e.points))
        for e in edges(doc, doc.geometry_for("p").id)
        if len(e.points) == 2 and all(p[1] == 5 for p in e.points)
    }
    assert cut_edges == {((11, 5), (12, 5))}


def test_transformed_path_is_cut_where_the_line_is_on_screen():
    document = drawing(
        wrap=('<g transform="translate(20 0)"><g transform="scale(2)">', "</g></g>")
    )
    # On screen the square spans x 20..40; x=25 is a quarter of the way in.
    editor, pieces = cut(document, (25, -5), (25, 30))
    doc = editor.snapshot.document
    assert sorted(area(doc, oid) for oid in pieces) == pytest.approx([25, 75])
    assert all(doc.ancestry(oid)[-2].get("transform") == "scale(2)" for oid in pieces)


def test_short_cut_respects_its_endpoints_in_transformed_user_space():
    document = drawing(
        U_SHAPE, wrap=('<g transform="translate(20 0) scale(2)">', "</g>")
    )
    editor, pieces = cut(document, (18, 12), (28, 12))
    doc = editor.snapshot.document
    assert sorted(area(doc, oid) for oid in pieces) == pytest.approx([12, 60])
    assert curve_path(doc.geometry_for("p")).contains((8, 8))


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


@pytest.mark.parametrize("lock", ["geometry", "structure"])
def test_locked_paths_are_refused(lock):
    document = drawing()
    editor = Editor(document, selection=select("p"))
    editor.set_locks("p", frozenset({lock}))
    before = editor.snapshot.document
    with (
        pytest.raises(EditRejectedError, match="locked"),
        editor.transaction("Cut with knife") as tx,
    ):
        tx.cut_paths((5, -5), (5, 15))
    assert editor.snapshot.document == before


@pytest.mark.parametrize(
    "data",
    [
        "M0 0H10V10H0Z",
        "M0 0H2H10V10H2H0V2Z",  # Collinear nodes, including the closing edge.
        "M0.123456789 0.234567891H10.123456789V10.234567891H0.123456789Z",
        CIRCLE,
        "M0 0H10V10H0Z M1 1V3H3V1Z",  # An untouched hole.
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_cut_preserves_pinned_endpoint_positions_and_identities(data, reverse):
    editor = Editor(drawing(data), selection=select("p"))
    nodes = [
        n for s in editor.snapshot.document.geometry_for("p").subpaths for n in s.nodes
    ]
    editor.pin_nodes(("p", n.id) for n in nodes)
    before = editor.snapshot.document
    start, end = (5, -5), (5, 25)
    if reverse:
        start, end = end, start
    with editor.transaction("Cut with knife") as tx:
        pieces = tx.cut_paths(start, end)
    after = editor.snapshot.document
    surviving = {
        n.id: (oid, n)
        for oid in pieces
        for s in after.geometry_for(oid).subpaths
        for n in s.nodes
    }
    for old in nodes:
        oid, node = surviving[old.id]
        assert node.pinned
        assert node.endpoint == old.endpoint
        with (
            pytest.raises(EditRejectedError, match="pinned"),
            editor.transaction("Move pinned endpoint") as tx,
        ):
            tx.update_node(oid, node.id, (*node.values[:-2], 100, 100))
    assert sum(area(after, oid) for oid in pieces) == pytest.approx(area(before, "p"))
    after.validate()
    assert editor.undo().document == before
    assert editor.redo().document == after


def test_pin_on_a_subdivided_cubic_survives_cutting_elsewhere():
    editor = Editor(drawing(CIRCLE), selection=select("p"))
    curve = editor.snapshot.document.geometry_for("p").subpaths[0].nodes[1]
    with editor.transaction("Subdivide curve") as tx:
        added = tx.split_edge("p", curve.id, 0.37)
    editor.pin_node("p", added[0])
    before = editor.snapshot.document.geometry_for("p").node(added[0])
    with editor.transaction("Cut with knife") as tx:
        pieces = tx.cut_paths((5, -5), (5, 25))
    surviving = [
        n
        for oid in pieces
        for s in editor.snapshot.document.geometry_for(oid).subpaths
        for n in s.nodes
        if n.id == before.id
    ]
    assert len(surviving) == 1
    assert surviving[0].pinned
    assert surviving[0].endpoint == before.endpoint


def test_cut_that_would_remove_an_interior_pin_is_refused():
    # The boolean removes a redundant contour inside the filled square.
    document = drawing("M0 0H10V10H0Z M1 1H3V3H1Z")
    editor = Editor(document, selection=select("p"))
    pin = document.geometry_for("p").subpaths[1].nodes[0]
    editor.pin_node("p", pin.id)
    before = editor.snapshot.document
    with (
        pytest.raises(EditRejectedError, match="pinned endpoints would be removed"),
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
