"""Selective hole filling preserves remaining curves and validates cleanup scope."""

import pytest

from tests.document.test_document import select
from tests.ui.test_session import send
from vectrify.document import DocumentError, Editor, EditRejectedError, import_svg
from vectrify.document.holes import (
    document_hole_shape,
    enclosed_objects,
    filled_region,
    find_holes,
)
from vectrify.document.join import path_style
from vectrify.ui.session import Session

OUTER = "M0 0H100V100H0Z"
SMALL = "M10 10V30H30V10Z"
LARGE = "M50 50V90H90V50Z"


def drawing(data=None, attrs="", shapes=""):
    return import_svg(
        '<svg width="200" height="200">'
        f'<path id="p" d="{data or OUTER + SMALL + LARGE}" {attrs}/>{shapes}</svg>'
    )


@pytest.mark.parametrize("rule", ["nonzero", "evenodd"])
def test_selective_fill_preserves_other_contours_and_undo(rule):
    doc = drawing(attrs=f'fill-rule="{rule}"')
    holes = find_holes(doc, "p")
    assert [h.shape.area for h in holes] == [400, 1600]
    before = doc.geometry_for("p")
    editor = Editor(doc, selection=select("p"))
    with editor.transaction("Fill holes") as tx:
        tx.fill_holes("p", frozenset({holes[0].id}))
    assert editor.snapshot.document.geometry_for("p").subpaths == (
        before.subpaths[0],
        before.subpaths[2],
    )
    assert len(find_holes(editor.snapshot.document, "p")) == 1
    assert editor.undo().document == doc
    assert len(find_holes(editor.redo().document, "p")) == 1


def test_winding_same_direction_is_not_a_hole_but_evenodd_is():
    data = OUTER + "M10 10H30V30H10Z"
    assert not find_holes(drawing(data), "p")
    assert len(find_holes(drawing(data, 'fill-rule="evenodd"'), "p")) == 1
    assert not find_holes(drawing(attrs='fill="none"'), "p")


def test_filling_parent_hole_removes_nested_island_contours():
    doc = drawing(OUTER + LARGE + "M60 60H80V80H60Z M65 65V75H75V65Z")
    holes = find_holes(doc, "p")
    assert len(holes) == 2
    editor = Editor(doc, selection=select("p"))
    with editor.transaction("Fill holes") as tx:
        tx.fill_holes("p", frozenset({holes[-1].id}))
    assert (
        editor.snapshot.document.geometry_for("p").subpaths
        == doc.geometry_for("p").subpaths[:1]
    )


@pytest.mark.parametrize("lock", ["geometry", "structure"])
def test_fill_obeys_locks(lock):
    editor = Editor(drawing(), selection=select("p"))
    editor.set_locks("p", frozenset({lock}))
    before = editor.snapshot
    hole = find_holes(before.document, "p")[0]
    with (
        pytest.raises(EditRejectedError, match="locked"),
        editor.transaction("Fill") as tx,
    ):
        tx.fill_holes("p", frozenset({hole.id}))
    assert editor.snapshot == before


def test_pinned_hole_cannot_be_removed():
    editor = Editor(drawing(), selection=select("p"))
    hole = find_holes(editor.snapshot.document, "p")[0]
    editor.pin_node(
        "p",
        editor.snapshot.document.geometry_for("p").subpaths[1].nodes[0].id,
        pinned=True,
    )
    before = editor.snapshot
    with (
        pytest.raises(EditRejectedError, match="Unpin"),
        editor.transaction("Fill") as tx,
    ):
        tx.fill_holes("p", frozenset({hole.id}))
    assert editor.snapshot == before


def test_only_wholly_enclosed_shapes_are_candidates_with_transforms():
    doc = drawing(
        attrs='transform="translate(10 20) scale(2)"',
        shapes="""
    <rect id="inside" x="35" y="45" width="10" height="10"/>
    <rect id="overlap" x="5" y="5" width="100" height="100"/>
    <rect id="outside" x="180" y="180" width="10" height="10"/>
    <rect id="wide-stroke" x="31" y="41" width="10" height="10"
          stroke="black" stroke-width="10"/>
    """,
    )
    holes = find_holes(doc, "p")
    assert document_hole_shape(doc, "p", (holes[0],)).area == 1600
    assert enclosed_objects(doc, "p", (holes[0],)) == frozenset({"inside"})


def test_fill_and_explicit_cleanup_are_one_undoable_action():
    session = Session(
        drawing(
            shapes='<rect id="inside" x="15" y="15" width="10" height="10"/>'
            '<rect id="keep" x="17" y="17" width="2" height="2"/>'
        )
    )
    send(session, "select", objects=["p"])
    before = session.editor.snapshot.document
    hole = find_holes(before, "p")[0]
    result = send(
        session, "fill_holes", object="p", holes=[hole.id], delete_objects=["inside"]
    )
    assert "inside" not in {e.id for e in session.editor.snapshot.document.elements()}
    assert "keep" in {e.id for e in session.editor.snapshot.document.elements()}
    assert result["undo"] == ["Fill holes"]
    send(session, "undo")
    assert session.editor.snapshot.document == before


def test_cleanup_failure_leaves_holes_and_objects_intact():
    session = Session(
        drawing(
            shapes='<rect id="outside" x="180" y="180" width="10" height="10"/>'
            '<rect id="locked" x="15" y="15" width="10" height="10"/>'
        )
    )
    send(session, "select", objects=["locked"])
    send(session, "locks", object="locked", locks=["structure"])
    send(session, "select", objects=["p"])
    before = session.editor.snapshot
    hole = find_holes(before.document, "p")[0]
    for cleanup, error in [("outside", "fully inside"), ("locked", "locked")]:
        with pytest.raises(DocumentError, match=error):
            send(
                session,
                "fill_holes",
                object="p",
                holes=[hole.id],
                delete_objects=[cleanup],
            )
        assert session.editor.snapshot == before
    with pytest.raises(DocumentError, match="Choose existing"):
        send(session, "fill_holes", object="p", holes=["missing"])
    assert session.editor.snapshot == before


@pytest.mark.parametrize("rule", ["nonzero", "evenodd"])
def test_coincident_and_intersecting_contours_are_not_offered_as_holes(rule):
    assert not find_holes(drawing(OUTER + SMALL + SMALL, f'fill-rule="{rule}"'), "p")
    assert not find_holes(
        drawing(OUTER + "M90 90V110H110V90Z", f'fill-rule="{rule}"'), "p"
    )


def test_fill_changes_only_pixels_inside_the_chosen_hole():
    import numpy as np

    from tests.document.test_document import render
    from vectrify.document import export_svg

    doc = drawing()
    editor = Editor(doc, selection=select("p"))
    hole = find_holes(doc, "p")[0]
    with editor.transaction("Fill") as tx:
        tx.fill_holes("p", frozenset({hole.id}))
    before, after = (
        render(export_svg(doc)),
        render(export_svg(editor.snapshot.document)),
    )
    changed = np.any(before != after, axis=2)
    assert changed[10:30, 10:30].all()
    changed[10:30, 10:30] = False
    assert not changed.any()


def test_self_touching_contours_keep_their_holes_editable():
    # Two same-winding lobes meet at (0, 0); the original curve is untouched.
    outer = "M0 0H100V100H0V0L-100 0V-100H0V0Z"
    doc = drawing(outer + SMALL)
    holes = find_holes(doc, "p")
    assert len(holes) == 1
    editor = Editor(doc, selection=select("p"))
    with editor.transaction("Fill") as tx:
        tx.fill_holes("p", frozenset({holes[0].id}))
    assert (
        editor.snapshot.document.geometry_for("p").subpaths
        == doc.geometry_for("p").subpaths[:1]
    )


def painted(document, object_id):
    style = path_style(document, document.element(object_id))
    return filled_region(document.geometry_for(object_id), style["fill-rule"])


@pytest.mark.parametrize("rule", ["nonzero", "evenodd"])
@pytest.mark.parametrize("inner", [SMALL, "M10 10H30V30H10Z"])
def test_cut_out_hole_adds_the_inner_contour_and_undo_restores(rule, inner):
    doc = drawing(OUTER, f'fill-rule="{rule}"', f'<path id="q" d="{inner}"/>')
    editor = Editor(doc, selection=select("p", "q"))
    with editor.transaction("Cut out as hole") as tx:
        assert tx.cut_out_hole(frozenset({"p", "q"})) == "p"
    after = editor.snapshot.document
    assert [e.id for e in after.elements() if e.tag == "path"] == ["p"]
    geometry = after.geometry_for("p")
    # The outer contour keeps its nodes; the hole is appended.
    assert geometry.subpaths[0] == doc.geometry_for("p").subpaths[0]
    assert len(geometry.subpaths) == 2
    assert painted(after, "p").area == pytest.approx(10000 - 400)
    assert len(find_holes(after, "p")) == 1
    assert editor.undo().document == doc


def test_cutting_out_a_ring_leaves_an_island_in_its_middle():
    doc = drawing(OUTER, shapes=f'<path id="q" d="{LARGE} M60 60H80V80H60Z"/>')
    editor = Editor(doc, selection=select("p", "q"))
    with editor.transaction("Cut out as hole") as tx:
        tx.cut_out_hole(frozenset({"p", "q"}))
    after = editor.snapshot.document
    assert len(after.geometry_for("p").subpaths) == 3
    assert painted(after, "p").area == pytest.approx(10000 - 1600 + 400)


def test_cut_out_hole_maps_between_groups_and_keeps_the_outer_paint():
    doc = drawing(
        OUTER,
        'fill="#123456" transform="translate(10 0)"',
        '<g transform="scale(2)"><path id="q" d="M10 10H20V20H10Z"/></g>',
    )
    editor = Editor(doc, selection=select("p", "q"))
    with editor.transaction("Cut out as hole") as tx:
        tx.cut_out_hole(frozenset({"p", "q"}))
    after = editor.snapshot.document
    assert after.element("p").get("fill") == "#123456"
    (hole,) = find_holes(after, "p")
    assert hole.shape.bounds == pytest.approx((10, 20, 30, 40))


def test_partly_overlapping_front_shape_cuts_the_back_one():
    doc = drawing(OUTER, shapes='<path id="q" d="M50 50H150V150H50Z"/>')
    editor = Editor(doc, selection=select("p", "q"))
    with editor.transaction("Cut out as hole") as tx:
        assert tx.cut_out_hole(frozenset({"p", "q"})) == "p"
    after = editor.snapshot.document
    assert "q" not in {e.id for e in after.elements()}
    assert painted(after, "p").area == pytest.approx(10000 - 2500)


def test_cut_out_hole_refuses_disjoint_locked_and_single_paths():
    far = drawing(OUTER, shapes='<path id="q" d="M150 150H190V190H150Z"/>')
    far_editor = Editor(far, selection=select("p", "q"))
    with (
        pytest.raises(EditRejectedError, match="do not overlap"),
        far_editor.transaction("Cut") as tx,
    ):
        tx.cut_out_hole(frozenset({"p", "q"}))
    doc = drawing(OUTER, shapes=f'<path id="q" d="{SMALL}"/>')
    editor = Editor(doc, selection=select("p", "q"))
    editor.set_locks("p", frozenset({"geometry"}))
    with (
        pytest.raises(EditRejectedError, match="locked"),
        editor.transaction("Cut") as tx,
    ):
        tx.cut_out_hole(frozenset({"p", "q"}))
    assert editor.snapshot.document.geometry_for("p") == doc.geometry_for("p")
    single = Editor(doc, selection=select("p"))
    with (
        pytest.raises(EditRejectedError, match="two paths"),
        single.transaction("Cut") as tx,
    ):
        tx.cut_out_hole(frozenset({"p"}))


def test_hole_to_shape_moves_the_contour_into_a_new_path_above():
    doc = drawing(attrs='fill="#abcdef"', shapes='<path id="top" d="M0 0H1V1Z"/>')
    small = find_holes(doc, "p")[0]
    editor = Editor(doc, selection=select("p"))
    with editor.transaction("Holes to shapes") as tx:
        (shape,) = tx.holes_to_shapes("p", frozenset({small.id}))
    after = editor.snapshot.document
    order = [e.id for e in after.elements() if e.tag == "path"]
    assert order == ["p", shape, "top"]
    assert after.element(shape).get("fill") == "#abcdef"
    assert painted(after, shape).area == pytest.approx(400)
    assert len(find_holes(after, "p")) == 1
    assert painted(after, "p").area == pytest.approx(10000 - 1600)
    assert editor.undo().document == doc


@pytest.mark.parametrize("rule", ["nonzero", "evenodd"])
def test_hole_to_shape_keeps_islands_open_in_the_new_shape(rule):
    island = "M60 60H80V80H60Z"
    doc = drawing(OUTER + LARGE + island, f'fill-rule="{rule}"')
    large = find_holes(doc, "p")[-1]
    editor = Editor(doc, selection=select("p"))
    with editor.transaction("Holes to shapes") as tx:
        (shape,) = tx.holes_to_shapes("p", frozenset({large.id}))
    after = editor.snapshot.document
    assert painted(after, shape).area == pytest.approx(1600 - 400)
    assert painted(after, "p").area == pytest.approx(10000)


def test_hole_to_shape_refuses_pinned_contours():
    doc = drawing()
    small = find_holes(doc, "p")[0]
    node = next(s for s in doc.geometry_for("p").subpaths if s.id == small.id).nodes[1]
    editor = Editor(doc, selection=select("p"))
    editor.pin_node("p", node.id, pinned=True)
    with (
        pytest.raises(EditRejectedError, match="Unpin"),
        editor.transaction("Holes to shapes") as tx,
    ):
        tx.holes_to_shapes("p", frozenset({small.id}))
