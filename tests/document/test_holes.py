"""Selective hole filling preserves remaining curves and validates cleanup scope."""

import pytest

from tests.document.test_document import select
from tests.ui.test_session import send
from vectrify.document import DocumentError, Editor, EditRejectedError, import_svg
from vectrify.document.holes import document_hole_shape, enclosed_objects, find_holes
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
