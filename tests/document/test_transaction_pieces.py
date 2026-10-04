"""Paths created by structural edits remain usable in the same transaction."""

import pytest

from tests.document.test_document import select
from vectrify.document import DocumentError, Editor, EditRejectedError, import_svg
from vectrify.document.holes import find_holes


@pytest.mark.parametrize("command", ["knife", "region", "hole"])
def test_generated_pieces_can_be_edited_in_the_same_transaction(command):
    data = {
        "knife": "M0 0H10V10H0Z",
        "region": "M0 0H10V10H0Z M20 0H30V10H20Z",
        "hole": "M0 0H100V100H0Z M10 10V30H30V10Z",
    }[command]
    document = import_svg(
        f'<svg width="100" height="100"><g id="g">'
        f'<path id="p" fill="red" d="{data}"/>'
        '<path id="front" d="M0 0L1 1"/></g></svg>'
    )
    editor = Editor(document, selection=select("p"))
    with editor.transaction("Create and paint piece") as tx:
        if command == "knife":
            _, piece = tx.cut_paths((5, -5), (5, 15))
        elif command == "region":
            ((_, piece),) = tx.extract_region(
                [(19, -1), (31, -1), (31, 11), (19, 11)], cut=False
            )
            assert piece is not None
        else:
            (piece,) = tx.holes_to_shapes(
                "p", frozenset({find_holes(document, "p")[0].id})
            )
        assert piece in tx.scope
        tx.set_attributes(piece, {"fill": "blue"})
    after = editor.snapshot.document
    assert [child.id for child in after.element("g").children] == ["p", piece, "front"]
    assert after.element(piece).get("fill") == "blue"
    assert after.geometry_users(after.geometry_for(piece).id) == frozenset({piece})
    selected = select("p", piece) if command == "knife" else select("p")
    assert editor.snapshot.selection == selected
    assert editor.undo().document == document
    assert editor.redo().document == after


def test_failed_edit_after_piece_creation_prevents_partial_commit():
    document = import_svg(
        '<svg width="100" height="100"><path id="p" d="M0 0H10V10H0Z"/></svg>'
    )
    editor = Editor(document, selection=select("p"))
    tx = editor.transaction("Create piece then fail")
    _, piece = tx.cut_paths((5, -5), (5, 15))
    with pytest.raises(DocumentError, match="Unsupported attribute"):
        tx.set_attributes(piece, {"not-an-svg-attribute": "value"})
    with pytest.raises(EditRejectedError, match="failed edit"):
        tx.commit()
    assert editor.snapshot.document == document
