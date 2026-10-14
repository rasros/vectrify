"""Setting the artboard frames the artwork without changing it."""

import pytest

from vectrify.document import DocumentError, Editor, EditRejectedError, import_svg
from vectrify.document.model import EditKind, Selection

SVG = """<svg width="50" height="40" viewBox="0 0 100 80"><g id="layer">
<path id="a" fill="red" d="M10 10 L30 10 L30 30 Z"/>
<rect id="b" x="60" y="50" width="20" height="20" fill="blue"/>
</g></svg>"""


def artboard(editor, box, **options):
    with editor.transaction("Change artboard", **options) as tx:
        tx.set_artboard(*box)


def test_artboard_moves_the_viewbox_and_keeps_artwork_and_scale():
    editor = Editor(import_svg(SVG))
    before = editor.snapshot.document
    artboard(editor, (-20, 5, 300, 40))
    after = editor.snapshot.document
    assert after.artboard() == (-20, 5, 300, 40)
    # Width and height keep their ratio to the viewBox: half a unit each.
    assert after.root.get("width") == "150"
    assert after.root.get("height") == "20"
    # The artwork is untouched, so it stays where it is on the page.
    assert after.root.children == before.root.children
    assert after.geometries == before.geometries
    assert editor.undo_labels == ("Change artboard",)
    editor.undo()
    assert editor.snapshot.document == before


def test_artboard_without_viewbox_gains_one():
    editor = Editor(
        import_svg('<svg width="100" height="60"><path d="M0 0H5V5Z"/></svg>')
    )
    artboard(editor, (0, 0, 200, 60))
    root = editor.snapshot.document.root
    assert root.get("viewBox") == "0 0 200 60"
    assert (root.get("width"), root.get("height")) == ("200", "60")


def test_artboard_without_size_attributes_sets_only_the_viewbox():
    editor = Editor(import_svg('<svg viewBox="0 0 10 10"><path d="M0 0H5V5Z"/></svg>'))
    artboard(editor, (1.5, 2, 7, 8))
    root = editor.snapshot.document.root
    assert root.get("viewBox") == "1.5 2 7 8"
    assert root.get("width") is None
    assert root.get("height") is None


@pytest.mark.parametrize(
    "box", [(0, 0, 0, 10), (0, 0, 10, -1), (float("nan"), 0, 10, 10)]
)
def test_artboard_refuses_an_empty_or_invalid_box(box):
    editor = Editor(import_svg(SVG))
    with pytest.raises(DocumentError):
        artboard(editor, box)
    assert editor.snapshot.revision == 0


def test_artboard_ignores_descendant_locks_and_needs_permission():
    document = import_svg(SVG)
    editor = Editor(document, selection=Selection(object_ids=frozenset({"a"})))
    editor.set_locks("a", frozenset({EditKind.GEOMETRY.value}))
    # A locked shape stays put however the page changes.
    artboard(editor, (0, 0, 120, 80))
    assert editor.snapshot.document.artboard() == (0, 0, 120, 80)
    with pytest.raises(EditRejectedError, match="not permitted"):
        artboard(editor, (0, 0, 10, 10), allowed=frozenset({EditKind.PAINT}))
